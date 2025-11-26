from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn.functional as F
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def remaining_time_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the remaining time fraction in the episode."""
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    remaining_time = 1.0 - (env.episode_length_buf[:, None] * env.step_dt) / env.max_episode_length
    return remaining_time

@dataclass
class HeightScanRandCfg:
    """高度扫描随机化配置，针对 scanner 输出为 (N, num_rays, 3) 的情况."""

    # 网格尺寸：H * W = num_rays_per_scan
    grid_height: int = 31
    grid_width: int = 11

    # --------------------
    # 1. 高斯噪声
    # --------------------
    gaussian_std_xy: float = 0.02  # ~2cm
    gaussian_std_h: float = 0.1  # ~10cm

    # --------------------
    # 2. 随机 dropout
    # --------------------
    dropout_prob: float = 0.05
    missing_value: float = 0.0

    # --------------------
    # 3. 腿遮挡模型
    # --------------------
    # 距离脚在 xy 平面上的半径内的射线，有一定概率被遮挡
    leg_occlusion_radius: float = 0.1  # 10cm 半径
    leg_occlusion_prob: float = 0.7    # 在半径内被遮挡的概率

    # --------------------
    # 4. 地形边缘增强噪声
    # --------------------
    enable_edge_noise: bool = True
    edge_grad_threshold: float = 0.05
    edge_noise_std: float = 0.03


def _compute_edge_mask(height_2d: torch.Tensor, threshold: float) -> torch.Tensor:
    """对单个 env 的 2D 高度图计算 edge mask (True 表示地形边缘附近)."""
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=height_2d.device, dtype=height_2d.dtype)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=height_2d.device, dtype=height_2d.dtype)

    h = height_2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    grad_x = F.conv2d(h, sobel_x.view(1, 1, 3, 3), padding=1)
    grad_y = F.conv2d(h, sobel_y.view(1, 1, 3, 3), padding=1)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).squeeze(0).squeeze(0)
    return grad_mag > threshold


def randomized_height_scanner(
    env: ManagerBasedRLEnv,
    cfg: HeightScanRandCfg,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """对 scanner 输出 (N, R, 3) 做增强版随机化，返回 (N, R) 高度观测.

    包含：
        1. 高斯噪声
        2. 随机 dropout
        3. 腿遮挡模型
        4. 地形边缘增强噪声

    Args:
        scan_points: (N, R, 3)，最后一维为 (x, y, z).
        cfg: 随机化配置.
        asset_cfg: 用于获取脚位置的资产配置.
        sensor_cfg: 用于获取扫描器的传感器配置.

    Returns:
        heights_out: (N, R) 随机化后的高度。
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    scan_points =sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    
    if scan_points.ndim != 3 or scan_points.shape[-1] != 3:
        raise ValueError(f"scan_points must be (N, R, 3), got {scan_points.shape}")

    num_envs, num_rays, _ = scan_points.shape
    H, W = cfg.grid_height, cfg.grid_width
    assert H * W == num_rays, (
        f"grid_height * grid_width must equal num_rays_per_scan, "
        f"but {H} * {W} != {num_rays}"
    )

    device = scan_points.device

    # 提取 xyz，并 reshape 成 (N, H, W)
    xs = scan_points[..., 0].view(num_envs, H, W)
    ys = scan_points[..., 1].view(num_envs, H, W)
    zs = scan_points[..., 2].view(num_envs, H, W)  # 真正的高度

    x = xs.clone()
    y = ys.clone()
    h = zs.clone()

    # --------------------------------------------------
    # 1. 高斯噪声: h = h + N(0, sigma^2)
    # --------------------------------------------------
    if cfg.gaussian_std_xy > 0.0:
        noise_x = torch.randn_like(x) * cfg.gaussian_std_xy
        x = x + noise_x
        noise_y = torch.randn_like(y) * cfg.gaussian_std_xy
        y = y + noise_y
    if cfg.gaussian_std_h > 0.0:
        noise_h = torch.randn_like(h) * cfg.gaussian_std_h
        h = h + noise_h

    # --------------------------------------------------
    # 2. 随机 dropout: 每个 cell 以 p 概率被置为 missing_value
    # --------------------------------------------------
    if cfg.dropout_prob > 0.0:
        drop_mask = (torch.rand_like(h) < cfg.dropout_prob)
        h = torch.where(drop_mask, torch.full_like(h, cfg.missing_value), h)

    # --------------------------------------------------
    # 3. 腿遮挡模型:
    #    如果提供了 foot_positions，则在半径内的射线有一定概率被置为 missing_value
    # --------------------------------------------------
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if foot_positions is not None:
        # foot_positions: (N, num_feet, 3)
        assert foot_positions.ndim == 3 and foot_positions.shape[0] == num_envs
        num_feet = foot_positions.shape[1]

        # 把脚坐标展开成 (N, num_feet, 1, 1) 方便广播
        foot_x = foot_positions[..., 0].view(num_envs, num_feet, 1, 1)
        foot_y = foot_positions[..., 1].view(num_envs, num_feet, 1, 1)

        # xs, ys: (N, 1, H, W)
        xs_exp = xs.unsqueeze(1)
        ys_exp = ys.unsqueeze(1)

        # 计算每个 cell 到每只脚的距离 (N, num_feet, H, W)
        dist = torch.sqrt((xs_exp - foot_x) ** 2 + (ys_exp - foot_y) ** 2)

        # 距离小于半径的区域
        near_foot = dist < cfg.leg_occlusion_radius  # (N, num_feet, H, W)
        # 对多个脚取 OR
        near_any_foot = near_foot.any(dim=1)        # (N, H, W)

        # 在这些区域内再以 leg_occlusion_prob 决定是否真的遮挡
        rand_mask = torch.rand_like(h) < cfg.leg_occlusion_prob
        occlude_mask = near_any_foot & rand_mask

        h = torch.where(occlude_mask, torch.full_like(h, cfg.missing_value), h)

    # --------------------------------------------------
    # 4. 地形边缘增强噪声:
    #    使用 Sobel 计算梯度，大于阈值的地方加更大噪声
    # --------------------------------------------------
    if cfg.enable_edge_noise and cfg.edge_noise_std > 0.0:
        for env_id in range(num_envs):
            edge_mask = _compute_edge_mask(h[env_id], cfg.edge_grad_threshold)
            if edge_mask.any():
                edge_noise = torch.randn_like(h[env_id]) * cfg.edge_noise_std
                h[env_id][edge_mask] += edge_noise[edge_mask]

    x_out = x.view(num_envs, num_rays) # (N, R)
    y_out = y.view(num_envs, num_rays) # (N, R)
    heights_out = h.view(num_envs, num_rays) # (N, R)
    
    return torch.cat([x_out, y_out, heights_out], dim=-1)  # (N, R, 3)
