from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from lab_nav.tasks.manager_based.position.mdp.commands import *

def task_reward(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
    """Compute the task reward based on the distance to the target position and the remaining time.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        Tr (float): The time window before the end of the episode to start rewarding.

    Returns:
        torch.Tensor: The computed reward tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    robot_pos = command.robot_pos
    target_pos = command.target_pos

    distance = torch.norm(robot_pos - target_pos, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0 / Tr / (1.0 + distance), 0.0)

    return reward

def exploration_reward(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
    """Compute the exploration reward based on the orientation of the robot.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        Tr (float): The time window before the end of the episode to start rewarding.

    Returns:
        torch.Tensor: The computed reward tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    robot_vel = command.robot_velocity # (num_envs, 3)
    target_vec = command.target_pos - command.robot_pos # (num_envs, 3)
    distance = torch.norm(target_vec, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr) & (distance <= 1.0)
    
    # Calculate cosine similarity
    # Dot product for the numerator
    dot_product = torch.sum(robot_vel * target_vec, dim=-1)
    # Norms for the denominator
    robot_vel_norm = torch.norm(robot_vel, dim=-1)
    target_vec_norm = torch.norm(target_vec, dim=-1)
    
    # Calculate reward using torch.where for vectorized operation
    cosine_sim = dot_product / (robot_vel_norm * target_vec_norm + 1e-8)
    
    reward = torch.where(condition, 0.0, cosine_sim)
    return reward

def stalling_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the stalling penalty based on the robot's velocity.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.

    Returns:
        torch.Tensor: The computed penalty tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.robot_velocity, dim=-1)  # (num_envs,)
    distance = torch.norm(command.robot_pos - command.target_pos, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (speed < 0.1) & (distance > 0.5)
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0, 0.0)

    return reward

def feet_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for high feet acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # (num_envs, num_feet, 3)
    penalty = torch.norm(feet_acc, dim=-1)  # (num_envs, num_feet)
    reward = torch.sum(torch.square(penalty), dim=-1)  # (num_envs,)
    return reward