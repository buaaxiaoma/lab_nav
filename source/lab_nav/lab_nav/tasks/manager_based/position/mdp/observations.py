from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def remaining_time_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the remaining time fraction in the episode."""
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    remaining_time = 1.0 - (env.episode_length_buf[:, None] * env.step_dt) / env.max_episode_length
    return remaining_time   