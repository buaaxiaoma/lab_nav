from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from lab_nav.tasks.manager_based.position.mdp.commands import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_pos(env: ManagerBasedRLEnv, env_ids: Sequence[int], threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired position.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded position.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    command: TerrainBasedPoseCommand = env.command_manager.get_term("target_position")
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the distance to the target position
    distance = torch.norm(command.robot_pos[env_ids, :] - command.target_pos[env_ids, :], dim=1)
    
    # robots that walked close enough to target position go to harder terrains
    move_up = distance <= threshold
    
    # robots that walked less than half of their required distance go to simpler terrains
    initial_distance = torch.norm(terrain.env_origins[env_ids, :] + asset.data.default_root_state[env_ids, :3] - command.target_pos[env_ids, :], dim=1)
    move_down = distance > initial_distance / 2.0
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

# def update_domain_randomization_events(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     event_names: list[str],
#     start_iteration: int = 2500,
# ) -> None:
#     """
#     Enables domain randomization events after a certain number of training iterations.

#     This function checks the current training iteration. If it exceeds `start_iteration`,
#     it iterates through the provided `event_names` and re-enables them in the
#     environment's event manager by retrieving their saved configurations.

#     Args:
#         env (ManagerBasedRLEnv): The environment instance.
#         env_ids (Sequence[int]): The list of environment IDs to consider (not used in this function).
#         start_iteration (int): The training iteration number to start enabling the events.
#         event_names (list[str]): A list of event names to enable.
#     """
#     # Get the current training iteration count from the runner
#     current_iteration = env.common_step_counter

#     # If the current iteration is past the starting threshold, enable the events
#     if current_iteration >= start_iteration:
#         # Check if the temporary storage for disabled events exists
#         if not hasattr(env.cfg, "_disabled_events"):
#             raise RuntimeError(
#                 "The environment configuration does not have a '_disabled_events' attribute. "
#                 "Ensure that the events were disabled and their configurations were stored properly."
#             )

#         for event_name in event_names:
#             # Check if the event is currently not active in the manager
#             if env.event_manager.find_terms(event_name) is None:
#                 # Retrieve the original event configuration from the storage
#                 original_event_cfg = env.cfg._disabled_events.get(event_name)
#                 if original_event_cfg is not None:
#                     # Add the event back to the event manager
#                     env.event_manager.set_term_cfg(event_name, original_event_cfg)
#                     print(f"Iteration {current_iteration}: Enabled domain randomization event '{event_name}'.")