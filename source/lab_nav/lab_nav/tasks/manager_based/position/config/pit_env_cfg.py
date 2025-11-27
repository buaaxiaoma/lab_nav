# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import *
from lab_nav.tasks.manager_based.position.terrain import PIT_CFG

@configclass
class UnitreeGo2PitEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.terrain.terrain_generator = PIT_CFG
        self.commands.target_position.min_dist = 2.0
        
        self.rewards.flat_orientation.weight = -1.0
        self.rewards.base_lin_vel_z.weight = 0
        self.rewards.base_ang_vel_xy.weight = 0
        self.rewards.air_time_variance.weight = 0
        self.rewards.feet_gait.weight = 0
        self.rewards.joint_mirror.weight = 0
        self.rewards.feet_edge.weight = -5.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2PitEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeGo2PitEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = PIT_CFG
