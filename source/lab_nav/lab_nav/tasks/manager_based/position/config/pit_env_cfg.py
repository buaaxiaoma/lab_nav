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
        
        self.rewards.flat_orientation = None
        self.rewards.base_lin_vel_z = None
        self.rewards.base_ang_vel_xy = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2PitEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeGo2PitEnvCfg_PLAY(UnitreeGo2PitEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
