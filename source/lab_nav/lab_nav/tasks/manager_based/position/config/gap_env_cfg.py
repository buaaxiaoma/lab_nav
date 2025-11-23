# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import *
from lab_nav.tasks.manager_based.position.terrain import GAP_CFG

@configclass
class UnitreeGo2GapEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.terrain.terrain_generator = GAP_CFG
        
        self.rewards.flat_orientation.weight = 0
        self.rewards.base_lin_vel_z.weight = 0
        self.rewards.feet_gait.weight = 0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2GapEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeGo2GapEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = GAP_CFG
