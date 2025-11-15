# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .gap_env_cfg import *
from lab_nav.tasks.manager_based.position.terrain import PIT_CFG

@configclass
class UnitreeGo2PitEnvCfg(UnitreeGo2GapEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.terrain.terrain_generator = PIT_CFG

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2PitEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeGo2PitEnvCfg_PLAY(UnitreeGo2GapEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable sensor corruption