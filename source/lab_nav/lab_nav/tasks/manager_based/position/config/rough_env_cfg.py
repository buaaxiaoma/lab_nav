from isaaclab.utils import configclass

from .gap_env_cfg import *
from lab_nav.tasks.manager_based.position.terrain import ROUGH_CFG

@configclass
class UnitreeGo2RoughEnvCfg(UnitreeGo2GapEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_generator = ROUGH_CFG

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg":
            self.disable_zero_weight_rewards()
            
@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2GapEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable sensor corruption