"""GraspAndPlace environment — Inspire hand grasps the bottle and places it in the bowl."""
from tasks.base.base_env import BaseManipEnv
from .env_cfg import GraspAndPlaceEnvCfg


class GraspAndPlaceEnv(BaseManipEnv):
    cfg: GraspAndPlaceEnvCfg


TaskEnv = GraspAndPlaceEnv
