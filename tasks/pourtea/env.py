"""Pourtea environment — Inspire hand picks up the bottle and pours tea into the bowl."""
from tasks.base.base_env import BaseManipEnv
from .env_cfg import PourteaEnvCfg


class PourteaEnv(BaseManipEnv):
    cfg: PourteaEnvCfg


TaskEnv = PourteaEnv
