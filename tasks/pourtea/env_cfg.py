"""Configuration for the Pourtea environment."""
from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from tasks.base.base_env_cfg import BaseManipEnvCfg, OBS_STATE_DIM, OBS_CLOUD_DIM

_ASSETS_DIR = Path(__file__).parent / "assets"


@configclass
class PourteaEnvCfg(BaseManipEnvCfg):
    # ── Task description for Robometer VLM reward model ───────────────────────
    robometer_task: str = "pick up the bottle and pour tea into the bowl"

    # ── Initial poses (world frame) ───────────────────────────────────────────
    # Updated by prepare_objects.py for this task's specific object geometry.
    grasp_object_init_pos: tuple = (0.0, 0.0, 0.42)
    target_object_init_pos:   tuple = (0.3538, -0.0034, 0.42)

    # ── Asset URDF paths ──────────────────────────────────────────────────────
    grasp_object_urdf: str = str(_ASSETS_DIR / "grasp"  / "coacd_decomposed_object_one_link.urdf")
    target_object_urdf:   str = str(_ASSETS_DIR / "target" / "coacd_decomposed_object_one_link.urdf")


TaskEnvCfg = PourteaEnvCfg
