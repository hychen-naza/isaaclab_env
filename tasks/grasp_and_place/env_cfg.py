"""Configuration for the GraspAndPlace environment."""
from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from tasks.base.base_env_cfg import BaseManipEnvCfg

# Re-export observation dimension constants (used by train.py / eval.py / vis.py)
from tasks.base.base_env_cfg import OBS_JOINT_DIM, N_POINTS, N_TABLE_POINTS, OBS_STATE_DIM, OBS_CLOUD_DIM  # noqa: F401

_ASSETS_DIR = Path(__file__).parent / "assets"


@configclass
class GraspAndPlaceEnvCfg(BaseManipEnvCfg):
    # ── Task description for Robometer VLM reward model ───────────────────────
    robometer_task: str = "grasp the bottle and place it in the bowl"

    # ── Initial poses (world frame) ───────────────────────────────────────────
    # Bottle placed at table centre; bowl offset from video estimation.
    # z values set so each object's bottom face rests on the table top (z=0.40 m).
    grasp_object_init_pos: tuple = (0.0, 0.0, 0.405)    # 20mm above table top to avoid initial penetration
    target_object_init_pos:   tuple = (0.3228, 0.0197, 0.405)

    # ── Asset URDF paths ──────────────────────────────────────────────────────
    grasp_object_urdf: str = str(_ASSETS_DIR / "grasp"  / "coacd_decomposed_object_one_link.urdf")
    target_object_urdf:   str = str(_ASSETS_DIR / "target" / "coacd_decomposed_object_one_link.urdf")


TaskEnvCfg = GraspAndPlaceEnvCfg
