"""Inspire right-hand articulation config shared across all tasks.

USD conversion is done lazily via ``ensure_hand_usd(fix_base)``:
  - fix_base=True  → inspire_hand_right/      (for visualization, static base)
  - fix_base=False → inspire_hand_right_rl/   (for RL, floating base)

Call ``ensure_hand_usd()`` once inside _setup_scene() before constructing
the ArticulationCfg.
"""
from __future__ import annotations

from pathlib import Path

INSPIRE_DIR = Path(__file__).parent / "inspire"
HAND_URDF = str(INSPIRE_DIR / "inspire_hand_right.urdf")

# Cached USD output paths
# UrdfConverter places output as: usd_dir / <urdf_stem> / <urdf_stem>.usda
HAND_USD_VIZ = str(INSPIRE_DIR / "inspire_hand_right"    / "inspire_hand_right" / "inspire_hand_right.usda")
HAND_USD_RL  = str(INSPIRE_DIR / "inspire_hand_right_rl" / "inspire_hand_right" / "inspire_hand_right.usda")

# Independent (non-mimic) joints driven by actuators
FINGER_JOINTS = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

# Per-joint position limits [lower, upper] (rad), matching the URDF limits
FINGER_JOINT_LIMITS = {
    "thumb_proximal_yaw_joint":   (0.0, 1.308),
    "thumb_proximal_pitch_joint": (0.0, 0.6),
    "index_proximal_joint":       (0.0, 1.47),
    "middle_proximal_joint":      (0.0, 1.47),
    "ring_proximal_joint":        (0.0, 1.47),
    "pinky_proximal_joint":       (0.0, 1.47),
}


def ensure_hand_usd(fix_base: bool = False) -> str:
    """Return the USD path for the Inspire hand, asserting it was pre-converted.

    Run ``obj_utils/setup_assets.py`` once before training or visualisation to
    create the USD files.  The URDF importer is NOT available in the default
    headless kit experience used by training, so conversion must happen separately.
    """
    usd_path = HAND_USD_VIZ if fix_base else HAND_USD_RL
    if not Path(usd_path).exists():
        raise FileNotFoundError(
            f"Hand USD not found: {usd_path}\n"
            "Run 'python obj_utils/setup_assets.py' first to pre-convert all URDF assets."
        )
    return usd_path


def make_inspire_hand_cfg(
    prim_path: str,
    fix_base: bool = False,
    init_pos: tuple = (0.0, 0.0, 0.9),
):
    """Build and return an ArticulationCfg for the Inspire right hand.

    ``ensure_hand_usd(fix_base)`` must have been called before this.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    usd_path = HAND_USD_VIZ if fix_base else HAND_USD_RL
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                fix_root_link=False,  # dynamic root with fixed joint (like IsaacGym fix_base_link)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={j: 0.0 for j in FINGER_JOINTS},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=FINGER_JOINTS,
                stiffness=3.0,
                damping=0.1,
            ),
        },
    )
