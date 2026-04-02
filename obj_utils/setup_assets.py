"""Pre-convert all URDF assets to USD.

Must be run once before training or visualisation.  Uses minimal.kit so that
the URDF importer extension is available.

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python setup_assets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).parent.resolve()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pre-convert URDF → USD for all task assets")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless  = True
args.experience = str(BASE / "minimal.kit")

launcher       = AppLauncher(args)
simulation_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

INSPIRE_DIR = BASE.parent / "robots" / "inspire"
ASSETS_DIR = BASE / "tasks" / "grasp_and_place" / "assets"


def convert(urdf_path: Path, usd_dir: Path, fix_base: bool,
            joint_drive=None) -> str:
    # UrdfConverter outputs to: usd_dir / <stem> / <stem>.usda
    usd_path = usd_dir / urdf_path.stem / f"{urdf_path.stem}.usda"
    if usd_path.exists():
        print(f"  [skip]  {usd_path}")
        return str(usd_path)
    print(f"  [convert] {urdf_path} → {usd_path}")
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_dir),
        fix_base=fix_base,
        merge_fixed_joints=True,
        joint_drive=joint_drive,
        self_collision=False,
        merge_mesh=False,
    )
    UrdfConverter(cfg)
    print(f"  [done]  {usd_path}")
    return str(usd_path)


print("\n=== Converting Inspire hand (fix_base=True, for visualisation) ===")
convert(
    INSPIRE_DIR / "inspire_hand_right.urdf",
    INSPIRE_DIR / "inspire_hand_right",
    fix_base=True,
)

print("\n=== Converting Inspire hand (fix_base=False, for RL) ===")
convert(
    INSPIRE_DIR / "inspire_hand_right.urdf",
    INSPIRE_DIR / "inspire_hand_right_rl",
    fix_base=False,
)

print("\n=== Converting grasp_and_place objects ===")
convert(
    ASSETS_DIR / "grasp"  / "coacd_decomposed_object_one_link.urdf",
    ASSETS_DIR / "grasp"  / "coacd_decomposed_object_one_link",
    fix_base=False,
    joint_drive=None,
)
convert(
    ASSETS_DIR / "target" / "coacd_decomposed_object_one_link.urdf",
    ASSETS_DIR / "target" / "coacd_decomposed_object_one_link",
    fix_base=False,
    joint_drive=None,
)

print("\n=== All assets ready ===\n")
simulation_app.close()
