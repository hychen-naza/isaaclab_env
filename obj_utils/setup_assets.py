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
parser.add_argument("--task_dir", default="grasp_and_place",
                    help="Task subdirectory under tasks/ whose assets to convert (default: grasp_and_place)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless  = True
args.experience = str(BASE.parent / "minimal.kit")

launcher       = AppLauncher(args)
simulation_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

INSPIRE_DIR = BASE.parent / "robots" / "inspire"
ASSETS_DIR  = BASE.parent / "tasks" / args.task_dir / "assets"


def _capsule_for_mesh(coacd_obj: Path) -> tuple[float, float, float]:
    """Return (radius, cyl_height, translate_z) for a capsule that tightly wraps the mesh.

    The mesh has its bottom at z≈0 after center_mesh().  The capsule is aligned
    along Z so it bottoms-out at z=0 and covers the full mesh height.
    """
    import numpy as np
    verts = []
    with open(coacd_obj) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append((float(x), float(y), float(z)))
    verts = np.array(verts)
    x_ext = float(verts[:, 0].max() - verts[:, 0].min())
    y_ext = float(verts[:, 1].max() - verts[:, 1].min())
    z_ext = float(verts[:, 2].max() - verts[:, 2].min())
    radius     = max(x_ext, y_ext) / 2.0 * 1.10       # 10% margin
    cyl_height = max(0.0, z_ext - 2.0 * radius)
    translate_z = cyl_height / 2.0 + radius            # bottom of capsule at z≈0
    return radius, cyl_height, translate_z


def _inject_capsule_collision(base_usda: Path, radius: float,
                               cyl_height: float, translate_z: float) -> None:
    """Insert a capsule collision prim inside 'link_original' in base.usda."""
    text = base_usda.read_text()
    capsule_block = (
        f'\n            # Simple capsule collision for reliable GPU-PhysX contact.\n'
        f'            def Capsule "collision_capsule" (\n'
        f'                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]\n'
        f'            )\n'
        f'            {{\n'
        f'                double radius = {radius:.6f}\n'
        f'                double height = {cyl_height:.6f}\n'
        f'                uniform token axis = "Z"\n'
        f'                uniform token purpose = "guide"\n'
        f'                double3 xformOp:translate = (0.0, 0.0, {translate_z:.6f})\n'
        f'                uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        f'            }}\n'
    )
    # Insert the capsule right after the opening of "link_original"'s body
    import re
    text, n = re.subn(
        r'(def Xform "link_original"\s*\{)',
        r'\1' + capsule_block,
        text,
        count=1,
    )
    if n == 0:
        print(f"  [patch] WARNING: could not find 'link_original' in {base_usda.name}")
        return
    base_usda.write_text(text)
    print(f"  [patch] base.usda   → capsule r={radius:.4f} h={cyl_height:.4f} "
          f"tz={translate_z:.4f}: {base_usda}")


def _remove_fixed_root_joint(usd_dir: Path, stem: str, coacd_obj: Path | None = None) -> None:
    """Patch physics.usda and physx.usda to produce a pure RigidBody (no articulation).

    Newer versions of the URDF converter treat single-link URDFs as articulations,
    adding PhysicsArticulationRootAPI / NewtonArticulationRootAPI and a
    PhysicsFixedJoint that pins the object to the world.  We strip all of this
    so the object behaves as a free dynamic rigid body, matching the expected
    behavior of the grasp_and_place assets.

    Changes to physics.usda:
      - apiSchemas: keep only PhysicsRigidBodyAPI and PhysicsMassAPI
      - Remove the def PhysicsFixedJoint "root_joint" { ... } block

    Changes to physx.usda:
      - Remove the articulation apiSchemas overrides on link_original
        (delete/prepend PhysxArticulationAPI) and the physxArticulation property
    """
    import re
    physics_dir = usd_dir / stem / "payloads" / "Physics"

    # ── physics.usda ──────────────────────────────────────────────────────────
    physics_usda = physics_dir / "physics.usda"
    if physics_usda.exists():
        text = physics_usda.read_text()
        # 1. Strip articulation APIs from the apiSchemas list, keep only
        #    PhysicsRigidBodyAPI and PhysicsMassAPI.
        text = re.sub(
            r'prepend apiSchemas = \[.*?PhysicsRigidBodyAPI.*?\]',
            'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            text,
        )
        # 2. Remove the PhysicsFixedJoint "root_joint" block (no nested braces inside).
        text = re.sub(
            r'[ \t]*def PhysicsFixedJoint "root_joint"\s*\{[^}]*\}\n?',
            '',
            text,
        )
        physics_usda.write_text(text)
        print(f"  [patch] physics.usda → pure RigidBody, no FixedJoint: {physics_usda}")
    else:
        print(f"  [patch] physics.usda not found at {physics_usda}, skipping")

    # ── physx.usda ────────────────────────────────────────────────────────────
    physx_usda = physics_dir / "physx.usda"
    if physx_usda.exists():
        text = physx_usda.read_text()
        # Remove the articulation-specific override block on link_original:
        #   over "link_original" (
        #       delete apiSchemas = ["NewtonArticulationRootAPI"]
        #       prepend apiSchemas = ["PhysxArticulationAPI"]
        #   )
        #   {
        #       bool physxArticulation:enabledSelfCollisions = 0
        #       <mesh overs…>
        #   }
        # Replace with a plain over "link_original" { <mesh overs> } block.
        # Strategy: remove only the apiSchemas lines inside the prim metadata
        # and the physxArticulation property line.
        text = re.sub(
            r'[ \t]*delete apiSchemas = \["NewtonArticulationRootAPI"\]\n',
            '',
            text,
        )
        text = re.sub(
            r'[ \t]*prepend apiSchemas = \["PhysxArticulationAPI"\]\n',
            '',
            text,
        )
        text = re.sub(
            r'[ \t]*bool physxArticulation:enabledSelfCollisions = \d+\n',
            '',
            text,
        )
        # Clean up empty prim metadata parentheses: "over ... (\n        )\n" → "over ..."
        text = re.sub(r' \(\s*\)', '', text)
        physx_usda.write_text(text)
        print(f"  [patch] physx.usda  → removed PhysxArticulationAPI overrides: {physx_usda}")
    else:
        print(f"  [patch] physx.usda not found at {physx_usda}, skipping")

    # ── base.usda: inject capsule collision shape ─────────────────────────────
    if coacd_obj is not None and coacd_obj.exists():
        base_usda = usd_dir / stem / "payloads" / "base.usda"
        if base_usda.exists():
            r, h, tz = _capsule_for_mesh(coacd_obj)
            _inject_capsule_collision(base_usda, r, h, tz)
        else:
            print(f"  [patch] base.usda not found at {base_usda}, skipping capsule injection")
    else:
        print(f"  [patch] coacd_obj not provided or missing, skipping capsule injection")


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

print(f"\n=== Converting {args.task_dir} objects ===")
_GRASP_STEM  = "coacd_decomposed_object_one_link"
_TARGET_STEM = "coacd_decomposed_object_one_link"
convert(
    ASSETS_DIR / "grasp"  / f"{_GRASP_STEM}.urdf",
    ASSETS_DIR / "grasp"  / _GRASP_STEM,
    fix_base=False,
    joint_drive=None,
)
_remove_fixed_root_joint(ASSETS_DIR / "grasp"  / _GRASP_STEM,  _GRASP_STEM,
                         coacd_obj=ASSETS_DIR / "grasp"  / "coacd_allinone.obj")
convert(
    ASSETS_DIR / "target" / f"{_TARGET_STEM}.urdf",
    ASSETS_DIR / "target" / _TARGET_STEM,
    fix_base=False,
    joint_drive=None,
)
_remove_fixed_root_joint(ASSETS_DIR / "target" / _TARGET_STEM, _TARGET_STEM,
                         coacd_obj=ASSETS_DIR / "target" / "coacd_allinone.obj")

print("\n=== All assets ready ===\n")
simulation_app.close()
