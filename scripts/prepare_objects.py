"""prepare_objects.py

Full pipeline: copy reconstructed meshes → CoACD/URDF → compute z heights
→ update env_cfg.py → convert URDF → USD (setup_assets.py).

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python scripts/prepare_objects.py \\
        --grasp_mesh  /path/to/water_bottle/scaled_mesh.obj \\
        --target_mesh /path/to/red_bowl/scaled_mesh.obj \\
        --positions_json /path/to/object_positions.json

All three arguments are required.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # isaacsim_scene/
ASSETS     = SCRIPT_DIR / "tasks" / "grasp_and_place" / "assets"

TABLE_TOP_Z = 0.40    # table centre z (0.38) + half thickness (0.02)
CLEARANCE   = 0.005   # 5 mm to avoid initial penetration


# ── helpers ───────────────────────────────────────────────────────────────────

def copy_mesh_with_sidecars(src: Path, dst: Path) -> None:
    """Copy an .obj file and any .mtl / .png / .jpg sidecars from its directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied: {src} → {dst}")
    for ext in ("mtl", "png", "jpg"):
        for sidecar in src.parent.glob(f"*.{ext}"):
            shutil.copy2(sidecar, dst.parent / sidecar.name)
            print(f"  copied: {sidecar}")


def compute_init_z(obj_path: Path) -> float:
    """Return init_z so the mesh bottom rests on the table surface.

    Isaac Lab places the root prim origin at init_pos.  The mesh is Z-up with
    its bottom at z ≈ 0, so init_z = TABLE_TOP + z_max + CLEARANCE puts the
    mesh top at TABLE_TOP + z_max and the bottom exactly on the table.
    """
    import trimesh
    m     = trimesh.load(str(obj_path), force="mesh")
    z_max = float(m.vertices[:, 2].max())
    return TABLE_TOP_Z + z_max + CLEARANCE


def replace_pos(text: str, var: str, new_tuple: tuple) -> str:
    pat    = rf"({re.escape(var)}\s*:\s*tuple\s*=\s*)\([^)]*\)"
    result = re.sub(pat, r"\g<1>" + repr(new_tuple), text)
    if result == text:
        print(f"  WARNING: could not find {var} in env_cfg.py", file=sys.stderr)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Isaac Sim assets from reconstructed meshes")
    parser.add_argument("--grasp_mesh",      required=True, type=Path)
    parser.add_argument("--target_mesh",     required=True, type=Path)
    parser.add_argument("--positions_json",  required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print(" prepare_objects.py")
    print(f"  grasp_mesh    : {args.grasp_mesh}")
    print(f"  target_mesh   : {args.target_mesh}")
    print(f"  positions_json: {args.positions_json}")
    print("=" * 60)

    # ── Step 1: copy meshes ──────────────────────────────────────────────────
    print("\n[1/6] Copying meshes to assets directories ...")
    copy_mesh_with_sidecars(args.grasp_mesh,  ASSETS / "grasp"  / "frame_000000_object_mesh_grasp.obj")
    copy_mesh_with_sidecars(args.target_mesh, ASSETS / "target" / "frame_000000_object_mesh_target.obj")

    # ── Step 2: reorient meshes to Z-up ─────────────────────────────────────
    print("\n[2/6] Reorienting meshes to Z-up (-Y→Z, -90° around X) ...")
    sys.path.insert(0, str(SCRIPT_DIR / "obj_utils"))
    from preprocess_meshes import reorient_to_z_up   # noqa: E402
    for obj in [
        ASSETS / "grasp"  / "frame_000000_object_mesh_grasp.obj",
        ASSETS / "target" / "frame_000000_object_mesh_target.obj",
    ]:
        reorient_to_z_up(obj, obj)
        print(f"  reoriented: {obj}")

    # ── Step 3: CoACD decomposition + URDF ──────────────────────────────────
    print("\n[3/6] Running preprocess_meshes.py (center + CoACD + URDF) ...")
    subprocess.run([sys.executable, str(SCRIPT_DIR / "obj_utils" / "preprocess_meshes.py")],
                   cwd=str(SCRIPT_DIR), check=True)

    # ── Step 4: compute z heights + update env_cfg.py ───────────────────────
    print("\n[4/6] Computing z heights and updating env_cfg.py ...")
    bottle_z = compute_init_z(ASSETS / "grasp"  / "coacd_allinone.obj")
    bowl_z   = compute_init_z(ASSETS / "target" / "coacd_allinone.obj")
    print(f"  bottle init_z = {bottle_z:.4f} m")
    print(f"  bowl   init_z = {bowl_z:.4f} m")

    with open(args.positions_json) as f:
        pos = json.load(f)
    labels       = list(pos.keys())
    grasp_label  = labels[0]
    target_label = labels[1] if len(labels) > 1 else None
    bowl_x = pos[target_label]["table_x"] if target_label else 0.0
    bowl_y = pos[target_label]["table_y"] if target_label else 0.0
    print(f"  bowl table offset: x={bowl_x:+.3f}  y={bowl_y:+.3f} m  (rel to '{grasp_label}')")

    cfg_path = SCRIPT_DIR / "tasks" / "grasp_and_place" / "env_cfg.py"
    text = cfg_path.read_text()
    text = replace_pos(text, "bottle_init_pos",
                       (round(0.0,     4), round(0.0,     4), round(bottle_z, 4)))
    text = replace_pos(text, "bowl_init_pos",
                       (round(bowl_x,  4), round(bowl_y,  4), round(bowl_z,   4)))
    cfg_path.write_text(text)
    print("  env_cfg.py updated.")

    # ── Step 5: delete stale USD caches ─────────────────────────────────────
    print("\n[5/6] Removing stale USD caches ...")
    for stale in [
        ASSETS / "grasp"  / "coacd_decomposed_object_one_link",
        ASSETS / "target" / "coacd_decomposed_object_one_link",
    ]:
        if stale.exists():
            shutil.rmtree(stale)
            print(f"  removed: {stale}")

    # ── Step 6: URDF → USD conversion ───────────────────────────────────────
    print("\n[6/6] Running setup_assets.py (URDF → USD conversion) ...")
    subprocess.run([sys.executable, str(SCRIPT_DIR / "obj_utils" / "setup_assets.py")],
                   cwd=str(SCRIPT_DIR), check=True)

    print("\n" + "=" * 60)
    print(" All done. Objects are ready for simulation.")
    print(f"  grasp  URDF : {ASSETS}/grasp/coacd_decomposed_object_one_link.urdf")
    print(f"  target URDF : {ASSETS}/target/coacd_decomposed_object_one_link.urdf")
    print("  env_cfg.py updated with new init positions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
