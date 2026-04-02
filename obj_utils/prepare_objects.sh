#!/usr/bin/env bash
# prepare_objects.sh
#
# Full pipeline: copy reconstructed meshes → CoACD/URDF → compute z heights
# → update env_cfg.py → convert URDF→USD (setup_assets.py).
#
# Usage:
#   cd /home/hongyi/scalevideomanip/isaacsim_scene
#   bash prepare_objects.sh \
#       --grasp_mesh  /path/to/water_bottle/scaled_mesh.obj \
#       --target_mesh /path/to/red_bowl/scaled_mesh.obj \
#       --positions_json /path/to/object_positions.json
#
# All three arguments are required.

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
GRASP_MESH=""
TARGET_MESH=""
POSITIONS_JSON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --grasp_mesh)     GRASP_MESH="$2";     shift 2 ;;
        --target_mesh)    TARGET_MESH="$2";    shift 2 ;;
        --positions_json) POSITIONS_JSON="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$GRASP_MESH" || -z "$TARGET_MESH" || -z "$POSITIONS_JSON" ]]; then
    echo "Usage: bash prepare_objects.sh --grasp_mesh <obj> --target_mesh <obj> --positions_json <json>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="$SCRIPT_DIR/tasks/grasp_and_place/assets"

echo "============================================================"
echo " prepare_objects.sh"
echo "  grasp_mesh    : $GRASP_MESH"
echo "  target_mesh   : $TARGET_MESH"
echo "  positions_json: $POSITIONS_JSON"
echo "============================================================"

# ── Step 1: Copy meshes + sidecar files ───────────────────────────────────────
echo ""
echo "[1/6] Copying meshes to assets directories ..."

copy_mesh_with_sidecars() {
    local src_obj="$1"
    local dst_obj="$2"
    local dst_dir
    dst_dir="$(dirname "$dst_obj")"

    cp "$src_obj" "$dst_obj"
    echo "  copied: $src_obj → $dst_obj"

    local src_dir
    src_dir="$(dirname "$src_obj")"
    for ext in mtl png jpg; do
        for sidecar in "$src_dir"/*.$ext; do
            if [[ -e "$sidecar" ]]; then
                cp "$sidecar" "$dst_dir/"
                echo "  copied: $sidecar"
            fi
        done
    done
}

copy_mesh_with_sidecars "$GRASP_MESH"  "$ASSETS/grasp/frame_000000_object_mesh_grasp.obj"
copy_mesh_with_sidecars "$TARGET_MESH" "$ASSETS/target/frame_000000_object_mesh_target.obj"

# ── Step 2: Reorient meshes to Z-up ───────────────────────────────────────────
echo ""
echo "[2/6] Reorienting meshes to Z-up (-Y→Z, -90° around X) ..."
python - <<PYEOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")
from pathlib import Path
from preprocess_meshes import reorient_to_z_up

assets = Path("$ASSETS")
reorient_to_z_up(
    assets / "grasp/frame_000000_object_mesh_grasp.obj",
    assets / "grasp/frame_000000_object_mesh_grasp.obj",
)
reorient_to_z_up(
    assets / "target/frame_000000_object_mesh_target.obj",
    assets / "target/frame_000000_object_mesh_target.obj",
)
PYEOF

# ── Step 3: CoACD decomposition + URDF generation ─────────────────────────────
echo ""
echo "[3/6] Running preprocess_meshes.py (center + CoACD + URDF) ..."
cd "$SCRIPT_DIR"
python preprocess_meshes.py

# ── Step 4: Compute z heights + update env_cfg.py ─────────────────────────────
echo ""
echo "[4/6] Computing z heights and updating env_cfg.py ..."
python - <<PYEOF
import json, re, sys, numpy as np
import trimesh

TABLE_TOP_Z = 0.40   # 0.38 (table centre z) + 0.02 (half thickness)
CLEARANCE   = 0.005  # 5 mm to avoid initial penetration

assets = "$ASSETS"
grasp_coacd  = f"{assets}/grasp/coacd_allinone.obj"
target_coacd = f"{assets}/target/coacd_allinone.obj"

def compute_init_z(obj_path):
    # Mesh is Z-up with bottom at local z=0 (center_mesh guarantees this).
    # The root prim origin is placed at init_pos; z_max is the height of the object.
    # Isaac Lab's USD spawn positions the prim so that init_pos.z = TABLE_TOP_Z + z_max
    # places the mesh bottom on the table (object spans [TABLE_TOP, TABLE_TOP+z_max]).
    m = trimesh.load(obj_path, force="mesh")
    z_max = float(m.vertices[:, 2].max())
    return TABLE_TOP_Z + z_max + CLEARANCE

bottle_z = compute_init_z(grasp_coacd)
bowl_z   = compute_init_z(target_coacd)
print(f"  bottle init_z = {bottle_z:.4f} m")
print(f"  bowl   init_z = {bowl_z:.4f} m")

# Read XY positions from JSON (first label = grasp/origin, second = target)
with open("$POSITIONS_JSON") as f:
    pos = json.load(f)
labels      = list(pos.keys())
grasp_label  = labels[0]
target_label = labels[1] if len(labels) > 1 else None

bowl_x = pos[target_label]["table_x"] if target_label else 0.0
bowl_y = pos[target_label]["table_y"] if target_label else 0.0
print(f"  bowl table offset: x={bowl_x:+.3f}  y={bowl_y:+.3f} m  (rel to '{grasp_label}')")

# Patch env_cfg.py  — update the three init_pos tuples
cfg_path = "$SCRIPT_DIR/tasks/grasp_and_place/env_cfg.py"
text     = open(cfg_path).read()

def replace_pos(text, var, new_tuple):
    pat = rf"({re.escape(var)}\s*:\s*tuple\s*=\s*)\([^)]*\)"
    rep = r"\g<1>" + repr(new_tuple)
    result = re.sub(pat, rep, text)
    if result == text:
        print(f"  WARNING: could not find {var} in env_cfg.py", file=sys.stderr)
    return result

text = replace_pos(text, "bottle_init_pos",
                   (round(float(0.0),     4), round(float(0.0),     4), round(float(bottle_z), 4)))
text = replace_pos(text, "bowl_init_pos",
                   (round(float(bowl_x),  4), round(float(bowl_y),  4), round(float(bowl_z),   4)))

open(cfg_path, "w").write(text)
print(f"  env_cfg.py updated.")
PYEOF

# ── Step 5: Delete stale USD caches ───────────────────────────────────────────
echo ""
echo "[5/6] Removing stale USD caches ..."
rm -rf "$ASSETS/grasp/coacd_decomposed_object_one_link"
rm -rf "$ASSETS/target/coacd_decomposed_object_one_link"
echo "  removed old USD directories."

# ── Step 6: Convert URDF → USD ────────────────────────────────────────────────
echo ""
echo "[6/6] Running setup_assets.py (URDF → USD conversion) ..."
python setup_assets.py

echo ""
echo "============================================================"
echo " All done. Objects are ready for simulation."
echo "  grasp  URDF : $ASSETS/grasp/coacd_decomposed_object_one_link.urdf"
echo "  target URDF : $ASSETS/target/coacd_decomposed_object_one_link.urdf"
echo "  env_cfg.py updated with new init positions."
echo "============================================================"
