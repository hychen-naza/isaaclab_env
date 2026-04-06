#!/usr/bin/env bash
# prepare_objects.sh
#
# Full pipeline: copy reconstructed meshes → CoACD/URDF → compute z heights
# → update env_cfg.py → convert URDF→USD → copy artifacts to task output dir.
#
# Usage:
#   bash /path/to/isaacsim_scene/obj_utils/prepare_objects.sh \
#       --grasp_mesh  /path/to/water_bottle/scaled_mesh.obj \
#       --target_mesh /path/to/red_bowl/scaled_mesh.obj \
#       --positions_json /path/to/object_positions.json \
#       --task_name  pourtea
#
# --task_name is optional; inferred from the grasp_mesh path if omitted.
# --task_dir  is optional; the tasks/ subdirectory (default = task_name).
# --task_output_dir is optional; defaults to scalevideomanip/output/<task_name>.

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
GRASP_MESH=""
TARGET_MESH=""
POSITIONS_JSON=""
TASK_NAME=""
TASK_DIR=""
TASK_OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --grasp_mesh)      GRASP_MESH="$2";      shift 2 ;;
        --target_mesh)     TARGET_MESH="$2";     shift 2 ;;
        --positions_json)  POSITIONS_JSON="$2";  shift 2 ;;
        --task_name)       TASK_NAME="$2";       shift 2 ;;
        --task_dir)        TASK_DIR="$2";        shift 2 ;;
        --task_output_dir) TASK_OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$GRASP_MESH" || -z "$TARGET_MESH" || -z "$POSITIONS_JSON" ]]; then
    echo "Usage: bash prepare_objects.sh --grasp_mesh <obj> --target_mesh <obj> --positions_json <json> [--task_name NAME] [--task_dir DIR]"
    exit 1
fi

# Resolve all input paths to absolute so cd later doesn't break them
GRASP_MESH="$(realpath "$GRASP_MESH")"
TARGET_MESH="$(realpath "$TARGET_MESH")"
POSITIONS_JSON="$(realpath "$POSITIONS_JSON")"

# Resolve key directories from the script's own location
OBJ_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/isaacsim_scene/obj_utils
ISAACSIM_DIR="$(cd "$OBJ_UTILS_DIR/.." && pwd)"                  # …/isaacsim_scene
SCALEVIDEOMANIP_DIR="$ISAACSIM_DIR"            # …/scalevideomanip

# Derive task name from the grasp_mesh path if not given
if [[ -z "$TASK_NAME" ]]; then
    TASK_NAME="$(python3 -c "
from pathlib import Path
p = Path('$GRASP_MESH').resolve()
parts = p.parts
try:
    idx = parts.index('output')
    print(parts[idx + 1])
except (ValueError, IndexError):
    print(p.parent.parent.parent.name)
")"
fi

# --task_dir defaults to task_name
if [[ -z "$TASK_DIR" ]]; then
    TASK_DIR="$TASK_NAME"
fi

# task output dir defaults to scalevideomanip/output/<task_name>
if [[ -z "$TASK_OUTPUT_DIR" ]]; then
    TASK_OUTPUT_DIR="$SCALEVIDEOMANIP_DIR/output/$TASK_NAME"
fi

ASSETS="$ISAACSIM_DIR/tasks/$TASK_DIR/assets"
mkdir -p "$TASK_OUTPUT_DIR" "$ASSETS/grasp" "$ASSETS/target"

echo "============================================================"
echo " prepare_objects.sh"
echo "  task_name     : $TASK_NAME"
echo "  task_dir      : $TASK_DIR"
echo "  task_output   : $TASK_OUTPUT_DIR"
echo "  assets        : $ASSETS"
echo "  grasp_mesh    : $GRASP_MESH"
echo "  target_mesh   : $TARGET_MESH"
echo "  positions_json: $POSITIONS_JSON"
echo "============================================================"

# ── Step 1: Copy meshes + sidecar files ───────────────────────────────────────
echo ""
echo "[1/5] Copying meshes to assets directories ..."

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

# ── Step 2: CoACD decomposition + URDF generation ─────────────────────────────
# Note: reorient_to_z_up is now applied upstream in generate_and_scale_meshes
# (track3d_pipeline_hamer_debug.py) before meshes are handed to this script.
echo ""
echo "[2/5] Running preprocess_meshes.py (center + CoACD + URDF) ..."
cd "$ISAACSIM_DIR"
python "$SCALEVIDEOMANIP_DIR/preprocess_meshes.py" \
    --grasp_asset  "$ASSETS/grasp/frame_000000_object_mesh_grasp.obj" \
    --target_asset "$ASSETS/target/frame_000000_object_mesh_target.obj"

# ── Step 3: Compute z heights + update env_cfg.py ─────────────────────────────
echo ""
echo "[3/5] Computing z heights and updating env_cfg.py ..."
python3 - <<PYEOF
import json, re, sys

TABLE_TOP_Z = 0.40   # 0.38 (table centre z) + 0.02 (half thickness)
CLEARANCE   = 0.005  # 5 mm gap between mesh bottom and table surface

# The URDF link origin is at the mesh bottom (z=0 after center_mesh).
# So z_init = TABLE_TOP_Z + CLEARANCE places the bottom just above the table.
# We do NOT add mesh height here — that would float the object above the table.
grasp_object_z = TABLE_TOP_Z + CLEARANCE
target_object_z = TABLE_TOP_Z + CLEARANCE
print(f"  grasp_object init_z = {grasp_object_z:.4f} m")
print(f"  target_object init_z = {target_object_z:.4f} m")

with open("$POSITIONS_JSON") as f:
    pos = json.load(f)
labels       = list(pos.keys())
grasp_label  = labels[0]
target_label = labels[1] if len(labels) > 1 else None

target_object_x = pos[target_label]["table_x"] if target_label else 0.0
target_object_y = pos[target_label]["table_y"] if target_label else 0.0
print(f"  target_object table offset: x={target_object_x:+.3f}  y={target_object_y:+.3f} m  (rel to '{grasp_label}')")

# Write to grasp_and_place/env_cfg.py (base class); task env_cfg inherits from it.
cfg_path = "$ISAACSIM_DIR/tasks/$TASK_DIR/env_cfg.py"
text     = open(cfg_path).read()

def replace_pos(text, var, new_tuple):
    pat = rf"({re.escape(var)}\s*:\s*tuple\s*=\s*)\([^)]*\)"
    rep = r"\g<1>" + repr(new_tuple)
    result = re.sub(pat, rep, text)
    if result == text:
        print(f"  WARNING: could not find {var} in env_cfg.py", file=sys.stderr)
    return result

text = replace_pos(text, "grasp_object_init_pos",
                   (round(float(0.0),    4), round(float(0.0),    4), round(float(grasp_object_z), 4)))
text = replace_pos(text, "target_object_init_pos",
                   (round(float(target_object_x), 4), round(float(target_object_y), 4), round(float(target_object_z),   4)))

open(cfg_path, "w").write(text)
print(f"  env_cfg.py updated: {cfg_path}")
PYEOF

# ── Step 4: Delete stale USD caches ───────────────────────────────────────────
echo ""
echo "[4/5] Removing stale USD caches ..."
rm -rf "$ASSETS/grasp/coacd_decomposed_object_one_link"
rm -rf "$ASSETS/target/coacd_decomposed_object_one_link"
echo "  removed old USD directories."

# ── Step 5: Convert URDF → USD ────────────────────────────────────────────────
echo ""
echo "[5/5] Running setup_assets.py (URDF → USD conversion) ..."
cd "$ISAACSIM_DIR"
python "$OBJ_UTILS_DIR/setup_assets.py" --task_dir "$TASK_DIR"

# ── Copy artifacts to task output dir ─────────────────────────────────────────
echo ""
echo "[Done] Copying artifacts to task output dir: $TASK_OUTPUT_DIR ..."

SIM_ASSETS_OUT="$TASK_OUTPUT_DIR/sim_assets"
mkdir -p "$SIM_ASSETS_OUT/grasp" "$SIM_ASSETS_OUT/target"

cp "$ASSETS/grasp/frame_000000_object_mesh_grasp_centered.obj" "$SIM_ASSETS_OUT/grasp/" 2>/dev/null || true
cp "$ASSETS/target/frame_000000_object_mesh_target_centered.obj" "$SIM_ASSETS_OUT/target/" 2>/dev/null || true
cp "$ASSETS/grasp/coacd_decomposed_object_one_link.urdf"  "$SIM_ASSETS_OUT/grasp/" 2>/dev/null || true
cp "$ASSETS/target/coacd_decomposed_object_one_link.urdf" "$SIM_ASSETS_OUT/target/" 2>/dev/null || true
cp "$POSITIONS_JSON" "$TASK_OUTPUT_DIR/object_positions.json" 2>/dev/null || true

echo "  sim_assets/grasp/  and  sim_assets/target/  written."

echo ""
echo "============================================================"
echo " All done. Objects are ready for simulation."
echo "  task          : $TASK_NAME ($TASK_DIR)"
echo "  task output   : $TASK_OUTPUT_DIR"
echo "  grasp  URDF   : $ASSETS/grasp/coacd_decomposed_object_one_link.urdf"
echo "  target URDF   : $ASSETS/target/coacd_decomposed_object_one_link.urdf"
echo "  env_cfg.py    : $ISAACSIM_DIR/tasks/$TASK_DIR/env_cfg.py  (z heights updated)"
echo "============================================================"
