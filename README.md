# Isaac Lab Grasp-and-Place Environment

End-to-end pipeline: video → object reconstruction → Isaac Lab RL environment
with physics-accurate object placement, depth camera observations, and PPO training.

---

## Directory structure

```
isaacsim_scene/
├── scripts/
│   └── prepare_objects.py      # asset preparation pipeline (mesh → USD)
├── preprocess_meshes.py        # center mesh + CoACD decomposition + URDF
├── obj2urdf.py                 # CoACD → URDF helper
├── setup_assets.py             # URDF → USD conversion via Isaac Lab
├── train.py                    # PPO training entry point
├── vis.py                      # scene visualisation / debugging
├── tasks/
│   └── grasp_and_place/
│       ├── env.py              # DirectRLEnv implementation
│       ├── env_cfg.py          # config (init poses, reward, camera, …)
│       └── assets/
│           ├── grasp/          # grasp-object mesh + URDF + USD
│           └── target/         # target-object mesh + URDF + USD
└── robots/
    └── inspire_hand_cfg.py     # Inspire hand articulation config
```

---

## Prerequisites

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Add the above to `~/.bashrc` so it is always set.

---

## Visualisation (`vis.py`)

`vis.py` runs the environment headlessly for a number of physics steps, then
saves images to `vis/grasp_and_place/`.

```bash
cd /home/hongyi/scalevideomanip/isaacsim_scene

# With depth camera — saves rgb.png, depth.png, pointcloud.png
python vis.py --use_camera

# Run more steps before capturing (default: 60)
python vis.py --use_camera --steps 120

# Open Isaac Sim GUI window
python vis.py --gui

# Test robot action response (writes result to /tmp/vis_action_test.txt)
python vis.py --use_camera --test_actions
```

**Output files:**

| File | Description |
|------|-------------|
| `vis/grasp_and_place/rgb.png` | RGB camera image |
| `vis/grasp_and_place/depth.png` | Depth map (plasma colormap) |
| `vis/grasp_and_place/pointcloud.png` | World-frame point cloud (bottle-centred) |
| `/tmp/vis_positions.txt` | Bottle / bowl / hand world positions |
| `/tmp/vis_action_test.txt` | Robot movement delta (with `--test_actions`) |

---

## Training (`train.py`)

PPO training using [skrl](https://skrl.readthedocs.io). Logs and checkpoints
are saved under `logs/`.

```bash
cd /home/hongyi/scalevideomanip/isaacsim_scene

# Default: 256 envs, 50M steps, headless
python train.py

# Fewer environments (lower VRAM)
python train.py --num_envs 64

# Custom step budget
python train.py --num_envs 256 --max_steps 10_000_000

# Resume from a checkpoint
python train.py --checkpoint ./logs/GraspAndPlace-v0/checkpoints/agent_XXXXX.pt

# With depth camera + point-cloud observations
python train.py --use_camera --num_envs 64

# Custom log directory
python train.py --log_dir ./my_logs
```

**Key config knobs** (edit `tasks/grasp_and_place/env_cfg.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episode_length_s` | 10.0 | Episode length in seconds |
| `success_dist` | 0.08 m | Bottle-to-bowl distance for success |
| `sparse_reward` | 100.0 | Reward on success |
| `pos_action_scale` | 0.02 m | Max hand translation per RL step |
| `use_camera` | False | Enable depth camera observations |

---

## Asset preparation (`scripts/prepare_objects.py`)

Run this whenever you have new reconstructed meshes from the video pipeline.

```bash
cd /home/hongyi/scalevideomanip/isaacsim_scene

python scripts/prepare_objects.py \
    --grasp_mesh     /path/to/water_bottle/scaled_mesh.obj \
    --target_mesh    /path/to/red_bowl/scaled_mesh.obj     \
    --positions_json /path/to/object_positions.json
```

**What it does:**

| Step | Action |
|------|--------|
| 1 | Copy meshes + `.mtl` / `.png` sidecars to `assets/{grasp,target}/` |
| 2 | Reorient meshes to Z-up (−90° around X) |
| 3 | Center mesh + CoACD decomposition + generate URDF |
| 4 | Compute `init_z` from `z_max` of CoACD mesh; read XY from JSON; patch `env_cfg.py` |
| 5 | Delete stale USD caches |
| 6 | Convert URDF → USD via Isaac Lab |

After this step `env_cfg.py` is automatically updated with the correct initial poses.

---

## Object placement

- **Table top** is at world `z = 0.40 m` (cuboid centred at `z = 0.38`, thickness `0.04 m`).
- **Init height formula:** `init_z = TABLE_TOP + z_max + CLEARANCE`
  Isaac Lab places the root prim origin at `init_pos`; the mesh is Z-up with
  its bottom at `z = 0`, so adding `z_max` ensures the mesh spans
  `[TABLE_TOP, TABLE_TOP + z_max]`.

---

## Camera

The depth camera is at `(0, −1.0, 0.9)`, facing along `+Y` with a slight
downward tilt. Resolution: **1920 × 1280**. Point cloud observations are
world-frame, bottle-centred, filtered to table surface and above
(`z_world > 0.36 m`), and down-sampled to **256 points**.

Enable the camera with `--use_camera` in both `vis.py` and `train.py`
(requires `camera_headless.kit` and `--enable_cameras`).
