"""Visualise the RL scene (object positions) for a task.

Runs the env for a few physics steps to let objects settle, then saves:
  vis/{task}/scene.png           — 3-D scatter plot of object / hand positions
  vis/{task}/depth.png           — depth image  (only when env.camera is active)
  vis/{task}/pointcloud.png      — point-cloud  (only when env.camera is active)

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python vis.py --task grasp_and_place
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")
_GUI_KIT    = str(Path(__file__).parent / "gui.kit")

parser = argparse.ArgumentParser(description="Point-cloud visualiser")
parser.add_argument("--task",        type=str,  default="grasp_and_place")
parser.add_argument("--steps",       type=int,  default=60,
                    help="Physics steps to run before capturing")
parser.add_argument("--test_actions", action="store_true",
                    help="Apply non-zero test actions to verify robot moves")
parser.add_argument("--use_camera",  action="store_true",
                    help="Enable TiledCamera (requires camera_headless.kit)")
parser.add_argument("--gui",         action="store_true",
                    help="Open Isaac Sim GUI window")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.use_camera:
    args.experience = _CAMERA_KIT
    args.enable_cameras = True
elif args.gui:
    args.headless = False
    args.experience = _GUI_KIT
    args.visualizer = "kit"

launcher     = AppLauncher(args)
simulation_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import importlib
importlib.import_module(f"tasks.{args.task}")
env_mod  = importlib.import_module(f"tasks.{args.task}.env")
cfg_mod  = importlib.import_module(f"tasks.{args.task}.env_cfg")
TaskEnv    = env_mod.TaskEnv
TaskEnvCfg = cfg_mod.TaskEnvCfg
OBS_STATE_DIM = cfg_mod.OBS_STATE_DIM
OBS_CLOUD_DIM = cfg_mod.OBS_CLOUD_DIM

# ── Create env (single instance) ─────────────────────────────────────────────
env_cfg = TaskEnvCfg()
env_cfg.scene.num_envs = 1
if args.use_camera:
    env_cfg.use_camera = True
    env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
env = TaskEnv(cfg=env_cfg, render_mode="human" if args.gui else None)

# Let physics settle with zero actions
obs, _ = env.reset()

# Print positions immediately after reset (step 0) for height debugging
grasp_object_pos_0  = wp.to_torch(env.grasp_object.data.root_pos_w)[0].cpu().numpy()
target_object_pos_0 = wp.to_torch(env.target_object.data.root_pos_w)[0].cpu().numpy()
print(f"\nScene state at step 0 (after reset, before physics):")
print(f"  Grasp object  : {grasp_object_pos_0}  (cfg init_pos z={env_cfg.grasp_object_init_pos[2]:.4f})")
print(f"  Target object : {target_object_pos_0}  (cfg init_pos z={env_cfg.target_object_init_pos[2]:.4f})")
print(f"  Table top z = 0.40 m")

zero_actions = torch.zeros(1, env_cfg.action_space, device=env.device)
for _ in range(args.steps):
    obs, *_ = env.step(zero_actions)

# ── Optional robot-action test ─────────────────────────────────────────────
if args.test_actions:
    hand_before = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy().copy()
    test_act = torch.zeros(1, env_cfg.action_space, device=env.device)
    test_act[0, 0] =  0.0   # dx
    test_act[0, 1] =  1.0   # dy — toward grasp object (hand starts at y=-0.15)
    test_act[0, 2] =  0.5   # dz — lift
    test_act[0, 3:] = 0.5   # finger targets (half-closed)
    N_TEST = 30
    for _ in range(N_TEST):
        obs, *_ = env.step(test_act)
    hand_after = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy()
    with open("/home/runze/isaaclab_env/vis_action_test.txt", "w") as _f:
        _f.write(f"[test_actions] Hand before {N_TEST} steps: {hand_before}\n")
        _f.write(f"[test_actions] Hand after  {N_TEST} steps: {hand_after}\n")
        delta = hand_after - hand_before
        _f.write(f"[test_actions] Delta: {delta}\n")
        _f.write(f"[test_actions] Robot MOVES: {abs(delta).max() > 0.001}\n")

# Object / hand world positions (Warp arrays → Torch → NumPy)
grasp_object_pos  = wp.to_torch(env.grasp_object.data.root_pos_w)[0].cpu().numpy()
target_object_pos = wp.to_torch(env.target_object.data.root_pos_w)[0].cpu().numpy()
hand_pos          = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy()

with open("/home/runze/isaaclab_env/vis_positions.txt", "w") as _f:
    _f.write(f"Grasp object  : {grasp_object_pos}\n")
    _f.write(f"Target object : {target_object_pos}\n")
    _f.write(f"Hand          : {hand_pos}\n")

# ── Output directory ──────────────────────────────────────────────────────────
out_dir = Path("vis") / args.task
out_dir.mkdir(parents=True, exist_ok=True)

# ── Camera outputs (only when env.camera is active) ──────────────────────────
if env.camera is not None:
    depth_np = env.camera.data.output["depth"][0, :, :, 0].cpu().numpy()
    rgb_np   = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
    cam_pos  = env.camera.data.pos_w[0].cpu().numpy()

    # RGB image
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(rgb_np, origin="upper")
    ax.set_title(f"RGB image — {args.task}")
    ax.set_xlabel("pixel u"); ax.set_ylabel("pixel v")
    rgb_path = out_dir / "rgb.png"
    fig.savefig(rgb_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {rgb_path}")

    # Depth image
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(depth_np, cmap="plasma", origin="upper")
    plt.colorbar(im, ax=ax, label="Depth (m)")
    ax.set_title(f"Depth image — {args.task}")
    ax.set_xlabel("pixel u"); ax.set_ylabel("pixel v")
    depth_path = out_dir / "depth.png"
    fig.savefig(depth_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {depth_path}")

    # Point cloud — use env._compute_pointcloud() directly (same as RL observation)
    import torch
    grasp_pts_t, target_pts_t, table_pts_t = env._compute_pointcloud()
    grasp_np  = grasp_pts_t[0].cpu().numpy()   # (256, 3) grasp-object-centric
    target_np = target_pts_t[0].cpu().numpy()  # (256, 3) grasp-object-centric
    table_np  = table_pts_t[0].cpu().numpy()   # (256, 3) grasp-object-centric
    pts       = np.concatenate([grasp_np, target_np, table_np], axis=0)

    views = [
        ("Perspective",  25, -60),
        ("Top (XY)",     90,   0),
        ("Front (XZ)",    0,  90),
        ("Side (YZ)",     0,   0),
    ]
    fig = plt.figure(figsize=(14, 10))
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.scatter(grasp_np[:,0],  grasp_np[:,1],  grasp_np[:,2],
                   c="dodgerblue", s=6, alpha=0.8, label="grasp")
        ax.scatter(target_np[:,0], target_np[:,1], target_np[:,2],
                   c="tomato",     s=6, alpha=0.8, label="target")
        ax.scatter(table_np[:,0],  table_np[:,1],  table_np[:,2],
                   c="orange",     s=4, alpha=0.5, label="table")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
        if idx == 1:
            ax.legend(fontsize=8)
    fig.suptitle(f"Obs point cloud (grasp-object-centred) — {args.task}  "
                 f"grasp={len(grasp_np)} target={len(target_np)} table={len(table_np)}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    cloud_path = out_dir / "pointcloud.png"
    fig.savefig(cloud_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {cloud_path}")

# ── Interactive Plotly point-cloud visualisation ──────────────────────────────
if env.camera is not None:
    try:
        import plotly.graph_objects as go

        traces = [
            go.Scatter3d(x=grasp_np[:,0],  y=grasp_np[:,1],  z=grasp_np[:,2],
                         mode='markers', marker=dict(size=2, color='steelblue', opacity=0.8),
                         name=f'grasp object ({len(grasp_np)} pts)'),
            go.Scatter3d(x=target_np[:,0], y=target_np[:,1], z=target_np[:,2],
                         mode='markers', marker=dict(size=2, color='tomato', opacity=0.8),
                         name=f'target object ({len(target_np)} pts)'),
            go.Scatter3d(x=table_np[:,0],  y=table_np[:,1],  z=table_np[:,2],
                         mode='markers', marker=dict(size=2, color='saddlebrown', opacity=0.4),
                         name=f'table ({len(table_np)} pts)'),
        ]
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Grasp-object-centric point cloud — {args.task}  (origin = grasp object bottom)",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            height=700,
        )
        html_path = out_dir / "pc_vis.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        print(f"Saved → {html_path}  (interactive Plotly, open in browser)")
    except ImportError:
        print("[Vis] plotly not installed — skipping HTML visualisation (pip install plotly)")
    except Exception as _e:
        print(f"[Vis] interactive vis failed: {_e}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
if args.gui:
    print("[GUI] Running — close the window to exit.")
    while simulation_app.is_running():
        simulation_app.update()

env.close()
simulation_app.close()