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

import tasks.grasp_and_place  # noqa: F401  register gym env
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg

# ── Create env (single instance) ─────────────────────────────────────────────
env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = 1
if args.use_camera:
    env_cfg.use_camera = True
    env_cfg.observation_space = 29 + 256 * 3  # OBS_STATE_DIM + OBS_CLOUD_DIM
env = GraspAndPlaceEnv(cfg=env_cfg, render_mode="human" if args.gui else None)

# Let physics settle with zero actions
obs, _ = env.reset()

# Print positions immediately after reset (step 0) for height debugging
bottle_pos_0 = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()
bowl_pos_0   = wp.to_torch(env.bowl.data.root_pos_w)[0].cpu().numpy()
print(f"\nScene state at step 0 (after reset, before physics):")
print(f"  Bottle : {bottle_pos_0}  (cfg init_pos z={env_cfg.bottle_init_pos[2]:.4f})")
print(f"  Bowl   : {bowl_pos_0}  (cfg init_pos z={env_cfg.bowl_init_pos[2]:.4f})")
print(f"  Table top z = 0.40 m")

zero_actions = torch.zeros(1, env_cfg.action_space, device=env.device)
for _ in range(args.steps):
    obs, *_ = env.step(zero_actions)

# ── Optional robot-action test ─────────────────────────────────────────────
if args.test_actions:
    hand_before = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy().copy()
    # Move hand: +Y (toward bottle), +Z (lift), close fingers slightly
    test_act = torch.zeros(1, env_cfg.action_space, device=env.device)
    test_act[0, 0] =  0.0   # dx
    test_act[0, 1] =  1.0   # dy — toward bottle (hand starts at y=-0.15)
    test_act[0, 2] =  0.5   # dz — lift
    test_act[0, 3:] = 0.5   # finger targets (half-closed)
    N_TEST = 30
    for _ in range(N_TEST):
        obs, *_ = env.step(test_act)
    hand_after = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy()
    with open("/tmp/vis_action_test.txt", "w") as _f:
        _f.write(f"[test_actions] Hand before {N_TEST} steps: {hand_before}\n")
        _f.write(f"[test_actions] Hand after  {N_TEST} steps: {hand_after}\n")
        delta = hand_after - hand_before
        _f.write(f"[test_actions] Delta: {delta}\n")
        _f.write(f"[test_actions] Robot MOVES: {abs(delta).max() > 0.001}\n")

# Object / hand world positions (Warp arrays → Torch → NumPy)
bottle_pos = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()
bowl_pos   = wp.to_torch(env.bowl.data.root_pos_w)[0].cpu().numpy()
hand_pos   = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy()

with open("/tmp/vis_positions.txt", "w") as _f:
    _f.write(f"Bottle : {bottle_pos}\n")
    _f.write(f"Bowl   : {bowl_pos}\n")
    _f.write(f"Hand   : {hand_pos}\n")

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

    # Point cloud — back-project depth → world frame, zero-centred on bottle
    K  = env.camera.data.intrinsic_matrices[0].cpu().numpy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W   = depth_np.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_np.ravel()
    valid = np.isfinite(d) & (d > 0) & (d < 3.0)   # ~3 m captures scene, drops distant floor
    d = d[valid]

    # Isaac Lab TiledCamera: X-forward (depth), Y-right (pixel u), Z-up (flip v)
    x_cam =  d
    y_cam =  ((us.ravel()[valid] - cx) / fx) * d
    z_cam = -((vs.ravel()[valid] - cy) / fy) * d   # flip: pixel-v goes down, Z is up
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

    # Camera pose in world frame
    cam_pos_w  = env.camera.data.pos_w[0].cpu().numpy()          # (3,)
    quat_xyzw  = env.camera.data.quat_w_world[0].cpu().numpy()   # (x,y,z,w)
    qx, qy, qz, qw = quat_xyzw
    R_wc = np.array([                                              # world-from-cam
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])
    pts_world = (R_wc @ pts_cam.T).T + cam_pos_w   # (N, 3)

    # Keep only points at/above table level (exclude floor at world Z ≈ 0)
    table_z_thresh = 0.36   # world frame (m) — just below table bottom
    on_table = pts_world[:, 2] > table_z_thresh
    pts_world = pts_world[on_table]

    # Zero-centre around bottle (grasp object)
    pts = pts_world - bottle_pos

    print(f"  Valid cloud points: {valid.sum()} / {depth_np.size}")
    print(f"  Depth range: [{d.min():.3f}, {d.max():.3f}]")
    print(f"  World-frame X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  World-frame Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  World-frame Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    views = [
        ("Perspective",  25, -60),
        ("Top (XY)",     90,   0),
        ("Front (XZ)",    0,  90),
        ("Side (YZ)",     0,   0),
    ]
    z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    fig = plt.figure(figsize=(14, 10))
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=z_norm, cmap="viridis", s=4, alpha=0.7)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
    fig.colorbar(sc, ax=fig.axes, label="Z (normalised, bottle=0)", shrink=0.5)
    fig.suptitle(f"World-frame point cloud (bottle-centred) — {args.task}  ({valid.sum()} pts)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    cloud_path = out_dir / "pointcloud.png"
    fig.savefig(cloud_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {cloud_path}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
if args.gui:
    print("[GUI] Running — close the window to exit.")
    while simulation_app.is_running():
        simulation_app.update()

env.close()
simulation_app.close()
