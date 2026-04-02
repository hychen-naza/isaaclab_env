"""Replay recorded robot states from pc_data_gravity_aligned.npy in the Isaac Sim env.

Loads the gravity-aligned, bottle-centred qpos trajectory, sets the hand pose and
finger joints directly in the sim each frame, captures RGB from the TiledCamera,
and saves a video.

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python replay.py \
        --pc_data ../output/calibrated_pourtea/pc_data_gravity_aligned.npy \
        --output   ../output/calibrated_pourtea/replay.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")

parser = argparse.ArgumentParser(description="Replay recorded qpos in Isaac Sim")
parser.add_argument("--pc_data", type=str,
                    default="../output/calibrated_pourtea/pc_data_gravity_aligned.npy")
parser.add_argument("--output",  type=str,
                    default="../output/calibrated_pourtea/replay.mp4")
parser.add_argument("--fps",     type=int,   default=30,
                    help="Output video frame rate")
parser.add_argument("--settle",  type=int,   default=30,
                    help="Physics steps to run before replay starts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.experience      = _CAMERA_KIT
args.enable_cameras  = True
args.headless        = True

launcher       = AppLauncher(args)
simulation_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
import imageio
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tasks.grasp_and_place  # noqa: F401  register gym env
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg

# ── Load trajectory ───────────────────────────────────────────────────────────
print(f"[Replay] Loading {args.pc_data}")
data = np.load(args.pc_data, allow_pickle=True).item()

robot_qpos = data['robot_qpos']           # {frame_idx: (18,) float32}
frames     = sorted(robot_qpos.keys())
print(f"[Replay] {len(frames)} frames  (indices {frames[0]}–{frames[-1]})")

# ── Create env ────────────────────────────────────────────────────────────────
env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs  = 1
env_cfg.use_camera      = True
env_cfg.observation_space = 29 + 256 * 3
# Rotate camera 180° around Z: move to opposite side (0,+1,0.9) looking in -Y
env_cfg.camera_pos = (0.0, 1.0, 0.9)
env_cfg.camera_rot = (0.0, 0.5605, 0.8284, 0.0)  # (x,y,z,w)
# Move bowl to left side from new camera view (+X = left when facing -Y)

env = GraspAndPlaceEnv(cfg=env_cfg, render_mode=None)
device = env.device

# Resolve actuated-joint indices (same as training)
# qpos[6:18] follows the URDF joint order (12 joints):
#  0 thumb_proximal_yaw   1 thumb_proximal_pitch  2 thumb_intermediate  3 thumb_distal
#  4 index_proximal       5 index_intermediate    6 middle_proximal      7 middle_intermediate
#  8 ring_proximal        9 ring_intermediate    10 pinky_proximal      11 pinky_intermediate
# We write all 12 directly via write_joint_state_to_sim (bypasses PD controller).

# Bottle world-frame position: z = z_max (top of bottle as placed in sim)
bottle_world = torch.tensor(env_cfg.bottle_init_pos, dtype=torch.float32, device=device)

# bottle_center: same XY, but z = centroid height.
# From pc_data: bottle point cloud z_max relative to centroid ≈ +0.095 m.
# So centroid is 0.095 m below z_max in world frame.
_obj_pcs = data.get('object_pcs', {})
_bottle_pcs = _obj_pcs.get('bottle', {})
if _bottle_pcs:
    _first_pts = next(iter(v for v in _bottle_pcs.values() if v is not None and len(v) > 0))
    _z_top_offset = float(_first_pts[:, 2].max())   # z_max relative to centroid in pc_data
    _y_top_offset = float(_first_pts[:, 1].max())   # y_max relative to centroid in pc_data
else:
    _z_top_offset = 0.095  # fallback
    _y_top_offset = 0.0
bottle_center = bottle_world.clone()
bottle_center[2] = bottle_world[2] - _z_top_offset
bottle_center[1] = bottle_world[1] - _y_top_offset
print(f"[Replay] bottle_world z={bottle_world[2]:.4f}, y={bottle_world[1]:.4f} "
      f"→ bottle_center z={bottle_center[2]:.4f} (z_top_offset={_z_top_offset:.4f}), "
      f"y={bottle_center[1]:.4f} (y_top_offset={_y_top_offset:.4f})")

bottle_world_np = bottle_world.cpu().numpy()

# ── Point-cloud helper ────────────────────────────────────────────────────────
def _make_combined_frame(rgb_np: np.ndarray, depth_np: np.ndarray,
                          bottle_pos_w: np.ndarray) -> np.ndarray:
    """Return a combined uint8 image: RGB on left, top-down point cloud on right."""
    K        = env.camera.data.intrinsic_matrices[0].cpu().numpy()
    fx, fy   = K[0, 0], K[1, 1]
    cx, cy   = K[0, 2], K[1, 2]
    H, W     = depth_np.shape
    us, vs   = np.meshgrid(np.arange(W), np.arange(H))
    d        = depth_np.ravel()
    valid    = np.isfinite(d) & (d > 0) & (d < 3.0)
    d        = d[valid]
    # Isaac Lab TiledCamera convention: X=forward (depth), Y=left (negated u), Z=up
    # Negate Y so that image-left maps to +X_world for this camera (at Y=+1 looking -Y).
    x_cam    =  d
    y_cam    = -((us.ravel()[valid] - cx) / fx) * d
    z_cam    = -((vs.ravel()[valid] - cy) / fy) * d
    pts_cam  = np.stack([x_cam, y_cam, z_cam], axis=1)

    cam_pos_w = env.camera.data.pos_w[0].cpu().numpy()
    qxyzw     = env.camera.data.quat_w_world[0].cpu().numpy()
    qx, qy, qz, qw = qxyzw
    R_wc = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])
    if not hasattr(_make_combined_frame, '_printed'):
        print(f"[DEBUG] cam_pos_w={cam_pos_w}")
        print(f"[DEBUG] quat_w_world(xyzw)={qxyzw}")
        print(f"[DEBUG] R_wc=\n{np.round(R_wc, 4)}")
        _make_combined_frame._printed = True
    pts_world  = (R_wc @ pts_cam.T).T + cam_pos_w
    if not hasattr(_make_combined_frame, '_printed2'):
        print(f"[DEBUG] pts_world X range: {pts_world[:,0].min():.3f} to {pts_world[:,0].max():.3f}")
        print(f"[DEBUG] pts_world Y range: {pts_world[:,1].min():.3f} to {pts_world[:,1].max():.3f}")
        print(f"[DEBUG] pts_world Z range: {pts_world[:,2].min():.3f} to {pts_world[:,2].max():.3f}")
        print(f"[DEBUG] bottle_pos_w={bottle_pos_w}")
        _make_combined_frame._printed2 = True
    on_table   = pts_world[:, 2] > 0.36          # shared floor mask
    pts        = pts_world[on_table] - bottle_pos_w

    if len(pts) > 8000:
        idx = np.random.choice(len(pts), 8000, replace=False)
        pts = pts[idx]

    z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)

    fig = plt.figure(figsize=(16, 10), dpi=100)

    # Top-left: RGB
    ax_rgb = fig.add_subplot(2, 2, 1)
    ax_rgb.imshow(rgb_np)
    ax_rgb.axis("off")
    ax_rgb.set_title("RGB")

    # Top-right: Depth
    ax_depth = fig.add_subplot(2, 2, 2)
    im = ax_depth.imshow(depth_np, cmap="plasma", origin="upper")
    plt.colorbar(im, ax=ax_depth, label="Depth (m)")
    ax_depth.axis("off")
    ax_depth.set_title("Depth")

    # Bottom: bottle-centred world-frame point cloud (spans full width)
    bowl_pos_w  = wp.to_torch(env.bowl.data.root_pos_w)[0].cpu().numpy()
    bowl_rel    = bowl_pos_w - bottle_pos_w
    ax_pc = fig.add_subplot(2, 1, 2, projection="3d")
    ax_pc.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=z_norm, cmap="viridis", s=2, alpha=0.6)
    ax_pc.scatter([bowl_rel[0]], [bowl_rel[1]], [bowl_rel[2]],
                  c='red', s=80, marker='*', label=f'bowl ({bowl_rel[0]:.2f},{bowl_rel[2]:.2f})')
    ax_pc.scatter([0], [0], [0], c='cyan', s=80, marker='*', label='bottle(0,0)')
    ax_pc.legend(fontsize=7)
    ax_pc.set_xlabel("X (m)"); ax_pc.set_ylabel("Y (m)"); ax_pc.set_zlabel("Z (m)")
    ax_pc.set_title("Point cloud (world, bottle-centred)")
    #ax_pc2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    ow, oh = fig.canvas.get_width_height()
    combined = buf.reshape(oh, ow, 4)[:, :, :3]
    plt.close(fig)
    return combined

# ── Settle physics (let objects come to rest at init poses) ──────────────────
obs, _ = env.reset()
zero_act = torch.zeros(1, env_cfg.action_space, device=device)
for _ in range(args.settle):
    env.step(zero_act)
print(f"[Replay] Physics settled ({args.settle} steps).")

# ── Replay loop ───────────────────────────────────────────────────────────────
# We bypass env.step() entirely here — it calls _apply_action() which would
# overwrite our pose writes with zero-action outputs.  Instead we write poses
# directly then step the simulator and update the scene ourselves.
rgb_frames = []
prev_pos_world  = None
prev_joint_pos  = None
dt = env.physics_dt
bottle_pw = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()

for frame_idx in frames:
    if not simulation_app.is_running():
        print(f"[Replay] simulation_app stopped at frame {frame_idx}, breaking early")
        break
    qpos = robot_qpos[frame_idx].astype(np.float64)   # (18,)

    # Root position: bottle-centric → world frame (use bottle_center, not z_max)
    pos_world = (
        torch.tensor(qpos[:3], dtype=torch.float32, device=device) + bottle_center
    ).unsqueeze(0)                                      # (1, 3)
    print(f"[Replay] pos_world={pos_world}")
    # Root orientation: Euler XYZ → quat (x, y, z, w) for Isaac Lab
    rot       = Rotation.from_euler('XYZ', qpos[3:6])
    quat_xyzw = rot.as_quat()                                       # scipy → (x,y,z,w)
    quat_wxyz = torch.tensor(
        [quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)                                                   # (1, 4) w,x,y,z

    all_joint_pos = torch.tensor(qpos[6:18], dtype=torch.float32, device=device).unsqueeze(0)

    # Compute velocities from finite differences so PhysX generates contact forces
    if prev_pos_world is not None:
        lin_vel  = (pos_world - prev_pos_world) / dt          # (1, 3)
        joint_vel = (all_joint_pos - prev_joint_pos) / dt     # (1, 12)
    else:
        lin_vel   = torch.zeros(1, 3, device=device)
        joint_vel = torch.zeros_like(all_joint_pos)
    ang_vel  = torch.zeros(1, 3, device=device)
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)           # (1, 6)

    root_pose = torch.cat([pos_world, quat_wxyz], dim=-1)      # (1, 7)

    # Write pose + velocity so PhysX sees the hand as moving (enables contact forces)
    env.robot.write_root_pose_to_sim(root_pose)
    env.robot.write_root_velocity_to_sim(root_vel)
    env.robot.write_joint_state_to_sim(all_joint_pos, joint_vel)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    # Re-pin robot to trajectory pose after physics step
    env.robot.write_root_pose_to_sim(root_pose)
    env.robot.write_root_velocity_to_sim(root_vel)
    env.scene.write_data_to_sim()
    env.sim.render()
    env.scene.update(dt=dt)

    prev_pos_world = pos_world.clone()
    prev_joint_pos = all_joint_pos.clone()

    # Collision/contact check: print if bottle moves
    bottle_pos_now = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()
    bottle_disp    = np.linalg.norm(bottle_pos_now - bottle_world_np)
    hand_pos_np    = pos_world[0].cpu().numpy()
    hand_bottle_dist = np.linalg.norm(hand_pos_np - bottle_pos_now)
    if bottle_disp > 0.005 or hand_bottle_dist < 0.15:
        print(f"[Contact] frame={frame_idx}  hand-bottle dist={hand_bottle_dist:.3f}m  "
              f"bottle moved={bottle_disp:.4f}m  bottle_pos={np.round(bottle_pos_now,3)}")

    # Capture RGB + point cloud → combined frame (use fixed initial bottle pos for centering)
    rgb      = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    depth    = env.camera.data.output["depth"][0, :, :, 0].cpu().numpy()
    combined = _make_combined_frame(rgb, depth, bottle_world_np)
    rgb_frames.append(combined)

print(f"[Replay] Captured {len(rgb_frames)} frames.", flush=True)

# ── Save video ────────────────────────────────────────────────────────────────
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in rgb_frames:
    writer.append_data(frame)
writer.close()
print(f"[Replay] Saved → {out_path}  ({len(rgb_frames)} frames @ {args.fps} fps)", flush=True)

# ── Cleanup ───────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
