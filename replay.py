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
parser.add_argument("--task",    type=str,   default="pourtea",
                    help="Task name (subdirectory under tasks/); also sets default pc_data/output paths")
parser.add_argument("--pc_data", type=str,   default=None,
                    help="Path to pc_data_gravity_aligned.npy (default: ../output/<task>/pc_data_gravity_aligned.npy)")
parser.add_argument("--output",  type=str,   default=None,
                    help="Output video path (default: ../output/<task>/replay.mp4)")
parser.add_argument("--fps",     type=int,   default=30,
                    help="Output video frame rate")
parser.add_argument("--settle",  type=int,   default=30,
                    help="Physics steps to run before replay starts")
parser.add_argument("--debug_frame_dir", type=str, default=None,
                    help="Save N observation debug PNGs to this dir (tests base env point cloud)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Derive paths from task name if not specified
if args.pc_data is None:
    args.pc_data = f"../output/{args.task}/pc_data_gravity_aligned.npy"
if args.output is None:
    args.output  = f"../output/{args.task}/replay.mp4"

args.experience      = _CAMERA_KIT
args.enable_cameras  = True
args.headless        = True

launcher       = AppLauncher(args)
simulation_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import importlib
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
import imageio
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dynamically import and register the task module
importlib.import_module(f"tasks.{args.task}")           # registers gym env
env_mod  = importlib.import_module(f"tasks.{args.task}.env")
cfg_mod  = importlib.import_module(f"tasks.{args.task}.env_cfg")
TaskEnv    = env_mod.TaskEnv
TaskEnvCfg = cfg_mod.TaskEnvCfg
OBS_STATE_DIM = cfg_mod.OBS_STATE_DIM
OBS_CLOUD_DIM = cfg_mod.OBS_CLOUD_DIM

# ── Load trajectory ───────────────────────────────────────────────────────────
print(f"[Replay] task={args.task}  Loading {args.pc_data}")
data = np.load(args.pc_data, allow_pickle=True).item()

robot_qpos = data['robot_qpos']           # {frame_idx: (18,) float32}
frames     = sorted(robot_qpos.keys())
print(f"[Replay] {len(frames)} frames  (indices {frames[0]}–{frames[-1]})")

# ── Create env ────────────────────────────────────────────────────────────────
env_cfg = TaskEnvCfg()
env_cfg.scene.num_envs  = 1
env_cfg.use_camera      = True
env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
if args.debug_frame_dir:
    env_cfg.debug_frame_dir   = args.debug_frame_dir
    env_cfg.debug_frame_count = 5
env = TaskEnv(cfg=env_cfg, render_mode=None)
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
# bottle_world is the physics origin = bottom of bottle.
# bottle_center is the centroid (mid-height), which is bottom + z_top_offset
# (z_top_offset = max z of centroid-centred pc_data = half-height above centroid).
bottle_center[2] = bottle_world[2] + _z_top_offset
bottle_center[1] = bottle_world[1] + 0.05 #_y_top_offset
print(f"[Replay] bottle_world z={bottle_world[2]:.4f}, y={bottle_world[1]:.4f} "
      f"→ bottle_center z={bottle_center[2]:.4f} (z_top_offset={_z_top_offset:.4f}), "
      f"y={bottle_center[1]:.4f} (y_top_offset={_y_top_offset:.4f})")

bottle_world_np = bottle_world.cpu().numpy()

# ── Collision-shape visualization helpers ─────────────────────────────────────

def _capsule_rings(cx, cy, cz, r, h, n_pts=32):
    """Yield (xs, ys, zs) for capsule wireframe: cylinder rings + hemisphere rings + verticals."""
    theta = np.linspace(0, 2 * np.pi, n_pts + 1)
    # Cylinder section: 3 rings
    for dz in (-h / 2, 0.0, h / 2):
        yield (cx + r * np.cos(theta), cy + r * np.sin(theta),
               np.full(n_pts + 1, cz + dz))
    # Hemisphere rings (2 latitudes per cap)
    for cap_s in (1, -1):
        for sin_lat in (0.5, 0.866):
            r_ring = r * np.sqrt(max(0.0, 1 - sin_lat ** 2))
            z_ring = cz + cap_s * (h / 2 + r * sin_lat)
            yield (cx + r_ring * np.cos(theta), cy + r_ring * np.sin(theta),
                   np.full(n_pts + 1, z_ring))
    # 4 generatrix vertical lines
    for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        yield (np.array([cx + r * np.cos(ang)] * 2),
               np.array([cy + r * np.sin(ang)] * 2),
               np.array([cz - h / 2, cz + h / 2]))


def _world_to_px(pts_w, cam_pos, Rwc, fx, fy, cx_img, cy_img):
    """Project Nx3 world points → pixel (u, v); NaN for behind-camera pts."""
    pc = (Rwc.T @ (np.asarray(pts_w) - cam_pos).T).T
    d  = pc[:, 0]
    u  = np.where(d > 0.01, -pc[:, 1] / np.maximum(d, 1e-6) * fx + cx_img, np.nan)
    v  = np.where(d > 0.01, -pc[:, 2] / np.maximum(d, 1e-6) * fy + cy_img, np.nan)
    return u, v


def _draw_col_capsule(ax3d, ax2d, cx_w, cy_w, cz_w, r, h, c3d, c2d,
                       cam_pos, Rwc, fx, fy, cx_img, cy_img,
                       label=None, img_w=1920, img_h=1280,
                       draw_3d=True, draw_2d=True):
    """Draw capsule wireframe on 3D axes AND/OR projected on 2D RGB image axes.

    All coordinates (cx_w, cy_w, cz_w, cam_pos) must be in the same reference frame.
    For ax3d (bottle-ref frame): pass bottle-ref coords + shifted cam_pos.
    For ax2d (world frame): pass world coords + world cam_pos.
    Since (pts - cam_pos) is frame-invariant, a single call with either convention
    produces correct 2D projections AND correct 3D wireframe positions.
    """
    first = True
    for xs, ys, zs in _capsule_rings(cx_w, cy_w, cz_w, r, h):
        if draw_3d:
            kw3 = dict(color=c3d, linewidth=0.9, alpha=0.75)
            if first and label:
                kw3['label'] = label
                first = False
            ax3d.plot(xs, ys, zs, **kw3)
        if draw_2d:
            pts = np.stack([xs, ys, zs], axis=1)
            u, v = _world_to_px(pts, cam_pos, Rwc, fx, fy, cx_img, cy_img)
            vis = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < img_w) & (v < img_h)
            u[~vis] = np.nan; v[~vis] = np.nan
            if np.isfinite(u).sum() > 1:
                ax2d.plot(u, v, color=c2d, linewidth=1.2, alpha=0.75)


# ── Point-cloud helper ────────────────────────────────────────────────────────
def _make_combined_frame(rgb_np: np.ndarray, depth_np: np.ndarray,
                          bottle_pos_w: np.ndarray) -> np.ndarray:
    """Return a combined uint8 image with RGB, depth, 3-D point cloud and collision meshes."""
    K        = env.camera.data.intrinsic_matrices[0].cpu().numpy()
    fx, fy   = K[0, 0], K[1, 1]
    cx, cy   = K[0, 2], K[1, 2]
    H, W     = depth_np.shape
    us, vs   = np.meshgrid(np.arange(W), np.arange(H))
    d        = depth_np.ravel()
    valid    = np.isfinite(d) & (d > 0) & (d < 3.0)
    d        = d[valid]
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
    on_table   = pts_world[:, 2] > 0.36
    pts        = pts_world[on_table] - bottle_pos_w

    if len(pts) > 8000:
        idx = np.random.choice(len(pts), 8000, replace=False)
        pts = pts[idx]

    z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)

    fig = plt.figure(figsize=(20, 12), dpi=100)

    # Top-left: RGB only (clean, no overlays)
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

    # Shared data for bottom panels
    bowl_pos_w  = wp.to_torch(env.bowl.data.root_pos_w)[0].cpu().numpy()
    bowl_rel    = bowl_pos_w - bottle_pos_w
    bot_pos_now = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()
    bot_rel     = bot_pos_now - bottle_pos_w  # bottle current pos in bottle-ref frame

    tz = 0.40 - bottle_pos_w[2]   # table-top z in bottle-ref frame
    tx0, tx1 = -0.6 - bottle_pos_w[0], 0.6 - bottle_pos_w[0]
    ty0, ty1 = -0.35 - bottle_pos_w[1], 0.35 - bottle_pos_w[1]
    rect_x = np.array([tx0, tx1, tx1, tx0, tx0])
    rect_y = np.array([ty0, ty0, ty1, ty1, ty0])
    cam_ref = cam_pos_w - bottle_pos_w  # camera pos in bottle-ref frame

    # Bottom-left: 3D point cloud (scatter only, no wireframes)
    ax_pc = fig.add_subplot(2, 2, 3, projection="3d")
    ax_pc.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=z_norm, cmap="viridis", s=2, alpha=0.6)
    ax_pc.scatter([bowl_rel[0]], [bowl_rel[1]], [bowl_rel[2]],
                  c='red', s=80, marker='*', label=f'bowl z={bowl_rel[2]+bottle_pos_w[2]:.3f}')
    ax_pc.scatter([bot_rel[0]], [bot_rel[1]], [bot_rel[2]],
                  c='cyan', s=80, marker='*', label=f'bottle z={bot_pos_now[2]:.3f}')
    ax_pc.legend(fontsize=7)
    ax_pc.set_xlabel("X"); ax_pc.set_ylabel("Y"); ax_pc.set_zlabel("Z")
    ax_pc.set_title("Point cloud (bottle-centred)")

    # Bottom-right: Collision mesh wireframes only
    ax_col = fig.add_subplot(2, 2, 4, projection="3d")
    # Table top outline
    ax_col.plot(rect_x, rect_y, np.full(5, tz), 'g-', linewidth=2, alpha=0.9, label='table top')
    # Bottle capsule
    _draw_col_capsule(
        ax_col, None,
        cx_w=bot_rel[0], cy_w=bot_rel[1], cz_w=bot_rel[2] + 0.157,
        r=0.055, h=0.206,
        c3d='dodgerblue', c2d='dodgerblue',
        cam_pos=cam_ref, Rwc=R_wc,
        fx=fx, fy=fy, cx_img=cx, cy_img=cy,
        label='bottle capsule',
        img_w=W, img_h=H,
        draw_3d=True, draw_2d=False,
    )
    # Bowl capsule
    _draw_col_capsule(
        ax_col, None,
        cx_w=bowl_rel[0], cy_w=bowl_rel[1], cz_w=bowl_rel[2] + 0.111,
        r=0.092, h=0.038,
        c3d='darkorange', c2d='darkorange',
        cam_pos=cam_ref, Rwc=R_wc,
        fx=fx, fy=fy, cx_img=cx, cy_img=cy,
        label='bowl capsule',
        img_w=W, img_h=H,
        draw_3d=True, draw_2d=False,
    )
    ax_col.legend(fontsize=7)
    ax_col.set_xlabel("X"); ax_col.set_ylabel("Y"); ax_col.set_zlabel("Z")
    ax_col.set_title(f"Collision capsules  bottle_z={bot_pos_now[2]:.3f}  bowl_z={bowl_pos_w[2]:.3f}")

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

# Note: GPU PhysX articulation-rigidbody contact does not generate impulses, so
# the hand passes through the bottle in simulation. We rely on table-capsule collision
# to keep objects on the table; no manual contact force is applied.

# ── Velocity sanity check ─────────────────────────────────────────────────────
# Kick the bottle at 1 m/s +X for 5 steps to verify it can actually be moved.
pos_before = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy().copy()
kick_vel = torch.zeros(1, 6, device=device); kick_vel[0, 0] = 1.0
for _k in range(5):
    env.bottle.write_root_velocity_to_sim(kick_vel)
    env.sim.step(render=False)
    env.sim.render()
    env.scene.update(dt=dt)
pos_after = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy().copy()
print(f"[Sanity] bottle pos before kick: {np.round(pos_before,4)}", flush=True)
print(f"[Sanity] bottle pos after  kick: {np.round(pos_after,4)}", flush=True)
# Reset bottle to original position
env.bottle.write_root_pose_to_sim(
    torch.tensor([[*env_cfg.bottle_init_pos, 0.0, 0.0, 0.0, 1.0]], device=device)  # xyzw identity, matches _OBJ_QUAT_NP
)
env.bottle.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
env.sim.step(render=False); env.sim.render(); env.scene.update(dt=dt)
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

    # Track bottle/bowl positions (no manual contact force — let PhysX table collision handle it)
    bottle_pos_now = wp.to_torch(env.bottle.data.root_pos_w)[0].cpu().numpy()
    bowl_pos_now   = wp.to_torch(env.bowl.data.root_pos_w)[0].cpu().numpy()
    bottle_disp    = np.linalg.norm(bottle_pos_now - bottle_world_np)
    if frame_idx % 30 == 0:
        print(f"[Replay] frame={frame_idx}  bottle_z={bottle_pos_now[2]:.4f}  "
              f"bowl_z={bowl_pos_now[2]:.4f}", flush=True)

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
