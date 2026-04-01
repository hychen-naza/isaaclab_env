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
import pdb
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

env = GraspAndPlaceEnv(cfg=env_cfg, render_mode=None)
device = env.device

# Resolve actuated-joint indices (same as training)
# qpos[6:18] follows the URDF joint order (12 joints):
#  0 thumb_proximal_yaw   1 thumb_proximal_pitch  2 thumb_intermediate  3 thumb_distal
#  4 index_proximal       5 index_intermediate    6 middle_proximal      7 middle_intermediate
#  8 ring_proximal        9 ring_intermediate    10 pinky_proximal      11 pinky_intermediate
# We write all 12 directly via write_joint_state_to_sim (bypasses PD controller).

# Bottle world-frame position (origin for the bottle-centric qpos)
bottle_world = torch.tensor(env_cfg.bottle_init_pos, dtype=torch.float32, device=device)

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

for frame_idx in frames:
    qpos = robot_qpos[frame_idx].astype(np.float64)   # (18,)

    # Root position: bottle-centric → world frame (user confirmed same XYZ frame)
    pos_world = (
        torch.tensor(qpos[:3], dtype=torch.float32, device=device) + bottle_world
    ).unsqueeze(0)                                                   # (1, 3)

    # Root orientation: Euler XYZ → quat (w, x, y, z) for Isaac Lab
    rot       = Rotation.from_euler('XYZ', qpos[3:6])
    quat_xyzw = rot.as_quat()                                       # scipy → (x,y,z,w)
    quat_wxyz = torch.tensor(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)                                                   # (1, 4) w,x,y,z

    # Prepare tensors once per frame
    root_pose     = torch.cat([pos_world, quat_wxyz], dim=-1)       # (1, 7)
    zero_vel      = torch.zeros(1, 6, device=device)
    all_joint_pos = torch.tensor(qpos[6:18], dtype=torch.float32, device=device).unsqueeze(0)
    zero_jvel     = torch.zeros_like(all_joint_pos)
    print(f"root_pose: {root_pose}")
    # pdb.set_trace()
    # Write at each decimation sub-step so gravity/physics can't drift the pose
    for _ in range(env_cfg.decimation):
        env.robot.write_root_pose_to_sim(root_pose)
        env.robot.write_root_velocity_to_sim(zero_vel)
        env.robot.write_joint_state_to_sim(all_joint_pos, zero_jvel)
        env.robot.write_data_to_sim()   # flush all writes to PhysX
        env.sim.step()

    env.scene.update(dt=env.cfg.sim.dt * env_cfg.decimation)

    # Capture RGB
    rgb = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    rgb_frames.append(rgb)

print(f"[Replay] Captured {len(rgb_frames)} frames.")

# ── Save video ────────────────────────────────────────────────────────────────
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in rgb_frames:
    writer.append_data(frame)
writer.close()
print(f"[Replay] Saved → {out_path}  ({len(rgb_frames)} frames @ {args.fps} fps)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
