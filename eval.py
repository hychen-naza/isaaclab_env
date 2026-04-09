"""Evaluate IL policy in the GraspAndPlace Isaac Lab env.

Usage
-----
    python eval.py --il_checkpoint /path/to/policy_checkpoint/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pdb
# ── 1. App launcher (MUST come before any omni / isaaclab imports) ─────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")

parser = argparse.ArgumentParser(description="Evaluate IL policy in GraspAndPlace env")
parser.add_argument("--il_checkpoint", type=str,
                    default="isaacsim_scene/policy/il_dmp/checkpoints/pourtea/best.pt",
                    help=".pt checkpoint (BC transformer + scene PCs)")
parser.add_argument("--il_model_config", type=str, default=None,
                    help="YAML for MPTransformer (default: policy/eval_il_model_config.yaml)")
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=1,
                    help="Minimum completed episodes to collect before stopping")
parser.add_argument("--max_steps",    type=int, default=25,
                    help="Hard cap on simulation steps")
parser.add_argument("--save_video",   action="store_true", default=True,
                    help="Save RGB frames as a video (enables camera)")
parser.add_argument("--video_env",    type=int, default=0,
                    help="Which env index to record")
parser.add_argument("--video_out",    type=str, default="eval_il_policy.mp4")
parser.add_argument("--video_fps",    type=int, default=30)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless       = True
args.experience     = _CAMERA_KIT
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app  = launcher.app

# ── 2. Post-launch imports ─────────────────────────────────────────────────────
import imageio
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent / "policy" / "il_dmp"))

import tasks.grasp_and_place  # noqa: F401
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg
from tasks.grasp_and_place.env_cfg import N_POINTS, N_TABLE_POINTS, OBS_CLOUD_DIM, OBS_STATE_DIM

from policy.il_utils import build_il_obs_with_scene_pcs, load_il_policy_pt

# ── 3. Build environment ───────────────────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs    = args.num_envs
env_cfg.use_camera        = True
env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
env    = GraspAndPlaceEnv(cfg=env_cfg)
device = env.device

# ── 4. Load policy ────────────────────────────────────────────────────────────

il_model = load_il_policy_pt(
    args.il_checkpoint, device,
    model_config_yaml=args.il_model_config, freeze=True,
)

# ── 5. Evaluation loop (direct pose writing, like replay.py) ─────────────────

bottle_center = torch.tensor(env_cfg.grasp_object_init_pos, dtype=torch.float32, device=device)
print(f"[eval] bottle_center (grasp_object_init_pos) = {bottle_center.cpu().tolist()}")

print(f"\n[eval] IL policy  |  {args.num_envs} env(s)"
      f"  |  collecting ≥{args.num_episodes} episodes (max {args.max_steps} steps)")
if args.save_video:
    print(f"[eval] Recording env {args.video_env} → {args.video_out}\n")

obs_dict, _ = env.reset()
dt = env.physics_dt
ws_min = torch.tensor(env_cfg.workspace_min, device=device)
ws_max = torch.tensor(env_cfg.workspace_max, device=device)

# ── Load raw trajectory for reference / debugging ────────────────────────────
_raw_traj_path = Path(__file__).parent / "policy" / "il_dmp" / "data" / "raw_data" / "pourtea" / "pc_data_gravity_aligned.npy"
_raw_traj = np.load(str(_raw_traj_path), allow_pickle=True).item()
_raw_qpos = _raw_traj["robot_qpos"]  # dict {frame_idx: (18,)} or array (T, 18)
if isinstance(_raw_qpos, dict):
    _raw_frames = sorted(_raw_qpos.keys())
    _raw_qpos_arr = np.stack([_raw_qpos[k] for k in _raw_frames], axis=0)  # (T, 18)
else:
    _raw_qpos_arr = np.array(_raw_qpos)
print(f"[eval] Loaded raw trajectory: {_raw_traj_path.name}  T={_raw_qpos_arr.shape[0]}  keys={list(_raw_traj.keys())}")

# ── Initialize hand to training data frame-0 pose ────────────────────────────
import h5py
_src_hdf5 = Path(__file__).parent / "policy" / "il_dmp" / "data" / "datasets" / "source" / "pourtea" / "trajectory_dataset.hdf5"
with h5py.File(str(_src_hdf5), "r") as _f:
    _frame0_wrist  = _f["trajectories/traj_000000/qpos_wrist_pose"][0]    # (6,) bottle-centred
    _frame0_fingers = _f["trajectories/traj_000000/qpos_finger_joints"][0]  # (12,)
print(f"[eval] Training data frame-0: wrist_pos_bc={_frame0_wrist[:3].tolist()}"
      f"  euler={_frame0_wrist[3:6].tolist()}  fingers={_frame0_fingers.tolist()}")

# Teleport hand to training frame-0: bottle-centred → world
_init_pos_world = (torch.tensor(_frame0_wrist[:3], dtype=torch.float32, device=device)
                   + bottle_center).unsqueeze(0)
_init_euler = _frame0_wrist[3:6].astype(np.float64)
_init_rot = Rotation.from_euler('XYZ', _init_euler)
_init_qxyzw = _init_rot.as_quat()
_init_quat = torch.tensor(
    [[_init_qxyzw[0], _init_qxyzw[1], _init_qxyzw[2], _init_qxyzw[3]]],
    dtype=torch.float32, device=device,
)
_init_joints = torch.tensor(_frame0_fingers, dtype=torch.float32, device=device).unsqueeze(0)
_init_pose = torch.cat([_init_pos_world, _init_quat.expand(args.num_envs, -1)], dim=-1)

env.robot.write_root_pose_to_sim(_init_pose)
env.robot.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=device))
env.robot.write_joint_state_to_sim(_init_joints, torch.zeros_like(_init_joints))
env.scene.write_data_to_sim()
env.sim.step(render=False)
# Re-pin after physics step
env.robot.write_root_pose_to_sim(_init_pose)
env.robot.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=device))
env.scene.write_data_to_sim()
# Render so the camera captures the teleported pose
env.sim.render()
env.scene.update(dt=dt)
env._hand_pos_desired[:] = _init_pos_world

# Confirm robot state matches what we wrote
_actual_pos  = wp.to_torch(env.robot.data.root_pos_w)[0]
_actual_quat = wp.to_torch(env.robot.data.root_quat_w)[0]
_actual_joints = wp.to_torch(env.robot.data.joint_pos)[0]
print(f"[eval] Frame-0 init — commanded vs actual:")
print(f"  pos_world  cmd={_init_pos_world[0].cpu().tolist()}  actual={_actual_pos.cpu().tolist()}")
print(f"  pos_bc     cmd={_frame0_wrist[:3].tolist()}")
print(f"  euler      cmd={_init_euler.tolist()}")
print(f"  quat_xyzw  cmd={_init_qxyzw.tolist()}  actual={_actual_quat.cpu().tolist()}")
print(f"  joints     cmd={_init_joints[0].cpu().tolist()}")
print(f"             actual={_actual_joints.cpu().tolist()}")

# Do a second render pass so camera buffer is fully updated with new pose
env.sim.render()
env.scene.update(dt=dt)

# Save frame-0 RGB snapshot
_frame0_path = Path(args.video_out).with_name("eval_frame0_init.png")
_rgb = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
imageio.imwrite(str(_frame0_path), _rgb)
print(f"[eval] Saved frame-0 init snapshot → {_frame0_path}")

# Refresh observations from the new hand pose
obs_dict = env._get_observations()

episodes_done    = 0
episodes_success = 0
episode_steps    = torch.zeros(args.num_envs, device=device)
step_idx         = 0
rgb_frames: list[np.ndarray] = []

prev_pos_world = _init_pos_world.clone()
prev_joint_pos = _init_joints.clone()
prev_euler = np.array(_init_euler, dtype=np.float64)  # track last commanded euler for continuity
eval_log = []  # collect per-step robot state and predictions

while episodes_done < args.num_episodes and step_idx < args.max_steps:
    policy_full = obs_dict["policy"]
    # Sanitize NaN in observations (depth camera can produce NaN points)
    policy_full = torch.nan_to_num(policy_full, nan=0.0)

    # Compute action from IL policy
    il_obs = build_il_obs_with_scene_pcs(
        policy_full, device,
        state_dim=OBS_STATE_DIM,
        n_obj_points=N_POINTS,
        n_table_points=N_TABLE_POINTS,
    )
    # Sanitize NaN in all il_obs tensors (synthetic hand PC or point clouds may have NaN)
    for k in il_obs:
        il_obs[k] = torch.nan_to_num(il_obs[k], nan=0.0)

    # Fix euler discontinuity: override observed euler with the last commanded euler.
    # The sim quat→euler conversion can land on a different branch each time,
    # but we know the hand is at the orientation we last commanded.
    obs_euler_raw = il_obs["qpos_wrist_pose"][0, -1, 3:6].cpu().numpy()
    il_obs["qpos_wrist_pose"][0, -1, 3:6] = torch.tensor(prev_euler, dtype=torch.float32, device=device)

    obs_wrist = il_obs["qpos_wrist_pose"][0, -1].cpu().numpy()
    print(f"[eval chunk_start step={step_idx}]  obs_euler_raw={obs_euler_raw.tolist()}  obs_euler_override={prev_euler.tolist()}")

    # Get model prediction: execute first N_EXEC steps from the chunk, then re-observe
    N_EXEC = 5  # number of chunk steps to execute before re-querying the model
    with torch.no_grad():
        raw_pred = il_model.forward(il_obs)  # (N, chunk, action_dim) raw deltas
        il_act = il_model.get_action(il_obs)  # (N, chunk, action_dim) absolute targets

    # NaN guard on full chunk
    if torch.isnan(il_act).any():
        print(f"[eval step={step_idx}] WARNING: NaN in model output, holding previous pose")
        step_idx += 1
        episode_steps += 1
        continue

    # Finger joints: keep current from observation (model doesn't predict fingers)
    all_joint_pos = il_obs["qpos_finger_joints"][:, -1, :]  # (N, 12)

    n_to_exec = min(N_EXEC, il_act.shape[1], args.max_steps - step_idx)
    for chunk_step in range(n_to_exec):
        wrist_target = il_act[:, chunk_step, :6]  # (N, 6)

        # Position: bottle-centred → world frame, clamped to workspace
        pos_world = (wrist_target[:, :3] + bottle_center.unsqueeze(0)).clamp(ws_min, ws_max)

        # Orientation: use model's predicted euler if action_dim >= 6
        euler_np = wrist_target[0, 3:6].cpu().numpy().astype(np.float64)
        rot = Rotation.from_euler('XYZ', euler_np)
        quat_xyzw = rot.as_quat()
        quat_tensor = torch.tensor(
            [[quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]],
            dtype=torch.float32, device=device,
        )
        
        # Compute velocities from finite differences
        lin_vel   = (pos_world - prev_pos_world) / dt
        joint_vel = (all_joint_pos - prev_joint_pos) / dt
        ang_vel   = torch.zeros(args.num_envs, 3, device=device)
        root_vel  = torch.cat([lin_vel, ang_vel], dim=-1)

        root_pose = torch.cat([pos_world, quat_tensor.expand(args.num_envs, -1)], dim=-1)

        # Write pose + velocity, step physics, re-pin
        env.robot.write_root_pose_to_sim(root_pose)
        env.robot.write_root_velocity_to_sim(root_vel)
        env.robot.write_joint_state_to_sim(all_joint_pos, joint_vel)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.robot.write_root_pose_to_sim(root_pose)
        env.robot.write_root_velocity_to_sim(root_vel)
        env.scene.write_data_to_sim()
        env.sim.render()
        env.scene.update(dt=dt)

        env._hand_pos_desired[:] = pos_world
        prev_pos_world = pos_world.clone()
        prev_joint_pos = all_joint_pos.clone()

        # Capture frame
        if args.save_video and env.camera is not None:
            _frame = env.camera.data.output["rgb"][args.video_env, :, :, :3].cpu().numpy().astype(np.uint8)
            rgb_frames.append(_frame)

        # Log: current hand pose (from sim) and action (from model)
        hand_pos_w = wp.to_torch(env.robot.data.root_pos_w)[0].cpu().numpy()
        hand_quat_w = wp.to_torch(env.robot.data.root_quat_w)[0].cpu().numpy()
        # Convert sim quat (xyzw) to euler for comparison
        hand_euler = Rotation.from_quat(hand_quat_w).as_euler('XYZ')
        step_log = {
            "obs_wrist_bc": il_obs["qpos_wrist_pose"][0, -1].cpu().numpy(),
            "raw_pred": raw_pred[0, chunk_step].cpu().numpy(),
            "decoded_wrist": wrist_target[0].cpu().numpy(),
            "pos_world": pos_world[0].cpu().numpy(),
            "euler_action": euler_np,
            "hand_pos_w": hand_pos_w,
            "hand_euler": hand_euler,
        }
        eval_log.append(step_log)
        print(f"[eval step={step_idx:3d} chunk={chunk_step}]"
              f"  hand_pos_w=[{hand_pos_w[0]:.4f}, {hand_pos_w[1]:.4f}, {hand_pos_w[2]:.4f}]"
              f"  hand_euler=[{hand_euler[0]:.4f}, {hand_euler[1]:.4f}, {hand_euler[2]:.4f}]"
              f"  action_pos_bc=[{wrist_target[0,0]:.4f}, {wrist_target[0,1]:.4f}, {wrist_target[0,2]:.4f}]"
              f"  action_euler=[{euler_np[0]:.4f}, {euler_np[1]:.4f}, {euler_np[2]:.4f}]")

        prev_euler = euler_np.copy()  # update for next chunk's unwrap

        episode_steps += 1
        step_idx      += 1

    # Refresh observations after executing the chunk
    obs_dict = env._get_observations()

    # Check termination (grasp object near target)
    grasp_pos = wp.to_torch(env.grasp_object.data.root_pos_w)
    target_obj_pos = wp.to_torch(env.target_object.data.root_pos_w)
    dist = torch.norm(grasp_pos - target_obj_pos, dim=-1)
    terminated = dist < env_cfg.success_dist
    truncated = torch.zeros_like(terminated)

    done    = terminated | truncated
    n_done  = int(done.sum().item())
    if n_done > 0:
        n_success        = int((terminated & done).sum().item())
        episodes_done    += n_done
        episodes_success += n_success
        episode_steps[done] = 0

    if step_idx % 100 == 0:
        sr = episodes_success / max(episodes_done, 1)
        print(f"  step={step_idx:5d}  episodes={episodes_done:4d}"
              f"  success={episodes_success:4d}  SR={sr:.3f}")

# ── 8. Save video ─────────────────────────────────────────────────────────────

if args.save_video and rgb_frames:
    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Saving {len(rgb_frames)} frames → {out_path}  (fps={args.video_fps})")
    with imageio.get_writer(str(out_path), fps=args.video_fps, quality=8) as writer:
        for frame in rgb_frames:
            writer.append_data(frame)
    print(f"[eval] Video saved: {out_path}")

env.close()
sim_app.close()
