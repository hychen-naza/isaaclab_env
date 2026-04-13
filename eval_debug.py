"""Evaluate IL policy in the GraspAndPlace Isaac Lab env, using the new RH56E2
hand and a physics-based PD wrist tracker (no kinematic teleport — collisions
with the table/bottle/bowl are respected).

Identical to eval.py; this copy exists for ad-hoc debugging tweaks.

Mechanism:
  - Before importing tasks.grasp_and_place, we monkey-patch the
    robots.inspire_hand_cfg module so that base_env.py picks up an
    RH56E2-backed make_*_hand_cfg() and the matching FINGER_JOINTS list.
  - The IL policy's wrist target is reached via set_external_force_and_torque
    (PD on the floating root), so the hand cannot penetrate the table or
    other dynamic objects.
  - The eval loop applies HAND_ROOT_OFFSET (a -π/2 Y rotation) to every
    commanded wrist quaternion so the new hand mesh aligns with the old
    inspire wrist convention used by the training trajectories.
  - The IL observation's finger-joint slot is overridden with the constant
    training-frame-0 finger pose (in OLD URDF order) so the policy isn't
    confused by the renamed/reordered RH56E2 joints.

Usage
-----
    python eval_debug.py --il_checkpoint /path/to/policy_checkpoint/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pdb
# ── 1. App launcher (MUST come before any omni / isaaclab imports) ─────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")
_HERE = Path(__file__).parent.resolve()
_RH56E2_DIR = _HERE / "RH56E2_R_2026_1_5"
_RH56E2_URDF = _RH56E2_DIR / "urdf" / "RH56E2_R_2026_1_5_abs.urdf"
# We use the "freeroot" USD with the auto-generated PhysicsFixedJoint root_joint
# stripped, so the root body can actually move under external forces (required
# for the physics-based PD wrist tracker below).
_RH56E2_USD_DIR = _RH56E2_DIR / "usd_freeroot"
_RH56E2_USD = _RH56E2_USD_DIR / _RH56E2_URDF.stem / f"{_RH56E2_URDF.stem}.usda"

# ── Physics-based wrist tracker gains ────────────────────────────────────────
# We replace kinematic write_root_pose_to_sim with a force/torque PD controller
# so that the hand respects collisions (e.g. the table) instead of teleporting
# through them.
_PD_KP_POS = 1000.0  # N/m   (high so the hand tracks the IL trajectory closely)
_PD_KD_POS = 60.0    # N·s/m (≈ 0.95 critical damping for ~1.5 kg total hand mass)
_PD_F_MAX  = 200.0   # N (max linear force per axis)
# Hand inertia is small (~5e-3 kg·m² total). Rotation gains tuned for ζ ≈ 1
# critical damping at ω ≈ 8 rad/s, with low torque clamp to avoid overshoot.
_PD_KP_ROT = 0.30    # N·m/rad
_PD_KD_ROT = 0.08    # N·m·s/rad
_PD_T_MAX  = 1.0     # N·m (max torque per axis)

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
parser.add_argument("--hold_init",    action="store_true",
                    help="Ignore IL policy and hold the frame-0 wrist pose throughout — "
                         "isolates PD controller stability from IL tracking")

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

# ── 2a. Convert RH56E2 URDF → freeroot USD (no auto root_joint) if not cached
def _strip_root_fixed_joint(usd_dir: Path):
    """Remove the auto-generated PhysicsFixedJoint 'root_joint' that the URDF
    importer adds to URDFs whose root link is the actual physical body, even
    when fix_base=False. Without this, the articulation root body is pinned
    and cannot move under external forces (only kinematic write_root_pose_to_sim
    can override the pin)."""
    physics_usda = usd_dir / _RH56E2_URDF.stem / "payloads" / "Physics" / "physics.usda"
    if not physics_usda.exists():
        print(f"[eval] WARNING: physics.usda not found at {physics_usda}")
        return
    text = physics_usda.read_text()
    import re
    pattern = r'[ \t]*def PhysicsFixedJoint "root_joint"\s*\{[^}]*\}\n?'
    new_text, n = re.subn(pattern, '', text)
    if n > 0:
        physics_usda.write_text(new_text)
        print(f"[eval] Stripped {n} root_joint block(s) from {physics_usda.name}")


if not _RH56E2_USD.exists():
    print(f"[eval] Converting {_RH56E2_URDF} → {_RH56E2_USD}")
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    UrdfConverter(UrdfConverterCfg(
        asset_path=str(_RH56E2_URDF),
        usd_dir=str(_RH56E2_USD_DIR),
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        merge_mesh=False,
    ))
    _strip_root_fixed_joint(_RH56E2_USD_DIR)
print(f"[eval] RH56E2 USD: {_RH56E2_USD}")

# ── 2b. Monkey-patch robots.inspire_hand_cfg BEFORE tasks.* import ───────────
# This makes base_env.py spawn the RH56E2 hand instead of the old inspire one,
# without editing any shared source files. base_env.py does
# `from robots.inspire_hand_cfg import FINGER_JOINTS, make_inspire_hand_cfg, ...`
# which evaluates those names AT IMPORT TIME, so we must mutate the module
# attributes BEFORE that import runs.
import robots.inspire_hand_cfg as _inspire_cfg

_RH56E2_DRIVEN_JOINTS = [
    "right_thumb_1_joint",   # thumb yaw
    "right_thumb_2_joint",   # thumb pitch (drives thumb_3, thumb_4 via mimic in URDF)
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]

_RH56E2_FINGER_JOINT_LIMITS = {
    "right_thumb_1_joint":   (0.0, 1.658),
    "right_thumb_2_joint":   (0.0, 0.62),
    "right_index_1_joint":   (0.0, 1.4381),
    "right_middle_1_joint":  (0.0, 1.4381),
    "right_ring_1_joint":    (0.0, 1.4381),
    "right_little_1_joint":  (0.0, 1.4381),
}


def _make_rh56e2_hand_cfg(prim_path, fix_base=False, init_pos=(0.0, 0.0, 0.9)):
    """Drop-in replacement for inspire_hand_cfg.make_inspire_hand_cfg()."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_RH56E2_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                fix_root_link=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={j: 0.0 for j in _RH56E2_DRIVEN_JOINTS},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=_RH56E2_DRIVEN_JOINTS,
                stiffness=1000.0,
                damping=200.0,
            ),
        },
    )


def _ensure_rh56e2_usd(fix_base: bool = False) -> str:
    """Drop-in replacement for inspire_hand_cfg.ensure_hand_usd()."""
    return str(_RH56E2_USD)


_inspire_cfg.FINGER_JOINTS = _RH56E2_DRIVEN_JOINTS
_inspire_cfg.FINGER_JOINT_LIMITS = _RH56E2_FINGER_JOINT_LIMITS
_inspire_cfg.make_inspire_hand_cfg = _make_rh56e2_hand_cfg
_inspire_cfg.ensure_hand_usd = _ensure_rh56e2_usd
print(f"[eval] Monkey-patched robots.inspire_hand_cfg → RH56E2")

# ── Orientation offset between OLD inspire root frame and NEW RH56E2 wrist ──
# Training data Eulers were recorded against the OLD hand which had a baked
# rpy=(0,-π,-π/2) in base→hand_base_link. Empirically, composing the new hand
# with R(0, -π/2, 0) on the right gives the same visual orientation.
HAND_ROOT_OFFSET = Rotation.from_euler("XYZ", [0.0, -1.5708, 0.0])


def apply_root_offset(quat_xyzw_train: np.ndarray) -> np.ndarray:
    """Compose training-frame quat with HAND_ROOT_OFFSET so the new RH56E2
    mesh ends up in the same visual orientation as the old hand.
    Returns the resulting (x, y, z, w) quat as numpy."""
    rot_train = Rotation.from_quat(quat_xyzw_train)
    rot_new = rot_train * HAND_ROOT_OFFSET
    return rot_new.as_quat()


def compute_pd_wrench(target_pos_t: "torch.Tensor",        # (3,) world
                      target_quat_xyzw_np: np.ndarray,     # (4,) xyzw world
                      cur_pos_t: "torch.Tensor",           # (3,)
                      cur_quat_xyzw_t: "torch.Tensor",     # (4,) xyzw from sim
                      cur_lin_vel_t: "torch.Tensor",       # (3,)
                      cur_ang_vel_t: "torch.Tensor",       # (3,)
                      device,
                      ):
    """PD position+orientation controller for the floating articulation root.

    Both target and current quaternions are in (x, y, z, w) order — Isaac Lab
    uses this convention for both reading (root_quat_w) and writing
    (write_root_pose_to_sim).

    Returns (force_world (3,), torque_world (3,)) torch tensors, clamped.
    """
    import torch as _torch
    # Position PD
    pos_err = target_pos_t - cur_pos_t
    F = _PD_KP_POS * pos_err - _PD_KD_POS * cur_lin_vel_t
    F = _torch.clamp(F, -_PD_F_MAX, _PD_F_MAX)

    # Orientation PD: compute rotvec error in world frame
    cur_quat_xyzw_np = cur_quat_xyzw_t.cpu().numpy().astype(np.float64)
    target_rot = Rotation.from_quat(target_quat_xyzw_np)
    cur_rot    = Rotation.from_quat(cur_quat_xyzw_np)
    err_rot    = target_rot * cur_rot.inv()      # world-frame rotation from cur → target
    err_rotvec = err_rot.as_rotvec()              # axis * angle, world frame
    err_rotvec_t = _torch.tensor(err_rotvec, dtype=_torch.float32, device=device)
    T = _PD_KP_ROT * err_rotvec_t - _PD_KD_ROT * cur_ang_vel_t
    T = _torch.clamp(T, -_PD_T_MAX, _PD_T_MAX)
    return F, T


import tasks.grasp_and_place  # noqa: F401
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg
from tasks.grasp_and_place.env_cfg import N_POINTS, N_TABLE_POINTS, OBS_CLOUD_DIM, OBS_STATE_DIM

from policy.il_utils import build_il_obs_with_scene_pcs, load_il_policy_pt, _ACTUATED_JOINT_IDX

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
_init_qxyzw_train = _init_rot.as_quat()
# Apply RH56E2 wrist orientation offset
_init_qxyzw = apply_root_offset(_init_qxyzw_train)
_init_quat = torch.tensor(
    [[_init_qxyzw[0], _init_qxyzw[1], _init_qxyzw[2], _init_qxyzw[3]]],
    dtype=torch.float32, device=device,
)
_init_joints = torch.tensor(_frame0_fingers, dtype=torch.float32, device=device).unsqueeze(0)
_init_pose = torch.cat([_init_pos_world, _init_quat.expand(args.num_envs, -1)], dim=-1)

# Build remapping from training data joint order (OLD URDF) to RH56E2 sim joint order
# Training data is in OLD URDF order:
#   [thumb_yaw, thumb_pitch, thumb_inter, thumb_distal, index_prox, index_inter,
#    middle_prox, middle_inter, ring_prox, ring_inter, pinky_prox, pinky_inter]
_TRAINING_JOINT_NAMES = [
    "thumb_proximal_yaw_joint",   "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",   "thumb_distal_joint",
    "index_proximal_joint",       "index_intermediate_joint",
    "middle_proximal_joint",      "middle_intermediate_joint",
    "ring_proximal_joint",        "ring_intermediate_joint",
    "pinky_proximal_joint",       "pinky_intermediate_joint",
]
# RH56E2 sim joint name → semantically-equivalent OLD URDF joint name
RH56E2_TO_OLD_NAME = {
    "right_thumb_1_joint":   "thumb_proximal_yaw_joint",
    "right_thumb_2_joint":   "thumb_proximal_pitch_joint",
    "right_thumb_3_joint":   "thumb_intermediate_joint",
    "right_thumb_4_joint":   "thumb_distal_joint",
    "right_index_1_joint":   "index_proximal_joint",
    "right_index_2_joint":   "index_intermediate_joint",
    "right_middle_1_joint":  "middle_proximal_joint",
    "right_middle_2_joint":  "middle_intermediate_joint",
    "right_ring_1_joint":    "ring_proximal_joint",
    "right_ring_2_joint":    "ring_intermediate_joint",
    "right_little_1_joint":  "pinky_proximal_joint",
    "right_little_2_joint":  "pinky_intermediate_joint",
}
_sim_joint_names = list(env.robot.joint_names)
# For each sim slot, find the corresponding training slot
TRAIN_TO_SIM_PERM = [
    _TRAINING_JOINT_NAMES.index(RH56E2_TO_OLD_NAME[n]) for n in _sim_joint_names
]
print(f"[eval] sim joint order: {_sim_joint_names}")
print(f"[eval] train→sim permutation: {TRAIN_TO_SIM_PERM}")

def remap_train_to_sim(joints_train: torch.Tensor) -> torch.Tensor:
    """Remap (N, 12) joints from training (OLD URDF) order → RH56E2 sim order."""
    return joints_train[:, TRAIN_TO_SIM_PERM]

_init_joints_sim = remap_train_to_sim(_init_joints)

# Frame-0 init: kinematically teleport the hand to the start pose AND set the
# finger actuator targets to the init pose. Without setting the actuator
# targets, the joint PD controllers would try to drive the fingers from their
# kinematically-written init pose back to their default target (0), generating
# large reaction torques that propagate through the articulation and destabilise
# the floating root body.
env.robot.write_root_pose_to_sim(_init_pose)
env.robot.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=device))
env.robot.write_joint_state_to_sim(_init_joints_sim, torch.zeros_like(_init_joints_sim))
env.robot.set_joint_position_target(_init_joints_sim)
env.scene.write_data_to_sim()
env.sim.step(render=False)
# Re-pin after physics step.
env.robot.write_root_pose_to_sim(_init_pose)
env.robot.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=device))
env.robot.write_joint_state_to_sim(_init_joints_sim, torch.zeros_like(_init_joints_sim))
env.robot.set_joint_position_target(_init_joints_sim)
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

    # Override the policy's finger-joint observation with the training-frame
    # frame-0 values (in OLD URDF order). The env now exposes finger joints in
    # RH56E2 sim order which the IL policy was not trained on, so we feed it
    # the constant training values it expects.
    il_obs["qpos_finger_joints"][0, -1, :] = _init_joints[0]

    obs_wrist = il_obs["qpos_wrist_pose"][0, -1].cpu().numpy()
    print(f"[eval chunk_start step={step_idx}]  obs_euler_raw={obs_euler_raw.tolist()}  obs_euler_override={prev_euler.tolist()}")

    # Get model prediction: execute first N_EXEC steps from the chunk, then re-observe
    N_EXEC = 5  # number of chunk steps to execute before re-querying the model
    with torch.no_grad():
        raw_pred = il_model.forward(il_obs)  # (N, chunk, action_dim) raw deltas
        il_act = il_model.get_action(il_obs)  # (N, chunk, action_dim) absolute targets

    # NaN guard on full chunk: hold previous pose and capture frames so the
    # video doesn't stop just because the policy went off the rails.
    if torch.isnan(il_act).any():
        print(f"[eval step={step_idx}] WARNING: NaN in model output, holding previous pose")
        n_hold = min(N_EXEC, args.max_steps - step_idx)
        for _ in range(n_hold):
            env.sim.render()
            env.scene.update(dt=dt)
            if args.save_video and env.camera is not None:
                _frame = env.camera.data.output["rgb"][args.video_env, :, :, :3].cpu().numpy().astype(np.uint8)
                rgb_frames.append(_frame)
            step_idx += 1
            episode_steps += 1
        continue

    # Finger joints: hold the init pose throughout the trajectory (no oscillation)
    all_joint_pos = _init_joints  # (N, 12) — fixed from training data frame-0

    n_to_exec = min(N_EXEC, il_act.shape[1], args.max_steps - step_idx)
    for chunk_step in range(n_to_exec):
        if args.hold_init:
            # DIAGNOSTIC: ignore IL output, hold the frame-0 pose
            pos_world = _init_pos_world.clone()
            quat_xyzw = np.asarray(_init_qxyzw, dtype=np.float64)
            euler_np = np.array(_init_euler, dtype=np.float64)
        else:
            wrist_target = il_act[:, chunk_step, :6]  # (N, 6)

            # Position: bottle-centred → world frame, clamped to workspace
            pos_world = (wrist_target[:, :3] + bottle_center.unsqueeze(0)).clamp(ws_min, ws_max)

            # Orientation: model predicts euler in OLD-hand convention; compose with
            # HAND_ROOT_OFFSET to get the equivalent quat for the RH56E2 root.
            euler_np = wrist_target[0, 3:6].cpu().numpy().astype(np.float64)
            quat_xyzw_train = Rotation.from_euler('XYZ', euler_np).as_quat()
            quat_xyzw = apply_root_offset(quat_xyzw_train)
        quat_tensor = torch.tensor(
            [[quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]],
            dtype=torch.float32, device=device,
        )
        
        # ── Physics-based PD wrist tracker (replaces kinematic teleport) ─────
        # The IL policy outputs an absolute target pose. We drive the floating
        # articulation root toward it via external force/torque, so collisions
        # (e.g. with the table or the bottle) actually stop the hand instead of
        # being kinematically penetrated.
        target_pos_t = pos_world[0]                                  # (3,)

        cur_pos_t   = wp.to_torch(env.robot.data.root_pos_w)[0]      # (3,)
        cur_quat_t  = wp.to_torch(env.robot.data.root_quat_w)[0]     # (4,) wxyz
        cur_linv_t  = wp.to_torch(env.robot.data.root_lin_vel_w)[0]  # (3,)
        cur_angv_t  = wp.to_torch(env.robot.data.root_ang_vel_w)[0]  # (3,)

        F, T = compute_pd_wrench(
            target_pos_t, np.asarray(quat_xyzw, dtype=np.float64),
            cur_pos_t, cur_quat_t, cur_linv_t, cur_angv_t, device,
        )

        # Apply external wrench to the root body (id=0). Gravity is disabled
        # on the hand via inspire_hand_cfg, so the only forces acting on the
        # root are this PD wrench + collision constraints. The wrench is in
        # WORLD frame so we must pass is_global=True (otherwise the API
        # interprets forces in the body's local frame, which would rotate them
        # by the hand's current orientation and cause the controller to
        # diverge whenever the hand isn't at identity rotation).
        forces_t  = F.view(1, 1, 3)
        torques_t = T.view(1, 1, 3)
        env.robot.set_external_force_and_torque(
            forces_t, torques_t, body_ids=[0], is_global=True,
        )

        # Fingers: drive via PD targets (NOT kinematic writes — those inject
        # impulses into the root body via constraint forces and destabilise
        # the wrist tracker).
        all_joint_pos_sim = remap_train_to_sim(all_joint_pos)
        env.robot.set_joint_position_target(all_joint_pos_sim)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)
        env.sim.render()

        # Bookkeeping for finite-difference velocity (still used by logs)
        lin_vel = (pos_world - prev_pos_world) / dt
        joint_vel = (all_joint_pos - prev_joint_pos) / dt
        root_pose = torch.cat([pos_world, quat_tensor.expand(args.num_envs, -1)], dim=-1)

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
        actual_joint_pos = wp.to_torch(env.robot.data.joint_pos)[0].cpu().numpy()  # sim order
        target_joint_pos = all_joint_pos_sim[0].cpu().numpy()  # sim order
        # Convert sim quat (xyzw) to euler for comparison
        hand_euler = Rotation.from_quat(hand_quat_w).as_euler('XYZ')
        step_log = {
            "obs_wrist_bc": il_obs["qpos_wrist_pose"][0, -1].cpu().numpy(),
            "raw_pred": raw_pred[0, chunk_step].cpu().numpy(),
            "decoded_wrist": pos_world[0].cpu().numpy(),  # use commanded pos (works in both --hold_init and IL modes)
            "pos_world": pos_world[0].cpu().numpy(),
            "euler_action": euler_np,
            "hand_pos_w": hand_pos_w,
            "hand_euler": hand_euler,
            "target_joints": target_joint_pos,
            "actual_joints": actual_joint_pos,
        }
        eval_log.append(step_log)
        # Print only the 6 actuated joints (sim ids 0-5) for clarity
        sim_names = list(env.robot.joint_names)
        actuated_print = []
        for sid in env._finger_joint_ids:
            actuated_print.append(f"{sim_names[sid][:15]}: tgt={target_joint_pos[sid]:+.3f} act={actual_joint_pos[sid]:+.3f} err={actual_joint_pos[sid]-target_joint_pos[sid]:+.3f}")
        print(f"[eval step={step_idx:3d} chunk={chunk_step}]  pos_w={[round(x,3) for x in hand_pos_w.tolist()]}")
        for line in actuated_print:
            print(f"    {line}")

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
