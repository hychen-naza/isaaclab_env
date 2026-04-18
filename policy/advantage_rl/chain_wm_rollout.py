"""Sim rollout for advantage RL: DP3 → Isaac Lab grasp_and_place env → sim-true obs.

Sim IS the world model — no video WM imagination needed. For each of N init states,
runs K stochastic DP3 samples (different diffusion seeds) from the same init.
The stochasticity + same init gives a per-init baseline for advantage computation.

Output layout:
  <output_dir>/
    init<i>_sample<k>/
      rollout.mp4              # sim RGB (for Robometer scoring)
      actions.npz              # (T, 6) executed rotvec+xyz deltas
      qpos_traj.npz            # (T, 18) DRO qpos per step
      agent_pos_traj.npz       # (T, 7) xyz + quat_xyzw (DP3 input format)
      pcs_traj.npz             # (T, 1280, 4) point cloud obs
    manifest.json              # flat list of {sample_dir, video, init_i, sample_k, ...}

The rollout logic mirrors eval_dp3.py verbatim — eval_dp3.py is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HERE_ADV   = Path(__file__).parent.resolve()                       # policy/advantage_rl/
_REPO_ROOT  = _HERE_ADV.parent.parent                               # isaacsim_scene/
_CAMERA_KIT = str(_REPO_ROOT / "camera_headless.kit")
_RH56E2_DIR = _REPO_ROOT / "RH56E2_R_2026_1_5"
_RH56E2_URDF = _RH56E2_DIR / "urdf" / "RH56E2_R_2026_1_5_abs.urdf"
_RH56E2_USD_DIR = _RH56E2_DIR / "usd_freeroot"
_RH56E2_USD = _RH56E2_USD_DIR / _RH56E2_URDF.stem / f"{_RH56E2_URDF.stem}.usda"
_DP3_ROOT = Path("/home/hongyi/scalevideomanip/3D-Diffusion-Policy/3D-Diffusion-Policy")

parser = argparse.ArgumentParser(description="DP3 sim-rollout for advantage RL")
parser.add_argument("--dp3_checkpoint",     type=str, required=True)
parser.add_argument("--output_dir",         type=str, required=True)
parser.add_argument("--num_init_states",    type=int, default=5)
parser.add_argument("--n_samples_per_init", type=int, default=4)
parser.add_argument("--max_steps",          type=int, default=250)
parser.add_argument("--rand_offset",        type=int, default=0,
                    help="Seed offset; per-init seed = rand_offset + init_i")
parser.add_argument("--obj_trans_range_xy", type=float, default=0.05,
                    help="±m XY perturbation of grasp/target object per init state")
parser.add_argument("--video_fps",          type=int, default=30)
parser.add_argument("--action_steps_per_query", type=int, default=2,
                    help="Executes first N of the n_action_steps DP3 predicts, then requeries")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.experience = _CAMERA_KIT
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

# ── Post-launch heavy imports ────────────────────────────────────────────────
import imageio
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

# ── URDF → USD (same as eval_dp3.py) ─────────────────────────────────────────
def _strip_root_fixed_joint(usd_dir: Path):
    physics_usda = usd_dir / _RH56E2_URDF.stem / "payloads" / "Physics" / "physics.usda"
    if not physics_usda.exists():
        return
    import re
    text = physics_usda.read_text()
    new_text, n = re.subn(r'[ \t]*def PhysicsFixedJoint "root_joint"\s*\{[^}]*\}\n?', '', text)
    if n > 0:
        physics_usda.write_text(new_text)

if not _RH56E2_USD.exists():
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    UrdfConverter(UrdfConverterCfg(
        asset_path=str(_RH56E2_URDF), usd_dir=str(_RH56E2_USD_DIR),
        fix_base=False, merge_fixed_joints=True, self_collision=False, merge_mesh=False,
    ))
    _strip_root_fixed_joint(_RH56E2_USD_DIR)

# ── Monkey-patch for RH56E2 hand (same as eval_dp3.py) ──────────────────────
# robots/ lives under old/ in this repo; add both to sys.path so the import resolves
# the same way it does in eval_dp3.py (which runs from the repo root).
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "old"))
import robots.inspire_hand_cfg as _inspire_cfg

_RH56E2_DRIVEN_JOINTS = [
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint", "right_little_1_joint",
]
_RH56E2_FINGER_JOINT_LIMITS = {
    "right_thumb_1_joint": (0.0, 1.658), "right_thumb_2_joint": (0.0, 0.62),
    "right_index_1_joint": (0.0, 1.4381), "right_middle_1_joint": (0.0, 1.4381),
    "right_ring_1_joint": (0.0, 1.4381), "right_little_1_joint": (0.0, 1.4381),
}

def _make_rh56e2_hand_cfg(prim_path, fix_base=False, init_pos=(0.0, 0.0, 0.9)):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_RH56E2_USD), activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, fix_root_link=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos, joint_pos={j: 0.0 for j in _RH56E2_DRIVEN_JOINTS}),
        actuators={"fingers": ImplicitActuatorCfg(
            joint_names_expr=_RH56E2_DRIVEN_JOINTS, stiffness=10.0, damping=0.2)},
    )

_inspire_cfg.FINGER_JOINTS = _RH56E2_DRIVEN_JOINTS
_inspire_cfg.FINGER_JOINT_LIMITS = _RH56E2_FINGER_JOINT_LIMITS
_inspire_cfg.make_inspire_hand_cfg = _make_rh56e2_hand_cfg
_inspire_cfg.ensure_hand_usd = lambda fix_base=False: str(_RH56E2_USD)
print(f"[rollout] Monkey-patched → RH56E2")

# ── Load DP3 policy ──────────────────────────────────────────────────────────
sys.path.insert(0, str(_DP3_ROOT))
import dill
from diffusion_policy_3d.policy.dp3 import DP3

print(f"[rollout] Loading DP3 checkpoint: {args.dp3_checkpoint}")
payload = torch.load(args.dp3_checkpoint, pickle_module=dill, map_location="cpu")
dp3_cfg = payload["cfg"]
n_obs_steps = dp3_cfg.n_obs_steps
n_action_steps = dp3_cfg.n_action_steps
print(f"[rollout] n_obs_steps={n_obs_steps}  n_action_steps={n_action_steps}  horizon={dp3_cfg.horizon}")

from hydra.utils import instantiate
policy: DP3 = instantiate(dp3_cfg.policy)
policy.load_state_dict(payload["state_dicts"]["model"])
if "normalizer" in payload["pickles"]:
    normalizer_state = dill.loads(payload["pickles"]["normalizer"])
    policy.normalizer.load_state_dict(normalizer_state.state_dict()
                                       if hasattr(normalizer_state, "state_dict")
                                       else normalizer_state)
policy.eval().to("cuda:0")
print(f"[rollout] Policy loaded ({sum(p.numel() for p in policy.parameters()):,} params)")

# ── Build environment (once) ─────────────────────────────────────────────────
import tasks.grasp_and_place  # noqa
from tasks.grasp_and_place.env import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg
from tasks.grasp_and_place.env_cfg import N_POINTS, N_TABLE_POINTS, OBS_CLOUD_DIM, OBS_STATE_DIM
from policy.il_utils import build_il_obs_with_scene_pcs

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.use_camera = True
env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
env = GraspAndPlaceEnv(cfg=env_cfg)
device = env.device

# Defaults (reference positions; actual per-init positions are default + offset)
_DEFAULT_GRASP  = torch.tensor(env_cfg.grasp_object_init_pos,  dtype=torch.float32, device=device)
_DEFAULT_TARGET = torch.tensor(env_cfg.target_object_init_pos, dtype=torch.float32, device=device)
_ID_QUAT        = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)  # xyzw
ws_min = torch.tensor(env_cfg.workspace_min, device=device)
ws_max = torch.tensor(env_cfg.workspace_max, device=device)

# ── Frame-0 init from pourtea_rh56e2 source hdf5 ────────────────────────────
import h5py
_src_hdf5 = _REPO_ROOT / "policy/il_dmp/data/datasets/source/pourtea_rh56e2/trajectory_dataset.hdf5"
if not _src_hdf5.exists():
    _src_hdf5 = _REPO_ROOT / "policy/il_dmp/data/datasets/source/pourtea/trajectory_dataset.hdf5"
with h5py.File(str(_src_hdf5), "r") as _f:
    _frame0_wrist   = _f["trajectories/traj_000000/qpos_wrist_pose"][0]
    _frame0_fingers = _f["trajectories/traj_000000/qpos_finger_joints"][0]
_init_euler  = _frame0_wrist[3:6].astype(np.float64)
_init_qxyzw  = Rotation.from_euler('XYZ', _init_euler).as_quat()
_init_joints = torch.tensor(_frame0_fingers, dtype=torch.float32, device=device).unsqueeze(0)

# ── Joint remap (training-order ↔ sim-order) ───────────────────────────────
_TRAINING_JOINT_NAMES = [
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
]
RH56E2_TO_OLD_NAME = {
    "right_thumb_1_joint": "thumb_proximal_yaw_joint",
    "right_thumb_2_joint": "thumb_proximal_pitch_joint",
    "right_thumb_3_joint": "thumb_intermediate_joint",
    "right_thumb_4_joint": "thumb_distal_joint",
    "right_index_1_joint": "index_proximal_joint",
    "right_index_2_joint": "index_intermediate_joint",
    "right_middle_1_joint": "middle_proximal_joint",
    "right_middle_2_joint": "middle_intermediate_joint",
    "right_ring_1_joint": "ring_proximal_joint",
    "right_ring_2_joint": "ring_intermediate_joint",
    "right_little_1_joint": "pinky_proximal_joint",
    "right_little_2_joint": "pinky_intermediate_joint",
}
_sim_joint_names = list(env.robot.joint_names)
TRAIN_TO_SIM_PERM = [_TRAINING_JOINT_NAMES.index(RH56E2_TO_OLD_NAME[n]) for n in _sim_joint_names]

def remap_train_to_sim(joints_train: torch.Tensor) -> torch.Tensor:
    return joints_train[:, TRAIN_TO_SIM_PERM]

_init_joints_sim = remap_train_to_sim(_init_joints)

# ── RH56E2 FK hand-PC sampler (matches training pipeline) ────────────────────
sys.path.insert(0, "/home/hongyi/scalevideomanip")
from hand_utils.rh56e2_retargeting import RH56E2HandRetargeting
_rh56e2_retarget = RH56E2HandRetargeting()
_dro2pin = np.argsort(_rh56e2_retarget.retarget2dro)
print("[rollout] Loaded RH56E2 FK sampler")

def fk_hand_pc(pos_bc_np, quat_xyzw_np, finger_joints_train_np):
    euler = Rotation.from_quat(quat_xyzw_np).as_euler("XYZ")
    dro = np.concatenate([pos_bc_np.astype(np.float64),
                          euler.astype(np.float64),
                          finger_joints_train_np.astype(np.float64)])
    pin = dro[_dro2pin]
    pts = _rh56e2_retarget.sample_pts(pin)
    if hasattr(pts, "cpu"):
        pts = pts.cpu().numpy()
    return np.asarray(pts, dtype=np.float32)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_n(pc: torch.Tensor, n: int) -> torch.Tensor:
    pc = torch.clamp(pc, -100.0, 100.0)
    if pc.shape[0] < n:
        reps = (n + pc.shape[0] - 1) // pc.shape[0]
        pc = pc.repeat(reps, 1)
    return pc[:n]

def sample_init_offsets(rng: np.random.RandomState, xy_range: float):
    """Sample random XY perturbations for grasp and target objects."""
    g = np.array([rng.uniform(-xy_range, xy_range),
                  rng.uniform(-xy_range, xy_range), 0.0], dtype=np.float32)
    t = np.array([rng.uniform(-xy_range, xy_range),
                  rng.uniform(-xy_range, xy_range), 0.0], dtype=np.float32)
    return g, t

def reset_env_with_offsets(grasp_offset_np: np.ndarray, target_offset_np: np.ndarray):
    """Reset env; write grasp/target object poses with XY offsets; teleport hand to frame-0.
    Returns the effective bottle-center world position (used for bottle-centric frame).
    """
    env.reset()
    dt = env.physics_dt

    # Object poses (identity orientation)
    grasp_offset  = torch.tensor(grasp_offset_np,  dtype=torch.float32, device=device)
    target_offset = torch.tensor(target_offset_np, dtype=torch.float32, device=device)
    grasp_pos     = (_DEFAULT_GRASP  + grasp_offset ).unsqueeze(0)
    target_pos    = (_DEFAULT_TARGET + target_offset).unsqueeze(0)
    id_quat_b     = _ID_QUAT.unsqueeze(0)
    env.grasp_object.write_root_pose_to_sim(torch.cat([grasp_pos,  id_quat_b], dim=-1))
    env.target_object.write_root_pose_to_sim(torch.cat([target_pos, id_quat_b], dim=-1))
    env.grasp_object.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    env.target_object.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    # Teleport hand to frame-0 relative to the NEW bottle position
    bottle_w      = grasp_pos[0]  # (3,) world-frame bottle center after offset
    init_pos_w    = (torch.tensor(_frame0_wrist[:3], dtype=torch.float32, device=device)
                     + bottle_w).unsqueeze(0)
    init_quat     = torch.tensor([[_init_qxyzw[0], _init_qxyzw[1],
                                   _init_qxyzw[2], _init_qxyzw[3]]],
                                 dtype=torch.float32, device=device)
    init_pose     = torch.cat([init_pos_w, init_quat], dim=-1)

    for _ in range(2):
        env.robot.write_root_pose_to_sim(init_pose)
        env.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        env.robot.write_joint_state_to_sim(_init_joints_sim, torch.zeros_like(_init_joints_sim))
        env.robot.set_joint_position_target(_init_joints_sim)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)
    env.sim.render()
    env.scene.update(dt=dt)
    env._hand_pos_desired[:] = init_pos_w
    return bottle_w  # (3,) on device

def run_rollout(bottle_w: torch.Tensor, max_steps: int, seed: int,
                action_steps_per_query: int):
    """Run one DP3 rollout from the current env state. Returns trajectory dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cur_pos_bc_np    = np.asarray(_frame0_wrist[:3], dtype=np.float64)
    cur_quat_xyzw_np = Rotation.from_euler("XYZ", _init_euler).as_quat()

    obs_history_pc    = []
    obs_history_agent = []

    rgb_frames    = []
    action_log    = []
    qpos_log      = []
    agent_pos_log = []
    pc_log        = []

    obs_dict = env._get_observations()
    dt = env.physics_dt
    step_idx = 0

    while step_idx < max_steps:
        if not sim_app.is_running():
            break

        policy_full = torch.nan_to_num(obs_dict["policy"], nan=0.0)
        il_obs = build_il_obs_with_scene_pcs(
            policy_full, device,
            state_dim=OBS_STATE_DIM, n_obj_points=N_POINTS, n_table_points=N_TABLE_POINTS,
        )
        for k in il_obs:
            il_obs[k] = torch.nan_to_num(il_obs[k], nan=0.0)

        grasp_pc_raw  = il_obs["grasp_object_pc"][0, -1]  if il_obs["grasp_object_pc"].dim()  == 4 else il_obs["grasp_object_pc"][0]
        target_pc_raw = il_obs["target_object_pc"][0, -1] if il_obs["target_object_pc"].dim() == 4 else il_obs["target_object_pc"][0]
        table_pc_raw  = il_obs["table_pc"][0, -1]         if il_obs["table_pc"].dim()         == 4 else il_obs["table_pc"][0]

        fk_pts = fk_hand_pc(cur_pos_bc_np, cur_quat_xyzw_np, _init_joints[0].cpu().numpy())
        hand_pc_raw = torch.from_numpy(fk_pts).to(device)
        hand_pc   = _ensure_n(hand_pc_raw,   512)
        grasp_pc  = _ensure_n(grasp_pc_raw,  256)
        target_pc = _ensure_n(target_pc_raw, 256)
        table_pc  = _ensure_n(table_pc_raw,  256)
        pc_3d = torch.cat([hand_pc, grasp_pc, target_pc, table_pc], dim=0)
        intensity = torch.ones(pc_3d.shape[0], 1, device=device)
        pc_4d = torch.cat([pc_3d, intensity], dim=-1)  # (1280, 4)

        cur_pos_t  = torch.tensor(cur_pos_bc_np,    dtype=torch.float32, device=device)
        cur_quat_t = torch.tensor(cur_quat_xyzw_np, dtype=torch.float32, device=device)
        agent_pos  = torch.cat([cur_pos_t, cur_quat_t])  # (7,)

        obs_history_pc.append(pc_4d)
        obs_history_agent.append(agent_pos)
        if len(obs_history_pc) > n_obs_steps:
            obs_history_pc    = obs_history_pc[-n_obs_steps:]
            obs_history_agent = obs_history_agent[-n_obs_steps:]
        while len(obs_history_pc) < n_obs_steps:
            obs_history_pc.insert(0, obs_history_pc[0])
            obs_history_agent.insert(0, obs_history_agent[0])

        dp3_obs = {
            "point_cloud": torch.stack(obs_history_pc,    dim=0).unsqueeze(0),
            "agent_pos":   torch.stack(obs_history_agent, dim=0).unsqueeze(0),
        }
        with torch.no_grad():
            result = policy.predict_action(dp3_obs)
        actions = result["action"][0]  # (n_action_steps, 6) deltas

        if torch.isnan(actions).any():
            print(f"  [step={step_idx}] NaN in DP3 output, holding pose")
            env.sim.render(); env.scene.update(dt=dt)
            if env.camera is not None:
                rgb_frames.append(env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8))
            step_idx += 1
            obs_dict = env._get_observations()
            continue

        for ai in range(min(action_steps_per_query, max_steps - step_idx)):
            wrist_delta = actions[ai]
            delta_np = wrist_delta.cpu().numpy().astype(np.float64)

            cur_pos_bc_np = cur_pos_bc_np + delta_np[:3]
            pos_world = (torch.tensor(cur_pos_bc_np, dtype=torch.float32, device=device)
                         + bottle_w).unsqueeze(0).clamp(ws_min, ws_max)

            delta_quat_xyzw = Rotation.from_rotvec(delta_np[3:6]).as_quat()
            new_rot = Rotation.from_quat(delta_quat_xyzw) * Rotation.from_quat(cur_quat_xyzw_np)
            cur_quat_xyzw_np = new_rot.as_quat()
            quat_tensor = torch.tensor([[cur_quat_xyzw_np[0], cur_quat_xyzw_np[1],
                                         cur_quat_xyzw_np[2], cur_quat_xyzw_np[3]]],
                                        dtype=torch.float32, device=device)

            root_pose = torch.cat([pos_world, quat_tensor], dim=-1)
            env.robot.write_root_pose_to_sim(root_pose)
            env.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            env.robot.set_joint_position_target(_init_joints_sim)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=dt)
            env.sim.render()
            env._hand_pos_desired[:] = pos_world

            if env.camera is not None:
                rgb_frames.append(env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8))

            action_log.append(delta_np.copy().astype(np.float32))
            agent_pos_log.append(agent_pos.cpu().numpy().copy())  # agent_pos BEFORE this delta
            qpos_log.append(np.concatenate([
                cur_pos_bc_np.astype(np.float32),
                Rotation.from_quat(cur_quat_xyzw_np).as_euler("XYZ").astype(np.float32),
                _init_joints[0].cpu().numpy().astype(np.float32),
            ]))
            pc_log.append(pc_4d.cpu().numpy().copy())
            step_idx += 1

        obs_dict = env._get_observations()

    return {
        "rgb_frames":     rgb_frames,
        "actions":        np.stack(action_log,    axis=0) if action_log    else np.zeros((0, 6), dtype=np.float32),
        "qpos_traj":      np.stack(qpos_log,      axis=0) if qpos_log      else np.zeros((0, 18), dtype=np.float32),
        "agent_pos_traj": np.stack(agent_pos_log, axis=0) if agent_pos_log else np.zeros((0, 7), dtype=np.float32),
        "pcs_traj":       np.stack(pc_log,        axis=0) if pc_log        else np.zeros((0, 1280, 4), dtype=np.float32),
    }

# ── Main loop ────────────────────────────────────────────────────────────────
output_root = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)

manifest = []

for init_i in range(args.num_init_states):
    rng = np.random.RandomState(args.rand_offset + init_i)
    grasp_offset_np, target_offset_np = sample_init_offsets(rng, args.obj_trans_range_xy)
    print(f"\n[init {init_i+1}/{args.num_init_states}] grasp_off={grasp_offset_np.tolist()}  target_off={target_offset_np.tolist()}")

    for sample_k in range(args.n_samples_per_init):
        sample_seed = args.rand_offset * 1000 + init_i * 100 + sample_k
        sample_dir_name = f"init{init_i:03d}_sample{sample_k:02d}"
        sample_dir = output_root / sample_dir_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Reset env + apply init offsets + teleport hand (same init across K samples)
        bottle_w = reset_env_with_offsets(grasp_offset_np, target_offset_np)

        # Save an init-frame png for debugging
        env.sim.render(); env.scene.update(dt=env.physics_dt)
        if env.camera is not None:
            init_rgb = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            imageio.imwrite(str(sample_dir / "init_frame.png"), init_rgb)

        # Run rollout with unique diffusion seed
        traj = run_rollout(bottle_w, args.max_steps, sample_seed, args.action_steps_per_query)

        # Save video + trajectories
        video_path = sample_dir / "rollout.mp4"
        if traj["rgb_frames"]:
            with imageio.get_writer(str(video_path), fps=args.video_fps, quality=8) as w:
                for frame in traj["rgb_frames"]:
                    w.append_data(frame)
        np.savez(sample_dir / "actions.npz",        actions=traj["actions"])
        np.savez(sample_dir / "qpos_traj.npz",      qpos=traj["qpos_traj"])
        np.savez(sample_dir / "agent_pos_traj.npz", agent_pos=traj["agent_pos_traj"])
        np.savez(sample_dir / "pcs_traj.npz",       pcs=traj["pcs_traj"])

        entry = {
            "sample_dir":    sample_dir_name,
            "video":         f"{sample_dir_name}/rollout.mp4",
            "init_i":        int(init_i),
            "sample_k":      int(sample_k),
            "seed":          int(sample_seed),
            "n_steps":       int(len(traj["rgb_frames"])),
            "grasp_offset":  grasp_offset_np.tolist(),
            "target_offset": target_offset_np.tolist(),
        }
        manifest.append(entry)
        print(f"  sample {sample_k+1}/{args.n_samples_per_init}  → {sample_dir_name}  ({entry['n_steps']} frames, seed={sample_seed})")

        # Incrementally persist manifest so partial runs are usable
        with open(output_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

print(f"\n[rollout done] {len(manifest)} samples  →  {output_root / 'manifest.json'}")

env.close()
sim_app.close()
