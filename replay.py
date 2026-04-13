"""Replay recorded trajectory using the RH56E2 hand in Isaac Sim.

Three control modes:
  hybrid (default) — kinematic wrist + physics-based finger PD.
                     Matches base_env.py training convention.
  physical         — PD force/torque on wrist + PD finger targets.
                     Collisions fully respected; wrist tracking is imperfect.
  kinematic        — teleport everything each frame (no physics interaction).

Trajectory format (from pc_data_gravity_aligned.npy):
    robot_qpos: dict[int, np.ndarray(18,)]
        [0:3]  = root pos (bottle-centred)
        [3:6]  = root euler XYZ
        [6:18] = 12 finger joints (training URDF order, remapped to sim order)

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        python replay_debug.py --task pourtea
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
RH56E2_DIR = HERE / "RH56E2_R_2026_1_5"
RH56E2_URDF_ABS = RH56E2_DIR / "urdf" / "RH56E2_R_2026_1_5_abs.urdf"
RH56E2_USD_DIR = RH56E2_DIR / "usd_freeroot"
_CAMERA_KIT = str(HERE / "camera_headless.kit")

# ── PD gains for "physical" control mode ──────────────────────────────────────
_PD_KP_POS = 1000.0  # N/m
_PD_KD_POS = 60.0    # N·s/m
_PD_F_MAX  = 400.0   # N
_PD_KP_ROT = 7.0     # N·m/rad
_PD_KD_ROT = 2.0     # N·m·s/rad (tuned for 480 Hz)
_PD_T_MAX  = 1.5     # N·m

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay trajectory with RH56E2 hand")
parser.add_argument("--task", type=str, default="pourtea")
parser.add_argument("--pc_data", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--max_frames", type=int, default=0,
                    help="Truncate replay (0 = use all frames)")
parser.add_argument("--offset_rpy", type=str, default="0,-1.5708,0",
                    help="Euler XYZ offset (radians) composed with training orientation.")
parser.add_argument("--control", type=str, default="hybrid",
                    choices=["physical", "kinematic", "hybrid"],
                    help="'hybrid' = kinematic wrist + dynamic fingers (default); "
                         "'physical' = PD wrist + PD fingers; "
                         "'kinematic' = teleport everything.")
parser.add_argument("--physics_hz", type=int, default=480)
parser.add_argument("--finger_stiffness", type=float, default=10.0,
                    help="Finger actuator stiffness (G1+Inspire standard: 10.0)")
parser.add_argument("--finger_damping", type=float, default=0.2,
                    help="Finger actuator damping (G1+Inspire standard: 0.2)")
# PD overrides (only used in "physical" mode)
parser.add_argument("--kp_pos", type=float, default=_PD_KP_POS)
parser.add_argument("--kd_pos", type=float, default=_PD_KD_POS)
parser.add_argument("--f_max",  type=float, default=_PD_F_MAX)
parser.add_argument("--kp_rot", type=float, default=_PD_KP_ROT)
parser.add_argument("--kd_rot", type=float, default=_PD_KD_ROT)
parser.add_argument("--t_max",  type=float, default=_PD_T_MAX)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
_PD_KP_POS = args.kp_pos
_PD_KD_POS = args.kd_pos
_PD_F_MAX  = args.f_max
_PD_KP_ROT = args.kp_rot
_PD_KD_ROT = args.kd_rot
_PD_T_MAX  = args.t_max
args.headless = True
args.experience = _CAMERA_KIT
args.enable_cameras = True

if args.pc_data is None:
    args.pc_data = str(HERE / "policy/il_dmp/data/raw_data" / args.task / "pc_data_gravity_aligned.npy")
if args.output is None:
    args.output = str(HERE / f"replay_rh56e2_{args.task}.mp4")

# Load trajectory BEFORE AppLauncher (Isaac Sim's bundled numpy conflicts).
import numpy as _pre_np
print(f"[replay] Loading trajectory: {args.pc_data}")
_pc_data_dict = _pre_np.load(args.pc_data, allow_pickle=True).item()
print(f"[replay] Loaded {len(_pc_data_dict.get('robot_qpos', {}))} frames")

launcher = AppLauncher(args)
sim_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import importlib
import imageio
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

# ── 1. Convert RH56E2 URDF → freeroot USD if needed ──────────────────────────
RH56E2_USD = RH56E2_USD_DIR / RH56E2_URDF_ABS.stem / f"{RH56E2_URDF_ABS.stem}.usda"


def _strip_root_fixed_joint(usd_dir: Path):
    """Remove the auto-generated PhysicsFixedJoint 'root_joint' so the root
    body can move under external forces."""
    physics_usda = usd_dir / RH56E2_URDF_ABS.stem / "payloads" / "Physics" / "physics.usda"
    if not physics_usda.exists():
        return
    import re
    text = physics_usda.read_text()
    new_text, n = re.subn(r'[ \t]*def PhysicsFixedJoint "root_joint"\s*\{[^}]*\}\n?', '', text)
    if n > 0:
        physics_usda.write_text(new_text)
        print(f"[replay] Stripped {n} root_joint block(s) from {physics_usda.name}")


if not RH56E2_USD.exists():
    print(f"[replay] Converting {RH56E2_URDF_ABS} → {RH56E2_USD}")
    UrdfConverter(UrdfConverterCfg(
        asset_path=str(RH56E2_URDF_ABS),
        usd_dir=str(RH56E2_USD_DIR),
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        merge_mesh=False,
    ))
    _strip_root_fixed_joint(RH56E2_USD_DIR)
print(f"[replay] RH56E2 USD: {RH56E2_USD}")


# ── PD wrench (only used in "physical" mode) ─────────────────────────────────
def compute_pd_wrench(target_pos_t, target_quat_xyzw_np,
                      cur_pos_t, cur_quat_xyzw_t,
                      cur_lin_vel_t, cur_ang_vel_t, device):
    """PD position+orientation controller. Quats are (x, y, z, w)."""
    F = _PD_KP_POS * (target_pos_t - cur_pos_t) - _PD_KD_POS * cur_lin_vel_t
    F = torch.clamp(F, -_PD_F_MAX, _PD_F_MAX)

    cur_quat_np = cur_quat_xyzw_t.cpu().numpy().astype(np.float64)
    err_rotvec = (Rotation.from_quat(target_quat_xyzw_np) *
                  Rotation.from_quat(cur_quat_np).inv()).as_rotvec()
    err_t = torch.tensor(err_rotvec, dtype=torch.float32, device=device)
    T = _PD_KP_ROT * err_t - _PD_KD_ROT * cur_ang_vel_t
    T = torch.clamp(T, -_PD_T_MAX, _PD_T_MAX)
    return F, T


# ── 2. Load task config ──────────────────────────────────────────────────────
importlib.import_module(f"tasks.{args.task}")
cfg_mod = importlib.import_module(f"tasks.{args.task}.env_cfg")
task_cfg = cfg_mod.TaskEnvCfg()

print(f"[replay] task={args.task}")
print(f"[replay] grasp_object_init_pos={task_cfg.grasp_object_init_pos}")
print(f"[replay] target_object_init_pos={task_cfg.target_object_init_pos}")


def _resolve_obj_usd(urdf_path: str) -> str:
    p = Path(urdf_path)
    return str(p.parent / p.stem / p.stem / f"{p.stem}.usda")


GRASP_USD = _resolve_obj_usd(task_cfg.grasp_object_urdf)
TARGET_USD = _resolve_obj_usd(task_cfg.target_object_urdf)
assert Path(GRASP_USD).exists(), f"missing {GRASP_USD}"
assert Path(TARGET_USD).exists(), f"missing {TARGET_USD}"

# ── 3. Trajectory ────────────────────────────────────────────────────────────
robot_qpos = _pc_data_dict["robot_qpos"]
frames = sorted(robot_qpos.keys())
if args.max_frames > 0:
    frames = frames[:args.max_frames]
print(f"[replay] {len(frames)} frames (first={frames[0]}, last={frames[-1]})")

# ── 4. Joint remap: training order → RH56E2 sim order ───────────────────────
# Training qpos[6:18] follows the DRO convention (thumb_1, thumb_2, ..., little_2).
# The Isaac Sim articulation has a different joint ordering.
OLD_TO_RH56E2_PERM = np.array([
    0,   # right_thumb_1   ← thumb_yaw
    4,   # right_index_1   ← index_proximal
    6,   # right_middle_1  ← middle_proximal
    8,   # right_ring_1    ← ring_proximal
    10,  # right_little_1  ← pinky_proximal
    1,   # right_thumb_2   ← thumb_pitch
    5,   # right_index_2   ← index_intermediate
    7,   # right_middle_2  ← middle_intermediate
    9,   # right_ring_2    ← ring_intermediate
    11,  # right_little_2  ← pinky_intermediate
    2,   # right_thumb_3   ← thumb_intermediate
    3,   # right_thumb_4   ← thumb_distal
], dtype=np.int64)


def remap_train_to_rh56e2(qpos12: np.ndarray) -> np.ndarray:
    return qpos12[OLD_TO_RH56E2_PERM]


# Orientation offset between training root frame and sim root frame
_offset_rpy = [float(x) for x in args.offset_rpy.split(",")]
HAND_ROOT_OFFSET = Rotation.from_euler("XYZ", _offset_rpy)
print(f"[replay] hand root offset rpy={_offset_rpy}")

# ── 5. Build scene ───────────────────────────────────────────────────────────
RH56E2_DRIVEN_JOINTS = [
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint",  "right_little_1_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.0

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.7, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
    )
    grasp_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GRASP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=task_cfg.grasp_object_init_pos),
    )
    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TARGET_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=task_cfg.target_object_init_pos),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(RH56E2_USD),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, fix_root_link=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7),
            joint_pos={j: 0.0 for j in RH56E2_DRIVEN_JOINTS},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=RH56E2_DRIVEN_JOINTS,
                stiffness=args.finger_stiffness,
                damping=args.finger_damping,
            ),
        },
    )
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 1.0, 0.9),
            rot=(0.0, 0.5605, 0.8284, 0.0),  # xyzw
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=38.01, clipping_range=(0.05, 5.0),
        ),
        width=1280, height=960,
    )


# ── 6. Initialise sim + scene ────────────────────────────────────────────────
_phys_dt = 1.0 / args.physics_hz
print(f"[replay] Physics: {args.physics_hz} Hz  (dt={_phys_dt*1000:.2f} ms)")
sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=_phys_dt))
scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()

robot: Articulation = scene["robot"]
print(f"[replay] joint_names: {robot.data.joint_names}")
print(f"[replay] body_names:  {robot.data.body_names}")

# ── 7. Replay loop ───────────────────────────────────────────────────────────
device = robot.device
dt = _phys_dt
bottle_center = torch.tensor(task_cfg.grasp_object_init_pos, dtype=torch.float32, device=device)
n_substeps = max(1, args.physics_hz // 120)

# Settle physics
for _ in range(30):
    sim.step()
    scene.update(dt=dt)

# Teleport hand to frame 0
_qpos0 = robot_qpos[frames[0]].astype(np.float64)
_init_pos = (torch.tensor(_qpos0[:3], dtype=torch.float32, device=device) + bottle_center).unsqueeze(0)
_init_rot = Rotation.from_euler("XYZ", _qpos0[3:6]) * HAND_ROOT_OFFSET
_init_q = _init_rot.as_quat()
_init_quat_t = torch.tensor([[_init_q[0], _init_q[1], _init_q[2], _init_q[3]]],
                             dtype=torch.float32, device=device)
_init_joints_t = torch.tensor(remap_train_to_rh56e2(_qpos0[6:18]).astype(np.float32),
                               dtype=torch.float32, device=device).unsqueeze(0)
_init_pose = torch.cat([_init_pos, _init_quat_t], dim=-1)

for _ in range(2):
    robot.write_root_pose_to_sim(_init_pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    robot.write_joint_state_to_sim(_init_joints_t, torch.zeros_like(_init_joints_t))
    robot.set_joint_position_target(_init_joints_t)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt=dt)
sim.render()
print(f"[replay] Frame-0 teleport done; control={args.control}, substeps={n_substeps}")

rgb_frames: list[np.ndarray] = []
pos_errs: list[float] = []
rot_errs: list[float] = []

for fi, frame_idx in enumerate(frames):
    if not sim_app.is_running():
        break
    qpos = robot_qpos[frame_idx].astype(np.float64)

    target_pos_t = torch.tensor(qpos[:3], dtype=torch.float32, device=device) + bottle_center
    rot = Rotation.from_euler("XYZ", qpos[3:6]) * HAND_ROOT_OFFSET
    target_quat_xyzw = rot.as_quat()
    target_quat_t = torch.tensor(
        [[target_quat_xyzw[0], target_quat_xyzw[1],
          target_quat_xyzw[2], target_quat_xyzw[3]]],
        dtype=torch.float32, device=device)

    joints_t = torch.tensor(remap_train_to_rh56e2(qpos[6:18]).astype(np.float32),
                             dtype=torch.float32, device=device).unsqueeze(0)

    if args.control == "physical":
        for _ in range(n_substeps):
            cur_pos_t  = wp.to_torch(robot.data.root_pos_w)[0]
            cur_quat_t = wp.to_torch(robot.data.root_quat_w)[0]
            cur_linv_t = wp.to_torch(robot.data.root_lin_vel_w)[0]
            cur_angv_t = wp.to_torch(robot.data.root_ang_vel_w)[0]
            F, T = compute_pd_wrench(
                target_pos_t, np.asarray(target_quat_xyzw, dtype=np.float64),
                cur_pos_t, cur_quat_t, cur_linv_t, cur_angv_t, device)
            robot.set_external_force_and_torque(
                F.view(1, 1, 3), T.view(1, 1, 3), body_ids=[0], is_global=True)
            robot.set_joint_position_target(joints_t)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=dt)
        sim.render()

    elif args.control == "hybrid":
        root_pose = torch.cat([target_pos_t.unsqueeze(0), target_quat_t], dim=-1)
        for _ in range(n_substeps):
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            robot.set_joint_position_target(joints_t)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=dt)
        sim.render()

    else:  # kinematic
        root_pose = torch.cat([target_pos_t.unsqueeze(0), target_quat_t], dim=-1)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        robot.write_joint_state_to_sim(joints_t, torch.zeros_like(joints_t))
        robot.set_joint_position_target(joints_t)
        scene.write_data_to_sim()
        sim.step(render=False)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        robot.write_joint_state_to_sim(joints_t, torch.zeros_like(joints_t))
        scene.write_data_to_sim()
        sim.render()
        scene.update(dt=dt)

    rgb = scene["camera"].data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    rgb_frames.append(rgb)

    actual_root = wp.to_torch(robot.data.root_pos_w)[0].cpu().numpy()
    err = float(np.linalg.norm(target_pos_t.cpu().numpy() - actual_root))
    pos_errs.append(err)
    actual_quat = wp.to_torch(robot.data.root_quat_w)[0].cpu().numpy()
    rot_err = float((Rotation.from_quat(target_quat_xyzw).inv() *
                     Rotation.from_quat(actual_quat)).magnitude() * 180.0 / np.pi)
    rot_errs.append(rot_err)
    if fi % 30 == 0:
        print(f"[replay] frame={frame_idx:4d}  pos={err*1000:.1f}mm  rot={rot_err:.1f}°")

# ── 8. Summary + save ────────────────────────────────────────────────────────
print(f"[replay] Captured {len(rgb_frames)} frames")
if pos_errs:
    p = np.asarray(pos_errs)
    r = np.asarray(rot_errs)
    print(f"[replay] POS: mean={p.mean()*1000:.1f}mm  max={p.max()*1000:.1f}mm")
    print(f"[replay] ROT: mean={r.mean():.1f}°  max={r.max():.1f}°")

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
print(f"[replay] Saving → {out_path}  ({len(rgb_frames)} frames @ {args.fps} fps)")
with imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                        quality=8, macro_block_size=1) as writer:
    for f in rgb_frames:
        writer.append_data(f)
print(f"[replay] Done.")

sim_app.close()
