"""Convert RH56E2_R_2026_1_5_abs.urdf → USD, spawn it in Isaac Lab, verify the
mimic chain works (driving right_index_1_joint should make right_index_2_joint
follow with multiplier 1.0843), and save a render snapshot.

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        python RH56E2_R_2026_1_5/test_mimic.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
BASE = HERE.parent  # isaacsim_scene
URDF = HERE / "urdf" / "RH56E2_R_2026_1_5_abs.urdf"
USD_DIR = HERE / "usd"

# ── App launch ────────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(BASE / "camera_headless.kit")

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.experience = _CAMERA_KIT
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import imageio
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation


def look_at_quat_xyzw_world(eye, target, world_up=(0, 0, 1)):
    """Return (x,y,z,w) quaternion for a camera in 'world' convention
    (forward = +X, up = +Z) at `eye` looking at `target`.

    Camera basis in world coords:
        +X (cam) = forward = (target - eye) normalized
        +Z (cam) = world_up projected perpendicular to +X
        +Y (cam) = +Z × +X
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    world_up = np.array(world_up, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    # Project world_up perpendicular to fwd to get cam +Z
    z_cam = world_up - np.dot(world_up, fwd) * fwd
    z_cam /= np.linalg.norm(z_cam)
    y_cam = np.cross(z_cam, fwd)
    R = np.stack([fwd, y_cam, z_cam], axis=1)  # columns: cam +X, +Y, +Z in world
    return Rotation.from_matrix(R).as_quat()  # xyzw

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

# ── 1. Convert URDF → USD ─────────────────────────────────────────────────────
print(f"\n[test_mimic] Converting {URDF}")
print(f"            → {USD_DIR}/{URDF.stem}/{URDF.stem}.usda")

usd_path = USD_DIR / URDF.stem / f"{URDF.stem}.usda"
if usd_path.exists():
    print(f"[test_mimic] USD already exists, skipping conversion")
else:
    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(USD_DIR),
        fix_base=True,           # static base for visualisation test
        merge_fixed_joints=True, # collapse the many fixed force-sensor joints
        self_collision=False,
        merge_mesh=False,
    )
    UrdfConverter(cfg)
print(f"[test_mimic] USD: {usd_path}")
assert usd_path.exists(), f"conversion failed, USD missing"

# ── 2. Build a minimal scene ──────────────────────────────────────────────────
DRIVEN_JOINTS = [
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]

# Mimic relationships from RH56E2_R_2026_1_5.urdf:
#   right_thumb_3_joint  = 0.8392 * right_thumb_2_joint
#   right_thumb_4_joint  = 0.891  * right_thumb_3_joint  (chained)
#   right_index_2_joint  = 1.0843 * right_index_1_joint
#   right_middle_2_joint = 1.0843 * right_middle_1_joint
#   right_ring_2_joint   = 1.0843 * right_ring_1_joint
#   right_little_2_joint = 1.0843 * right_little_1_joint
#
# Isaac Lab on PhysX strips the URDF <mimic> constraints (deletes
# NewtonMimicAPI in physx.usda), so we enforce them in user code.
MIMIC_RULES = [
    # (target_joint, source_joint, multiplier)
    ("right_index_2_joint",  "right_index_1_joint",  1.0843),
    ("right_middle_2_joint", "right_middle_1_joint", 1.0843),
    ("right_ring_2_joint",   "right_ring_1_joint",   1.0843),
    ("right_little_2_joint", "right_little_1_joint", 1.0843),
    ("right_thumb_3_joint",  "right_thumb_2_joint",  0.8392),
    ("right_thumb_4_joint",  "right_thumb_3_joint",  0.891),
]


# ── compute camera params before building the scene cfg ──────────────────────
HAND_POS = (-0.28, 0.13, 0.52)
# Hand extends upward from wrist (~16cm tall); aim camera at vertical midpoint of hand
HAND_CENTER = np.array([-0.28, 0.14, 0.62])
_eye = HAND_CENTER + np.array([0.18, 0.18, 0.06])  # 25cm front-right-up
_target = HAND_CENTER
_quat = look_at_quat_xyzw_world(_eye, _target)
CAM_EYE_PLACEHOLDER = tuple(float(v) for v in _eye)
# Isaac Lab's TiledCameraCfg.OffsetCfg.rot is (x, y, z, w)
CAM_QUAT_PLACEHOLDER = (float(_quat[0]), float(_quat[1]), float(_quat[2]), float(_quat[3]))
print(f"[test_mimic] precomputed cam pos={CAM_EYE_PLACEHOLDER} rot_xyzw={CAM_QUAT_PLACEHOLDER}")


def apply_mimics(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Set mimic'd joint positions from their driven sources. Operates in-place
    on the (N, n_joints) tensor and returns it. Order matters: thumb_4 depends
    on thumb_3 which depends on thumb_2, so apply rules in declaration order.
    """
    idx = {n: i for i, n in enumerate(joint_names)}
    for tgt, src, mult in MIMIC_RULES:
        if tgt in idx and src in idx:
            joint_pos[..., idx[tgt]] = mult * joint_pos[..., idx[src]]
    return joint_pos


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 1.0

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                fix_root_link=False,  # base is already fixed inside the USD via fix_base=True
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.28, 0.13, 0.52),  # match the location used in eval_vis_debug for visual comparison
            joint_pos={j: 0.0 for j in DRIVEN_JOINTS},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=DRIVEN_JOINTS,
                stiffness=1000.0,
                damping=200.0,
            ),
        },
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAM_EYE_PLACEHOLDER,
            rot=CAM_QUAT_PLACEHOLDER,
            convention="world",  # forward = +X, up = +Z (matches look_at_quat_xyzw_world)
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=38.01,
            clipping_range=(0.01, 5.0),
        ),
        width=1280,
        height=960,
    )


# ── 3. Initialise sim + scene ─────────────────────────────────────────────────
sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=1.0 / 60.0))
sim.set_camera_view(eye=(0.4, 0.3, 0.7), target=(0.0, 0.0, 0.5))

scene = InteractiveScene(TestSceneCfg(num_envs=1, env_spacing=1.0))
sim.reset()
scene.write_data_to_sim()

robot: Articulation = scene["robot"]
print(f"\n[test_mimic] Articulation created")
print(f"  body_names ({len(robot.data.body_names)}): {robot.data.body_names}")
print(f"  joint_names ({len(robot.data.joint_names)}): {robot.data.joint_names}")

# Where actually is each link?
sim.step()
scene.update(dt=1.0/60.0)
_bp = wp.to_torch(robot.data.body_pos_w)[0].cpu().numpy()
print("[test_mimic] body world positions after spawn:")
for _bn, _p in zip(robot.data.body_names, _bp):
    print(f"  {_bn:24s} {_p.tolist()}")

# ── 4. Snapshot at zero pose ──────────────────────────────────────────────────
def render_and_save(name: str):
    sim.render()
    scene.update(dt=1.0 / 60.0)
    sim.render()
    cam = scene["camera"]
    rgb = cam.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    out = HERE / f"test_mimic_{name}.png"
    imageio.imwrite(str(out), rgb)
    print(f"[test_mimic] Saved {out.name}")
    return out


# Step physics a few times so the robot settles
for _ in range(5):
    sim.step()
    scene.update(dt=1.0 / 60.0)
render_and_save("zero")

# ── 5. Drive joints and report mimic results ─────────────────────────────────
def drive(name: str, joint_dict: dict[str, float]):
    """Set joint targets for the named driven joints, apply mimic rules to fill
    in the dependent joints, kinematically write all 12, render, and report.
    """
    n_joints = len(robot.data.joint_names)
    target = torch.zeros(1, n_joints, device=robot.device)
    for jname, jval in joint_dict.items():
        if jname not in robot.data.joint_names:
            print(f"  [drive] WARNING: '{jname}' not in articulation joint_names")
            continue
        idx = robot.data.joint_names.index(jname)
        target[0, idx] = jval
    # Apply mimic rules so the dependent joints are filled in
    apply_mimics(target, robot.data.joint_names)
    # Kinematic write
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    scene.write_data_to_sim()
    for _ in range(3):
        sim.step()
        scene.update(dt=1.0 / 60.0)
        # Re-pin every step so unactuated joints don't drift
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        scene.write_data_to_sim()
    render_and_save(name)
    # Print actual joint positions for the driven + mimic'd joint pairs
    actual = wp.to_torch(robot.data.joint_pos)[0].cpu().numpy()
    print(f"\n[drive {name}] joint state after driving:")
    for jname in robot.data.joint_names:
        idx = robot.data.joint_names.index(jname)
        print(f"  {jname:36s} cmd={target[0, idx].item():+.4f}  actual={actual[idx]:+.4f}")


# Test 1: curl just the index finger by driving right_index_1_joint
drive("index_curled", {"right_index_1_joint": 1.4})

# Test 2: curl just the thumb chain (drive thumb_2 → thumb_3 (mimic) → thumb_4 (mimic))
drive("thumb_curled", {"right_thumb_2_joint": 0.6})

# Test 3: full curl on all 4 non-thumb fingers
drive("all_fingers_curled", {
    "right_index_1_joint":  1.4,
    "right_middle_1_joint": 1.4,
    "right_ring_1_joint":   1.4,
    "right_little_1_joint": 1.4,
})

print("\n[test_mimic] Done. Check the saved PNGs in:", HERE)
sim_app.close()
