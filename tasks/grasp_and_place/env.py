"""GraspAndPlace DirectRLEnv implementation.

Task:  The Inspire right hand must grasp the bottle and carry it to the bowl.
Obs:   hand root pose + joint positions + bottle pose + bowl position
       + world-frame partial point cloud from a depth camera.
Act:   [dx, dy, dz, finger_0, …, finger_5]  (all normalised to [-1, 1]).
Reward: sparse — 0 every step; +100 when bottle centre is within success_dist
        of bowl centre.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat

from robots.inspire_hand_cfg import (
    FINGER_JOINT_LIMITS,
    FINGER_JOINTS,
    ensure_hand_usd,
    make_inspire_hand_cfg,
)
from .env_cfg import GraspAndPlaceEnvCfg

# ── Asset paths ───────────────────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).parent / "assets"
BOTTLE_URDF = str(_ASSETS_DIR / "grasp"  / "coacd_decomposed_object_one_link.urdf")
BOWL_URDF   = str(_ASSETS_DIR / "target" / "coacd_decomposed_object_one_link.urdf")

# Objects are pre-processed to Z-up + centered by prepare_objects.sh,
# so identity quaternion (w=1) is correct — no additional rotation needed.
_OBJ_QUAT_NP = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_obj_usd(urdf_path: str) -> str:
    """Return the pre-converted USD path for an object URDF.

    Raises FileNotFoundError if not found — run setup_assets.py first.
    """
    base_name = Path(urdf_path).stem
    usd_path  = str(Path(urdf_path).parent / base_name / base_name / f"{base_name}.usda")
    if not Path(usd_path).exists():
        raise FileNotFoundError(
            f"Object USD not found: {usd_path}\n"
            "Run 'python setup_assets.py' first."
        )
    return usd_path


def _make_obj_cfg(prim_path: str, usd_path: str,
                  init_pos: tuple, kinematic: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=init_pos,
            rot=tuple(_OBJ_QUAT_NP.tolist()),
        ),
    )


# ── Environment ───────────────────────────────────────────────────────────────

class GraspAndPlaceEnv(DirectRLEnv):
    cfg: GraspAndPlaceEnvCfg

    def __init__(self, cfg: GraspAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        self._ws_min = torch.tensor(cfg.workspace_min, device="cpu")
        self._ws_max = torch.tensor(cfg.workspace_max, device="cpu")
        self._hand_init_pos   = torch.tensor(cfg.hand_init_pos,   device="cpu")
        self._bottle_init_pos = torch.tensor(cfg.bottle_init_pos, device="cpu")
        self._bowl_init_pos   = torch.tensor(cfg.bowl_init_pos,   device="cpu")

        # Placeholders — finger joint info resolved after super().__init__() starts physics
        self._finger_joint_ids:    list[int]    = []
        self._finger_joint_mins:   torch.Tensor = torch.zeros(6)
        self._finger_joint_ranges: torch.Tensor = torch.ones(6)

        super().__init__(cfg, render_mode, **kwargs)

        # Move tensors to the device resolved by super().__init__()
        self._ws_min          = self._ws_min.to(self.device)
        self._ws_max          = self._ws_max.to(self.device)
        self._hand_init_pos   = self._hand_init_pos.to(self.device)
        self._bottle_init_pos = self._bottle_init_pos.to(self.device)
        self._bowl_init_pos   = self._bowl_init_pos.to(self.device)
        self._obj_quat        = torch.tensor(_OBJ_QUAT_NP, dtype=torch.float32, device=self.device)
        self._actions         = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # Desired hand position — updated in _apply_action to avoid reading stale sensor data
        # within decimation sub-steps (root_pos_w is only refreshed after scene.update()).
        self._hand_pos_desired = self._hand_init_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Resolve finger joint indices (articulation is now fully initialised)
        self._finger_joint_ids, _ = self.robot.find_joints(FINGER_JOINTS)
        mins   = torch.tensor([FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS], device=self.device)
        maxs   = torch.tensor([FINGER_JOINT_LIMITS[j][1] for j in FINGER_JOINTS], device=self.device)
        self._finger_joint_mins   = mins
        self._finger_joint_ranges = maxs - mins

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        # 1. Convert URDFs → USD (cached after first run)
        ensure_hand_usd(fix_base=False)
        bottle_usd = _ensure_obj_usd(BOTTLE_URDF)
        bowl_usd   = _ensure_obj_usd(BOWL_URDF)

        # Env prim prefix — matches the cartpole direct-rl example pattern.
        # InteractiveScene creates /World/envs/env_0, env_1, … for each env.
        _NS = "/World/envs/env_.*"

        # 2. Inspire hand (floating-base articulation)
        self.robot = Articulation(cfg=make_inspire_hand_cfg(
            prim_path=f"{_NS}/Robot",
            fix_base=False,
            init_pos=self.cfg.hand_init_pos,
        ))

        # 3. Bottle — dynamic rigid body
        self.bottle = RigidObject(cfg=_make_obj_cfg(
            prim_path=f"{_NS}/Bottle",
            usd_path=bottle_usd,
            init_pos=self.cfg.bottle_init_pos,
            kinematic=False,
        ))

        # 4. Bowl — dynamic rigid body (gravity-affected, not glued)
        self.bowl = RigidObject(cfg=_make_obj_cfg(
            prim_path=f"{_NS}/Bowl",
            usd_path=bowl_usd,
            init_pos=self.cfg.bowl_init_pos,
            kinematic=False,
        ))

        # 5. Table (static, shared across envs)
        table_cfg = sim_utils.CuboidCfg(
            size=(1.2, 0.7, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
        )
        table_cfg.func(f"{_NS}/Table", table_cfg, translation=(0.0, 0.0, 0.38))

        # 6. Depth camera — only when use_camera=True (requires omni.replicator.core)
        if self.cfg.use_camera:
            from isaaclab.sensors import TiledCamera, TiledCameraCfg
            self.camera = TiledCamera(cfg=TiledCameraCfg(
                prim_path=f"{_NS}/Camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=self.cfg.camera_pos,
                    rot=self.cfg.camera_rot,    # (x, y, z, w)
                    convention="opengl",
                ),
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self.cfg.camera_focal_length,
                    horizontal_aperture=self.cfg.camera_horiz_aperture,
                    clipping_range=self.cfg.camera_clipping,
                ),
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
            ))
        else:
            self.camera = None

        # 7. Ground plane and lighting (/World level, not per-env)
        spawn_ground_plane("/World/GroundPlane", GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
            "/World/DomeLight",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )

        # 8. Register all scene entities
        self.scene.articulations["robot"]  = self.robot
        self.scene.rigid_objects["bottle"] = self.bottle
        self.scene.rigid_objects["bowl"]   = self.bowl
        if self.camera is not None:
            self.scene.sensors["camera"]   = self.camera

        # Clone environments and filter collisions (required by DirectRLEnv)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])


    # ── Point cloud ───────────────────────────────────────────────────────────

    def _compute_pointcloud(self) -> torch.Tensor:
        """Back-project depth image → world-frame point cloud.

        Returns
        -------
        torch.Tensor
            Shape (num_envs, n_pointcloud_points, 3), world-frame XYZ.
        """
        # depth: (N, H, W, 1) → squeeze to (N, H, W)
        depth = self.camera.data.output["depth"].squeeze(-1)
        N, H, W = depth.shape

        # Intrinsic matrix: (N, 3, 3)
        K  = self.camera.data.intrinsic_matrices
        fx = K[:, 0, 0].view(N, 1, 1)
        fy = K[:, 1, 1].view(N, 1, 1)
        cx = K[:, 0, 2].view(N, 1, 1)
        cy = K[:, 1, 2].view(N, 1, 1)

        # Pixel coordinate grids: (1, H, W)
        u = torch.arange(W, device=self.device, dtype=torch.float32).view(1, 1, W)
        v = torch.arange(H, device=self.device, dtype=torch.float32).view(1, H, 1)

        # Isaac Lab TiledCamera convention: X-forward (depth along +X_cam),
        # Y-right (pixel u), Z-up (flip pixel v which goes downward).
        x_cam =  depth                     # X = depth (forward)
        y_cam =  (u - cx) / fx * depth    # Y = right (pixel u, no flip)
        z_cam = -(v - cy) / fy * depth    # Z = up (flip pixel v)

        # Stack → (N, H*W, 3) in camera frame
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(N, H * W, 3)

        # Transform to world frame: p_world = R_cam @ p_cam + t_cam
        # camera.data.pos_w:        (N, 3)
        # camera.data.quat_w_world: (N, 4) — (x, y, z, w); matrix_from_quat expects (w, x, y, z)
        quat_xyzw = self.camera.data.quat_w_world            # (N, 4) x,y,z,w
        quat_wxyz = torch.cat([quat_xyzw[:, 3:], quat_xyzw[:, :3]], dim=-1)  # → w,x,y,z
        R_cam = matrix_from_quat(quat_wxyz)                  # (N, 3, 3)
        t_cam = self.camera.data.pos_w.unsqueeze(1)          # (N, 1, 3)
        pts_world = pts_cam @ R_cam.transpose(-1, -2) + t_cam  # (N, H*W, 3)

        # Sample n_pointcloud_points from table + object pixels only (exclude floor).
        # Table surface is at world Z ≈ 0.40 m; floor is at Z = 0.
        # Weight = 1 for points at/above table level, 0 for floor / below table.
        n = self.cfg.n_pointcloud_points
        table_z_thresh = 0.36   # world frame (m) — just below table bottom (table top ~0.40 m)
        weights = (pts_world[:, :, 2] > table_z_thresh).float() + 1e-6  # (N, H*W)
        idx = torch.multinomial(weights, n, replacement=True)            # (N, n)
        pts = pts_world.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N, n, 3)

        # Zero-center around the grasp object (bottle) position
        bottle_pos = wp.to_torch(self.bottle.data.root_pos_w).unsqueeze(1)  # (N, 1, 3)
        pts = pts - bottle_pos
        return pts   # (N, n, 3) centred at bottle

    # ── Step ──────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Hand base translation — use self._hand_pos_desired instead of reading stale
        # root_pos_w (sensor buffer is only refreshed by scene.update() after all sub-steps,
        # so reading it within the decimation loop would not reflect the previous write).
        pos_delta = self._actions[:, :3] * self.cfg.pos_action_scale
        self._hand_pos_desired = (self._hand_pos_desired + pos_delta).clamp(
            self._ws_min, self._ws_max
        )
        self.robot.write_root_pose_to_sim(
            torch.cat([self._hand_pos_desired, wp.to_torch(self.robot.data.root_quat_w)], dim=-1)
        )
        # Zero root velocity so gravity doesn't accumulate between steps
        self.robot.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device)
        )

        # Finger joint targets: actions in [-1, 1] → [joint_min, joint_max]
        finger_norm    = (self._actions[:, 3:] + 1.0) * 0.5
        finger_targets = finger_norm * self._finger_joint_ranges + self._finger_joint_mins
        self.robot.set_joint_position_target(
            finger_targets, joint_ids=self._finger_joint_ids
        )
        self.robot.write_data_to_sim()

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # Convert Warp arrays → Torch tensors
        bot_pos   = wp.to_torch(self.bottle.data.root_pos_w)   # (N, 3) — origin for centering
        bot_quat  = wp.to_torch(self.bottle.data.root_quat_w)
        hand_pos  = wp.to_torch(self.robot.data.root_pos_w) - bot_pos   # zero-centred
        hand_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        bowl_pos  = wp.to_torch(self.bowl.data.root_pos_w)  - bot_pos   # zero-centred

        # bottle pos is always (0,0,0) in the centred frame; keep quat for orientation
        state_obs = torch.cat([
            hand_pos,                        # (N, 3) relative to bottle
            hand_quat,                       # (N, 4)
            joint_pos,                       # (N, num_joints)
            torch.zeros_like(bot_pos),       # (N, 3) bottle pos = origin
            bot_quat,                        # (N, 4)
            bowl_pos,                        # (N, 3) relative to bottle
        ], dim=-1)

        if self.cfg.use_camera and self.camera is not None:
            cloud_obs = self._compute_pointcloud().reshape(self.num_envs, -1)
            return {"policy": torch.cat([state_obs, cloud_obs], dim=-1),
                    "rgb":   self.camera.data.output["rgb"],    # (N, H, W, 3) uint8
                    "depth": self.camera.data.output["depth"]}  # (N, H, W, 1) float
        return {"policy": state_obs}

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        dist = torch.norm(
            wp.to_torch(self.bottle.data.root_pos_w) - wp.to_torch(self.bowl.data.root_pos_w), dim=-1
        )
        return torch.where(
            dist < self.cfg.success_dist,
            torch.full((self.num_envs,), self.cfg.sparse_reward, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )

    # ── Termination ───────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.norm(
            wp.to_torch(self.bottle.data.root_pos_w) - wp.to_torch(self.bowl.data.root_pos_w), dim=-1
        )
        terminated = dist < self.cfg.success_dist
        truncated  = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Reset desired hand position tracker for these envs
        self._hand_pos_desired[env_ids] = self._hand_init_pos

        # Robot root (identity orientation, zero velocity)
        root_state = torch.zeros(n, 13, device=self.device)
        root_state[:, :3] = self._hand_init_pos
        root_state[:, 3]  = 1.0   # w = 1 (identity quaternion)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        # Robot joints → all zeros (fingers open)
        num_joints = self.robot.data.joint_pos.shape[-1]
        joint_pos  = torch.zeros(n, num_joints, device=self.device)
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
        )

        # Bottle
        bottle_state = torch.zeros(n, 13, device=self.device)
        bottle_state[:, :3]  = self._bottle_init_pos
        bottle_state[:, 3:7] = self._obj_quat
        self.bottle.write_root_pose_to_sim(bottle_state[:, :7], env_ids=env_ids)
        self.bottle.write_root_velocity_to_sim(bottle_state[:, 7:], env_ids=env_ids)

        # Bowl — dynamic, reset to its init pose with zero velocity
        bowl_state = torch.zeros(n, 13, device=self.device)
        bowl_state[:, :3]  = self._bowl_init_pos
        bowl_state[:, 3:7] = self._obj_quat
        self.bowl.write_root_pose_to_sim(bowl_state[:, :7], env_ids=env_ids)
        self.bowl.write_root_velocity_to_sim(bowl_state[:, 7:], env_ids=env_ids)
