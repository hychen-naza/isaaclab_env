"""Base DirectRLEnv for tabletop manipulation with the Inspire hand.

Shared logic:
  Scene:   Inspire hand + two rigid objects (grasp target + goal) + table + optional depth camera.
  Obs:     hand root pose + joint positions + object poses + optional world-frame point cloud.
  Act:     [dx, dy, dz, finger_0, …, finger_5]  (all normalised to [-1, 1]).
  Reward:  sparse — 0 every step; +100 when grasp object is within success_dist of goal object.
           Optionally dense — Robometer progress signal (set use_robometer=True).

Task-specific behaviour is introduced by subclassing BaseManipEnvCfg and BaseManipEnv.
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
from .base_env_cfg import BaseManipEnvCfg

# Objects are pre-processed to Z-up + centered by prepare_objects.sh,
# so identity quaternion (w=1) is correct — no additional rotation needed.
_OBJ_QUAT_NP = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, z, w) — identity


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
            "Run 'python obj_utils/setup_assets.py' first."
        )
    return usd_path


def _make_obj_cfg(prim_path: str, usd_path: str,
                  init_pos: tuple, kinematic: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                linear_damping=0.5,
                angular_damping=8.0,
                max_linear_velocity=2.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=init_pos,
            rot=tuple(_OBJ_QUAT_NP.tolist()),
        ),
    )


# ── Environment ───────────────────────────────────────────────────────────────

class BaseManipEnv(DirectRLEnv):
    cfg: BaseManipEnvCfg

    def __init__(self, cfg: BaseManipEnvCfg, render_mode: str | None = None, **kwargs):
        self._ws_min = torch.tensor(cfg.workspace_min, device="cpu")
        self._ws_max = torch.tensor(cfg.workspace_max, device="cpu")
        self._hand_init_pos   = torch.tensor(cfg.hand_init_pos,   device="cpu")
        self._grasp_object_init_pos = torch.tensor(cfg.grasp_object_init_pos, device="cpu")
        self._target_object_init_pos   = torch.tensor(cfg.target_object_init_pos,   device="cpu")

        # Fixed point-cloud centering origin: frozen at the configured object's
        # initial position so the frame is stable even as the object moves.
        _center_src = cfg.target_object_init_pos if cfg.pointcloud_center == "target_object" else cfg.grasp_object_init_pos
        self._pc_center_init_pos = torch.tensor(_center_src, dtype=torch.float32, device="cpu")
        self._debug_frames_saved = 0   # counter — incremented in _save_debug_frame

        # Placeholders — finger joint info resolved after super().__init__() starts physics
        self._finger_joint_ids:    list[int]    = []
        self._finger_joint_mins:   torch.Tensor = torch.zeros(6)
        self._finger_joint_ranges: torch.Tensor = torch.ones(6)

        # Robometer placeholders (populated after super().__init__ if use_robometer=True)
        self._robometer_model      = None
        self._rbm_tokenizer        = None
        self._rbm_processor        = None
        self._rbm_exp_config       = None
        self._rbm_collator         = None
        self._rbm_frame_buf        = None   # (N_envs, buf_size, H, W, 3) uint8, CPU
        self._rbm_last_progress    = None   # (N_envs,) float32 — progress at last call
        self._rbm_cached_reward    = None   # (N_envs,) float32 — last robometer reward
        self._rbm_step_counter     = 0

        super().__init__(cfg, render_mode, **kwargs)

        # Move tensors to the device resolved by super().__init__()
        self._ws_min              = self._ws_min.to(self.device)
        self._ws_max              = self._ws_max.to(self.device)
        self._hand_init_pos       = self._hand_init_pos.to(self.device)
        self._grasp_object_init_pos     = self._grasp_object_init_pos.to(self.device)
        self._target_object_init_pos       = self._target_object_init_pos.to(self.device)
        self._pc_center_init_pos  = self._pc_center_init_pos.to(self.device)
        self._obj_quat        = torch.tensor(_OBJ_QUAT_NP, dtype=torch.float32, device=self.device)
        self._actions         = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # Desired hand position — updated in _apply_action to avoid reading stale sensor data
        # within decimation sub-steps (root_pos_w is only refreshed after scene.update()).
        self._hand_pos_desired = self._hand_init_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Load Robometer reward model if requested
        if cfg.use_robometer:
            self._load_robometer()

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
        grasp_object_usd = _ensure_obj_usd(self.cfg.grasp_object_urdf)
        target_object_usd = _ensure_obj_usd(self.cfg.target_object_urdf)

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
        self.grasp_object = RigidObject(cfg=_make_obj_cfg(
            prim_path=f"{_NS}/GraspObject",
            usd_path=grasp_object_usd,
            init_pos=self.cfg.grasp_object_init_pos,
            kinematic=False,
        ))

        # 4. Bowl — dynamic rigid body (gravity-affected, not glued)
        self.target_object = RigidObject(cfg=_make_obj_cfg(
            prim_path=f"{_NS}/TargetObject",
            usd_path=target_object_usd,
            init_pos=self.cfg.target_object_init_pos,
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
        spawn_ground_plane("/World/GroundPlane", GroundPlaneCfg(physics_material=None, color=None))
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
            "/World/DomeLight",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )

        # 8. Register all scene entities
        self.scene.articulations["robot"]  = self.robot
        self.scene.rigid_objects["grasp_object"] = self.grasp_object
        self.scene.rigid_objects["target_object"]   = self.target_object
        if self.camera is not None:
            self.scene.sensors["camera"]   = self.camera

        # Clone environments and filter collisions (required by DirectRLEnv)
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])
        # Capsule collision shapes are embedded directly in the bottle/bowl USD files
        # (base.usda) so they are registered before physics init — no runtime addition needed.

    # ── Robometer reward model ────────────────────────────────────────────────

    def _load_robometer(self) -> None:
        """Load the Robometer VLM reward model (called once at init)."""
        import unsloth  # must be imported before transformers for patching
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        cls = type(self).__name__
        print(f"[{cls}] Loading Robometer from '{self.cfg.robometer_model_path}' …")
        rbm_device = torch.device(self.cfg.robometer_device)
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=self.cfg.robometer_model_path,
            device=rbm_device,
        )
        reward_model.eval()
        self._robometer_device = rbm_device
        self._robometer_model  = reward_model
        self._rbm_tokenizer    = tokenizer
        self._rbm_processor    = processor
        self._rbm_exp_config   = exp_config
        self._rbm_collator     = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

        # Camera is required; auto-enable if not set
        if not self.cfg.use_camera:
            print(f"[{cls}] WARNING: use_robometer=True but use_camera=False. "
                  "Robometer needs RGB frames — enable use_camera or rewards will be zero.")

        N   = self.num_envs
        buf = self.cfg.robometer_frame_buffer_size
        H, W = self.cfg.robometer_frame_size   # downscaled resolution for VLM
        # Allocate on CPU (frames are uint8, avoid VRAM pressure)
        self._rbm_frame_buf     = np.zeros((N, buf, H, W, 3), dtype=np.uint8)
        self._rbm_last_progress = np.zeros(N, dtype=np.float32)
        self._rbm_cached_reward = torch.zeros(N, device=self.device)
        print(f"[{cls}] Robometer ready. "
              f"reward_freq={self.cfg.robometer_reward_freq}, "
              f"buffer={buf} frames, eval_envs={self.cfg.robometer_eval_envs}")

    def _robometer_step(self) -> torch.Tensor:
        """
        Run Robometer on a random subset of envs and return per-env dense rewards.

        Returns
        -------
        torch.Tensor
            Shape (num_envs,).  Envs not evaluated this call get reward=0.
        """
        import time
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        t_start = time.perf_counter()
        rewards = torch.zeros(self.num_envs, device=self.device)
        n_eval  = self.cfg.robometer_eval_envs
        if n_eval < 0 or n_eval >= self.num_envs:
            eval_ids = list(range(self.num_envs))
        else:
            eval_ids = np.random.choice(self.num_envs, size=n_eval, replace=False).tolist()

        loss_cfg     = getattr(self._rbm_exp_config, "loss", None)
        is_discrete  = (
            getattr(loss_cfg, "progress_loss_type", "l2").lower() == "discrete"
            if loss_cfg else False
        )
        num_bins = (
            getattr(loss_cfg, "progress_discrete_bins", None)
            or getattr(self._rbm_exp_config.model, "progress_discrete_bins", 10)
        )

        for env_id in eval_ids:
            frames = self._rbm_frame_buf[env_id]          # (buf, H, W, 3) uint8
            T      = frames.shape[0]
            traj   = Trajectory(
                frames=frames,
                frames_shape=tuple(frames.shape),
                task=self.cfg.robometer_task,
                id=str(env_id),
                metadata={"subsequence_length": T},
                video_embeddings=None,
            )
            sample = ProgressSample(trajectory=traj, sample_type="progress")
            batch  = self._rbm_collator([sample])
            prog_inputs = batch["progress_inputs"]
            prog_inputs = {k: v.to(self._robometer_device) if hasattr(v, "to") else v
                           for k, v in prog_inputs.items()}

            with torch.no_grad():
                results = compute_batch_outputs(
                    self._robometer_model,
                    self._rbm_tokenizer,
                    prog_inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )

            prog_pred = results.get("progress_pred", [])
            if prog_pred and len(prog_pred) > 0 and len(prog_pred[0]) > 0:
                # Use the last-frame progress score
                latest_progress = float(np.array(prog_pred[0])[-1])
            else:
                latest_progress = self._rbm_last_progress[env_id]

            delta = latest_progress - self._rbm_last_progress[env_id]
            rewards[env_id] = float(self.cfg.robometer_reward_scale * max(float(delta), 0.0))
            self._rbm_last_progress[env_id] = latest_progress

        elapsed = time.perf_counter() - t_start
        n_eval  = len(eval_ids)
        print(f"[Robometer] step={self._rbm_step_counter}  "
              f"envs={n_eval}  time={elapsed:.2f}s  "
              f"({elapsed/max(n_eval,1)*1000:.0f} ms/env)  "
              f"reward_sum={rewards.sum().item():.3f}")
        return rewards

    def _rbm_push_frame(self, rgb_frames: torch.Tensor) -> None:
        """
        Shift the per-env frame buffer left and append a new (downscaled) frame.

        Args:
            rgb_frames: (N, H, W, 4) or (N, H, W, 3) uint8 tensor from TiledCamera.
        """
        import cv2
        frames_np = rgb_frames[..., :3].cpu().numpy().astype(np.uint8)  # (N, H, W, 3)
        tH, tW   = self.cfg.robometer_frame_size
        resized  = np.stack([
            cv2.resize(frames_np[i], (tW, tH), interpolation=cv2.INTER_AREA)
            for i in range(frames_np.shape[0])
        ])                                                              # (N, tH, tW, 3)
        self._rbm_frame_buf = np.roll(self._rbm_frame_buf, shift=-1, axis=1)
        self._rbm_frame_buf[:, -1] = resized

    # ── Point cloud ───────────────────────────────────────────────────────────

    @staticmethod
    def _fps_torch(pts: torch.Tensor, n: int) -> torch.Tensor:
        """Batched approximate Farthest Point Sampling.

        Args:
            pts: (B, N, 3) candidate points.
            n:   number of output points.

        Returns:
            (B, n, 3) selected points.
        """
        B, N, _ = pts.shape
        pool = min(N, 4096)
        if N > pool:
            ridx  = torch.randperm(N, device=pts.device)[:pool]
            cands = pts[:, ridx]
        else:
            cands = pts
            pool  = N

        selected = torch.zeros(B, n, dtype=torch.long, device=pts.device)
        dists    = torch.full((B, pool), float('inf'), device=pts.device)
        idx      = torch.zeros(B, dtype=torch.long, device=pts.device)

        for i in range(n):
            selected[:, i] = idx
            pt    = cands.gather(1, idx.view(B, 1, 1).expand(B, 1, 3))   # (B,1,3)
            d     = ((cands - pt) ** 2).sum(-1)                           # (B,pool)
            dists = torch.minimum(dists, d)
            idx   = dists.argmax(dim=-1)                                   # (B,)

        return cands.gather(1, selected.unsqueeze(-1).expand(B, n, 3))

    def _compute_pointcloud(self) -> tuple:
        """Back-project depth image → bottle-centric object and table point clouds.

        Convention (matches video pipeline after gravity alignment + centering):
          - Origin    = root_pos_w of bottle = bottom of bottle mesh (center_mesh sets Z_min=0)
          - Object pts: z > 0.005 m in bottle-centric frame  (above table surface)
          - Table pts : -0.06 < z < 0.005 m                  (at/near table surface)

        Returns
        -------
        obj_pts   : (num_envs, n_pointcloud_points, 3)  bottle-centric, object region
        table_pts : (num_envs, n_table_points, 3)        bottle-centric, table region
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

        # Isaac Lab TiledCamera convention matching replay.py/_make_combined_frame:
        #   X = depth (forward), Y = -(u-cx)/fx*d (left, not right), Z = -(v-cy)/fy*d (up).
        # Using the same sign convention as replay.py ensures sim points are
        # in-distribution with the video pipeline point clouds.
        x_cam =  depth
        y_cam = -(u - cx) / fx * depth
        z_cam = -(v - cy) / fy * depth

        # ── Depth validity mask (matches replay.py: finite, >0, <3 m) ──────────
        # Background/sky pixels have inf depth; without this filter they project
        # to garbage world positions that dominate the sampling weights.
        valid_mask = torch.isfinite(depth) & (depth > 0.01) & (depth < 3.0)  # (N, H, W)
        valid_flat = valid_mask.reshape(N, H * W)                              # (N, H*W)

        # Stack → (N, H*W, 3) in camera frame
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(N, H * W, 3)

        # Transform to world frame
        # matrix_from_quat expects (x, y, z, w) — same format as quat_w_world
        R_cam = matrix_from_quat(self.camera.data.quat_w_world)  # (N, 3, 3)
        t_cam     = self.camera.data.pos_w.unsqueeze(1)      # (N, 1, 3)
        pts_world = pts_cam @ R_cam.transpose(-1, -2) + t_cam  # (N, H*W, 3)

        # Centre on the fixed initial position of the configured object.
        # Using the initial position (not the live current position) keeps the
        # point-cloud coordinate frame stable throughout the trajectory,
        # matching the convention in the video pipeline (pc_data_gravity_aligned).
        pc_center = self._pc_center_init_pos.view(1, 1, 3)   # (1, 1, 3) → broadcast
        pts_cent  = pts_world - pc_center                      # (N, H*W, 3)

        GLOBAL_Z_MIN  = -0.10   # discard anything > 10 cm below table (noise/floor)
        OBJECT_Z_MIN  =  0.005  # above table surface in centred frame
        TABLE_Z_MIN   = -0.06   # just below table surface
        n_obj         = self.cfg.n_pointcloud_points
        n_table       = self.cfg.n_table_points
        OBJECT_RADIUS =  0.20   # metres — sample only within this radius of each object

        # Global floor filter applied once; all subsequent masks inherit it
        in_scene   = valid_flat & (pts_cent[:, :, 2] > GLOBAL_Z_MIN)    # (N, H*W)
        above_table = in_scene  & (pts_cent[:, :, 2] > OBJECT_Z_MIN)

        def _sample_near(obj_pos_w: torch.Tensor) -> torch.Tensor:
            """Sample n_obj pts within OBJECT_RADIUS of obj_pos_w.

            obj_pos_w is the LIVE world position of the object (changes each step).
            We subtract _pc_center_init_pos to express it in the same fixed centred
            frame as pts_cent, then find nearby depth pixels.
            """
            obj_cent = (obj_pos_w - self._pc_center_init_pos).unsqueeze(1)  # (N,1,3)
            dist2    = ((pts_cent - obj_cent) ** 2).sum(-1)                  # (N, H*W)
            near     = above_table & (dist2 < OBJECT_RADIUS ** 2)
            w        = near.float() + 1e-6
            idx      = torch.multinomial(w, n_obj, replacement=True)
            return pts_cent.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))   # (N, n_obj, 3)

        # ── Grasp object (bottle) and target object (bowl) ────────────────────
        # Both positions are live (update each step as objects move).
        # Centering on _pc_center_init_pos keeps the coordinate frame stable.
        grasp_object_pos_w = wp.to_torch(self.grasp_object.data.root_pos_w)  # (N, 3) live
        target_object_pos_w   = wp.to_torch(self.target_object.data.root_pos_w)    # (N, 3) live
        grasp_pts  = _sample_near(grasp_object_pos_w)
        target_pts = _sample_near(target_object_pos_w)

        # ── Table points: FPS over table-surface band ─────────────────────────
        table_mask    = in_scene & (pts_cent[:, :, 2] > TABLE_Z_MIN) & (pts_cent[:, :, 2] < OBJECT_Z_MIN)
        table_weights = table_mask.float() + 1e-6
        pool          = min(pts_cent.shape[1], 4096)
        cand_idx      = torch.multinomial(table_weights, pool, replacement=False)
        table_cands   = pts_cent.gather(1, cand_idx.unsqueeze(-1).expand(-1, -1, 3))
        table_pts     = self._fps_torch(table_cands, n_table)

        # ── Optional debug frame saving ───────────────────────────────────────
        if (self.cfg.debug_frame_dir
                and self._debug_frames_saved < self.cfg.debug_frame_count):
            rgb = self.camera.data.output["rgb"]
            self._save_debug_frame(
                rgb[0].cpu().numpy(),
                pts_cent[0].cpu().numpy(),
                grasp_pts[0].cpu().numpy(),
                target_pts[0].cpu().numpy(),
                table_pts[0].cpu().numpy(),
            )

        return grasp_pts, target_pts, table_pts  # (N,256,3), (N,256,3), (N,256,3)

    def _save_debug_frame(self, rgb_np: np.ndarray,
                          pts_cent_all: np.ndarray,
                          grasp_pts: np.ndarray,
                          target_pts: np.ndarray,
                          table_pts: np.ndarray) -> None:
        """Save a debug visualisation PNG: RGB + sampled point clouds.

        Args:
            rgb_np:       (H, W, 3/4) uint8 from the camera (first env only).
            pts_cent_all: (H*W, 3) all back-projected points in centered frame.
            grasp_pts:    (n_obj, 3)   grasp object (bottle) points.
            target_pts:   (n_obj, 3)   target object (bowl) points.
            table_pts:    (n_table, 3) table surface points.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

        out_dir = Path(self.cfg.debug_frame_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 6), dpi=100)

        # Left: RGB
        ax_rgb = fig.add_subplot(1, 3, 1)
        ax_rgb.imshow(rgb_np[..., :3])
        ax_rgb.axis("off")
        ax_rgb.set_title(f"RGB  (frame {self._debug_frames_saved})")

        # Centre: all depth points coloured by Z
        ax_all = fig.add_subplot(1, 3, 2, projection="3d")
        # subsample for speed
        n_show = min(len(pts_cent_all), 4000)
        idx    = np.random.choice(len(pts_cent_all), n_show, replace=False)
        p      = pts_cent_all[idx]
        valid  = np.isfinite(p).all(axis=1) & (np.abs(p) < 5.0).all(axis=1)
        p      = p[valid]
        if len(p) > 0:
            # clip to ±3 m from center to exclude any residual outliers
            valid = (np.abs(p) < 3.0).all(axis=1)
            p = p[valid]
        if len(p) > 0:
            z_norm = (p[:, 2] - p[:, 2].min()) / (p[:, 2].max() - p[:, 2].min() + 1e-6)
            ax_all.scatter(p[:, 0], p[:, 1], p[:, 2], c=z_norm, cmap="viridis", s=1, alpha=0.4)
        ax_all.set_xlabel("X"); ax_all.set_ylabel("Y"); ax_all.set_zlabel("Z")
        center_name = self.cfg.pointcloud_center
        ax_all.set_title(f"All depth pts ({center_name}-centred)")

        # Right: sampled per-object + table points
        ax_pts = fig.add_subplot(1, 3, 3, projection="3d")
        ax_pts.scatter(grasp_pts[:, 0],  grasp_pts[:, 1],  grasp_pts[:, 2],
                       c="dodgerblue", s=4, alpha=0.8, label=f"grasp  ({len(grasp_pts)})")
        ax_pts.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
                       c="tomato",     s=4, alpha=0.8, label=f"target ({len(target_pts)})")
        ax_pts.scatter(table_pts[:, 0],  table_pts[:, 1],  table_pts[:, 2],
                       c="orange",     s=4, alpha=0.5, label=f"table  ({len(table_pts)})")
        ax_pts.legend(fontsize=8)
        ax_pts.set_xlabel("X"); ax_pts.set_ylabel("Y"); ax_pts.set_zlabel("Z")
        ax_pts.set_title(f"Sampled obs pts ({center_name}-centred)")

        plt.tight_layout()
        out_path = out_dir / f"obs_frame_{self._debug_frames_saved:04d}.png"
        plt.savefig(str(out_path))
        plt.close(fig)
        print(f"[BaseManipEnv] Saved debug frame → {out_path}")
        self._debug_frames_saved += 1

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
        # All positions centred on the fixed initial position of the pointcloud_center
        # object — same frame as the point cloud observation.
        pc_center = self._pc_center_init_pos   # (3,) fixed
        hand_pos  = wp.to_torch(self.robot.data.root_pos_w)  - pc_center  # (N, 3)
        hand_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        bot_pos   = wp.to_torch(self.grasp_object.data.root_pos_w) - pc_center  # (N, 3) live
        bot_quat  = wp.to_torch(self.grasp_object.data.root_quat_w)
        bowl_pos  = wp.to_torch(self.target_object.data.root_pos_w)   - pc_center  # (N, 3) live

        state_obs = torch.cat([
            hand_pos,   # (N, 3)
            hand_quat,  # (N, 4)
            joint_pos,  # (N, num_joints)
            bot_pos,    # (N, 3) live bottle pos in fixed centred frame
            bot_quat,   # (N, 4)
            bowl_pos,   # (N, 3) live bowl pos in fixed centred frame
        ], dim=-1)

        if self.cfg.use_camera and self.camera is not None:
            rgb = self.camera.data.output["rgb"]               # (N, H, W, 3/4) uint8
            if self.cfg.use_robometer and self._rbm_frame_buf is not None:
                self._rbm_push_frame(rgb)
            grasp_pts, target_pts, table_pts = self._compute_pointcloud()
            cloud_obs = torch.cat([
                grasp_pts.reshape(self.num_envs, -1),   # (N, 256*3)
                target_pts.reshape(self.num_envs, -1),  # (N, 256*3)
                table_pts.reshape(self.num_envs, -1),   # (N, 256*3)
            ], dim=-1)
            return {"policy": torch.cat([state_obs, cloud_obs], dim=-1),
                    "rgb":   rgb,
                    "depth": self.camera.data.output["depth"]}  # (N, H, W, 1) float
        return {"policy": state_obs}

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        dist = torch.norm(
            wp.to_torch(self.grasp_object.data.root_pos_w) - wp.to_torch(self.target_object.data.root_pos_w), dim=-1
        )
        sparse = torch.where(
            dist < self.cfg.success_dist,
            torch.full((self.num_envs,), self.cfg.sparse_reward, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )

        if not self.cfg.use_robometer or self._robometer_model is None:
            return sparse

        # Dense Robometer reward: run inference every `robometer_reward_freq` steps
        self._rbm_step_counter += 1
        if self._rbm_step_counter % self.cfg.robometer_reward_freq == 0:
            self._rbm_cached_reward = self._robometer_step()

        return sparse + self._rbm_cached_reward

    # ── Termination ───────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.norm(
            wp.to_torch(self.grasp_object.data.root_pos_w) - wp.to_torch(self.target_object.data.root_pos_w), dim=-1
        )
        terminated = dist < self.cfg.success_dist
        truncated  = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Clear Robometer frame buffers and progress for reset envs
        if self.cfg.use_robometer and self._rbm_frame_buf is not None:
            ids_cpu = env_ids.cpu().numpy() if hasattr(env_ids, 'cpu') else env_ids
            self._rbm_frame_buf[ids_cpu]     = 0
            self._rbm_last_progress[ids_cpu] = 0.0
            self._rbm_cached_reward[env_ids] = 0.0   # torch tensor, CUDA index OK

        # Reset desired hand position tracker for these envs
        self._hand_pos_desired[env_ids] = self._hand_init_pos

        # Robot root (identity orientation, zero velocity)
        root_state = torch.zeros(n, 13, device=self.device)
        root_state[:, :3] = self._hand_init_pos
        root_state[:, 6]  = 1.0   # qw = 1 (identity quaternion; write_root_pose_to_sim uses xyzw)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        # Robot joints → all zeros (fingers open)
        num_joints = self.robot.data.joint_pos.shape[-1]
        joint_pos  = torch.zeros(n, num_joints, device=self.device)
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
        )

        # Bottle
        grasp_object_state = torch.zeros(n, 13, device=self.device)
        grasp_object_state[:, :3]  = self._grasp_object_init_pos
        grasp_object_state[:, 3:7] = self._obj_quat
        self.grasp_object.write_root_pose_to_sim(grasp_object_state[:, :7], env_ids=env_ids)
        self.grasp_object.write_root_velocity_to_sim(grasp_object_state[:, 7:], env_ids=env_ids)

        # Bowl — dynamic, reset to its init pose with zero velocity
        target_object_state = torch.zeros(n, 13, device=self.device)
        target_object_state[:, :3]  = self._target_object_init_pos
        target_object_state[:, 3:7] = self._obj_quat
        self.target_object.write_root_pose_to_sim(target_object_state[:, :7], env_ids=env_ids)
        self.target_object.write_root_velocity_to_sim(target_object_state[:, 7:], env_ids=env_ids)
