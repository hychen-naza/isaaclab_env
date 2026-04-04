"""Base configuration for tabletop manipulation environments with the Inspire hand.

Subclasses MUST override the task-specific fields marked below:
  - bottle_urdf, bowl_urdf   (asset URDF paths)
  - bottle_init_pos, bowl_init_pos  (initial object positions)
  - robometer_task           (task description string for the VLM reward model)
"""
from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# ── Observation layout (must match BaseManipEnv._get_observations) ────────────
#   hand_root_pos    (3)
#   hand_root_quat   (4)
#   hand_joint_pos   (12)   ← 6 actuated + 6 mimic joints
#   bottle_pos       (3)
#   bottle_quat      (4)
#   bowl_pos         (3)
#   [point_cloud     (N_PTS × 3)   ← only when use_camera=True]
#
# Update OBS_JOINT_DIM if mimic joints are collapsed by PhysX (check at runtime).
OBS_JOINT_DIM  = 12
N_POINTS       = 256   # points per object (grasp + target) sampled near each object
N_TABLE_POINTS = 256   # table surface points (FPS)
OBS_STATE_DIM  = 3 + 4 + OBS_JOINT_DIM + 3 + 4 + 3   # = 29
OBS_CLOUD_DIM  = N_POINTS * 2 * 3 + N_TABLE_POINTS * 3  # 512*3 + 256*3 = 2304


@configclass
class BaseManipEnvCfg(DirectRLEnvCfg):
    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.5)

    # ── Camera toggle ─────────────────────────────────────────────────────────
    # Set to True only when running with --enable_cameras (requires omni.replicator.core).
    # When False, point-cloud observations are omitted and observation_space shrinks.
    use_camera: bool = False

    # ── RL spaces ─────────────────────────────────────────────────────────────
    observation_space: int = OBS_STATE_DIM   # 29; becomes 1181 when use_camera=True
    action_space: int      = 9    # 3 (hand xyz delta) + 6 (finger joint targets)
    state_space: int       = 0    # no asymmetric critic

    # ── Decimation (physics steps per RL step) ────────────────────────────────
    decimation: int = 2

    # ── Episode ───────────────────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # ── Reward ────────────────────────────────────────────────────────────────
    success_dist: float  = 0.08    # metres — bottle centre within this of bowl centre
    sparse_reward: float = 100.0

    # ── Hand ──────────────────────────────────────────────────────────────────
    hand_init_pos:    tuple = (0.0, -0.15, 0.9)
    pos_action_scale: float = 0.02   # max metres the hand can translate per step
    workspace_min:    tuple = (-0.7, -0.6, 0.4)
    workspace_max:    tuple = (0.7, 0.6, 1.3)

    # ── Robometer reward ──────────────────────────────────────────────────────
    # Set use_robometer=True to replace the sparse reward with a dense progress
    # signal from the Robometer VLM reward model.  Requires a GPU with enough
    # VRAM (~10 GB for the 4B model).  The model is loaded once at env init.
    use_robometer:               bool  = False
    robometer_model_path:        str   = "robometer/Robometer-4B"
    # How often (in env steps) to run robometer inference and refresh rewards.
    # Lower = denser rewards but much slower training (4B model is expensive).
    robometer_reward_freq:       int   = 20
    # Number of recent frames kept per env for the sliding-window trajectory.
    robometer_frame_buffer_size: int   = 8
    # Scale applied to the delta-progress reward from robometer.
    robometer_reward_scale:      float = 10.0
    # How many envs to evaluate per robometer call (random subset each time).
    # Set to -1 to evaluate all envs (very slow with many envs).
    robometer_eval_envs:         int   = 16
    # Device for the Robometer model (separate from Isaac Lab sim device).
    # Use "cuda:1" to keep the sim on cuda:0 and avoid OOM.
    robometer_device:            str   = "cuda:1"
    # Resize frames to this resolution before VLM inference (saves VRAM).
    robometer_frame_size:        tuple = (448, 448)

    # ── Depth camera ──────────────────────────────────────────────────────────
    camera_pos: tuple           = (0.0, 1.0, 0.9)
    camera_rot: tuple           = (0.0, 0.5605, 0.8284, 0.0)  # (x,y,z,w) — looks along -Y, tilted down
    camera_width: int           = 1920
    camera_height: int          = 1280
    camera_focal_length: float  = 24.0
    camera_horiz_aperture: float = 38.01   # gives fx≈808 px at 1280 px width
    n_pointcloud_points: int    = N_POINTS
    n_table_points: int         = N_TABLE_POINTS
    camera_clipping: tuple      = (0.1, 5.0)   # (near, far) metres

    # ── Point-cloud centering ─────────────────────────────────────────────────
    # Which object's fixed initial position to use as the point-cloud origin.
    # "bottle" = grasp target (default), "bowl" = goal object.
    # The center is frozen at the initial position so the coordinate frame stays
    # stable even as the object moves during a trajectory.
    pointcloud_center: str      = "bottle"

    # ── Debug frame saving ────────────────────────────────────────────────────
    # Set debug_frame_dir to a directory path to save RGB + point-cloud
    # visualisation PNGs for the first debug_frame_count observation steps.
    debug_frame_dir:   str      = ""
    debug_frame_count: int      = 5

    # ── Task-specific fields — subclasses MUST override ───────────────────────
    # Asset URDF paths (task-specific objects).
    bottle_urdf: str = ""
    bowl_urdf:   str = ""
    # Initial world-frame positions; z set so object bottom rests on table top (z≈0.40 m).
    bottle_init_pos: tuple = (0.0, 0.0, 0.0)
    bowl_init_pos:   tuple = (0.0, 0.0, 0.0)
    # Natural-language task description for the Robometer VLM reward model.
    robometer_task: str = ""
