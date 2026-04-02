"""Configuration for the GraspAndPlace environment."""
from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# ── Observation layout (must match env._get_observations) ─────────────────────
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
N_POINTS       = 256   # number of sampled points from the depth image
OBS_STATE_DIM  = 3 + 4 + OBS_JOINT_DIM + 3 + 4 + 3   # = 29
OBS_CLOUD_DIM  = N_POINTS * 3                           # = 768


@configclass
class GraspAndPlaceEnvCfg(DirectRLEnvCfg):
    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.5)

    # ── Camera toggle ─────────────────────────────────────────────────────────
    # Set to True only when running with --enable_cameras (requires omni.replicator.core).
    # When False, point-cloud observations are omitted and observation_space shrinks.
    use_camera: bool = False

    # ── RL spaces ─────────────────────────────────────────────────────────────
    observation_space: int = OBS_STATE_DIM   # 29; becomes 797 when use_camera=True
    action_space: int      = 9    # 3 (hand xyz delta) + 6 (finger joint targets)
    state_space: int       = 0    # no asymmetric critic

    # ── Decimation (physics steps per RL step) ────────────────────────────────
    decimation: int = 2

    # ── Episode ───────────────────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # ── Task ──────────────────────────────────────────────────────────────────
    success_dist: float  = 0.08    # metres — bottle centre within this of bowl centre
    sparse_reward: float = 100.0

    # ── Initial poses (world frame) ───────────────────────────────────────────
    # Bottle (grasp object) placed at table centre; bowl offset from video estimation.
    # z values set so each object's bottom face rests on the table top (z=0.40 m).
    # Bowl at x=-0.354 m estimated from camera-frame centroids (pourtea.mp4).
    hand_init_pos:   tuple = (0.0, -0.15, 0.9)
    bottle_init_pos: tuple = (0.0, 0.0, 0.7215)
    bowl_init_pos:   tuple = (0.3538, -0.0034, 0.4810)

    # ── Action scales ─────────────────────────────────────────────────────────
    pos_action_scale: float = 0.02   # max metres the hand can translate per step

    # ── Workspace clamp (world frame) ─────────────────────────────────────────
    workspace_min: tuple = (-0.7, -0.6, 0.4)
    workspace_max: tuple = (0.7, 0.6, 1.3)

    # ── Robometer reward ─────────────────────────────────────────────────────
    # Set use_robometer=True to replace the sparse reward with a dense progress
    # signal from the Robometer VLM reward model.  Requires a GPU with enough
    # VRAM (~10 GB for the 4B model).  The model is loaded once at env init.
    use_robometer:              bool  = False
    robometer_model_path:       str   = "robometer/Robometer-4B"
    robometer_task:             str   = "grasp the bottle and place it in the bowl"
    # How often (in env steps) to run robometer inference and refresh rewards.
    # Lower = denser rewards but much slower training (4B model is expensive).
    robometer_reward_freq:      int   = 20
    # Number of recent frames kept per env for the sliding-window trajectory.
    robometer_frame_buffer_size:int   = 8
    # Scale applied to the delta-progress reward from robometer.
    robometer_reward_scale:     float = 10.0
    # How many envs to evaluate per robometer call (random subset each time).
    # Set to -1 to evaluate all envs (very slow with many envs).
    robometer_eval_envs:        int   = 16
    # Device for the Robometer model (separate from Isaac Lab sim device).
    # Use "cuda:1" to keep the sim on cuda:0 and avoid OOM.
    robometer_device:           str   = "cuda:1"
    # Resize frames to this resolution before VLM inference (saves VRAM).
    robometer_frame_size:       tuple = (448, 448)

    # ── Depth camera ──────────────────────────────────────────────────────────
    # Camera at (0, -1.0, 0.9), closer to table, facing along +Y with downward
    # tilt toward (0, 0, 0.5).  Quaternion recomputed for new position then
    # rotated -90° around world Z (same pipeline as before):
    #   q_new = q_Rz(-90) * q_lookAt((1.0,0,0.9)→(0,0,0.5))
    #         = (x=0.5605, y=0, z=0, w=0.8284)
    #
    # Intrinsics match tmp/cam_K.txt horizontal FOV:
    #   focal_length=24.0, horizontal_aperture=38.01
    # Resolution: 1920×1280 (landscape)
    camera_pos: tuple          = (0.0, -1.0, 0.9)
    camera_rot: tuple          = (0.5605, 0.0, 0.0, 0.8284)  # (x,y,z,w) — looks along +Y, tilted down
    camera_width: int          = 1920
    camera_height: int         = 1280
    camera_focal_length: float = 24.0
    camera_horiz_aperture: float = 38.01   # gives fx≈808 px at 1280 px width
    n_pointcloud_points: int   = N_POINTS
    camera_clipping: tuple     = (0.1, 5.0)   # (near, far) metres
