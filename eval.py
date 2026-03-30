"""Evaluate the IL (imitation-learning) policy in the GraspAndPlace Isaac Lab env.

The checkpoint is a zip archive produced by the `nmp` training pipeline.
Observations from Isaac Lab are adapted on-the-fly to the IL policy's expected
input format (point clouds + wrist-pose + finger joints).

Usage
-----
    python eval.py \
        --checkpoint /home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip \
        --num_envs 16 \
        --num_episodes 50

The env auto-resets upon termination/truncation, so we just run for enough
steps to collect `num_episodes` completed episodes across all parallel envs.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

# ── 1. App launcher (MUST come before any omni / isaaclab imports) ─────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate IL policy in GraspAndPlace env")
parser.add_argument("--checkpoint",   type=str,
                    default="/home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip")
parser.add_argument("--num_envs",     type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=50,
                    help="Minimum number of completed episodes to collect before stopping")
parser.add_argument("--max_steps",    type=int, default=5000,
                    help="Hard cap on simulation steps (safety valve)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher  = AppLauncher(args)
sim_app   = launcher.app

# ── 2. Post-launch imports ─────────────────────────────────────────────────────
import numpy as np
import torch
import yaml
import warp as wp
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Make the nmp package importable
sys.path.insert(0, str(Path(__file__).parent / "policy" / "il_dmp"))

import tasks.grasp_and_place  # noqa: F401 — registers GraspAndPlace-v0
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg
from robots.inspire_hand_cfg       import FINGER_JOINTS, FINGER_JOINT_LIMITS

from nmp.model.custom_transformer  import MPTransformer, strip_prefix_from_state_dict


# ── 3. Helper utilities ────────────────────────────────────────────────────────

def load_il_policy(checkpoint_zip: str, device: torch.device) -> MPTransformer:
    """Load MPTransformer from the zip checkpoint."""
    z = zipfile.ZipFile(checkpoint_zip)
    model_cfg = OmegaConf.create(yaml.safe_load(z.read("model_config.yaml")))
    model: MPTransformer = instantiate(model_cfg)

    buf = io.BytesIO(z.read("best.pt"))
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    sd   = strip_prefix_from_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"[eval] Loaded IL policy  ({sum(p.numel() for p in model.parameters()):,} params)"
          f"  epoch={ckpt['epoch']}  eval_SR={ckpt.get('eval_success_rate', 'n/a')}")
    return model


def quat_wxyz_to_euler(q: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, 4) wxyz quaternion → (B, 3) roll-pitch-yaw (radians).
    Uses the standard ZYX Euler convention.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp =  2.0 * (w * x + y * z)
    cosr_cosp =  1.0 - 2.0 * (x * x + y * y)
    roll  = torch.atan2(sinr_cosp, cosr_cosp)

    sinp  = 2.0 * (w * y - z * x)
    sinp  = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp =  2.0 * (w * z + x * y)
    cosy_cosp =  1.0 - 2.0 * (z * z + y * y)
    yaw   = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def make_sphere_pc(centre: torch.Tensor, n: int, radius: float = 0.05) -> torch.Tensor:
    """
    Generate a synthetic spherical point cloud of `n` points around each centre.

    Args:
        centre: (B, 3)  world / scene frame
        n:      number of points per cloud
        radius: sphere radius in metres

    Returns:
        (B, n, 3)
    """
    B, device = centre.shape[0], centre.device
    theta  = torch.rand(B, n, device=device) * 2.0 * torch.pi
    phi    = torch.acos(2.0 * torch.rand(B, n, device=device) - 1.0)
    r      = radius * (torch.rand(B, n, device=device) ** (1.0 / 3.0))
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    pts = torch.stack([x, y, z], dim=-1)       # (B, n, 3)
    return pts + centre.unsqueeze(1)


def build_il_obs(policy_obs: torch.Tensor, device: torch.device) -> dict:
    """
    Convert Isaac Lab flat state observation → IL policy obs dict.

    Isaac Lab obs layout (29-D, bottle-centred):
      [0:3]   hand_pos     (relative to bottle)
      [3:7]   hand_quat    (w, x, y, z)
      [7:19]  joint_pos    (12 = 6 actuated + 6 mimic)
      [19:22] bottle_pos   (always 0 — centred)
      [22:26] bottle_quat
      [26:29] bowl_pos     (relative to bottle)

    IL policy obs keys:
      qpos_wrist_pose    (B, 1, 6)   = hand_pos(3) + euler(3)
      qpos_finger_joints (B, 1, 12)
      target_object_pc   (B, 1, 256, 3)   bottle PC — centred at origin
      env_object_pcs     (B, 1, 1, 256, 3) bowl PC
      robot_hand_pc      (B, 1, 64, 3)    hand PC
    """
    B = policy_obs.shape[0]

    hand_pos   = policy_obs[:, 0:3]    # (B, 3) relative to bottle
    hand_quat  = policy_obs[:, 3:7]    # (B, 4) wxyz
    joint_pos  = policy_obs[:, 7:19]   # (B, 12)
    bowl_pos   = policy_obs[:, 26:29]  # (B, 3) relative to bottle

    euler = quat_wxyz_to_euler(hand_quat)                           # (B, 3)
    qpos_wrist  = torch.cat([hand_pos, euler], dim=-1).unsqueeze(1) # (B, 1, 6)
    qpos_finger = joint_pos.unsqueeze(1)                            # (B, 1, 12)

    # Synthetic point clouds (centred in bottle frame = origin for target obj)
    origin = torch.zeros(B, 3, device=device)
    target_pc = make_sphere_pc(origin,    n=256, radius=0.06).unsqueeze(1)       # (B,1,256,3)
    bowl_pc   = make_sphere_pc(bowl_pos,  n=256, radius=0.08).unsqueeze(1) \
                                                              .unsqueeze(2)       # (B,1,1,256,3)
    hand_pc   = make_sphere_pc(hand_pos,  n=64,  radius=0.04).unsqueeze(1)       # (B,1,64,3)

    return {
        "qpos_wrist_pose":    qpos_wrist.float(),
        "qpos_finger_joints": qpos_finger.float(),
        "target_object_pc":   target_pc.float(),
        "env_object_pcs":     bowl_pc.float(),
        "robot_hand_pc":      hand_pc.float(),
    }


# Pre-compute finger joint limit tensors (same device will be used in the loop)
_FINGER_MINS   = torch.tensor([FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS],
                               dtype=torch.float32)
_FINGER_RANGES = torch.tensor(
    [FINGER_JOINT_LIMITS[j][1] - FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS],
    dtype=torch.float32
)


def il_actions_to_isaac(
    il_actions:       torch.Tensor,   # (B, chunk, 18)
    current_wrist_pos:torch.Tensor,   # (B, 3)  in bottle-centred frame
    pos_action_scale: float,
    device:           torch.device,
) -> torch.Tensor:
    """
    Map IL policy absolute-target output → Isaac Lab 9-D normalised actions.

    IL action layout (18-D absolute targets after delta decoding):
      [:6]  wrist target  = pos(3) + euler(3)
      [6:]  finger joints = 12 target angles (6 actuated + 6 mimic)

    Isaac Lab action layout (9-D, all in [-1, 1]):
      [:3]  position delta normalised by pos_action_scale
      [3:]  6 finger targets normalised to [-1, 1] via joint limits

    Only the first action in the chunk is used.
    """
    wrist_target  = il_actions[:, 0, :3]    # (B, 3) target position
    finger_target = il_actions[:, 0, 6:12]  # (B, 6) first 6 → actuated joints

    # Position: convert absolute target to normalised delta
    delta = wrist_target - current_wrist_pos                    # metres
    action_pos = (delta / pos_action_scale).clamp(-1.0, 1.0)   # (B, 3)

    # Fingers: absolute joint angles → normalised [-1, 1]
    mins   = _FINGER_MINS.to(device)
    ranges = _FINGER_RANGES.to(device)
    finger_norm = ((finger_target - mins) / ranges) * 2.0 - 1.0
    finger_norm = finger_norm.clamp(-1.0, 1.0)                  # (B, 6)

    return torch.cat([action_pos, finger_norm], dim=-1)          # (B, 9)


# ── 4. Build environment ───────────────────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = GraspAndPlaceEnv(cfg=env_cfg)
device = env.device

# ── 5. Load IL policy ──────────────────────────────────────────────────────────

il_model = load_il_policy(args.checkpoint, device)

# ── 6. Evaluation loop ────────────────────────────────────────────────────────

print(f"\n[eval] Running {args.num_envs} parallel envs, "
      f"collecting ≥{args.num_episodes} episodes (max {args.max_steps} steps)\n")

obs_dict, _ = env.reset()

episodes_done    = 0
episodes_success = 0
episode_steps    = torch.zeros(args.num_envs, device=device)
step_idx         = 0

while episodes_done < args.num_episodes and step_idx < args.max_steps:
    policy_obs = obs_dict["policy"]                       # (N, 29)
    current_wrist_pos = policy_obs[:, 0:3].clone()        # (N, 3) bottle-centred

    # Build IL obs and get actions
    il_obs = build_il_obs(policy_obs, device)
    with torch.no_grad():
        il_act = il_model.get_action(il_obs)              # (N, 10, 18)

    actions = il_actions_to_isaac(
        il_act, current_wrist_pos,
        pos_action_scale=env_cfg.pos_action_scale,
        device=device,
    )                                                      # (N, 9)

    obs_dict, rewards, terminated, truncated, _ = env.step(actions)
    episode_steps += 1
    step_idx      += 1

    done = terminated | truncated                          # (N,)
    n_done = int(done.sum().item())
    if n_done > 0:
        n_success        = int((terminated & done).sum().item())
        episodes_done    += n_done
        episodes_success += n_success
        episode_steps[done] = 0   # reset step counter for completed envs

    if step_idx % 100 == 0:
        sr = episodes_success / max(episodes_done, 1)
        print(f"  step={step_idx:5d}  episodes={episodes_done:4d}  "
              f"success={episodes_success:4d}  SR={sr:.3f}")

# ── 7. Results ────────────────────────────────────────────────────────────────

success_rate = episodes_success / max(episodes_done, 1)
print("\n" + "=" * 50)
print(f"  Episodes completed : {episodes_done}")
print(f"  Successes          : {episodes_success}")
print(f"  Success rate       : {success_rate:.4f}  ({success_rate*100:.1f}%)")
print("=" * 50 + "\n")

env.close()
sim_app.close()
