"""Shared IL policy utilities.

Imported by both eval.py and train.py (residual mode).
Callers must ensure policy/il_dmp is on sys.path before importing this module
so that `nmp` is resolvable.
"""
from __future__ import annotations

import io
import zipfile

import torch
import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate

from robots.inspire_hand_cfg import FINGER_JOINTS, FINGER_JOINT_LIMITS
from nmp.model.custom_transformer import MPTransformer, strip_prefix_from_state_dict

# Pre-computed finger joint limit tensors (CPU; callers move to device as needed)
_FINGER_MINS = torch.tensor(
    [FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS], dtype=torch.float32
)
_FINGER_RANGES = torch.tensor(
    [FINGER_JOINT_LIMITS[j][1] - FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS],
    dtype=torch.float32,
)


def load_il_policy(checkpoint_zip: str, device: torch.device,
                   freeze: bool = False) -> MPTransformer:
    """Load MPTransformer from zip checkpoint."""
    z = zipfile.ZipFile(checkpoint_zip)
    model_cfg = OmegaConf.create(yaml.safe_load(z.read("model_config.yaml")))
    model: MPTransformer = instantiate(model_cfg)
    buf  = io.BytesIO(z.read("best.pt"))
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_prefix_from_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    tag = "  [frozen]" if freeze else ""
    print(f"[il_utils] Loaded IL policy ({n_params:,} params)"
          f"  epoch={ckpt['epoch']}  eval_SR={ckpt.get('eval_success_rate', 'n/a')}{tag}")
    return model


def quat_wxyz_to_euler(q: torch.Tensor) -> torch.Tensor:
    """(B, 4) wxyz quaternion → (B, 3) roll-pitch-yaw (radians)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr  = 2.0 * (w * x + y * z)
    cosr  = 1.0 - 2.0 * (x * x + y * y)
    roll  = torch.atan2(sinr, cosr)
    sinp  = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    siny  = 2.0 * (w * z + x * y)
    cosy  = 1.0 - 2.0 * (z * z + y * y)
    yaw   = torch.atan2(siny, cosy)
    return torch.stack([roll, pitch, yaw], dim=-1)


def make_sphere_pc(centre: torch.Tensor, n: int, radius: float = 0.05) -> torch.Tensor:
    """Generate a synthetic spherical point cloud around each centre.

    Args:
        centre: (B, 3)
        n:      points per cloud
        radius: sphere radius in metres

    Returns:
        (B, n, 3)
    """
    B, device = centre.shape[0], centre.device
    theta = torch.rand(B, n, device=device) * 2.0 * torch.pi
    phi   = torch.acos(2.0 * torch.rand(B, n, device=device) - 1.0)
    r     = radius * (torch.rand(B, n, device=device) ** (1.0 / 3.0))
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=-1) + centre.unsqueeze(1)


def build_il_obs(policy_obs: torch.Tensor, device: torch.device) -> dict:
    """Convert Isaac Lab flat state obs (29-D) → IL policy obs dict.

    Isaac Lab obs layout:
      [0:3]   hand_pos   (relative to bottle)
      [3:7]   hand_quat  (w, x, y, z)
      [7:19]  joint_pos  (12 DOF)
      [19:22] bottle_pos (always 0 — centred frame)
      [22:26] bottle_quat
      [26:29] bowl_pos   (relative to bottle)
    """
    B         = policy_obs.shape[0]
    hand_pos  = policy_obs[:, 0:3]
    hand_quat = policy_obs[:, 3:7]
    joint_pos = policy_obs[:, 7:19]
    bowl_pos  = policy_obs[:, 26:29]
    origin    = torch.zeros(B, 3, device=device)
    euler     = quat_wxyz_to_euler(hand_quat)
    return {
        "qpos_wrist_pose":    torch.cat([hand_pos, euler], -1).unsqueeze(1).float(),
        "qpos_finger_joints": joint_pos.unsqueeze(1).float(),
        "target_object_pc":   make_sphere_pc(origin,   n=256, radius=0.06).unsqueeze(1).float(),
        "env_object_pcs":     make_sphere_pc(bowl_pos, n=256, radius=0.08).unsqueeze(1).unsqueeze(2).float(),
        "robot_hand_pc":      make_sphere_pc(hand_pos, n=64,  radius=0.04).unsqueeze(1).float(),
    }


def il_actions_to_isaac(
    il_actions:        torch.Tensor,  # (B, chunk, 18)
    current_wrist_pos: torch.Tensor,  # (B, 3)
    pos_action_scale:  float,
    device:            torch.device,
) -> torch.Tensor:
    """Map IL absolute-target output → Isaac Lab 9-D normalised action.

    IL layout:  [:6] wrist target = pos(3)+euler(3),  [6:] finger joints (12)
    Isaac Lab:  [:3] pos delta / scale,  [3:] 6 finger targets → [-1, 1]
    """
    wrist_target  = il_actions[:, 0, :3]
    finger_target = il_actions[:, 0, 6:12]
    delta         = (wrist_target - current_wrist_pos) / pos_action_scale
    action_pos    = delta.clamp(-1.0, 1.0)
    mins   = _FINGER_MINS.to(device)
    ranges = _FINGER_RANGES.to(device)
    finger_norm = ((finger_target - mins) / ranges * 2.0 - 1.0).clamp(-1.0, 1.0)
    return torch.cat([action_pos, finger_norm], dim=-1)
