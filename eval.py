"""Evaluate IL or residual RL policy in the GraspAndPlace Isaac Lab env.

Usage
-----
IL policy (default)::

    python eval.py

Residual RL policy::

    python eval.py --residual \\
        --residual_checkpoint logs/residual_rl/residual_ppo/checkpoints/agent_50000.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── 1. App launcher (MUST come before any omni / isaaclab imports) ─────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")

parser = argparse.ArgumentParser(description="Evaluate policy in GraspAndPlace env")
parser.add_argument("--il_checkpoint", type=str,
                    default="/home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip")
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Minimum completed episodes to collect before stopping")
parser.add_argument("--max_steps",    type=int, default=5000,
                    help="Hard cap on simulation steps")
parser.add_argument("--save_video",   action="store_true", default=True,
                    help="Save RGB frames as a video (enables camera)")
parser.add_argument("--video_env",    type=int, default=0,
                    help="Which env index to record")
parser.add_argument("--video_out",    type=str, default="eval_il_policy.mp4")
parser.add_argument("--video_fps",    type=int, default=30)

# Residual RL eval
parser.add_argument("--residual", action="store_true",
                    help="Evaluate a residual RL policy on top of the IL base")
parser.add_argument("--residual_checkpoint", type=str, default=None,
                    help="Saved residual PPO checkpoint (.pt)")
parser.add_argument("--residual_scale", type=float, default=0.3)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.save_video:
    args.experience     = _CAMERA_KIT
    args.enable_cameras = True

launcher  = AppLauncher(args)
sim_app   = launcher.app

# ── 2. Post-launch imports ─────────────────────────────────────────────────────
import imageio
import numpy as np
import torch
import warp as wp

# Make nmp importable before policy.il_utils is loaded
sys.path.insert(0, str(Path(__file__).parent / "policy" / "il_dmp"))

import tasks.grasp_and_place  # noqa: F401 — registers GraspAndPlace-v0
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg

from policy.il_utils import load_il_policy, build_il_obs, il_actions_to_isaac

# ── 3. Build environment ───────────────────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs
if args.save_video:
    from tasks.grasp_and_place.env_cfg import OBS_STATE_DIM, OBS_CLOUD_DIM
    env_cfg.use_camera        = True
    env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
env    = GraspAndPlaceEnv(cfg=env_cfg)
device = env.device

# ── 4. Load policy ────────────────────────────────────────────────────────────

il_model = load_il_policy(args.il_checkpoint, device, freeze=True)

residual_agent = None
if args.residual:
    from policy.residual_rl.residual_ppo import ResidualPolicy, ResidualValue
    from policy.residual_rl.wrapper      import ResidualEnvWrapper
    from skrl.agents.torch.ppo           import PPO, PPO_DEFAULT_CONFIG
    from skrl.envs.wrappers.torch        import wrap_env

    skrl_env = wrap_env(env)
    models = {
        "policy": ResidualPolicy(skrl_env.observation_space, skrl_env.action_space, device),
        "value":  ResidualValue( skrl_env.observation_space, skrl_env.action_space, device),
    }
    residual_agent = PPO(
        models=models, memory=None,
        cfg=PPO_DEFAULT_CONFIG.copy(),
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=device,
    )
    if args.residual_checkpoint:
        residual_agent.load(args.residual_checkpoint)
        print(f"[eval] Loaded residual checkpoint: {args.residual_checkpoint}")
    wrapped_env = ResidualEnvWrapper(env, il_model, args.residual_scale)

# ── 5. Evaluation loop ────────────────────────────────────────────────────────

mode_tag = "residual RL" if args.residual else "IL"
print(f"\n[eval] {mode_tag} policy  |  {args.num_envs} env(s)"
      f"  |  collecting ≥{args.num_episodes} episodes (max {args.max_steps} steps)")
if args.save_video:
    print(f"[eval] Recording env {args.video_env} → {args.video_out}\n")

obs_dict, _ = env.reset()

episodes_done    = 0
episodes_success = 0
episode_steps    = torch.zeros(args.num_envs, device=device)
step_idx         = 0
rgb_frames: list[np.ndarray] = []

while episodes_done < args.num_episodes and step_idx < args.max_steps:
    policy_obs = obs_dict["policy"][:, :29]   # always use 29-D state slice
    current_wrist_pos = policy_obs[:, 0:3].clone()

    # Capture frame before stepping
    if args.save_video and "rgb" in obs_dict:
        frame = obs_dict["rgb"][args.video_env]
        rgb_frames.append(frame[..., :3].cpu().numpy().astype(np.uint8))

    # Compute action
    il_obs = build_il_obs(policy_obs, device)
    with torch.no_grad():
        il_act = il_model.get_action(il_obs)   # (N, chunk, 18)
    base_action = il_actions_to_isaac(
        il_act, current_wrist_pos,
        env_cfg.pos_action_scale, device,
    )

    if args.residual and residual_agent is not None:
        with torch.no_grad():
            res_out = residual_agent.act({"states": policy_obs},
                                         timestep=step_idx, timesteps=args.max_steps)
        residual_actions = res_out[0]
        actions = (base_action + args.residual_scale * residual_actions).clamp(-1.0, 1.0)
    else:
        actions = base_action

    obs_dict, rewards, terminated, truncated, _ = env.step(actions)
    episode_steps += 1
    step_idx      += 1

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

# ── 6. Results ────────────────────────────────────────────────────────────────

success_rate = episodes_success / max(episodes_done, 1)
print("\n" + "=" * 50)
print(f"  Episodes completed : {episodes_done}")
print(f"  Successes          : {episodes_success}")
print(f"  Success rate       : {success_rate:.4f}  ({success_rate*100:.1f}%)")
print("=" * 50 + "\n")

# ── 7. Save video ─────────────────────────────────────────────────────────────

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
