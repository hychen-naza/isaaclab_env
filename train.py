"""PPO training entry point for Isaac Lab RL tasks.

Usage:
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python train.py --task GraspAndPlace-v0 --num_envs 256

Dependencies (install once):
    pip install skrl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── 1. App launcher (MUST come before any omni / isaaclab imports) ────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")

parser = argparse.ArgumentParser(description="Train an Isaac Lab RL task with PPO")
parser.add_argument("--task",        type=str,  default="GraspAndPlace-v0")
parser.add_argument("--num_envs",    type=int,  default=256)
parser.add_argument("--max_steps",   type=int,  default=50_000_000,
                    help="Total environment steps to train for")
parser.add_argument("--log_dir",     type=str,  default="./logs")
parser.add_argument("--checkpoint",  type=str,  default=None,
                    help="Path to a saved checkpoint to resume from")
parser.add_argument("--use_camera",  action="store_true",
                    help="Enable TiledCamera observations (requires camera_headless.kit)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True   # training always runs headless
if args.use_camera:
    args.experience = _CAMERA_KIT
    args.enable_cameras = True

launcher  = AppLauncher(args)
sim_app   = launcher.app

# ── 2. Post-launch imports ────────────────────────────────────────────────────
import torch
import torch.nn as nn

import gymnasium as gym

import tasks.grasp_and_place  # noqa: F401 — registers GraspAndPlace-v0
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg

from skrl.agents.torch.ppo      import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch   import wrap_env
from skrl.memories.torch        import RandomMemory
from skrl.models.torch          import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch        import SequentialTrainer

# ── 3. Network definitions ────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Stochastic policy network (actor)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        obs_dim    = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_param, {}


class Value(DeterministicMixin, Model):
    """Value network (critic)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        obs_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ── 4. Build environment ──────────────────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs
if args.use_camera:
    from tasks.grasp_and_place.env_cfg import OBS_STATE_DIM, OBS_CLOUD_DIM
    env_cfg.use_camera = True
    env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
env = GraspAndPlaceEnv(cfg=env_cfg)
env = wrap_env(env)          # skrl Isaac Lab wrapper

device = env.device

# ── 5. PPO agent ──────────────────────────────────────────────────────────────

models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value":  Value(env.observation_space,  env.action_space, device),
}

# Rollout buffer: store 16 steps × num_envs transitions before each update
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo.update({
    "rollouts":              16,      # steps before each update
    "learning_epochs":       8,
    "mini_batches":          4,
    "discount_factor":       0.99,
    "lambda":                0.95,
    "learning_rate":         3e-4,
    "grad_norm_clip":        1.0,
    "ratio_clip":            0.2,
    "value_clip":            0.2,
    "entropy_loss_scale":    0.01,
    "value_loss_scale":      1.0,
    "kl_threshold":          0.0,     # disabled
    "experiment": {
        "directory":         args.log_dir,
        "experiment_name":   args.task,
        "write_interval":    1000,
        "checkpoint_interval": 10_000,
    },
})

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

if args.checkpoint:
    agent.load(args.checkpoint)
    print(f"[train] Resumed from checkpoint: {args.checkpoint}")

# ── 6. Train ──────────────────────────────────────────────────────────────────

trainer_cfg = {"timesteps": args.max_steps, "headless": True}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

print(f"\n[train] Task: {args.task}  |  envs: {args.num_envs}  |  device: {device}")
print(f"[train] Logs → {os.path.abspath(args.log_dir)}\n")
trainer.train()

# ── 7. Cleanup ────────────────────────────────────────────────────────────────

env.close()
sim_app.close()
