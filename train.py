"""Training entry point for GraspAndPlace Isaac Lab tasks.

Two modes
---------
Vanilla PPO (default)::

    python train.py --num_envs 256

Residual RL — PPO correction on top of a frozen IL base policy::

    python train.py --residual \\
        --il_checkpoint /home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip \\
        --num_envs 256
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── 1. App launcher (MUST come before any omni / isaaclab imports) ─────────────
from isaaclab.app import AppLauncher

_CAMERA_KIT = str(Path(__file__).parent / "camera_headless.kit")

parser = argparse.ArgumentParser(description="Train GraspAndPlace with PPO or residual RL")
parser.add_argument("--num_envs",   type=int, default=256)
parser.add_argument("--max_steps",  type=int, default=50_000_000)
parser.add_argument("--log_dir",    type=str, default="./logs")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from a saved RL checkpoint")

# Camera / Robometer
parser.add_argument("--use_camera",    action="store_true")
parser.add_argument("--use_robometer", action="store_true",
                    help="Enable Robometer dense reward (implies --use_camera)")
parser.add_argument("--robometer_model_path",   type=str, default="robometer/Robometer-4B")
parser.add_argument("--robometer_task",         type=str,
                    default="grasp the bottle and place it in the bowl")
parser.add_argument("--robometer_reward_freq",  type=int, default=20)
parser.add_argument("--robometer_eval_envs",    type=int, default=16)
parser.add_argument("--robometer_reward_scale", type=float, default=10.0)
parser.add_argument("--robometer_device",       type=str, default="cuda:1")

# Residual RL
parser.add_argument("--residual", action="store_true",
                    help="Train a residual PPO policy on top of a frozen IL base")
parser.add_argument("--il_checkpoint", type=str,
                    default="/home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip",
                    help="IL zip checkpoint used as frozen base (--residual mode only)")
parser.add_argument("--residual_scale", type=float, default=0.3,
                    help="Scale applied to the residual action before adding to IL base")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.use_camera or args.use_robometer:
    args.experience     = _CAMERA_KIT
    args.enable_cameras = True

launcher  = AppLauncher(args)
sim_app   = launcher.app

# ── 2. Post-launch imports ─────────────────────────────────────────────────────
import torch
import torch.nn as nn

# Make nmp importable (needed when --residual loads the IL policy)
sys.path.insert(0, str(Path(__file__).parent / "policy" / "il_dmp"))

import tasks.grasp_and_place  # noqa: F401 — registers GraspAndPlace-v0
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg

from skrl.agents.torch.ppo    import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch      import RandomMemory
from skrl.models.torch        import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch      import SequentialTrainer

# ── 3. Vanilla PPO networks ───────────────────────────────────────────────────

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


# ── 4. Residual trainer ───────────────────────────────────────────────────────

class ResidualTrainer(SequentialTrainer):
    """SequentialTrainer override that blends IL base + PPO residual at each step."""

    def __init__(self, base_env, wrapped_env, il_model, pos_action_scale,
                 residual_scale, **kwargs):
        super().__init__(**kwargs)
        self._base_env         = base_env
        self._wrapped          = wrapped_env
        self._il               = il_model
        self._pos_action_scale = pos_action_scale
        self._residual_scale   = residual_scale

    def _step(self, timestep, timesteps):
        from policy.il_utils import build_il_obs, il_actions_to_isaac

        # 1. PPO actor produces residual actions
        with torch.no_grad():
            outputs = self.agents.act(self._obs, timestep=timestep, timesteps=timesteps)
        residual_actions = outputs[0]   # (N, 9)

        # 2. IL base action from current observation
        policy_obs = self._obs["states"] if isinstance(self._obs, dict) else self._obs
        il_obs = build_il_obs(policy_obs, self._base_env.device)
        with torch.no_grad():
            il_act = self._il.get_action(il_obs)
        base_action = il_actions_to_isaac(
            il_act, policy_obs[:, 0:3],
            self._pos_action_scale, self._base_env.device,
        )

        # 3. Combine and step the underlying env
        combined = (base_action + self._residual_scale * residual_actions).clamp(-1.0, 1.0)
        next_obs_dict, rewards, terminated, truncated, infos = self._base_env.step(combined)
        next_states = next_obs_dict["policy"]
        next_obs    = {"states": next_states}
        self._obs   = next_obs

        # 4. Record transition for PPO update
        self.agents.record_transition(
            states={"states": policy_obs},
            actions=residual_actions,
            rewards=rewards,
            next_states=next_obs,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        super()._post_interaction(timestep=timestep, timesteps=timesteps)


# ── 5. Build environment ──────────────────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs
if args.use_camera or args.use_robometer:
    from tasks.grasp_and_place.env_cfg import OBS_STATE_DIM, OBS_CLOUD_DIM
    env_cfg.use_camera        = True
    env_cfg.observation_space = OBS_STATE_DIM + OBS_CLOUD_DIM
if args.use_robometer:
    env_cfg.use_robometer          = True
    env_cfg.robometer_model_path   = args.robometer_model_path
    env_cfg.robometer_task         = args.robometer_task
    env_cfg.robometer_reward_freq  = args.robometer_reward_freq
    env_cfg.robometer_eval_envs    = args.robometer_eval_envs
    env_cfg.robometer_reward_scale = args.robometer_reward_scale
    env_cfg.robometer_device       = args.robometer_device

base_env  = GraspAndPlaceEnv(cfg=env_cfg)
device    = base_env.device
skrl_env  = wrap_env(base_env)

# ── 6. PPO agent (shared by both modes) ──────────────────────────────────────

if args.residual:
    from policy.residual_rl.residual_ppo import ResidualPolicy, ResidualValue
    models = {
        "policy": ResidualPolicy(skrl_env.observation_space, skrl_env.action_space, device),
        "value":  ResidualValue( skrl_env.observation_space, skrl_env.action_space, device),
    }
    experiment_name = "residual_ppo"
else:
    models = {
        "policy": Policy(skrl_env.observation_space, skrl_env.action_space, device),
        "value":  Value( skrl_env.observation_space, skrl_env.action_space, device),
    }
    experiment_name = "GraspAndPlace-v0"

memory  = RandomMemory(memory_size=16, num_envs=skrl_env.num_envs, device=device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo.update({
    "rollouts":              16,
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
    "kl_threshold":          0.0,
    "experiment": {
        "directory":           args.log_dir,
        "experiment_name":     experiment_name,
        "write_interval":      1000,
        "checkpoint_interval": 10_000,
    },
})

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg_ppo,
    observation_space=skrl_env.observation_space,
    action_space=skrl_env.action_space,
    device=device,
)

if args.checkpoint:
    agent.load(args.checkpoint)
    print(f"[train] Resumed from checkpoint: {args.checkpoint}")

# ── 7. Train ──────────────────────────────────────────────────────────────────

trainer_cfg = {"timesteps": args.max_steps, "headless": True}

if args.residual:
    from policy.il_utils import load_il_policy
    from policy.residual_rl.wrapper import ResidualEnvWrapper

    il_model    = load_il_policy(args.il_checkpoint, device, freeze=True)
    wrapped_env = ResidualEnvWrapper(base_env, il_model, args.residual_scale)

    obs_dict, _ = base_env.reset()
    trainer = ResidualTrainer(
        base_env=base_env,
        wrapped_env=wrapped_env,
        il_model=il_model,
        pos_action_scale=env_cfg.pos_action_scale,
        residual_scale=args.residual_scale,
        env=skrl_env,
        agents=agent,
        cfg=trainer_cfg,
    )
    trainer._obs = {"states": obs_dict["policy"]}
    print(f"\n[train] Residual RL  envs={args.num_envs}"
          f"  residual_scale={args.residual_scale}  device={device}")
else:
    trainer = SequentialTrainer(cfg=trainer_cfg, env=skrl_env, agents=agent)
    print(f"\n[train] Vanilla PPO  envs={args.num_envs}  device={device}")

print(f"[train] Logs → {os.path.abspath(args.log_dir)}\n")
trainer.train()

# ── 8. Cleanup ────────────────────────────────────────────────────────────────

base_env.close()
sim_app.close()
