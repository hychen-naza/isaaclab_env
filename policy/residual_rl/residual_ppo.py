"""Residual RL: PPO policy that learns a correction on top of a frozen IL base policy.

Architecture
------------
At every environment step:

    a_base     = IL_policy(obs)          # frozen — from the imitation checkpoint
    a_residual = PPO_actor(obs)          # learnable — trained with PPO
    a_env      = clip(a_base + a_residual, -1, 1)   # sent to Isaac Lab env

The IL policy sees the same hand-crafted obs dict it was trained on (point clouds,
wrist pose, finger joints) while the PPO actor receives the flat 29-D state vector
from the env.  This asymmetry is intentional: the IL policy supplies a strong prior
and the residual actor needs only a compact state to learn corrections.

Training
--------
    cd /home/hongyi/scalevideomanip/isaacsim_scene
    python policy/residual_rl/residual_ppo.py \
        --il_checkpoint /home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip \
        --num_envs 256 \
        --max_steps 50_000_000
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

# ── 1. App launcher (must come before omni / isaaclab) ────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Residual RL on top of IL base policy")
parser.add_argument("--il_checkpoint", type=str,
                    default="/home/hongyi/scalevideomanip/policy_checkpoint/impact_bottle_bowl.zip")
parser.add_argument("--num_envs",   type=int, default=256)
parser.add_argument("--max_steps",  type=int, default=50_000_000)
parser.add_argument("--log_dir",    type=str, default="./logs/residual_rl")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Residual RL checkpoint to resume from")
parser.add_argument("--residual_scale", type=float, default=0.3,
                    help="Scale factor applied to the residual action before adding to IL base")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
sim_app   = launcher.app

# ── 2. Post-launch imports ─────────────────────────────────────────────────────
import yaml
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).parent.parent / "il_dmp"))

import tasks.grasp_and_place  # noqa: F401
from tasks.grasp_and_place.env     import GraspAndPlaceEnv
from tasks.grasp_and_place.env_cfg import GraspAndPlaceEnvCfg
from robots.inspire_hand_cfg       import FINGER_JOINTS, FINGER_JOINT_LIMITS

from nmp.model.custom_transformer  import MPTransformer, strip_prefix_from_state_dict

from skrl.agents.torch.ppo    import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch      import RandomMemory
from skrl.models.torch        import DeterministicMixin, GaussianMixin, Model

# ── 3. Observation adapters (shared with eval.py) ─────────────────────────────

def quat_wxyz_to_euler(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll  = torch.atan2(sinr, cosr)
    sinp  = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    siny  = 2.0 * (w * z + x * y)
    cosy  = 1.0 - 2.0 * (z * z + y * y)
    yaw   = torch.atan2(siny, cosy)
    return torch.stack([roll, pitch, yaw], dim=-1)


def make_sphere_pc(centre: torch.Tensor, n: int, radius: float = 0.05) -> torch.Tensor:
    B, device = centre.shape[0], centre.device
    theta = torch.rand(B, n, device=device) * 2.0 * torch.pi
    phi   = torch.acos(2.0 * torch.rand(B, n, device=device) - 1.0)
    r     = radius * (torch.rand(B, n, device=device) ** (1.0 / 3.0))
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=-1) + centre.unsqueeze(1)


def build_il_obs(policy_obs: torch.Tensor, device: torch.device) -> dict:
    B          = policy_obs.shape[0]
    hand_pos   = policy_obs[:, 0:3]
    hand_quat  = policy_obs[:, 3:7]
    joint_pos  = policy_obs[:, 7:19]
    bowl_pos   = policy_obs[:, 26:29]
    origin     = torch.zeros(B, 3, device=device)
    euler      = quat_wxyz_to_euler(hand_quat)
    return {
        "qpos_wrist_pose":    torch.cat([hand_pos, euler], -1).unsqueeze(1).float(),
        "qpos_finger_joints": joint_pos.unsqueeze(1).float(),
        "target_object_pc":   make_sphere_pc(origin,   n=256, radius=0.06).unsqueeze(1).float(),
        "env_object_pcs":     make_sphere_pc(bowl_pos, n=256, radius=0.08).unsqueeze(1).unsqueeze(2).float(),
        "robot_hand_pc":      make_sphere_pc(hand_pos, n=64,  radius=0.04).unsqueeze(1).float(),
    }


_FINGER_MINS   = torch.tensor([FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS], dtype=torch.float32)
_FINGER_RANGES = torch.tensor(
    [FINGER_JOINT_LIMITS[j][1] - FINGER_JOINT_LIMITS[j][0] for j in FINGER_JOINTS],
    dtype=torch.float32,
)


def il_to_isaac_action(
    il_actions:        torch.Tensor,   # (B, chunk, 18)
    current_wrist_pos: torch.Tensor,   # (B, 3)
    pos_action_scale:  float,
    device:            torch.device,
) -> torch.Tensor:
    """Return 9-D Isaac Lab action from IL decoded absolute targets."""
    wrist_target  = il_actions[:, 0, :3]
    finger_target = il_actions[:, 0, 6:12]
    delta         = (wrist_target - current_wrist_pos) / pos_action_scale
    action_pos    = delta.clamp(-1.0, 1.0)
    mins, ranges  = _FINGER_MINS.to(device), _FINGER_RANGES.to(device)
    finger_norm   = ((finger_target - mins) / ranges * 2.0 - 1.0).clamp(-1.0, 1.0)
    return torch.cat([action_pos, finger_norm], dim=-1)


def load_il_policy(checkpoint_zip: str, device: torch.device) -> MPTransformer:
    z    = zipfile.ZipFile(checkpoint_zip)
    cfg  = OmegaConf.create(yaml.safe_load(z.read("model_config.yaml")))
    model: MPTransformer = instantiate(cfg)
    buf  = io.BytesIO(z.read("best.pt"))
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_prefix_from_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[residual_rl] IL policy loaded and frozen  "
          f"(epoch={ckpt['epoch']}, SR={ckpt.get('eval_success_rate', 'n/a')})")
    return model


# ── 4. Residual RL environment wrapper ────────────────────────────────────────

class ResidualEnvWrapper:
    """
    Wraps GraspAndPlaceEnv so that the PPO actor produces *residual* actions.

    The wrapped step():
      1. Gets base action from the frozen IL policy.
      2. Adds the PPO residual (scaled by residual_scale).
      3. Clips the combined action to [-1, 1].
      4. Steps the underlying Isaac Lab env.

    The PPO agent therefore optimises only the residual on top of IL.
    """

    def __init__(
        self,
        env:            GraspAndPlaceEnv,
        il_model:       MPTransformer,
        residual_scale: float,
    ):
        self._env            = env
        self._il             = il_model
        self.residual_scale  = residual_scale
        self.device          = env.device
        self.num_envs        = env.num_envs

        # Expose spaces expected by skrl
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.state_space       = env.state_space

    # Delegate attribute look-ups to the wrapped env (needed by skrl)
    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    def step(self, residual_actions: torch.Tensor):
        """
        Args:
            residual_actions: (N, 9) from the PPO actor, in [-1, 1]

        Returns:
            Standard (obs, reward, terminated, truncated, info) tuple.
        """
        obs_dict = self._env.observation_manager.compute() \
            if hasattr(self._env, "observation_manager") \
            else self._last_obs

        # Get base IL action from last stored observation
        il_obs  = build_il_obs(self._last_policy_obs, self.device)
        with torch.no_grad():
            il_act = self._il.get_action(il_obs)                 # (N, 10, 18)
        base_action = il_to_isaac_action(
            il_act, self._last_policy_obs[:, 0:3],
            self._env.cfg.pos_action_scale, self.device
        )

        combined = (base_action + self.residual_scale * residual_actions).clamp(-1.0, 1.0)
        obs, rew, term, trunc, info = self._env.step(combined)
        self._last_policy_obs = obs["policy"]
        return obs, rew, term, trunc, info

    def step_with_obs(self, policy_obs: torch.Tensor, residual_actions: torch.Tensor):
        """
        Convenience wrapper used inside the custom trainer below.
        `policy_obs` is the CURRENT observation (before this step).
        """
        il_obs  = build_il_obs(policy_obs, self.device)
        with torch.no_grad():
            il_act = self._il.get_action(il_obs)                 # (N, 10, 18)
        base_action = il_to_isaac_action(
            il_act, policy_obs[:, 0:3],
            self._env.cfg.pos_action_scale, self.device
        )
        combined = (base_action + self.residual_scale * residual_actions).clamp(-1.0, 1.0)
        return self._env.step(combined)


# ── 5. PPO actor / critic ─────────────────────────────────────────────────────

class ResidualPolicy(GaussianMixin, Model):
    """Stochastic residual policy (actor).

    Initialised with near-zero mean so it starts as a near-identity correction.
    """
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        obs_dim    = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, action_dim),
        )
        # Start near-zero so the policy initially passes through the IL base action
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)
        # Log-std starts negative → small initial std
        self.log_std_param = nn.Parameter(torch.full((action_dim,), -2.0))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_param, {}


class ResidualValue(DeterministicMixin, Model):
    """Value network (critic) for residual PPO."""
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ── 6. Build env and residual wrapper ─────────────────────────────────────────

env_cfg = GraspAndPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs

base_env = GraspAndPlaceEnv(cfg=env_cfg)
device   = base_env.device

il_model = load_il_policy(args.il_checkpoint, device)

wrapped_env = ResidualEnvWrapper(base_env, il_model, residual_scale=args.residual_scale)

# skrl wrapper (handles observation/action space introspection)
skrl_env = wrap_env(base_env)

# ── 7. PPO agent ──────────────────────────────────────────────────────────────

models = {
    "policy": ResidualPolicy(skrl_env.observation_space, skrl_env.action_space, device),
    "value":  ResidualValue( skrl_env.observation_space, skrl_env.action_space, device),
}

memory = RandomMemory(memory_size=16, num_envs=skrl_env.num_envs, device=device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo.update({
    "rollouts":           16,
    "learning_epochs":    8,
    "mini_batches":       4,
    "discount_factor":    0.99,
    "lambda":             0.95,
    "learning_rate":      3e-4,
    "grad_norm_clip":     1.0,
    "ratio_clip":         0.2,
    "value_clip":         0.2,
    "entropy_loss_scale": 0.01,
    "value_loss_scale":   1.0,
    "kl_threshold":       0.0,
    "experiment": {
        "directory":          args.log_dir,
        "experiment_name":    "residual_ppo",
        "write_interval":     1000,
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
    print(f"[residual_rl] Resumed agent from: {args.checkpoint}")

# ── 8. Custom training loop ───────────────────────────────────────────────────
# We use a manual loop instead of skrl SequentialTrainer so we can intercept
# each step and inject the IL base action before stepping the environment.

from skrl.trainers.torch import SequentialTrainer

class ResidualTrainer(SequentialTrainer):
    """
    Trainer that overrides the environment step to blend IL base + residual.

    The PPO agent produces residual_actions; we combine them with IL base
    actions before passing to the Isaac Lab env.
    """

    def __init__(self, env, wrapped_env, il_model, **kwargs):
        # Pass the skrl-wrapped base env so skrl handles obs/action spaces
        super().__init__(env=env, **kwargs)
        self._wrapped = wrapped_env
        self._il      = il_model

    def _step(self, timestep, timesteps):
        # --- 1. Sample actions from PPO agent (residual) ---
        with torch.no_grad():
            outputs = self.agents.act(self._obs, timestep=timestep, timesteps=timesteps)
        residual_actions = outputs[0]  # (N, 9)

        # --- 2. Combine with IL base action ---
        policy_obs = self._obs["states"] if isinstance(self._obs, dict) else self._obs

        il_obs      = build_il_obs(policy_obs, device)
        with torch.no_grad():
            il_act  = self._il.get_action(il_obs)
        base_action = il_to_isaac_action(
            il_act, policy_obs[:, 0:3],
            env_cfg.pos_action_scale, device,
        )
        combined = (base_action + args.residual_scale * residual_actions).clamp(-1.0, 1.0)

        # --- 3. Step the underlying Isaac Lab env directly ---
        next_obs_dict, rewards, terminated, truncated, infos = base_env.step(combined)
        next_states = next_obs_dict["policy"]

        # Package in the format skrl expects
        next_obs = {"states": next_states}
        self._obs = next_obs

        # --- 4. Record the transition in memory ---
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


# Initialise with a reset observation
obs_dict, _ = base_env.reset()
initial_obs  = {"states": obs_dict["policy"]}

trainer_cfg = {"timesteps": args.max_steps, "headless": True}
trainer = ResidualTrainer(
    env=skrl_env,
    wrapped_env=wrapped_env,
    il_model=il_model,
    agents=agent,
    cfg=trainer_cfg,
)
# Inject initial obs so _step() can read it
trainer._obs = initial_obs

print(f"\n[residual_rl] Training  envs={args.num_envs}  "
      f"residual_scale={args.residual_scale}  device={device}")
print(f"[residual_rl] Logs → {os.path.abspath(args.log_dir)}\n")

trainer.train()

base_env.close()
sim_app.close()
