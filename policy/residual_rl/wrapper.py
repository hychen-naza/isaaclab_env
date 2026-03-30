"""ResidualEnvWrapper: blends a frozen IL base policy with a PPO residual.

Used by both train.py (ResidualTrainer) and eval.py (residual eval mode).
"""
from __future__ import annotations

import torch

from policy.il_utils import build_il_obs, il_actions_to_isaac


class ResidualEnvWrapper:
    """Wraps GraspAndPlaceEnv so the PPO actor learns a correction on top of IL.

    At each step:
        a_base     = IL_policy(obs)                   # frozen
        a_residual = PPO_actor(obs)                   # learnable
        a_env      = clip(a_base + scale * a_residual, -1, 1)
    """

    def __init__(self, env, il_model, residual_scale: float):
        self._env           = env
        self._il            = il_model
        self.residual_scale = residual_scale
        self.device         = env.device
        self.num_envs       = env.num_envs

        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.state_space       = env.state_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    def step_with_obs(self, policy_obs: torch.Tensor,
                      residual_actions: torch.Tensor):
        """Step the env using IL base + scaled residual.

        Args:
            policy_obs:       (N, 29) flat state obs for the *current* step.
            residual_actions: (N, 9)  PPO actor output in [-1, 1].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        il_obs = build_il_obs(policy_obs, self.device)
        with torch.no_grad():
            il_act = self._il.get_action(il_obs)       # (N, chunk, 18)
        base_action = il_actions_to_isaac(
            il_act, policy_obs[:, 0:3],
            self._env.cfg.pos_action_scale, self.device,
        )
        combined = (base_action + self.residual_scale * residual_actions).clamp(-1.0, 1.0)
        return self._env.step(combined)
