"""Residual RL actor and critic networks for PPO."""
from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class ResidualPolicy(GaussianMixin, Model):
    """Stochastic residual actor.

    Initialised near-zero so the IL base action dominates at the start of training.
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
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)
        # Negative log-std → small initial std so residual starts near zero
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
