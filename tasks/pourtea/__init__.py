"""Pourtea task: pick up the bottle and pour tea into the bowl."""

import gymnasium as gym

gym.register(
    id="Pourtea-v0",
    entry_point="tasks.pourtea.env:PourteaEnv",
    kwargs={"cfg": None},
    disable_env_checker=True,
)
