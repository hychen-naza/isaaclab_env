"""Grasp-and-place task: pick up the bottle with the Inspire hand and move it to the bowl."""

import gymnasium as gym

gym.register(
    id="GraspAndPlace-v0",
    entry_point="tasks.grasp_and_place.env:GraspAndPlaceEnv",
    kwargs={"cfg": None},  # cfg supplied at make() time via kwargs
    disable_env_checker=True,
)
