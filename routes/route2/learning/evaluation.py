"""Batch deterministic policy calls across independently seeded held-out episodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def evaluate_batched(
    env_id: str,
    episodes: int,
    seed_start: int,
    policy_actions_callable: Callable[[np.ndarray], np.ndarray],
) -> list[dict[str, Any]]:
    """Evaluate each held-out seed once, batching all currently active states.

    The callback receives raw observations in increasing seed order and must
    return deterministic actions already transformed to environment units. It
    should use frozen policy/normalizer state and must not sample or update
    statistics. This helper does not seed, sample from, save, or restore any
    global RNG; each separate Gymnasium environment receives its own reset seed.

    Args:
        env_id: Gymnasium environment identifier.
        episodes: Number of distinct seeded episodes to evaluate.
        seed_start: First nonnegative held-out seed.
        policy_actions_callable: Batch of raw observations to batch of actions.

    Returns:
        Raw, undiscounted episode returns and lengths, ordered by seed.

    Raises:
        ValueError: Invalid counts/seeds or callback action batch size.
        Exception: Environment or policy failure. All created environments are
            closed even if another environment's close method raises.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or seed_start < 0
    ):
        raise ValueError("seed_start must be a nonnegative integer")
    if not callable(policy_actions_callable):
        raise ValueError("policy_actions_callable must be callable")

    # Keep CPU fake-environment tests usable without importing an installed gym.
    import gymnasium as gym

    environments = []
    exception_pending = False
    try:
        observations = []
        for index in range(episodes):
            environment = gym.make(env_id)
            environments.append(environment)
            observation, _ = environment.reset(seed=seed_start + index)
            observations.append(observation)

        returns = np.zeros(episodes, dtype=np.float64)
        lengths = np.zeros(episodes, dtype=np.int64)
        active = list(range(episodes))
        while active:
            observation_batch = np.stack([observations[index] for index in active])
            action_batch = np.asarray(policy_actions_callable(observation_batch))
            if action_batch.ndim == 0 or action_batch.shape[0] != len(active):
                raise ValueError(
                    "Policy action batch must have one leading entry per active "
                    f"environment; expected {len(active)}, got {action_batch.shape}"
                )
            next_active = []
            for index, action in zip(active, action_batch):
                observation, reward, terminated, truncated, _ = environments[
                    index
                ].step(action)
                returns[index] += float(reward)
                lengths[index] += 1
                observations[index] = observation
                if not (terminated or truncated):
                    next_active.append(index)
            active = next_active

        return [
            {
                "seed": seed_start + index,
                "return": float(returns[index]),
                "length": int(lengths[index]),
            }
            for index in range(episodes)
        ]
    except BaseException:
        exception_pending = True
        raise
    finally:
        close_errors = []
        for environment in environments:
            try:
                environment.close()
            except BaseException as error:
                close_errors.append(error)
        if close_errors and not exception_pending:
            raise close_errors[0]
