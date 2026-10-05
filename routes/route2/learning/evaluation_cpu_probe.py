"""Compare real CPU HalfCheetah serial/batched evaluation with a fixed policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from evaluation import evaluate_batched


class FixedLinearPolicy:
    """Use deterministic CPU coefficients and reductions, with no RNG sampling."""

    def __init__(self, observation_dim: int, low: np.ndarray, high: np.ndarray) -> None:
        action_dim = len(low)
        indices = np.arange(action_dim * observation_dim, dtype=np.float64).reshape(
            action_dim, observation_dim
        )
        self.weights = 0.004 * np.cos(indices + 1)
        self.bias = 0.02 * np.sin(np.arange(action_dim, dtype=np.float64) + 1)
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.calls = 0

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """Evaluate one batch with a batch-size-independent last-axis reduction."""
        self.calls += 1
        raw = np.asarray(observations, dtype=np.float64)
        # Identical reduction ordering for singleton and multi-episode batches
        # prevents BLAS shape choices from changing this chaotic simulator test.
        linear = (raw[:, None, :] * self.weights[None, :, :]).sum(axis=-1)
        return np.clip(linear + self.bias, self.low, self.high)


def serial_evaluation(
    env_id: str, episodes: int, seed_start: int, policy: FixedLinearPolicy
) -> list[dict[str, Any]]:
    """Evaluate one complete seeded episode at a time as an independent reference."""
    results = []
    for seed in range(seed_start, seed_start + episodes):
        environment = gym.make(env_id)
        try:
            observation, _ = environment.reset(seed=seed)
            total, length = 0.0, 0
            while True:
                action = policy(np.asarray(observation)[None])[0]
                observation, reward, terminated, truncated, _ = environment.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            results.append({"seed": seed, "return": total, "length": length})
        finally:
            environment.close()
    return results


def rng_snapshot() -> dict[str, Any]:
    """Read Python, NumPy and Torch CPU states without accessing GPU APIs."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def rng_unchanged(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    """Compare each state, including NumPy's cached Gaussian fields."""
    numpy_before, numpy_after = before["numpy"], after["numpy"]
    return {
        "python": before["python"] == after["python"],
        "numpy": numpy_before[0] == numpy_after[0]
        and np.array_equal(numpy_before[1], numpy_after[1])
        and numpy_before[2:] == numpy_after[2:],
        "torch_cpu": torch.equal(before["torch_cpu"], after["torch_cpu"]),
    }


def main() -> None:
    """Time two CPU-only evaluators and require paired outcome/RNG equivalence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error("episodes must be positive and seed-start nonnegative")

    # Resolve imports/model loading and spaces before timing the evaluators.
    setup_before = rng_snapshot()
    environment = gym.make(args.env_id)
    try:
        observation_dim = environment.observation_space.shape[0]
        low, high = (
            environment.action_space.low.copy(),
            environment.action_space.high.copy(),
        )
        observation, _ = environment.reset(seed=args.seed_start + 10000)
        warmup_policy = FixedLinearPolicy(observation_dim, low, high)
        for _ in range(20):
            observation, _, terminated, truncated, _ = environment.step(
                warmup_policy(np.asarray(observation)[None])[0]
            )
            if terminated or truncated:
                break
    finally:
        environment.close()
    setup_rng = rng_unchanged(setup_before, rng_snapshot())

    serial_policy = FixedLinearPolicy(observation_dim, low, high)
    batched_policy = FixedLinearPolicy(observation_dim, low, high)
    serial_before = rng_snapshot()
    started = time.perf_counter()
    serial = serial_evaluation(
        args.env_id, args.episodes, args.seed_start, serial_policy
    )
    serial_seconds = time.perf_counter() - started
    serial_rng = rng_unchanged(serial_before, rng_snapshot())

    batched_before = rng_snapshot()
    started = time.perf_counter()
    batched = evaluate_batched(
        args.env_id, args.episodes, args.seed_start, batched_policy
    )
    batched_seconds = time.perf_counter() - started
    batched_rng = rng_unchanged(batched_before, rng_snapshot())
    if [result["seed"] for result in serial] != [result["seed"] for result in batched]:
        raise AssertionError("Serial and batched evaluation seeds do not match")
    lengths_equal = [result["length"] for result in serial] == [
        result["length"] for result in batched
    ]
    serial_returns = np.array([result["return"] for result in serial])
    batched_returns = np.array([result["return"] for result in batched])
    np.testing.assert_allclose(batched_returns, serial_returns, rtol=1e-10, atol=1e-8)
    if not lengths_equal:
        raise AssertionError("Serial and batched episode lengths differ")
    if not all(all(group.values()) for group in (setup_rng, serial_rng, batched_rng)):
        raise AssertionError("CPU evaluation changed a global RNG state")

    report = {
        "status": "passed",
        "scope": "Real Gymnasium CPU simulation and fixed NumPy linear policy; no GPU API, no learning benchmark",
        "env_id": args.env_id,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gymnasium", "mujoco", "numpy", "torch")
        },
        "serial": serial,
        "batched": batched,
        "return_max_abs_error": float(np.abs(batched_returns - serial_returns).max()),
        "lengths_equal": lengths_equal,
        "rng_unchanged": {
            "setup": setup_rng,
            "serial": serial_rng,
            "batched": batched_rng,
        },
        "policy_calls": {
            "serial": serial_policy.calls,
            "batched": batched_policy.calls,
        },
        "timing_seconds": {"serial": serial_seconds, "batched": batched_seconds},
        "timing_serial_over_batched": serial_seconds / batched_seconds,
        "timing_scope": "One warmed paired run, including environment creation/reset/close; indicative CPU timing only",
        "comparison_tolerance": {"rtol": 1e-10, "atol": 1e-8},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
