"""CPU behavioral tests for active-episode batching, RNGs and exception cleanup."""

from __future__ import annotations

import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation import evaluate_batched


class FakeEnvironment:
    """Seed-dependent episode lengths with both true and time-limit endings."""

    def __init__(
        self,
        fail_reset: bool = False,
        fail_step: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.closed = False
        self.reset_count = 0
        self.step_count = 0
        self.done = False

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("reset failure")
        self.seed = seed
        self.limit = seed % 4 + 1
        self.offset = float(np.random.default_rng(seed).uniform(-0.5, 0.5))
        return np.array([self.seed, 0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise AssertionError("Completed episode was stepped again")
        if self.fail_step:
            raise RuntimeError("step failure")
        self.step_count += 1
        ended = self.step_count == self.limit
        self.done = ended
        # Even seeds terminate, odd seeds truncate; both must leave active set.
        terminated = ended and self.seed % 2 == 0
        truncated = ended and self.seed % 2 != 0
        reward = self.offset + float(action[0]) - 0.1 * self.step_count
        return (
            np.array([self.seed, self.step_count], dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failure")


def deterministic_actions(observations: np.ndarray) -> np.ndarray:
    """Use each seed and timestep so action/observation misrouting is detectable."""
    return ((observations[:, 0] % 5) * 0.2 + observations[:, 1] * 0.05)[:, None]


def serial_reference(episodes: int, seed_start: int) -> list[dict]:
    """Independent episode-at-a-time evaluation using the same policy contract."""
    results = []
    for seed in range(seed_start, seed_start + episodes):
        environment = FakeEnvironment()
        try:
            observation, _ = environment.reset(seed=seed)
            total, length = 0.0, 0
            while True:
                action = deterministic_actions(observation[None])[0]
                observation, reward, terminated, truncated, _ = environment.step(action)
                total += reward
                length += 1
                if terminated or truncated:
                    break
            results.append({"seed": seed, "return": total, "length": length})
        finally:
            environment.close()
    return results


class BatchedEvaluationTests(unittest.TestCase):
    """Behavioral coverage using fake Gym without requiring Torch or MuJoCo."""

    def test_variable_lengths_match_serial_and_batch_only_active_states(self) -> None:
        environments, calls = [], []

        def make(_: str) -> FakeEnvironment:
            environment = FakeEnvironment()
            environments.append(environment)
            return environment

        def policy(observations: np.ndarray) -> np.ndarray:
            calls.append(observations.copy())
            return deterministic_actions(observations)

        with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(make=make)}):
            observed = evaluate_batched("Fake-v0", 6, 20, policy)
        expected = serial_reference(6, 20)
        self.assertEqual(observed, expected)
        self.assertEqual([len(batch) for batch in calls], [6, 4, 2, 1])
        self.assertEqual(
            [batch[:, 0].tolist() for batch in calls],
            [[20, 21, 22, 23, 24, 25], [21, 22, 23, 25], [22, 23], [23]],
        )
        self.assertTrue(all(environment.closed for environment in environments))
        self.assertTrue(
            all(environment.reset_count == 1 for environment in environments)
        )
        self.assertEqual(
            [environment.step_count for environment in environments], [1, 2, 3, 4, 1, 2]
        )

    def test_helper_leaves_global_numpy_and_python_rng_states_unchanged(self) -> None:
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        with patch.dict(
            sys.modules,
            {"gymnasium": SimpleNamespace(make=lambda _: FakeEnvironment())},
        ):
            evaluate_batched("Fake-v0", 5, 101, deterministic_actions)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_creation_and_reset_failures_close_previously_created_environments(
        self,
    ) -> None:
        for failure in ("make", "reset"):
            with self.subTest(failure=failure):
                environments = []

                def make(_: str) -> FakeEnvironment:
                    if failure == "make" and len(environments) == 2:
                        raise RuntimeError("make failure")
                    environment = FakeEnvironment(
                        fail_reset=failure == "reset" and len(environments) == 2
                    )
                    environments.append(environment)
                    return environment

                with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(make=make)}):
                    with self.assertRaisesRegex(RuntimeError, f"{failure} failure"):
                        evaluate_batched("Fake-v0", 4, 20, deterministic_actions)
                self.assertTrue(all(environment.closed for environment in environments))

    def test_policy_step_and_shape_failures_close_all_without_masking_error(
        self,
    ) -> None:
        for failure in ("policy", "step", "shape"):
            with self.subTest(failure=failure):
                environments = []

                def make(_: str) -> FakeEnvironment:
                    environment = FakeEnvironment(
                        fail_step=failure == "step" and len(environments) == 1,
                        fail_close=len(environments) == 0,
                    )
                    environments.append(environment)
                    return environment

                def policy(observations: np.ndarray) -> np.ndarray:
                    if failure == "policy":
                        raise RuntimeError("policy failure")
                    if failure == "shape":
                        return np.zeros((len(observations) - 1, 1))
                    return deterministic_actions(observations)

                expected_error = ValueError if failure == "shape" else RuntimeError
                expected_message = (
                    "action batch" if failure == "shape" else f"{failure} failure"
                )
                with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(make=make)}):
                    with self.assertRaisesRegex(expected_error, expected_message):
                        evaluate_batched("Fake-v0", 4, 20, policy)
                self.assertTrue(all(environment.closed for environment in environments))

    def test_close_failure_still_closes_every_environment(self) -> None:
        environments = []

        def make(_: str) -> FakeEnvironment:
            environment = FakeEnvironment(fail_close=len(environments) == 0)
            environments.append(environment)
            return environment

        with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(make=make)}):
            with self.assertRaisesRegex(RuntimeError, "close failure"):
                evaluate_batched("Fake-v0", 4, 20, deterministic_actions)
        self.assertTrue(all(environment.closed for environment in environments))


if __name__ == "__main__":
    unittest.main()
