"""Boundary and layout tests needed before expanding the PPO training budget."""

import numpy as np
import torch

from mujoco_ppo_worker import bootstrap_truncations
from ppo_learning import IdentityMoments, LearningActorWorker, RunningMoments
from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_critic_loss


def test_identity_layout_accepts_single_and_strided_observations():
    source = np.arange(6, dtype=np.float32)[::2][None]
    normalized = IdentityMoments(3).normalize(source)
    np.testing.assert_array_equal(normalized, source)
    assert normalized.strides == (12, 4)
    assert not np.shares_memory(normalized, source)


def test_merging_complete_batches_matches_population_statistics():
    moments = RunningMoments(2)
    raw = np.array([[0, 5], [2, -1], [4, 3], [100, 7]], dtype=np.float32)
    moments.update(raw[:2])
    moments.update(raw[2:])
    initial = np.zeros((1, 2))
    expected_mean = raw.sum(axis=0) / (len(raw) + 1e-4)
    expected_var = ((raw - expected_mean) ** 2).sum(axis=0)
    expected_var = (expected_var + 1e-4 * (1 + (initial - expected_mean) ** 2)) / (
        len(raw) + 1e-4
    )
    np.testing.assert_allclose(moments.mean, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(moments.var, expected_var[0], rtol=1e-6)
    assert moments.count == 4.0001


def test_pendulum_action_scaling_and_clipping():
    actor = object.__new__(LearningActorWorker)
    actor.low = np.array([-2], dtype=np.float32)
    actor.high = np.array([2], dtype=np.float32)
    raw = np.array([[-10], [-0.5], [0], [0.5], [10]], dtype=np.float32)
    np.testing.assert_allclose(actor._env_actions(raw).ravel(), [-2, -1, 0, 1, 2])


def test_horizon_end_truncation_uses_final_value_once():
    corrected = bootstrap_truncations(
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[5.0, 5.0, 5.0]]),
        torch.tensor([[False, True, False]]),
        torch.tensor([[True, True, False]]),
        0.9,
    )
    _, returns = compute_gae_advantages_and_returns(
        rewards=corrected,
        values=torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 3.0]]),
        dones=torch.tensor([[False, False, False], [True, True, False]]),
        gamma=0.9,
        gae_lambda=1.0,
        normalize_advantages=False,
    )
    torch.testing.assert_close(returns, torch.tensor([[5.5, 1.0, 3.7]]))


def test_disabled_value_clip_preserves_gradient_toward_distant_target():
    value = torch.tensor([-1.0], requires_grad=True)
    loss, _ = compute_ppo_critic_loss(
        values=value,
        returns=torch.tensor([-50.0]),
        prev_values=torch.tensor([0.0]),
        value_clip=float("inf"),
        huber_delta=10.0,
    )
    loss.backward()
    torch.testing.assert_close(value.grad, torch.tensor([10.0]))
