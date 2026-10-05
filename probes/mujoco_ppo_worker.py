"""Bounded RLinf Worker/Channel PPO research runner for the original S4000 stack.

This uses a plain nn.Module, not the official FSDP embodied runner. It exercises
real RLinf scheduling, CPU tensor communication, GAE, and PPO losses on MUSA.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import ray
import torch
from torch import nn
from torch.distributions import Normal

try:
    import torch_musa  # noqa: F401
except ImportError:
    pass

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.scheduler import (
    Channel,
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
)


class RunningMoments:
    """Maintain CPU observation statistics with parallel variance merging."""

    def __init__(self, dimension: int) -> None:
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.var = np.ones(dimension, dtype=np.float64)
        self.count = 1e-4

    def update(self, observations: np.ndarray) -> None:
        """Merge a batch of observations into the running statistics."""
        batch_count = len(observations)
        batch_mean = observations.mean(axis=0)
        batch_var = observations.var(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        variance_sum = self.var * self.count + batch_var * batch_count
        variance_sum += delta**2 * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.var = variance_sum / total
        self.count = total

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        """Normalize using the current snapshot and cap extreme observations."""
        return np.clip(
            (observations - self.mean) / np.sqrt(self.var + 1e-8), -10, 10
        ).astype(np.float32)


class GaussianMLP(nn.Module):
    """FP32 diagonal Gaussian policy and independent value network."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()

        def network(output_dim: int, last_gain: float) -> nn.Sequential:
            layers = nn.Sequential(
                nn.Linear(observation_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, output_dim),
            )
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(layers[-1].weight, last_gain)
            return layers

        self.actor = network(action_dim, 0.01)
        self.critic = network(1, 1.0)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution_value(
        self, observations: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        """Return an unsquashed distribution and state value.

        Store sampled pre-clip actions for PPO likelihoods. The environment sees
        clipped/scaled actions; this is the standard Gaussian MuJoCo baseline.
        """
        mean = self.actor(observations)
        return Normal(mean, self.log_std.exp().expand_as(mean)), self.critic(
            observations
        ).squeeze(-1)


def bootstrap_truncations(
    rewards: torch.Tensor,
    terminal_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Bootstrap time limits once, while keeping episode traces separated."""
    time_limit = truncated & ~terminated
    return rewards + gamma * terminal_values * time_limit.to(rewards.dtype)


def population_standard_deviation(values: torch.Tensor) -> torch.Tensor:
    """Compute population standard deviation without the std.correction kernel."""
    centered = values - values.mean()
    return centered.square().mean().sqrt()


def clip_gradient_norm(parameters: Any, maximum_norm: float) -> torch.Tensor:
    """Clip the global L2 gradient norm using basic old-MUSA-supported ops."""
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Cannot clip an empty gradient set")
    norm = sum(gradient.square().sum() for gradient in gradients).sqrt()
    coefficient = (maximum_norm / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient)
    return norm


class MuJoCoEnvWorker(Worker):
    """CPU environments exchanging each batch of observations via RLinf Channel."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    @property
    def has_accelerator(self) -> bool:
        """Keep simulator communication on CPU despite node-wide visibility."""
        return False

    def initialize(self) -> dict[str, Any]:
        """Create independent environments and seed each once."""
        self.envs = [
            gym.make(self.config["env_id"]) for _ in range(self.config["num_envs"])
        ]
        self.observations = np.stack(
            [
                env.reset(seed=self.config["seed"] + index)[0]
                for index, env in enumerate(self.envs)
            ]
        ).astype(np.float32)
        self.episode_returns = np.zeros(len(self.envs), dtype=np.float64)
        self.episode_lengths = np.zeros(len(self.envs), dtype=np.int64)
        space = self.envs[0].action_space
        self.log_info("Initialized CPU MuJoCo environments for route 1")
        return {
            "observation_dim": self.observations.shape[-1],
            "action_dim": space.shape[0],
            "action_low": space.low.tolist(),
            "action_high": space.high.tolist(),
        }

    def respond(
        self, observations: Channel, actions: Channel, steps: int
    ) -> dict[str, int]:
        """Serve one horizon, resetting explicitly after each episode boundary."""
        observations.put({"observations": self.observations})
        for _ in range(steps):
            batch_actions = actions.get()
            rewards, terminated, truncated, final_observations, completed = (
                [],
                [],
                [],
                [],
                [],
            )
            next_observations = []
            for index, (env, action) in enumerate(zip(self.envs, batch_actions)):
                next_obs, reward, term, trunc, _ = env.step(action)
                self.episode_returns[index] += reward
                self.episode_lengths[index] += 1
                final_observations.append(next_obs.copy())
                rewards.append(reward)
                terminated.append(term)
                truncated.append(trunc)
                if term or trunc:
                    completed.append(
                        {
                            "return": float(self.episode_returns[index]),
                            "length": int(self.episode_lengths[index]),
                        }
                    )
                    self.episode_returns[index] = 0
                    self.episode_lengths[index] = 0
                    next_obs, _ = env.reset()
                next_observations.append(next_obs)
            self.observations = np.asarray(next_observations, dtype=np.float32)
            observations.put(
                {
                    "observations": self.observations,
                    "final_observations": np.asarray(
                        final_observations, dtype=np.float32
                    ),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "terminated": np.asarray(terminated, dtype=np.bool_),
                    "truncated": np.asarray(truncated, dtype=np.bool_),
                    "completed": completed,
                }
            )
        return {
            "transitions": steps * len(self.envs),
            "channel_messages": 2 * steps + 1,
        }

    def close(self) -> None:
        """Release simulator resources."""
        for env in self.envs:
            env.close()


class PlainPPOActorWorker(Worker):
    """Single-card rollout and update extension using the RLinf algorithm APIs."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def initialize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Initialize explicitly on the requested device; never silently fall back."""
        torch.set_num_threads(1)
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        self.device = torch.device(self.config["device"])
        if self.device.type == "musa":
            if not hasattr(torch, "musa") or not torch.musa.is_available():
                raise RuntimeError("MUSA is required for this requested run")
            torch.musa.set_device(0)
            torch.musa.manual_seed_all(self.config["seed"])
        self.model = GaussianMLP(
            metadata["observation_dim"], metadata["action_dim"]
        ).to(self.device)
        self.initial_weights = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], eps=1e-5
        )
        self.moments = RunningMoments(metadata["observation_dim"])
        self.low = np.asarray(metadata["action_low"], dtype=np.float32)
        self.high = np.asarray(metadata["action_high"], dtype=np.float32)
        self.version = 0
        self.optimizer_steps = 0
        self.log_info(f"Initialized ordinary-module PPO on {self.device}")
        return {
            "device": str(self.device),
            "torch_version": torch.__version__,
            "torch_musa_version": getattr(torch_musa, "__version__", "unknown")
            if self.device.type == "musa"
            else None,
            "accelerator_name": torch.musa.get_device_name(0)
            if self.device.type == "musa"
            else "CPU",
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def _tensor(self, value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        # Ray object-store NumPy arrays may be read-only: make an explicit copy.
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _env_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        return self.low + (np.clip(raw_actions, -1, 1) + 1) * (self.high - self.low) / 2

    @torch.no_grad()
    def collect(
        self, observations: Channel, actions: Channel, steps: int
    ) -> dict[str, Any]:
        """Collect a fixed policy horizon using actual CPU Worker communication."""
        data: dict[str, list[Any]] = {
            key: []
            for key in (
                "observations",
                "actions",
                "logprobs",
                "values",
                "rewards",
                "dones",
            )
        }
        completed = []
        message = observations.get()
        # Keep normalization fixed throughout a horizon so stored likelihoods
        # and bootstrap values describe the same policy representation.
        self.moments.update(message["observations"])
        data["dones"].append(
            torch.zeros(self.config["num_envs"], dtype=torch.bool, device=self.device)
        )
        for _ in range(steps):
            obs = self._tensor(self.moments.normalize(message["observations"]))
            distribution, value = self.model.distribution_value(obs)
            # Torch-MUSA 1.3 lacks normal.Tensor_Tensor; randn is supported.
            action = distribution.mean + distribution.stddev * torch.randn_like(
                distribution.mean
            )
            logprob = distribution.log_prob(action).sum(-1)
            actions.put(self._env_actions(action.cpu().numpy()))
            message = observations.get()
            _, final_values = self.model.distribution_value(
                self._tensor(self.moments.normalize(message["final_observations"]))
            )
            terminal = self._tensor(message["terminated"], torch.bool)
            truncation = self._tensor(message["truncated"], torch.bool)
            reward = bootstrap_truncations(
                self._tensor(message["rewards"]) * self.config["reward_scale"],
                final_values,
                terminal,
                truncation,
                self.config["gamma"],
            )
            for key, item in (
                ("observations", obs),
                ("actions", action),
                ("logprobs", logprob),
                ("values", value),
                ("rewards", reward),
            ):
                data[key].append(item)
            data["dones"].append(terminal | truncation)
            completed.extend(message["completed"])
        _, final_value = self.model.distribution_value(
            self._tensor(self.moments.normalize(message["observations"]))
        )
        data["values"].append(final_value)
        rollout = {key: torch.stack(items) for key, items in data.items()}
        advantages, returns = compute_gae_advantages_and_returns(
            rewards=rollout["rewards"],
            values=rollout["values"],
            dones=rollout["dones"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            normalize_advantages=False,
        )
        rollout["advantages"] = (advantages - advantages.mean()) / (
            population_standard_deviation(advantages) + 1e-8
        )
        rollout["returns"] = returns
        self.rollout = rollout
        return {
            "policy_version": self.version,
            "completed_episodes": completed,
            "finite_rollout": all(
                bool(torch.isfinite(x).all()) for x in rollout.values()
            ),
        }

    def update(self) -> dict[str, Any]:
        """Run RLinf PPO objectives, check gradients, and quantify parameter change."""
        started = time.monotonic()
        rollout = self.rollout
        tensors = {
            key: value.flatten(0, 1)
            for key, value in rollout.items()
            if key not in {"values", "dones", "rewards"}
        }
        tensors["values"] = rollout["values"][:-1].flatten(0, 1)
        count = tensors["observations"].shape[0]
        before = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        reports = []
        for _ in range(self.config["epochs"]):
            indices = torch.randperm(count, device="cpu").to(self.device)
            for offset in range(0, count, self.config["minibatch"]):
                batch = indices[offset : offset + self.config["minibatch"]]
                distribution, values = self.model.distribution_value(
                    tensors["observations"][batch]
                )
                logprobs = distribution.log_prob(tensors["actions"][batch]).sum(-1)
                actor_loss, actor_metrics = compute_ppo_actor_loss(
                    logprobs=logprobs,
                    old_logprobs=tensors["logprobs"][batch],
                    advantages=tensors["advantages"][batch],
                    clip_ratio_low=0.2,
                    clip_ratio_high=0.2,
                )
                critic_loss, _ = compute_ppo_critic_loss(
                    values=values,
                    returns=tensors["returns"][batch],
                    prev_values=tensors["values"][batch],
                    value_clip=0.2,
                    huber_delta=10.0,
                )
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - self.config["entropy_coef"] * entropy
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    p.grad is None or bool(torch.isfinite(p.grad).all())
                    for p in self.model.parameters()
                ):
                    raise RuntimeError("Non-finite PPO gradients")
                grad_norm = clip_gradient_norm(self.model.parameters(), 0.5)
                self.optimizer.step()
                reports.append(
                    {
                        "loss": float(loss.detach()),
                        "actor_loss": float(actor_loss.detach()),
                        "critic_loss": float(critic_loss.detach()),
                        "grad_norm": float(grad_norm),
                        "approx_kl": float(actor_metrics["actor/approx_kl"]),
                    }
                )
        delta = sum(
            float((value - before[key]).abs().sum())
            for key, value in self.model.state_dict().items()
        )
        if not delta > 0:
            raise RuntimeError("PPO update made no parameter change")
        self.version += 1
        self.optimizer_steps += len(reports)
        del self.rollout
        return {
            "policy_version": self.version,
            "parameter_l1_delta": delta,
            "optimizer_steps": len(reports),
            "update_seconds": time.monotonic() - started,
            "metrics": {
                key: float(np.mean([report[key] for report in reports]))
                for key in reports[0]
            },
        }

    @torch.no_grad()
    def evaluate(
        self, episodes: int, initial_weights: bool = False
    ) -> list[dict[str, Any]]:
        """Evaluate deterministically with frozen statistics and fixed held-out seeds."""
        env = gym.make(self.config["env_id"])
        results = []
        trained_weights = None
        if initial_weights:
            trained_weights = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.initial_weights)
        try:
            for episode in range(episodes):
                seed = self.config["seed"] + 10000 + episode
                obs, _ = env.reset(seed=seed)
                total, length = 0.0, 0
                while True:
                    distribution, _ = self.model.distribution_value(
                        self._tensor(self.moments.normalize(obs[None]))
                    )
                    action = self._env_actions(distribution.mean.cpu().numpy())[0]
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    length += 1
                    if terminated or truncated:
                        break
                results.append({"seed": seed, "return": total, "length": length})
        finally:
            env.close()
            if trained_weights is not None:
                self.model.load_state_dict(trained_weights)
        return results

    def checkpoint(self, path: str, include_rollout: bool = False) -> None:
        """Save optimizer, RNGs, normalizer, steps and optional pending PPO batch."""
        rng = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if self.device.type == "musa":
            rng["musa"] = torch.musa.get_rng_state_all()
        payload = {
            "model": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "initial_weights": self.initial_weights,
            "optimizer": self.optimizer.state_dict(),
            "mean": self.moments.mean.copy(),
            "var": self.moments.var.copy(),
            "count": self.moments.count,
            "rng": rng,
            "config": self.config,
            "policy_version": self.version,
            "optimizer_steps": self.optimizer_steps,
        }
        if include_rollout:
            payload["rollout"] = {
                key: value.detach().cpu() for key, value in self.rollout.items()
            }
        torch.save(payload, path)

    def restore(self, path: str) -> dict[str, int]:
        """Restore a trusted checkpoint produced by this isolated experiment.

        Environment simulator state is not included: resume starts from a fresh
        rollout boundary, or uses a saved pending PPO batch for the regression.
        """
        payload = torch.load(path, map_location="cpu")
        if payload["config"] != self.config:
            raise ValueError("Checkpoint configuration differs from this worker")
        self.model.load_state_dict(payload["model"])
        self.initial_weights = payload["initial_weights"]
        self.optimizer.load_state_dict(payload["optimizer"])
        self.moments.mean = payload["mean"]
        self.moments.var = payload["var"]
        self.moments.count = payload["count"]
        self.version = payload["policy_version"]
        self.optimizer_steps = payload["optimizer_steps"]
        if "rollout" in payload:
            self.rollout = {
                key: value.to(self.device) for key, value in payload["rollout"].items()
            }
        torch.set_rng_state(payload["rng"]["torch"])
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        if self.device.type == "musa":
            torch.musa.set_rng_state_all(payload["rng"]["musa"])
        return {"policy_version": self.version, "optimizer_steps": self.optimizer_steps}

    def verify_resume(self, path: str) -> dict[str, Any]:
        """Verify the same pending batch updates identically after a full restore."""
        self.checkpoint(path, include_rollout=True)
        first = self.update()
        expected_weights = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        expected_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.restore(path)
        resumed = self.update()
        max_error = 0.0
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value.detach().cpu(),
                expected_weights[key],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Resume parameter mismatch: {key}",
            )
            max_error = max(
                max_error,
                float((value.detach().cpu() - expected_weights[key]).abs().max()),
            )
        optimizer_error = 0.0
        for parameter_id, state in self.optimizer.state_dict()["state"].items():
            for key, value in state.items():
                expected = expected_optimizer["state"][parameter_id][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(
                        value.cpu(),
                        expected.cpu(),
                        rtol=1e-5,
                        atol=1e-6,
                        msg=f"Resume optimizer mismatch: {key}",
                    )
                    optimizer_error = max(
                        optimizer_error,
                        float((value.cpu() - expected.cpu()).abs().max()),
                    )
                elif value != expected:
                    raise AssertionError(f"Resume optimizer scalar mismatch: {key}")
        if (
            resumed["policy_version"] != first["policy_version"]
            or self.optimizer_steps
            != expected_optimizer["state"][next(iter(expected_optimizer["state"]))][
                "step"
            ]
        ):
            # Adam's step tracks minibatch updates for each trained parameter.
            raise AssertionError("Resume version or optimizer step mismatch")
        return {
            "status": "passed",
            "parameter_max_abs_error": max_error,
            "optimizer_max_abs_error": optimizer_error,
            "policy_version": self.version,
            "optimizer_steps": self.optimizer_steps,
            "resumed_update": resumed,
            "scope": "same PPO batch restart, no simulator-state resume",
        }


def main() -> None:
    """Launch two RLinf Worker groups and drive a bounded training experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa"), default="musa")
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument(
        "--source-commit", default="0505431899574619da86f551bad70b71e0ea2177"
    )
    parser.add_argument("--route-label", default="route1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for key in ("num_envs", "horizon", "iterations", "epochs", "minibatch"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.num_envs * args.horizon < 2:
        parser.error("The PPO batch must contain at least two transitions")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    cluster = Cluster(num_nodes=1)
    run_name = f"{args.route_label}_{os.getpid()}"
    env = MuJoCoEnvWorker.create_group(config).launch(
        cluster=cluster,
        name=f"{run_name}_env",
        placement_strategy=NodePlacementStrategy([0]),
    )
    placement = (
        PackedPlacementStrategy(start_hardware_rank=0, end_hardware_rank=0)
        if args.device == "musa"
        else NodePlacementStrategy([0])
    )
    actor = PlainPPOActorWorker.create_group(config).launch(
        cluster=cluster, name=f"{run_name}_actor", placement_strategy=placement
    )
    metadata = env.initialize().wait()[0]
    initialization = actor.initialize(metadata).wait()[0]
    observations = Channel.create(
        name=f"{run_name}_observations", distributed=False, transport="ray"
    )
    actions = Channel.create(
        name=f"{run_name}_actions", distributed=False, transport="ray"
    )
    report = {
        "scope": "custom ordinary-module RLinf Worker extension, not official FSDP runner",
        "source_commit": args.source_commit,
        "communication": "RLinf Channel existing Ray CPU-object APIs; collective tensor transport unverified",
        "config": config,
        "initialization": initialization,
        "iterations": [],
    }
    training_completed = False
    try:
        for iteration in range(args.iterations):
            started = time.monotonic()
            environment_work = env.respond(observations, actions, args.horizon)
            actor_work = actor.collect(observations, actions, args.horizon)
            collection = actor_work.wait()[0]
            communication = environment_work.wait()[0]
            if args.resume_check and iteration == 0:
                resume = actor.verify_resume(
                    str(output.with_suffix(".resume.pt"))
                ).wait()[0]
                report["resume_regression"] = resume
                update = resume["resumed_update"]
            else:
                update = actor.update().wait()[0]
            result = {
                "iteration": iteration,
                "seconds": time.monotonic() - started,
                "collection": collection,
                "communication": communication,
                "update": update,
            }
            report["iterations"].append(result)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        # Evaluate initial and trained weights with identical held-out seeds and
        # final frozen statistics; normalization alone must not count as learning.
        report["baseline_evaluation"] = (
            actor.evaluate(args.eval_episodes, True).wait()[0]
            if args.eval_episodes
            else []
        )
        report["final_evaluation"] = (
            actor.evaluate(args.eval_episodes).wait()[0] if args.eval_episodes else []
        )
        actor.checkpoint(str(output.with_suffix(".pt"))).wait()
        report["status"] = "update_passed"
        report["learning_claim"] = (
            "Not established by smoke runs; compare held-out evaluation across budgets and seeds."
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        training_completed = True
    finally:
        if training_completed and ray.is_initialized():
            env.close().wait()


if __name__ == "__main__":
    try:
        main()
    finally:
        # This also runs if Worker launch/initialization fails before collection.
        ray.shutdown()
