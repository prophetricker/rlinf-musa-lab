"""Learning baseline using real RLinf Workers, CPU Ray Channels and MUSA PPO.

The historical smoke runner is imported unchanged. This extends its ordinary
nn.Module actor, not RLinf's official FSDP Actor or EmbodiedRunner.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from evaluation import evaluate_batched
from mujoco_ppo_worker import (
    MuJoCoEnvWorker,
    PlainPPOActorWorker,
    RunningMoments,
    bootstrap_truncations,
    clip_gradient_norm,
    population_standard_deviation,
)
from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.scheduler import (
    Channel,
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)


class IdentityMoments(RunningMoments):
    """Keep checkpoint-compatible fields while leaving observations unnormalized."""

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        # obs[None] can have leading stride zero, rejected by old muDNN addmm.
        return np.array(observations, dtype=np.float32, copy=True, order="C")


class LearningActorWorker(PlainPPOActorWorker):
    """Add all-observation statistics and fixed, periodic evaluation."""

    def initialize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        result = super().initialize(metadata)
        if not self.config["normalize_observations"]:
            self.moments = IdentityMoments(metadata["observation_dim"])
        self.pending_observations = None
        return result

    @torch.no_grad()
    def collect(
        self, observations: Channel, actions: Channel, steps: int
    ) -> dict[str, Any]:
        """Freeze preprocessing until PPO finishes, then merge every raw input."""
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
        raw_observations = []
        completed = []
        clipped_actions = 0
        action_elements = 0
        message = observations.get()
        data["dones"].append(
            torch.zeros(self.config["num_envs"], dtype=torch.bool, device=self.device)
        )
        for _ in range(steps):
            raw = np.asarray(message["observations"], dtype=np.float32)
            raw_observations.append(raw.copy())
            obs = self._tensor(self.moments.normalize(raw))
            distribution, value = self.model.distribution_value(obs)
            action = distribution.mean + distribution.stddev * torch.randn_like(
                distribution.mean
            )
            logprob = distribution.log_prob(action).sum(-1)
            cpu_actions = action.cpu().numpy()
            clipped_actions += int((np.abs(cpu_actions) > 1).sum())
            action_elements += cpu_actions.size
            actions.put(self._env_actions(cpu_actions))
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
        self.pending_observations = np.concatenate(raw_observations)
        finite = all(bool(torch.isfinite(value).all()) for value in rollout.values())
        if not finite:
            raise RuntimeError("Non-finite rollout")
        return {
            "policy_version": self.version,
            "completed_episodes": completed,
            "finite_rollout": finite,
            "action_element_clip_fraction": clipped_actions / action_elements,
            "observation_samples": len(self.pending_observations),
        }

    def update(self) -> dict[str, Any]:
        """Use upstream losses with explicit critic settings and then update stats."""
        started = time.monotonic()
        tensors = {
            key: value.flatten(0, 1)
            for key, value in self.rollout.items()
            if key not in {"values", "dones", "rewards"}
        }
        tensors["values"] = self.rollout["values"][:-1].flatten(0, 1)
        count = tensors["observations"].shape[0]
        before = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        metrics = []
        for _ in range(self.config["epochs"]):
            indices = torch.randperm(count, device="cpu").to(self.device)
            for offset in range(0, count, self.config["minibatch"]):
                batch = indices[offset : offset + self.config["minibatch"]]
                distribution, values = self.model.distribution_value(
                    tensors["observations"][batch]
                )
                logprobs = distribution.log_prob(tensors["actions"][batch]).sum(-1)
                actor_loss, _ = compute_ppo_actor_loss(
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
                    value_clip=self.config["value_clip"],
                    huber_delta=self.config["huber_delta"],
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
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.model.parameters()
                ):
                    raise RuntimeError("Non-finite PPO gradients")
                norm = clip_gradient_norm(self.model.parameters(), 0.5)
                self.optimizer.step()
                log_ratio = logprobs.detach() - tensors["logprobs"][batch]
                metrics.append(
                    {
                        "loss": float(loss.detach()),
                        "actor_loss": float(actor_loss.detach()),
                        "critic_loss": float(critic_loss.detach()),
                        "grad_norm": float(norm),
                        "approx_kl_k3": float((log_ratio.exp() - 1 - log_ratio).mean()),
                        "latent_entropy": float(entropy.detach()),
                    }
                )
        delta = sum(
            float((value - before[key]).abs().sum())
            for key, value in self.model.state_dict().items()
        )
        if not delta > 0:
            raise RuntimeError("PPO update made no parameter change")
        self.version += 1
        self.optimizer_steps += len(metrics)
        del self.rollout
        if self.config["normalize_observations"]:
            self.moments.update(self.pending_observations)
        self.pending_observations = None
        if hasattr(self, "_resume_snapshots"):
            with torch.no_grad():
                distribution, fixed_values = self.model.distribution_value(
                    self._tensor(
                        self.moments.normalize(self._resume_pending_reference[:4])
                    )
                )
            self._resume_snapshots.append(
                {
                    "mean": self.moments.mean.copy(),
                    "var": self.moments.var.copy(),
                    "count": self.moments.count,
                    "policy_mean": distribution.mean.detach().cpu().clone(),
                    "values": fixed_values.detach().cpu().clone(),
                    "torch_rng": torch.get_rng_state().clone(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "musa_rng": [
                        state.clone() for state in torch.musa.get_rng_state_all()
                    ]
                    if self.device.type == "musa"
                    else [],
                }
            )
        return {
            "policy_version": self.version,
            "parameter_l1_delta": delta,
            "optimizer_steps": len(metrics),
            "update_seconds": time.monotonic() - started,
            "normalizer_count": self.moments.count,
            "metrics": {
                key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]
            },
        }

    @torch.no_grad()
    def evaluate(
        self,
        episodes: int,
        initial_weights: bool = False,
        evaluation_seed: int | None = None,
        initial_statistics: bool = False,
    ) -> list[dict[str, Any]]:
        """Use shared held-out seeds, raw task rewards and frozen current statistics."""
        saved = None
        saved_statistics = None
        if initial_statistics:
            saved_statistics = (
                self.moments.mean.copy(),
                self.moments.var.copy(),
                self.moments.count,
            )
            self.moments.mean.fill(0)
            self.moments.var.fill(1)
            self.moments.count = 1e-4
        if initial_weights:
            saved = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.initial_weights)
        try:

            def mean_actions(observations):
                distribution, _ = self.model.distribution_value(
                    self._tensor(self.moments.normalize(observations))
                )
                return self._env_actions(distribution.mean.cpu().numpy())

            seed_start = (
                self.config["eval_seed"] if evaluation_seed is None else evaluation_seed
            )
            if self.config["batch_evaluation"]:
                results = evaluate_batched(
                    self.config["env_id"], episodes, seed_start, mean_actions
                )
            else:
                results = []
                for episode in range(episodes):
                    results.extend(
                        evaluate_batched(
                            self.config["env_id"], 1, seed_start + episode, mean_actions
                        )
                    )
        finally:
            if saved is not None:
                self.model.load_state_dict(saved)
            if saved_statistics is not None:
                self.moments.mean, self.moments.var, self.moments.count = (
                    saved_statistics
                )
        return results

    def checkpoint(self, path: str, include_rollout: bool = False) -> None:
        super().checkpoint(path, include_rollout=include_rollout)
        payload = torch.load(path, map_location="cpu")
        payload["pending_observations"] = self.pending_observations
        torch.save(payload, path)

    def restore(self, path: str) -> dict[str, int]:
        result = super().restore(path)
        self.pending_observations = torch.load(path, map_location="cpu")[
            "pending_observations"
        ]
        if hasattr(self, "_resume_pending_reference"):
            np.testing.assert_array_equal(
                self.pending_observations, self._resume_pending_reference
            )
        return result

    def verify_resume(self, path: str) -> dict[str, Any]:
        """Extend model/Adam parity to deferred stats, fixed outputs and RNG state."""
        self._resume_pending_reference = self.pending_observations.copy()
        self._resume_snapshots = []
        try:
            result = super().verify_resume(path)
            expected, actual = self._resume_snapshots
            for key in ("mean", "var", "count", "numpy_rng", "python_rng"):
                np.testing.assert_equal(actual[key], expected[key])
            for key in ("policy_mean", "values", "torch_rng"):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            assert len(actual["musa_rng"]) == len(expected["musa_rng"])
            for actual_rng, expected_rng in zip(
                actual["musa_rng"], expected["musa_rng"]
            ):
                torch.testing.assert_close(actual_rng, expected_rng, atol=0, rtol=0)
            result.update(
                pending_observations_restored=True,
                normalizer_exact_parity=True,
                fixed_policy_output_exact_parity=True,
                rng_post_update_exact_parity=True,
                normalization_enabled=self.config["normalize_observations"],
            )
            return result
        finally:
            del self._resume_pending_reference
            del self._resume_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa"), default="musa")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=20000)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--batch-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-clip", type=float, default=float("inf"))
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for key in (
        "num_envs",
        "horizon",
        "iterations",
        "epochs",
        "minibatch",
        "eval_episodes",
        "eval_interval",
        "test_episodes",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    cluster = Cluster(num_nodes=1)
    run_name = f"learning_{os.getpid()}"
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
    actor = LearningActorWorker.create_group(config).launch(
        cluster=cluster, name=f"{run_name}_actor", placement_strategy=placement
    )
    metadata = env.initialize().wait()[0]
    initialization = actor.initialize(metadata).wait()[0]
    observations = Channel.create(
        name=f"{run_name}_obs", distributed=False, transport="ray"
    )
    actions = Channel.create(
        name=f"{run_name}_actions", distributed=False, transport="ray"
    )
    report = {
        "date": "2026-10-05",
        "scope": "ordinary-module RLinf Worker learning baseline, not official FSDP runner",
        "source_commit": args.source_commit,
        "runner_sha256": {
            "ppo_learning.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation.py": hashlib.sha256(
                Path(inspect.getfile(evaluate_batched)).read_bytes()
            ).hexdigest(),
            "mujoco_ppo_worker.py": hashlib.sha256(
                Path(inspect.getfile(PlainPPOActorWorker)).read_bytes()
            ).hexdigest(),
        },
        "config": config,
        "initialization": initialization,
        "iterations": [],
        "evaluations": [],
        "status": "running",
        "communication": "Ray CPU objects; rollout/update share an ActorWorker",
        "critic_clipping": "disabled" if np.isinf(args.value_clip) else args.value_clip,
        "evaluation_execution": "batched active CPU environments"
        if args.batch_evaluation
        else "serial episodes",
    }
    # Infinity is only an internal loss setting; record standards-compliant JSON.
    report["config"] = {
        **config,
        "value_clip": None if np.isinf(args.value_clip) else args.value_clip,
    }

    def save() -> None:
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def evaluate(transitions: int, final_test: bool = False) -> None:
        episode_count = args.test_episodes if final_test else args.eval_episodes
        evaluation_seed = args.test_seed if final_test else args.eval_seed
        baseline = actor.evaluate(episode_count, True, evaluation_seed, True).wait()[0]
        trained = actor.evaluate(episode_count, False, evaluation_seed).wait()[0]
        snapshot = {
            "transitions": transitions,
            "initial_policy": baseline,
            "trained_policy": trained,
            "initial_mean_return": float(
                np.mean([item["return"] for item in baseline])
            ),
            "trained_mean_return": float(np.mean([item["return"] for item in trained])),
            "split": "test" if final_test else "validation",
            "protocol": "same held-out seeds; initial vs trained complete policy+normalizer bundles; raw rewards",
        }
        if args.normalize_observations:
            snapshot["initial_weights_trained_normalizer"] = actor.evaluate(
                episode_count, True, evaluation_seed
            ).wait()[0]
        report["evaluations"].append(snapshot)
        print(json.dumps({"evaluation": snapshot}), flush=True)
        save()

    training_completed = False
    try:
        evaluate(0)
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
            row = {
                "iteration": iteration + 1,
                "transitions": (iteration + 1) * args.num_envs * args.horizon,
                "seconds": time.monotonic() - started,
                "collection": collection,
                "communication": communication,
                "update": update,
            }
            report["iterations"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if (
                iteration + 1
            ) % args.eval_interval == 0 or iteration + 1 == args.iterations:
                evaluate(row["transitions"])
                actor.checkpoint(str(output.with_suffix(".pt"))).wait()
        evaluate(args.iterations * args.num_envs * args.horizon, final_test=True)
        report["status"] = "completed"
        report["learning_claim"] = (
            "Inspect held-out returns across all predeclared seeds; updates alone are insufficient."
        )
        save()
        training_completed = True
    finally:
        if training_completed and ray.is_initialized():
            env.close().wait()


if __name__ == "__main__":
    try:
        main()
    finally:
        ray.shutdown()
