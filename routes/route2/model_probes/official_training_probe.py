#!/usr/bin/env python3
"""Audit Runner checkpoint continuation and final policy synchronization.

Use the official Env/Rollout collector and dispatcher: each environment produces
the configured number of action chunks, and each FULL_SHARD Actor rank receives
one complete temporal trajectory with its bootstrap row. Diagnostic subclasses observe
inherited initialization, GAE, PPO, synchronization, and Runner.run. They do not
manually construct, duplicate, or partition trajectories. No learning claim is
made by this execution test. The final sync is explicit. A resume starts fresh
Env/Rollout workers and new episodes; it does not restore simulator state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

from official_actor_rollout_probe import as_json as _base_as_json, tree_sha
from two_rank_rollout_probe import state_manifest


def as_json(value):
    """Include NumPy environment identifiers in otherwise CPU-safe evidence."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(nested) for nested in value]
    return _base_as_json(value)


def memory_snapshot(include_gpu=True):
    """Report per-process peak RSS and framework GPU allocation, in bytes."""
    import torch

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if include_gpu and torch.musa.is_initialized():
        result.update(gpu_allocated_bytes=int(torch.musa.memory_allocated()),
                      gpu_reserved_bytes=int(torch.musa.memory_reserved()),
                      gpu_peak_allocated_bytes=int(torch.musa.max_memory_allocated()),
                      gpu_peak_reserved_bytes=int(torch.musa.max_memory_reserved()))
    return result


def nested_tensor_shapes(value, path=()):
    import torch

    if isinstance(value, dict):
        result = {}
        for key, nested in value.items():
            result.update(nested_tensor_shapes(nested, (*path, key)))
        return result
    if isinstance(value, torch.Tensor):
        return {".".join(path): list(value.shape)}
    return {}


def nested_tensor_finite(value):
    """Inspect model input tensors without requiring a separate action field."""
    import torch

    if isinstance(value, dict):
        return all(nested_tensor_finite(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_tensor_finite(nested) for nested in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return True


def parse_placement(value: str) -> list[int]:
    """Parse the contiguous accelerator placement form used by the probes."""
    ranks = []
    for part in str(value).split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1 and bounds[0]:
            ranks.append(int(bounds[0]))
        elif len(bounds) == 2:
            start, end = (int(item) for item in bounds)
            if end < start:
                raise ValueError(f"invalid descending placement range: {part}")
            ranks.extend(range(start, end + 1))
        elif part.strip():
            raise ValueError(f"invalid placement range: {part}")
    result = sorted(set(ranks))
    if not result or result[0] < 0 or result != list(range(result[0], result[-1] + 1)):
        raise ValueError("placement must contain one non-empty contiguous range")
    return result


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("reset id must be non-negative or none")
    return result


def parse_task_ids(value: str) -> list[int] | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 0 for item in result):
        raise ValueError("task-id-filter must contain non-negative integer ids")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1,
                        help="additional official Runner iterations in this process")
    parser.add_argument("--steps-per-env", type=int, default=240,
                        help="simulator step budget per environment and iteration")
    parser.add_argument("--action-chunks", type=int, default=5,
                        help="actions executed per policy decision; must divide steps-per-env")
    parser.add_argument("--expected-state-count", type=int, default=907)
    parser.add_argument("--expected-selected-count", type=int, default=322)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--value-lr", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--actor-placement", default="0-1",
                        help="contiguous accelerator ranks for FULL_SHARD Actors")
    parser.add_argument("--rollout-placement", default="1",
                        help="accelerator ranks for the independent Rollout")
    parser.add_argument("--env-placement", default="0",
                        help="accelerator ranks for the Env worker")
    parser.add_argument("--expected-accelerators", type=int, default=2)
    parser.add_argument("--total-envs", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="defaults to total envs times policy decisions")
    parser.add_argument("--specific-reset-id", type=parse_optional_int, default=0,
                        help="LIBERO reset id; use 'none' for the full training pool")
    parser.add_argument("--task-id-filter", type=parse_task_ids, default=None,
                        help="optional comma-separated LIBERO task ids or 'all'")
    parser.add_argument("--ordered-training-resets", action="store_true",
                        help="walk the seeded reset pool instead of random training resets")
    parser.add_argument("--min-distinct-task-ids", type=int, default=1,
                        help="minimum distinct task ids required in the initial env report")
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--save-interval", type=int, default=-1,
                        help="positive interval enables inherited periodic Runner saving")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--resume-reference", type=Path,
                        help="successful save result to compare Actor state immediately after restore")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.steps_per_env < 1 or args.action_chunks < 1:
        parser.error("iteration, step and action budgets must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.value_lr)):
        parser.error("learning rates must be finite and positive")
    if (args.resume_dir is None) != (args.resume_reference is None):
        parser.error("resume-dir and resume-reference must be provided together")
    if args.steps_per_env % args.action_chunks:
        parser.error("action-chunks must divide steps-per-env")
    chunk_steps = args.steps_per_env // args.action_chunks
    try:
        actor_ranks = parse_placement(args.actor_placement)
        rollout_ranks = parse_placement(args.rollout_placement)
        env_ranks = parse_placement(args.env_placement)
        task_id_filter = args.task_id_filter
    except ValueError as error:
        parser.error(str(error))
    actor_world_size = len(actor_ranks)
    if len(rollout_ranks) != 1 or len(env_ranks) != 1:
        parser.error("this lifecycle probe requires one Rollout and one Env worker")
    if args.expected_accelerators < 1 or args.expected_accelerators <= max(actor_ranks + rollout_ranks + env_ranks):
        parser.error("expected-accelerators must cover every placement rank")
    if args.total_envs != actor_world_size:
        parser.error("this lifecycle probe requires exactly one environment per Actor rank")
    expected_samples = args.total_envs * chunk_steps
    global_batch_size = (expected_samples if args.global_batch_size is None
                         else args.global_batch_size)
    if (global_batch_size < 1 or global_batch_size % actor_world_size
            or global_batch_size != expected_samples):
        parser.error("global-batch-size must equal collected samples and be divisible by Actor world size")
    if args.min_distinct_task_ids < 1 or args.min_distinct_task_ids > args.total_envs:
        parser.error("min-distinct-task-ids must be between 1 and total-envs")
    if args.specific_reset_id is not None and args.min_distinct_task_ids != 1:
        parser.error("a specific reset id cannot cover multiple tasks")
    if args.save_interval == 0 or args.save_interval < -1:
        parser.error("save-interval must be -1 or positive")
    if args.save_final and args.save_interval > 0:
        parser.error("use either periodic saving or save-final")
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        parser.error("output and partial output must be new")
    if os.environ.get("RLINF_MUSA_FSDP_INDEPENDENT_INIT", "0") != "0":
        parser.error("official synchronized initialization is required")
    if os.environ.get("MCCL_P2P_DISABLE", "1") != "1":
        parser.error("validated Worker placement requires MCCL_P2P_DISABLE=1")
    paths = [str(args.rlinf_source.resolve()), str(args.gr00t_source.resolve()),
             str(Path(__file__).resolve().parent)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    os.environ["RLINF_MUSA_FSDP_LEGACY_NORM"] = "1"
    os.environ["RLINF_MUSA_FSDP_OPTIM_DEVICE_HANDLE"] = "1"
    os.environ["MCCL_P2P_DISABLE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "fail",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": f"official EmbodiedRunner.run; {actor_world_size} FULL_SHARD Actor ranks; "
                       f"Rollout placement {args.rollout_placement}; {args.total_envs} LIBERO envs",
              "iterations": args.iterations, "steps_per_environment": args.steps_per_env, "total_envs": args.total_envs,
              "actions_per_policy_decision": args.action_chunks,
              "policy_decisions_per_environment": chunk_steps,
              "expected_global_samples_per_iteration": expected_samples,
              "global_batch_size": global_batch_size,
              "actor_placement": args.actor_placement,
              "rollout_placement": args.rollout_placement,
              "env_placement": args.env_placement,
              "training_reset": {"specific_reset_id": args.specific_reset_id,
                                  "task_id_filter": task_id_filter,
                                  "ordered": args.ordered_training_resets},
              "sample_unit": "policy decisions / action chunks; simulator slots are reported separately",
              "scope_note": "Runner continuation/final-sync execution audit; no learning or benchmark-performance claim",
              "resume_scope": "Actor training state restored; Env/Rollout start new episodes, no simulator or rollout RNG restoration",
              "save_final_requested": args.save_final,
              "save_interval": args.save_interval,
              "saved_checkpoints": [],
              "trajectory_routing": "official TrajectoryCollector and least_loaded dispatcher; no manual partition",
              "memory_scope": "per-process RSS high-water marks and PyTorch MUSA allocator peaks; simulator child processes and Ray shared object storage are excluded",
              "synchronizations": [], "initialization": "sync_module_states=True",
              "transport_environment": {name: os.environ.get(name) for name in (
                  "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG", "MUSA_LAUNCH_BLOCKING",
                  "RLINF_MUSA_FSDP_LEGACY_NORM", "RLINF_MUSA_FSDP_OPTIM_DEVICE_HANDLE")}}
    started = time.monotonic()
    stage = "imports"

    def progress(name):
        nonlocal stage
        stage = name
        result["stage"] = name
        args.output.with_suffix(".partial.json").write_text(
            json.dumps(as_json(result), indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"stage": name, "elapsed_seconds": time.monotonic() - started}), flush=True)

    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from official_actor_config import make_config
        from rlinf.config import validate_cfg
        from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
        from rlinf.utils.utils import get_rng_state
        from rlinf.envs.utils import get_env_attr
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

        start_step = (parse_global_step_from_checkpoint_path(str(args.resume_dir.resolve()))
                      if args.resume_dir else 0)
        target_step = start_step + args.iterations
        reference = json.loads(args.resume_reference.read_text()) if args.resume_reference else None
        if reference is not None and (reference.get("status") != "pass" or not reference.get("checkpoint_state")
                          or not (reference.get("save_final_requested") is True or reference.get("saved_checkpoints"))
                          or reference["runner_global_step"] != start_step
                          or [row["rank"] for row in reference["checkpoint_state"]] != list(range(actor_world_size))
                          or Path(reference["checkpoint_dir"]).resolve() != args.resume_dir.resolve()):
            raise AssertionError("resume reference must be a successful saved Runner result at the requested step")
        if reference is not None:
            recorded_files = reference["checkpoint_files"]
            actual_files = {str(path.relative_to(args.resume_dir))
                            for path in args.resume_dir.rglob("*") if path.is_file()}
            if actual_files != {row["path"] for row in recorded_files}:
                raise AssertionError("checkpoint file coverage differs from save reference")
            for row in recorded_files:
                path = (args.resume_dir / row["path"]).resolve()
                if (not path.is_relative_to(args.resume_dir.resolve())
                        or path.stat().st_size != row["bytes"]
                        or file_sha256(path) != row["sha256"]):
                    raise AssertionError("checkpoint file bytes differ from save reference")
            result["checkpoint_file_hashes_verified_before_load"] = True
        result.update(start_global_step=start_step, target_global_step=target_step,
                      resume_reference_sha256=hashlib.sha256(args.resume_reference.read_bytes()).hexdigest()
                          if args.resume_reference else None)

        def seed(seed_value):
            import random
            import numpy as np

            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.musa.manual_seed(seed_value)
            if args.diagnostic:
                import faulthandler
                faulthandler.dump_traceback_later(120, repeat=True)

        def event(component, rank, name, data):
            row = {"component": component, "rank": rank, "event": name,
                   "unix_time": time.time(), **as_json(data)}
            path = args.output.with_name(f"{args.output.stem}.{component}.rank{rank}.events.jsonl")
            with path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"component": component, "rank": rank, "event": name}), flush=True)

        class AuditActor(EmbodiedFSDPActor):
            def seed_probe(self, seed_value):
                seed(seed_value)
                self.probe_records = []

            def snapshot_full_cpu(self):
                # All ranks enter reconstruction. Only rank zero retains a
                # complete CPU state dict; rank one's empty dict is expected.
                manifest = state_manifest(self.get_model_state_dict(
                    cpu_offload=True, full_state_dict=True))
                return {"rank": self._rank, "version": self.version,
                        "state_count": len(manifest), "manifest": manifest,
                        "model_sha256": tree_sha(manifest),
                        "selected_sync_names": list(self.param_names_need_sync),
                        "memory": memory_snapshot()}

            def snapshot_local(self):
                trainable = {name: param.detach() for name, param in self.model.named_parameters()
                             if param.requires_grad}
                active_states = [state for param, state in self.optimizer.state.items() if param.numel()]
                parameters_finite = all(not param.numel() or bool(torch.isfinite(param).all().item())
                                        for param in self.model.parameters())
                optimizer_finite = all(not isinstance(value, torch.Tensor) or not value.numel()
                    or bool(torch.isfinite(value).all().item())
                    for state in self.optimizer.state.values() for value in state.values())
                return {"rank": self._rank, "version": self.version,
                        "trainable_sha256": tree_sha(trainable),
                        "local_trainable_numel": sum(value.numel() for value in trainable.values()),
                        "optimizer_sha256": tree_sha(self.optimizer.state_dict()),
                        "optimizer_steps": self.optimizer_steps,
                        "active_adam_state_count": len(active_states),
                        "active_adam_steps": sorted(set(float(state["step"].item()) for state in active_states)),
                        "parameters_finite": parameters_finite, "optimizer_finite": optimizer_finite,
                        "memory": memory_snapshot()}

            def snapshot_resume_state(self):
                named = list(self.model.named_parameters())
                local_values = [param.detach() for _, param in named]
                trainable = {name: param.detach() for name, param in named if param.requires_grad}
                buffers = dict(self.model.named_buffers())
                topology = [(name, str(param.dtype), list(param.shape), param.requires_grad)
                            for name, param in named]
                active_optimizer = {
                    "state": {name: self.optimizer.state.get(param, {})
                              for name, param in named if param.requires_grad and param.numel()},
                    "param_groups": self.optimizer.state_dict()["param_groups"],
                }
                return {"rank": self._rank, "version": self.version,
                        "local_model_sha256": tree_sha(local_values), "trainable_sha256": tree_sha(trainable),
                        "buffers_sha256": tree_sha(buffers), "topology_sha256": tree_sha(topology),
                        "active_optimizer_sha256": tree_sha(active_optimizer),
                        "scheduler_sha256": tree_sha(self.lr_scheduler.state_dict()),
                        "rng_sha256": tree_sha(get_rng_state()), "optimizer_steps": self.optimizer_steps,
                        "nonempty_adam_steps": sorted(set(float(state["step"].item())
                            for param, state in self.optimizer.state.items() if param.numel()))}

            async def recv_rollout_trajectories(self, input_channel):
                await super().recv_rollout_trajectories(input_channel)
                batch = self.rollout_batch
                shapes = nested_tensor_shapes(batch)
                version_values = sorted(set(batch["versions"].reshape(-1).tolist()))
                # Independently count the prefix through the first ending action.
                # Repeated true boundary flags after termination are not episodes.
                done_slots = batch["dones"][1:].transpose(0, 1).reshape(1, -1)
                end_indices = torch.where(done_slots.any(dim=1),
                    done_slots.to(torch.int64).argmax(dim=1),
                    torch.full((1,), args.steps_per_env - 1, dtype=torch.int64,
                               device=done_slots.device))
                action_mask = torch.arange(args.steps_per_env, device=done_slots.device)[None, :] <= end_indices[:, None]
                expected_chunk_mask = action_mask.reshape(1, chunk_steps, args.action_chunks).transpose(0, 1).any(dim=-1, keepdim=True)
                checks = {"complete_temporal_sequence": tuple(batch["prev_logprobs"].shape[:2]) == (chunk_steps, 1),
                          "final_value_bootstrap": tuple(batch["prev_values"].shape[:2]) == (chunk_steps + 1, 1),
                          "initial_done_boundary": tuple(batch["dones"].shape[:2]) == (chunk_steps + 1, 1),
                          "initial_pre_action_done_false": not bool(batch["dones"][0].any()),
                          "on_policy_version": version_values == [float(self.version)],
                          "finite_policy_statistics": all(bool(torch.isfinite(batch[key]).all())
                              for key in ("prev_logprobs", "prev_values", "rewards")),
                          "finite_forward_inputs": nested_tensor_finite(batch["forward_inputs"]),
                          "loss_mask_includes_first_terminal_excludes_later": batch.get("loss_mask") is not None
                              and torch.equal(batch["loss_mask"].bool(), expected_chunk_mask)}
                row = {"iteration": len(self.probe_records), "rank": self._rank,
                       "actor_version": self.version, "stored_versions": version_values,
                       "received_sample_count": int(batch["prev_logprobs"].shape[0]
                                                    * batch["prev_logprobs"].shape[1]),
                       "simulator_action_slots": args.steps_per_env,
                       "valid_simulator_action_slots": int(action_mask.sum()),
                       "boundary_count_scope": "true per-action boundary flags; repeated post-terminal flags are not unique episodes",
                       "received_shapes": shapes, "receive_checks": checks,
                       "received_top_level_keys": sorted(batch),
                       "actions_field_present": "actions" in batch,
                       "action_contract": "actions are sent separately from Rollout to Env; Actor uses official forward_inputs",
                       "received_batch_sha256": tree_sha(batch),
                       "policy_data_sha256": tree_sha({key: batch[key] for key in ("forward_inputs", "prev_logprobs")}),
                       "rewards": {"sum": float(batch["rewards"].sum()),
                                   "max": float(batch["rewards"].max()),
                                   "nonzero_count": int(torch.count_nonzero(batch["rewards"])),
                                   "element_count": int(batch["rewards"].numel())},
                       # Omit the initial pre-action boundary from transition
                       # counts; retain its shape separately in receive checks.
                       "boundary_counts": {key: int(torch.count_nonzero(batch[key][1:]))
                                           for key in ("dones", "terminations", "truncations")},
                       "loss_mask": {"true_count": int(torch.count_nonzero(batch["loss_mask"])),
                                     "element_count": int(batch["loss_mask"].numel())}
                           if batch.get("loss_mask") is not None else None,
                       "memory_after_receive": memory_snapshot()}
                self.probe_records.append(row)
                event("actor", self._rank, "official_trajectory_received", row)
                if not all(checks.values()):
                    raise AssertionError(f"official collector/dispatcher trajectory audit failed: {checks}")

            def compute_advantages_and_returns(self):
                metrics = super().compute_advantages_and_returns()
                batch = self.rollout_batch
                row = {"metrics": as_json(metrics),
                       "advantages_shape": list(batch["advantages"].shape),
                       "returns_shape": list(batch["returns"].shape),
                       "advantages_sha256": tree_sha(batch["advantages"]),
                       "returns_sha256": tree_sha(batch["returns"]),
                       "finite": all(bool(torch.isfinite(batch[key]).all()) for key in ("advantages", "returns")),
                       "memory": memory_snapshot()}
                self.probe_records[-1]["gae"] = row
                event("actor", self._rank, "official_gae_complete", row)
                if not row["finite"]:
                    raise AssertionError("official GAE returned non-finite values")
                return metrics

            def run_training(self):
                before = self.snapshot_local()
                self.probe_micro_batch_count = 0
                self.probe_training_started = time.monotonic()
                event("actor", self._rank, "official_training_enter", before)
                metrics = super().run_training()
                after = self.snapshot_local()
                checks = {"changed_trainable_shard": before["trainable_sha256"] != after["trainable_sha256"],
                          "changed_optimizer": before["optimizer_sha256"] != after["optimizer_sha256"],
                          "one_optimizer_step": after["optimizer_steps"] == before["optimizer_steps"] + 1,
                          "active_adam_increment": bool(after["active_adam_state_count"])
                              and after["active_adam_steps"] == [before["optimizer_steps"] + 1.0],
                          "finite_parameters_and_optimizer": after["parameters_finite"] and after["optimizer_finite"],
                          "finite_nonzero_grad_norm": math.isfinite(float(metrics["actor/grad_norm"]))
                              and float(metrics["actor/grad_norm"]) > 0,
                          "finite_metrics": all(math.isfinite(float(value)) for value in metrics.values())}
                row = {"before": before, "after": after, "metrics": as_json(metrics), "checks": checks}
                self.probe_records[-1]["training"] = row
                event("actor", self._rank, "official_training_complete", row)
                if not all(checks.values()):
                    raise AssertionError(f"official PPO update audit failed: {checks}")
                return metrics

            def train_micro_batch(self, *positional, **keywords):
                value = super().train_micro_batch(*positional, **keywords)
                self.probe_micro_batch_count += 1
                if self.probe_micro_batch_count % 16 == 0 or keywords.get("is_last", False):
                    event("actor", self._rank, "official_micro_batches_submitted", {
                        "version": self.version,
                        "count": self.probe_micro_batch_count,
                        "expected": chunk_steps,
                        "elapsed_seconds": time.monotonic() - self.probe_training_started,
                        "scope": "inherited micro-batch calls returned; optimizer update audited separately"})
                return value

            def probe_report(self):
                return {"rank": self._rank, "iterations": self.probe_records,
                        "final": self.snapshot_local()}

        class AuditRollout(MultiStepRolloutWorker):
            def seed_probe(self, seed_value):
                seed(seed_value)

            def snapshot_cpu(self):
                # Runner's inherited generate() offloads the model after each
                # epoch. Synchronization audits precede the next generate().
                manifest = state_manifest(self.hf_model.state_dict())
                return {"rank": self._rank, "version": self.version, "global_step": self.global_step,
                        "manifest": manifest, "model_sha256": tree_sha(manifest),
                        "state_count": len(manifest), "memory": memory_snapshot(),
                        "fallback": self.hf_model.s4000_fallback_metadata}

        class AuditEnv(EnvWorker):
            def bootstrap_step(self):
                outputs = super().bootstrap_step()
                if not hasattr(self, "sampling_reports"):
                    self.sampling_reports = []
                report = self.probe_report()
                for stage_id, item in enumerate(report["identities"]):
                    descriptions = list(get_env_attr(self.env_list[stage_id], "task_descriptions"))
                    if list(outputs[stage_id].obs["task_descriptions"]) != descriptions:
                        raise AssertionError("policy observation task descriptions differ from active tasks")
                    item["task_descriptions"] = descriptions
                report["event"] = "training_bootstrap"
                self.sampling_reports.append(report)
                event("env", self._rank, "training_bootstrap", report)
                return outputs

            def finish_rollout(self, mode="train"):
                if mode == "train":
                    report = self.probe_report()
                    report["elapsed_simulator_steps"] = [
                        as_json(get_env_attr(env, "elapsed_steps")) for env in self.env_list]
                    report["success_once"] = [
                        as_json(get_env_attr(env, "success_once")) for env in self.env_list]
                    report["event"] = "training_horizon_complete"
                    self.sampling_reports.append(report)
                    event("env", self._rank, "training_horizon_complete", report)
                return super().finish_rollout(mode)

            def sampling_report(self):
                return getattr(self, "sampling_reports", [])

            def probe_report(self):
                identities = []
                for env in self.env_list:
                    identities.append({name: as_json(get_env_attr(env, name))
                                       for name in ("task_ids", "trial_ids")})
                distinct_task_ids = sorted({task for item in identities for task in item["task_ids"]})
                return {"rank": self._rank, "trajectory_step": self._trajectory_step,
                        "envs_per_stage": self.train_num_envs_per_stage,
                        "stages": self.stage_num, "identities": identities,
                        "distinct_task_ids": distinct_task_ids,
                        "distinct_task_count": len(distinct_task_ids),
                        "memory": memory_snapshot(include_gpu=False)}

        class AuditRunner(EmbodiedRunner):
            def _save_checkpoint(self):
                checkpoint_dir = (Path(self.cfg.runner.logger.log_path)
                    / self.cfg.runner.logger.experiment_name / "checkpoints"
                    / f"global_step_{self.global_step}")
                if checkpoint_dir.exists():
                    raise FileExistsError("refusing to overwrite a Runner checkpoint")
                if shutil.disk_usage(args.output.parent).free < 18 * 1024 ** 3:
                    raise RuntimeError("Runner DCP saving requires at least 18 GiB free")
                # The inherited loop increments Runner progress after training;
                # Actor version otherwise remains the version used for sampling.
                # Save the completed-step version without adding a rollout sync.
                self.actor.set_global_step(self.global_step).wait()
                before = self.actor.snapshot_resume_state().wait()
                progress(f"runner_step_{self.global_step}/official_checkpoint_save")
                super()._save_checkpoint()
                after = self.actor.snapshot_resume_state().wait()
                if before != after:
                    raise AssertionError("checkpoint save changed Actor training state")
                files = [{"path": str(path.relative_to(checkpoint_dir)),
                          "bytes": path.stat().st_size,
                          "sha256": file_sha256(path)}
                         for path in sorted(checkpoint_dir.rglob("*")) if path.is_file()]
                result["saved_checkpoints"].append({"global_step": self.global_step,
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "checkpoint_state": before, "post_save_actor_state": after,
                    "checkpoint_files": files,
                    "checkpoint_bytes": sum(row["bytes"] for row in files)})
                progress(f"runner_step_{self.global_step}/checkpoint_save_audited")

            def update_rollout_weights(self):
                progress(f"runner_step_{self.global_step}/official_weight_sync")
                super().update_rollout_weights()
                progress(f"runner_step_{self.global_step}/full_state_sync_audit")
                actor_states = self.actor.snapshot_full_cpu().wait()
                rollout_state = self.rollout.snapshot_cpu().wait()[0]
                leader = actor_states[0]
                selected = leader["selected_sync_names"]
                checks = {"actor_ranks_participated": [row["rank"] for row in actor_states] == list(range(actor_world_size)),
                          "complete_state_count": leader["state_count"] == args.expected_state_count,
                          "selected_count": len(selected) == args.expected_selected_count,
                          "selected_unique_and_covered": len(set(selected)) == len(selected)
                              and set(selected) <= set(leader["manifest"]),
                          "all_states_hashes_shapes_dtypes_equal": leader["manifest"] == rollout_state["manifest"],
                          "actor_versions": all(row["version"] == self.global_step for row in actor_states),
                          "rollout_version": rollout_state["version"] == self.global_step,
                          "optional_nonleader_states_equal": all(not row["manifest"]
                              or row["manifest"] == leader["manifest"] for row in actor_states),
                          "same_selected_names_on_ranks": all(row["selected_sync_names"] == selected
                              for row in actor_states)}
                row = {"runner_global_step": self.global_step, "actor_states": actor_states,
                       "rollout_state": rollout_state, "checks": checks}
                result["synchronizations"].append(row)
                progress(f"runner_step_{self.global_step}/full_state_sync_audit_complete")
                if not all(checks.values()):
                    raise AssertionError(f"official Runner synchronization audit failed: {checks}")

        cfg = make_config(args.rlinf_source, args.model_path, steps=args.steps_per_env)
        cfg.actor.optim.lr = args.lr
        cfg.actor.optim.value_lr = args.value_lr
        cfg.actor.seed = args.seed
        cfg.actor.model.num_action_chunks = args.action_chunks
        cfg.cluster.component_placement = {"actor": args.actor_placement,
                                           "rollout": args.rollout_placement,
                                           "env": args.env_placement}
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.use_orig_params = True
        cfg.actor.fsdp_config.torch22_state_dict_backend = "sharded_tensor"
        cfg.actor.global_batch_size = global_batch_size
        cfg.actor.micro_batch_size = 1
        cfg.algorithm.update_epoch = 1
        cfg.env.train.total_num_envs = args.total_envs
        cfg.env.train.specific_reset_id = args.specific_reset_id
        cfg.env.train.use_ordered_reset_state_ids = args.ordered_training_resets
        cfg.env.train.task_id_filter = task_id_filter
        cfg.env.train.rollout_epoch = 1
        cfg.env.train.max_steps_per_rollout_epoch = args.steps_per_env
        cfg.env.eval.total_num_envs = args.total_envs
        cfg.rollout.enable_offload = True
        cfg.weight_syncer.actor_state_mode = "full_cpu_rank0"
        cfg.runner.data_channel_transport = "ray"
        cfg.runner.max_epochs = target_step
        cfg.runner.max_steps = target_step
        cfg.runner.save_interval = args.save_interval
        cfg.runner.val_check_interval = -1
        cfg.runner.weight_sync_interval = 1
        cfg.runner.use_training_pipeline = False
        cfg.runner.overlap_env_bootstrap = False
        cfg.runner.resume_dir = str(args.resume_dir.resolve()) if args.resume_dir else None
        cfg.runner.logger = {"log_path": str(args.output.parent / f"{args.output.stem}.logs"),
                             "project_name": "rlinf-musa-lab", "experiment_name": args.output.stem,
                             "logger_backends": []}
        progress("validate_official_config")
        cfg = validate_cfg(cfg)
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        result["config_sha256"] = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
        result["source_fingerprints"] = {
            str(path.relative_to(args.rlinf_source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.rlinf_source / "rlinf").rglob("*.py")}
        result["gr00t_source_fingerprints"] = {
            str(path.relative_to(args.gr00t_source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.gr00t_source / "gr00t").rglob("*.py")}
        result["support_fingerprints"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("official_actor_config.py", "official_actor_rollout_probe.py",
                         "two_rank_rollout_probe.py", "actor_runtime_extension.py", "musa_fsdp_norm.py",
                         "musa_fsdp_optim_device.py", "gr00t_action_head_fallback.py")}
        if reference is not None:
            plain_cfg = result["config"]
            matching_cfg = {
                "actor_model": plain_cfg["actor"]["model"], "actor_optim": plain_cfg["actor"]["optim"],
                "actor_seed": plain_cfg["actor"]["seed"], "algorithm": plain_cfg["algorithm"],
                "actor_fsdp": plain_cfg["actor"]["fsdp_config"], "global_batch": plain_cfg["actor"]["global_batch_size"],
                "train_env": plain_cfg["env"]["train"],
                "placement": plain_cfg["cluster"]["component_placement"],
            }
            old_cfg = reference["config"]
            previous_cfg = {
                "actor_model": old_cfg["actor"]["model"], "actor_optim": old_cfg["actor"]["optim"],
                "actor_seed": old_cfg["actor"]["seed"], "algorithm": old_cfg["algorithm"],
                "actor_fsdp": old_cfg["actor"]["fsdp_config"], "global_batch": old_cfg["actor"]["global_batch_size"],
                "train_env": old_cfg["env"]["train"],
                "placement": old_cfg["cluster"]["component_placement"],
            }
            if matching_cfg != previous_cfg:
                raise AssertionError("resume configuration differs from the saved training configuration")
            if result["source_fingerprints"] != reference["source_fingerprints"]:
                raise AssertionError("resume production source differs from the saved run")
            if (result["gr00t_source_fingerprints"] != reference["gr00t_source_fingerprints"]
                    or result["support_fingerprints"] != reference["support_fingerprints"]
                    or result["probe_sha256"] != reference["probe_sha256"]):
                raise AssertionError("resume probe, helpers or GR00T source differ from the saved run")
        if (args.save_final or args.save_interval > 0) and shutil.disk_usage(args.output.parent).free < 18 * 1024 ** 3:
            raise RuntimeError("a full Runner DCP needs at least 18 GiB free; existing checkpoints are preserved")
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
        result["scheduler_collective_private_api_support"] = {
            name: hasattr(torch.distributed.distributed_c10d, name)
            for name in ("_process_group_color", "_register_process_group", "_DistributedBackendOptions")}
        cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
        if cluster.num_accelerators != args.expected_accelerators:
            raise RuntimeError(f"expected {args.expected_accelerators} accelerators, got {cluster.num_accelerators}")
        placement = HybridComponentPlacement(cfg, cluster)
        progress("launch_official_worker_groups")
        actor = AuditActor.create_group(cfg).launch(cluster=cluster, name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy("actor"))
        rollout = AuditRollout.create_group(cfg).launch(cluster=cluster, name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"))
        env = AuditEnv.create_group(cfg).launch(cluster=cluster, name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"))
        actor.seed_probe(cfg.actor.seed).wait()
        rollout.seed_probe(cfg.actor.seed).wait()
        runner = AuditRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
        result["data_channels"] = {name: bool(getattr(getattr(runner, name), "_use_ray_transport", False))
                                   for name in ("env_channel", "rollout_channel", "actor_channel")}
        if not all(result["data_channels"].values()):
            raise AssertionError("runner.data_channel_transport=ray was not applied to every data channel")
        progress("official_runner_init_workers")
        runner.init_workers()
        result["initial_actor_state"] = actor.snapshot_resume_state().wait()
        if runner.global_step != start_step:
            raise AssertionError("official Runner restored the wrong global step")
        if [row["rank"] for row in result["initial_actor_state"]] != list(range(actor_world_size)):
            raise AssertionError("Actor ranks did not initialize as expected")
        if reference is not None:
            expected_state = reference["checkpoint_state"]
            result["restore_checks"] = {
                f"rank{new['rank']}_{key}": new[key] == old[key]
                for new, old in zip(result["initial_actor_state"], expected_state, strict=True)
                for key in ("rank", "version", "local_model_sha256", "trainable_sha256", "buffers_sha256",
                            "topology_sha256", "active_optimizer_sha256", "scheduler_sha256", "rng_sha256",
                            "optimizer_steps", "nonempty_adam_steps")}
            if not all(result["restore_checks"].values()):
                raise AssertionError("Actor state did not restore exactly before fresh sampling")
        result["environment"] = env.probe_report().wait()
        if result["environment"][0]["distinct_task_count"] < args.min_distinct_task_ids:
            raise AssertionError(
                f"initial training reset pool covered only {result['environment'][0]['distinct_task_count']} "
                f"task ids; expected at least {args.min_distinct_task_ids}"
            )
        progress(f"official_runner_run_{args.iterations}_iterations")
        runner.run()
        progress("final_explicit_weight_sync")
        actor.set_global_step(runner.global_step).wait()
        rollout.set_global_step(runner.global_step).wait()
        env.set_global_step(runner.global_step).wait()
        runner.update_rollout_weights()
        result["checkpoint_state"] = actor.snapshot_resume_state().wait()
        if args.save_final:
            progress("official_runner_save_checkpoint")
            runner._save_checkpoint()
        if result["saved_checkpoints"]:
            last_save = result["saved_checkpoints"][-1]
            if last_save["global_step"] != target_step or last_save["checkpoint_state"] != result["checkpoint_state"]:
                raise AssertionError("last saved checkpoint differs from completed training state")
            for key in ("checkpoint_dir", "post_save_actor_state", "checkpoint_files", "checkpoint_bytes"):
                result[key] = last_save[key]
        progress("final_worker_audit")
        actors = actor.probe_report().wait()
        rollout_final = rollout.snapshot_cpu().wait()[0]
        result["actor_reports"] = actors
        result["rollout_final"] = rollout_final
        result["env_final"] = env.probe_report().wait()
        result["training_sampling_reports"] = env.sampling_report().wait()[0]
        bootstraps = [row for row in result["training_sampling_reports"]
                      if row["event"] == "training_bootstrap"]
        completions = [row for row in result["training_sampling_reports"]
                       if row["event"] == "training_horizon_complete"]
        identity_batches = [row["identities"] for row in bootstraps]
        result["training_sampling_checks"] = {
            "one_bootstrap_and_completion_per_iteration": len(bootstraps) == len(completions) == args.iterations,
            "completed_horizons_use_same_task_trial_as_bootstrap": all(
                [{key: item[key] for key in ("task_ids", "trial_ids")} for item in start["identities"]]
                == end["identities"] for start, end in zip(bootstraps, completions)),
            "multiple_tasks_in_actual_sampling": all(
                row["distinct_task_count"] >= args.min_distinct_task_ids for row in bootstraps),
            "requested_horizon_elapsed_on_all_lanes": all(
                all(value == args.steps_per_env for stage_values in row["elapsed_simulator_steps"] for value in stage_values)
                for row in completions),
            "reset_states_advance_between_iterations": (args.specific_reset_id is not None
                or args.iterations == 1 or all(left != right for left, right in zip(identity_batches, identity_batches[1:]))),
        }
        if not all(result["training_sampling_checks"].values()):
            raise AssertionError(f"training reset/trajectory checks failed: {result['training_sampling_checks']}")
        result["runner_global_step"] = runner.global_step
        checks = {"runner_completed_requested_steps": runner.global_step == target_step,
                  "syncs_at_all_iteration_versions_and_final": [row["runner_global_step"] for row in result["synchronizations"]] == list(range(start_step, target_step + 1)),
                  "actor_rank_reports": [row["rank"] for row in actors] == list(range(actor_world_size)),
                  "requested_trajectories_per_rank": all(len(row["iterations"]) == args.iterations for row in actors),
                  "requested_optimizer_updates_per_rank": all(row["final"]["optimizer_steps"] == target_step
                      and row["final"]["active_adam_steps"] == [float(target_step)] for row in actors),
                  "final_rollout_version": rollout_final["version"] == target_step,
                  "final_actor_version": all(row["final"]["version"] == target_step for row in actors)}
        result["final_checks"] = checks
        if not all(checks.values()):
            raise AssertionError(f"official Runner final audit failed: {checks}")
        per_iteration_checks = []
        for iteration in range(args.iterations):
            ranks = [row["iterations"][iteration] for row in actors]
            per_iteration_checks.append({"iteration": iteration,
                "global_sample_count": sum(row["received_sample_count"] for row in ranks) == expected_samples,
                "version_equals_global_iteration": all(row["stored_versions"] == [float(start_step + iteration)] for row in ranks),
                "different_received_batches": len({row["policy_data_sha256"] for row in ranks}) == actor_world_size,
                "both_receives_pass": all(all(row["receive_checks"].values()) for row in ranks),
                "both_gae_pass": all(row["gae"]["finite"] for row in ranks),
                "both_ppo_updates_pass": all(all(row["training"]["checks"].values()) for row in ranks),
                "same_reduced_grad_norm": len({row["training"]["metrics"]["actor/grad_norm"]
                                                for row in ranks}) == 1})
        result["per_iteration_checks"] = per_iteration_checks
        if not all(all(value for key, value in row.items() if key != "iteration") for row in per_iteration_checks):
            raise AssertionError("official Runner per-iteration audit failed")
        result["global_received_samples_per_iteration"] = [
            sum(row["iterations"][iteration]["received_sample_count"] for row in actors)
            for iteration in range(args.iterations)]
        result["total_received_action_chunk_count"] = sum(result["global_received_samples_per_iteration"])
        result["total_simulator_step_slots"] = result["total_received_action_chunk_count"] * args.action_chunks
        if args.action_chunks == 1:
            result["total_received_transition_count"] = result["total_received_action_chunk_count"]
        result["final_weight_scope"] = (
            f"Runner global_step and Actor/Rollout versions are {target_step}. "
            f"Rollout has weights after all {target_step} cumulative PPO updates; an explicit final sync was audited.")
        result["driver_memory"] = memory_snapshot(include_gpu=False)
        result["status"] = "pass"
    except Exception as error:
        result.update(stage=stage, error_class=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if "ray" in locals():
            ray.shutdown()
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(as_json(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "stage": stage, "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
