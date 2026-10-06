#!/usr/bin/env python3
"""Audit two actual EmbodiedRunner iterations on two S4000 GPUs.

Use the official Env/Rollout collector and dispatcher: two environments each
produce four action chunks, and each FULL_SHARD Actor rank receives one complete
four-step trajectory with its bootstrap row. Diagnostic subclasses observe
inherited initialization, GAE, PPO, synchronization, and Runner.run. They do not
manually construct, duplicate, or partition trajectories. No learning claim is
made by this execution test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int, default=907)
    parser.add_argument("--expected-selected-count", type=int, default=322)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    os.environ["MCCL_P2P_DISABLE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "fail",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "official EmbodiedRunner.run; two FULL_SHARD Actor ranks; one Rollout; two LIBERO envs",
              "iterations": 2, "steps_per_environment": 4, "total_envs": 2,
              "expected_global_samples_per_iteration": 8,
              "trajectory_routing": "official TrajectoryCollector and least_loaded dispatcher; no manual partition",
              "memory_scope": "per-process RSS high-water marks and PyTorch MUSA allocator peaks; simulator child processes and Ray shared object storage are excluded",
              "synchronizations": [], "initialization": "sync_module_states=True",
              "transport_environment": {name: os.environ.get(name) for name in (
                  "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG", "MUSA_LAUNCH_BLOCKING")}}
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
        from rlinf.envs.utils import get_env_attr
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

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

            async def recv_rollout_trajectories(self, input_channel):
                await super().recv_rollout_trajectories(input_channel)
                batch = self.rollout_batch
                shapes = nested_tensor_shapes(batch)
                version_values = sorted(set(batch["versions"].reshape(-1).tolist()))
                checks = {"complete_four_step_sequence": tuple(batch["prev_logprobs"].shape[:2]) == (4, 1),
                          "five_row_value_bootstrap": tuple(batch["prev_values"].shape[:2]) == (5, 1),
                          "five_row_done_boundary": tuple(batch["dones"].shape[:2]) == (5, 1),
                          "on_policy_version": version_values == [float(self.version)],
                          "finite_policy_statistics": all(bool(torch.isfinite(batch[key]).all())
                              for key in ("prev_logprobs", "prev_values", "rewards", "actions"))}
                row = {"iteration": len(self.probe_records), "rank": self._rank,
                       "actor_version": self.version, "stored_versions": version_values,
                       "received_sample_count": int(batch["prev_logprobs"].shape[0]
                                                    * batch["prev_logprobs"].shape[1]),
                       "received_shapes": shapes, "receive_checks": checks,
                       "received_batch_sha256": tree_sha(batch),
                       "policy_data_sha256": tree_sha({key: batch[key] for key in ("actions", "prev_logprobs")}),
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
            def probe_report(self):
                identities = []
                for env in self.env_list:
                    identities.append({name: as_json(get_env_attr(env, name))
                                       for name in ("task_ids", "trial_ids")})
                return {"rank": self._rank, "trajectory_step": self._trajectory_step,
                        "envs_per_stage": self.train_num_envs_per_stage,
                        "stages": self.stage_num, "identities": identities,
                        "memory": memory_snapshot(include_gpu=False)}

        class AuditRunner(EmbodiedRunner):
            def update_rollout_weights(self):
                progress(f"runner_step_{self.global_step}/official_weight_sync")
                super().update_rollout_weights()
                progress(f"runner_step_{self.global_step}/full_state_sync_audit")
                actor_states = self.actor.snapshot_full_cpu().wait()
                rollout_state = self.rollout.snapshot_cpu().wait()[0]
                leader = actor_states[0]
                selected = leader["selected_sync_names"]
                checks = {"two_actor_ranks_participated": [row["rank"] for row in actor_states] == [0, 1],
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

        cfg = make_config(args.rlinf_source, args.model_path, steps=4)
        cfg.cluster.component_placement = {"actor": "0-1", "rollout": "1", "env": "0"}
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.use_orig_params = True
        cfg.actor.fsdp_config.torch22_state_dict_backend = "sharded_tensor"
        cfg.actor.global_batch_size = 8
        cfg.actor.micro_batch_size = 1
        cfg.algorithm.update_epoch = 1
        cfg.env.train.total_num_envs = 2
        cfg.env.train.rollout_epoch = 1
        cfg.env.train.max_steps_per_rollout_epoch = 4
        cfg.env.eval.total_num_envs = 2
        cfg.rollout.enable_offload = True
        cfg.weight_syncer.actor_state_mode = "full_cpu_rank0"
        cfg.runner.data_channel_transport = "ray"
        cfg.runner.max_epochs = 2
        cfg.runner.max_steps = 2
        cfg.runner.save_interval = -1
        cfg.runner.val_check_interval = -1
        cfg.runner.weight_sync_interval = 1
        cfg.runner.use_training_pipeline = False
        cfg.runner.overlap_env_bootstrap = False
        cfg.runner.resume_dir = None
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
        result["support_fingerprints"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("official_actor_config.py", "official_actor_rollout_probe.py",
                         "two_rank_rollout_probe.py", "actor_runtime_extension.py", "musa_fsdp_norm.py")}
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
        result["scheduler_collective_private_api_support"] = {
            name: hasattr(torch.distributed.distributed_c10d, name)
            for name in ("_process_group_color", "_register_process_group", "_DistributedBackendOptions")}
        cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
        if cluster.num_accelerators != 2:
            raise RuntimeError(f"expected two accelerators, got {cluster.num_accelerators}")
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
        result["environment"] = env.probe_report().wait()
        progress("official_runner_run_two_iterations")
        runner.run()
        progress("final_worker_audit")
        actors = actor.probe_report().wait()
        rollout_final = rollout.snapshot_cpu().wait()[0]
        result["actor_reports"] = actors
        result["rollout_final"] = rollout_final
        result["env_final"] = env.probe_report().wait()
        result["runner_global_step"] = runner.global_step
        checks = {"runner_completed_two_steps": runner.global_step == 2,
                  "exactly_two_syncs_at_versions_zero_one": [row["runner_global_step"] for row in result["synchronizations"]] == [0, 1],
                  "two_actor_rank_reports": [row["rank"] for row in actors] == [0, 1],
                  "two_received_trajectories_per_rank": all(len(row["iterations"]) == 2 for row in actors),
                  "two_optimizer_updates_per_rank": all(row["final"]["optimizer_steps"] == 2
                      and row["final"]["active_adam_steps"] == [2.0] for row in actors),
                  "final_rollout_version_one": rollout_final["version"] == 1,
                  "final_actor_version_one": all(row["final"]["version"] == 1 for row in actors)}
        result["final_checks"] = checks
        if not all(checks.values()):
            raise AssertionError(f"official Runner final audit failed: {checks}")
        per_iteration_checks = []
        for iteration in range(2):
            ranks = [row["iterations"][iteration] for row in actors]
            per_iteration_checks.append({"iteration": iteration,
                "global_sample_count_eight": sum(row["received_sample_count"] for row in ranks) == 8,
                "version_equals_iteration": all(row["stored_versions"] == [float(iteration)] for row in ranks),
                "different_received_batches": len({row["policy_data_sha256"] for row in ranks}) == 2,
                "both_receives_pass": all(all(row["receive_checks"].values()) for row in ranks),
                "both_gae_pass": all(row["gae"]["finite"] for row in ranks),
                "both_ppo_updates_pass": all(all(row["training"]["checks"].values()) for row in ranks),
                "same_reduced_grad_norm": ranks[0]["training"]["metrics"]["actor/grad_norm"]
                    == ranks[1]["training"]["metrics"]["actor/grad_norm"]})
        result["per_iteration_checks"] = per_iteration_checks
        if not all(all(value for key, value in row.items() if key != "iteration") for row in per_iteration_checks):
            raise AssertionError("official Runner per-iteration audit failed")
        result["global_received_samples_per_iteration"] = [
            sum(row["iterations"][iteration]["received_sample_count"] for row in actors)
            for iteration in range(2)]
        result["final_weight_scope"] = (
            "Runner global_step is 2; Actor and Rollout versions remain 1. Rollout has weights after the first "
            "PPO update, used to collect iteration one. Actor has completed two updates. No extra final sync was run.")
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
