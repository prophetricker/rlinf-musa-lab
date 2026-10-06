#!/usr/bin/env python3
"""Audit the inherited RLinf evaluation path and actual LIBERO task/trial coverage.

Default: ten tasks, trial zero, two environment lanes, five actions per decision.
This is an initial-policy pilot, not the full suite or evidence of PPO learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from official_runner_probe import as_json, memory_snapshot, tree_sha


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials_per_task < 1:
        parser.error("trials-per-task must be positive")
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        parser.error("evidence output must be new")
    paths = [str(args.rlinf_source.resolve()), str(args.gr00t_source.resolve()),
             str(Path(__file__).resolve().parent)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["MCCL_P2P_DISABLE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"status": "fail", "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "inherited EmbodiedEvalRunner.init_workers/evaluate; initial Spatial policy; no Actor or PPO update",
              "trials_per_task_requested": args.trials_per_task,
              "policy_seed": 1234,
              "policy_rng_scope": "single whole-run seed before model initialization; no per-episode reseeding",
              "coverage_note": "one trial per task is a ten-task pilot; all actual init states are required for a full-suite baseline"}

    def progress(stage):
        result["stage"] = stage
        args.output.with_suffix(".partial.json").write_text(json.dumps(as_json(result), ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - started}), flush=True)

    try:
        import numpy as np
        from omegaconf import OmegaConf
        import random
        import torch
        import torch_musa
        import ray
        from official_actor_config import make_config
        from rlinf.config import validate_cfg
        from rlinf.envs.utils import get_env_attr
        from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

        def event(component, row):
            row = {"unix_time": time.time(), **as_json(row)}
            with args.output.with_name(f"{args.output.stem}.{component}.events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        class AuditEnv(EnvWorker):
            def env_evaluate_step(self, raw_actions, stage_id):
                env = self.eval_env_list[stage_id]
                before = set(get_env_attr(env, "_eval_seen_trials"))
                task_ids = np.array(get_env_attr(env, "task_ids"), copy=True)
                trial_ids = np.array(get_env_attr(env, "trial_ids"), copy=True)
                output, info = super().env_evaluate_step(raw_actions, stage_id)
                completed = set(get_env_attr(env, "_eval_seen_trials")) - before
                if completed:
                    episode = output.env_infos["final_info"]["episode"]
                    suite = get_env_attr(env, "task_suite")
                    for task, trial in sorted(completed):
                        index = int(np.flatnonzero((task_ids == task) & (trial_ids == trial))[0])
                        bins = get_env_attr(env, "trial_id_bins")
                        event("env", {"event": "episode_complete", "task_id": task, "trial_id": trial,
                                      "task_language": suite.get_task(task).language,
                                      "global_reset_id": int(sum(bins[:task]) + trial),
                                      "metrics": {key: as_json(value[index]) for key, value in episode.items()}})
                return output, info

            def coverage_report(self):
                rows = []
                for env in self.eval_env_list:
                    suite = get_env_attr(env, "task_suite")
                    tasks = []
                    for task_id in range(suite.get_num_tasks()):
                        task = suite.get_task(task_id)
                        states = suite.get_task_init_states(task_id)
                        tasks.append({"task_id": task_id, "language": task.language,
                                      "problem_folder": task.problem_folder, "bddl_file": task.bddl_file,
                                      "init_states_file": task.init_states_file,
                                      "init_state_count": len(states), "init_states_sha256": tree_sha(states)})
                    rows.append({"seen_trials": [list(pair) for pair in sorted(get_env_attr(env, "_eval_seen_trials"))],
                                 "task_stats": as_json(get_env_attr(env, "_task_success_stats")),
                                 "task_manifest": tasks,
                                 "init_state_counts": as_json(get_env_attr(env, "trial_id_bins"))})
                return {"rank": self._rank, "stages": rows, "memory": memory_snapshot(include_gpu=False)}

        class AuditRollout(MultiStepRolloutWorker):
            def seed_probe(self, value):
                random.seed(value)
                np.random.seed(value)
                torch.manual_seed(value)
                torch.musa.manual_seed(value)
                self.probe_predictions = 0

            def _predict_rollout_actions(self, *positional, **keywords):
                actions, details = super()._predict_rollout_actions(*positional, **keywords)
                finite = bool(torch.isfinite(actions).all()) if isinstance(actions, torch.Tensor) else bool(np.isfinite(actions).all())
                if not finite:
                    raise AssertionError("non-finite evaluation actions")
                self.probe_predictions += 1
                if self.probe_predictions % 48 == 0:
                    event("rollout", {"event": "finite_actions", "prediction_batches": self.probe_predictions})
                return actions, details

            def probe_report(self):
                return {"prediction_batches": self.probe_predictions, "version": self.version,
                        "memory": memory_snapshot(), "fallback": self.hf_model.s4000_fallback_metadata}

        cfg = make_config(args.rlinf_source, args.model_path)
        cfg.runner.task_type = "embodied_eval"
        cfg.runner.only_eval = True
        cfg.runner.data_channel_transport = "ray"
        cfg.runner.logger = {"log_path": str(args.output.parent / f"{args.output.stem}.logs"),
                             "project_name": "rlinf-musa-lab", "experiment_name": args.output.stem,
                             "logger_backends": []}
        cfg.rollout.model = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
        cfg.rollout.model.num_action_chunks = 5
        cfg.cluster.component_placement = {"rollout": "1", "env": "0"}
        cfg.env.eval.total_num_envs = 2
        cfg.env.eval.specific_reset_id = None
        cfg.env.eval.is_eval = True
        cfg.env.eval.auto_reset = True
        cfg.env.eval.ignore_terminations = True
        cfg.env.eval.rollout_epoch = 1
        cfg.env.eval.max_episode_steps = 240
        cfg.env.eval.max_steps_per_rollout_epoch = 240 * 5 * args.trials_per_task
        progress("validate_official_eval_config")
        cfg = validate_cfg(cfg)
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        result["config_sha256"] = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
        progress("initial_weight_provenance")
        index = args.model_path / "model.safetensors.index.json"
        if file_sha(index) != "bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746":
            raise AssertionError("Spatial weight index differs from the pinned revision")
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        weight_files = [index, args.model_path / "config.json", *(args.model_path / name for name in shards)]
        result["weight_files"] = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha(path)} for path in weight_files}
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
        result["source_fingerprints"] = {str(path.relative_to(args.rlinf_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (args.rlinf_source / "rlinf").rglob("*.py")}
        result["support_fingerprints"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("official_runner_probe.py", "official_actor_config.py", "actor_runtime_extension.py")}
        cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
        if cluster.num_accelerators != 2:
            raise RuntimeError("expected the two-S4000 node")
        placement = HybridComponentPlacement(cfg, cluster)
        rollout = AuditRollout.create_group(cfg).launch(cluster=cluster, name=cfg.rollout.group_name,
                                                       placement_strategy=placement.get_strategy("rollout"))
        env = AuditEnv.create_group(cfg).launch(cluster=cluster, name=cfg.env.group_name,
                                               placement_strategy=placement.get_strategy("env"))
        rollout.seed_probe(cfg.actor.seed).wait()
        runner = EmbodiedEvalRunner(cfg, rollout, env)
        result["ray_data_channels"] = all(channel._use_ray_transport for channel in (runner.env_channel, runner.rollout_channel))
        if not result["ray_data_channels"]:
            raise AssertionError("explicit evaluation Ray channels were not applied")
        progress("official_eval_init_workers")
        runner.init_workers()
        progress("official_eval_complete_episodes")
        result["metrics"] = as_json(runner.evaluate())
        result["environment"] = env.coverage_report().wait()
        rollout_reports = rollout.probe_report().wait()
        if len(rollout_reports) != 1:
            raise AssertionError("expected a single evaluation Rollout worker")
        result["rollout"] = rollout_reports[0]
        pairs = [tuple(pair) for worker in result["environment"] for stage in worker["stages"] for pair in stage["seen_trials"]]
        expected = {(task, trial) for trial in range(args.trials_per_task) for task in range(10)}
        counts = [stage["init_state_counts"] for worker in result["environment"] for stage in worker["stages"]]
        task_stats = [stage["task_stats"] for worker in result["environment"] for stage in worker["stages"]]
        checks = {"requested_unique_task_trials": set(pairs) == expected and len(pairs) == len(expected),
                  "ten_task_suite": all(len(bins) == 10 and min(bins) >= args.trials_per_task for bins in counts),
                  "official_trajectory_count": int(result["metrics"].get("num_trajectories", -1)) == 10 * args.trials_per_task,
                  "per_task_denominators": all(set(stats) == set(range(10))
                      and all(row["total"] == args.trials_per_task for row in stats.values()) for stats in task_stats),
                  "finite_metrics": all(math.isfinite(float(value)) for value in result["metrics"].values()),
                  "expected_policy_decisions": result["rollout"]["prediction_batches"] == 240 * args.trials_per_task}
        result["checks"] = checks
        result["full_suite_coverage"] = bool(counts) and all(all(n == args.trials_per_task for n in bins) for bins in counts)
        if not all(checks.values()):
            raise AssertionError(f"evaluation coverage audit failed: {checks}")
        result["status"] = "pass"
    except Exception as error:
        result.update(error_class=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if "ray" in locals():
            ray.shutdown()
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(as_json(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "stage": result.get("stage")}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
