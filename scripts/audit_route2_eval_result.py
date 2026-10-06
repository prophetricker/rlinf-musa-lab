#!/usr/bin/env python3
"""Cross-check inherited Spatial evaluation against per-episode event evidence.

This reads JSON only. It starts no Cluster, Ray, simulator or accelerator work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--eval-lock", type=Path, required=True)
    parser.add_argument("--weight-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be new")
    data = json.loads(args.result.read_text())
    source_lock = json.loads(args.source_lock.read_text())
    eval_lock = json.loads(args.eval_lock.read_text())
    weight_metadata = json.loads(args.weight_metadata.read_text())
    event_paths = {name: args.result.with_name(f"{args.result.stem}.{name}.events.jsonl")
                   for name in ("env", "rollout")}
    episodes = rows(event_paths["env"])
    predictions = rows(event_paths["rollout"])
    trials = data["trials_per_task_requested"]
    expected = {(task, trial) for task in range(10) for trial in range(trials)}
    pairs = [(row["task_id"], row["trial_id"]) for row in episodes]
    stages = [stage for worker in data["environment"] for stage in worker["stages"]]
    if len(stages) != 1:
        raise ValueError("this audit expects the pinned single-worker, single-stage layout")
    stage = stages[0]
    bins = stage["init_state_counts"]
    manifest = {row["task_id"]: row for row in stage["task_manifest"]}
    stats = {int(task): row for task, row in stage["task_stats"].items()}
    completed = Counter(row["task_id"] for row in episodes)
    successes = Counter(row["task_id"] for row in episodes if row["metrics"]["success_once"])
    metrics = data["metrics"]
    cfg = data["config"]
    env_cfg = cfg["env"]["eval"]
    source = data["source_fingerprints"]
    checks = {
        "raw_status_pass": data["status"] == "pass",
        "raw_checks_pass": all(data["checks"].values()),
        "probe_bytes_match": data["probe_sha256"] == sha(args.probe),
        "locked_production_files_match": all(source.get(row["path"]) == row["sha256"]
                                             for row in source_lock["production_files"]),
        "eval_runner_source_matches": source.get(eval_lock["production_file"]) == eval_lock["production_sha256"],
        "original_spatial_shards_match": all(
            data["weight_files"].get(row["rfilename"], {}).get("sha256") == row["lfs"]["sha256"]
            and data["weight_files"].get(row["rfilename"], {}).get("bytes") == row["size"]
            for row in weight_metadata["weights"]),
        "initial_policy_only_eval": cfg["runner"]["only_eval"]
            and set(cfg["cluster"]["component_placement"]) == {"env", "rollout"},
        "pinned_eval_configuration": env_cfg["auto_reset"] and env_cfg["ignore_terminations"]
            and env_cfg["is_eval"] and env_cfg["total_num_envs"] == 2
            and env_cfg["max_episode_steps"] == 240
            and env_cfg["max_steps_per_rollout_epoch"] == 240 * 5 * trials
            and env_cfg["rollout_epoch"] == 1 and cfg["rollout"]["model"]["num_action_chunks"] == 5,
        "explicit_ray_channels": data["ray_data_channels"] is True,
        "unique_episode_events_cover_request": set(pairs) == expected and len(pairs) == len(expected),
        "events_match_worker_seen_trials": set(pairs) == {tuple(pair) for pair in stage["seen_trials"]},
        "actual_init_state_manifest": set(manifest) == set(range(10)) and len(bins) == 10
            and all(manifest[task]["init_state_count"] == bins[task] >= trials
                    and len(manifest[task]["init_states_sha256"]) == 64 for task in range(10)),
        "event_reset_ids_and_languages": all(
            row["global_reset_id"] == sum(bins[:row["task_id"]]) + row["trial_id"]
            and row["task_language"] == manifest[row["task_id"]]["language"] for row in episodes),
        "complete_episode_lengths": all(row["metrics"]["episode_len"] == 240 for row in episodes),
        "finite_episode_metrics": all(math.isfinite(float(value)) for row in episodes
                                      for value in row["metrics"].values()),
        "per_task_stats_match_episode_events": set(stats) == set(range(10)) and all(
            stats[task] == {"success": successes[task], "total": completed[task]}
            and completed[task] == trials for task in range(10)),
        "official_count_matches_events": metrics["num_trajectories"] == len(episodes),
        "official_success_matches_events": math.isclose(float(metrics["success_once"]),
            sum(successes.values()) / len(expected), rel_tol=0, abs_tol=1e-7),
        "prediction_progress_complete": [row["prediction_batches"] for row in predictions]
            == list(range(48, 240 * trials + 1, 48))
            and all(row["event"] == "finite_actions" for row in predictions)
            and data["rollout"]["prediction_batches"] == 240 * trials,
        "full_suite_scope_matches_manifest": data["full_suite_coverage"] == all(n == trials for n in bins),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        "raw_result_sha256": sha(args.result), "probe_sha256": sha(args.probe),
        "source_lock_sha256": sha(args.source_lock), "eval_lock_sha256": sha(args.eval_lock),
        "event_sha256": {name: sha(path) for name, path in event_paths.items()},
        "unique_episodes": len(set(pairs)), "successes": sum(successes.values()),
        "success_rate": sum(successes.values()) / len(expected),
        "per_task": [{"task_id": task, "language": manifest[task]["language"],
                      "success": successes[task], "total": completed[task]} for task in range(10)],
        "simulator_step_slots": len(expected) * 240,
        "policy_decision_batches": data["rollout"]["prediction_batches"],
        "full_suite_coverage": data["full_suite_coverage"],
        "policy_rng_scope": data["policy_rng_scope"],
        "elapsed_seconds": data["elapsed_seconds"],
        "scope": "independent episode/worker/aggregate/hash cross-check; initial policy performance, no PPO learning claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": summary["status"], "checks": len(checks),
                      "unique_episodes": summary["unique_episodes"], "successes": summary["successes"]}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
