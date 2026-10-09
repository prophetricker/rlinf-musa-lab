#!/usr/bin/env python3
"""Independently audit the eight-GPU no-checkpoint Runner evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be new")
    result = json.loads(args.result.read_text())
    config = result["config"]
    actor_world_size = 7
    actor_events = []
    for rank in range(actor_world_size):
        path = args.events_dir / f"{args.result.stem}.actor.rank{rank}.events.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        training_rows = [row for row in rows if row["event"] == "official_training_complete"]
        actor_events.append({"rank": rank, "path": path.name, "sha256": sha(path),
                             "event_count": len(rows),
                             "received": sum(row["event"] == "official_trajectory_received" for row in rows),
                             "gae": sum(row["event"] == "official_gae_complete" for row in rows),
                             "training": len(training_rows),
                             "all_training_checks_pass": all(
                                 value for row in training_rows for value in row["checks"].values()),
                             "grad_norms": [row["metrics"]["actor/grad_norm"] for row in training_rows]})
    iterations = result["iterations"]
    checks = {
        "raw_result_pass": result["status"] == "pass",
        "no_checkpoint_requested": result["save_final_requested"] is False and result["save_interval"] == -1,
        "eight_gpu_layout": config["cluster"]["component_placement"] == {
            "actor": "0-6", "rollout": "7", "env": "0"},
        "seven_actor_ranks": len(actor_events) == actor_world_size,
        "seven_training_envs": config["env"]["train"]["total_num_envs"] == actor_world_size,
        "global_batch_preserved": config["actor"]["global_batch_size"] == 336,
        "two_iterations": iterations == 2,
        "chunk_sample_counts": result["global_received_samples_per_iteration"] == [336, 336],
        "separate_simulator_count": result["total_received_action_chunk_count"] == 672
            and result["total_simulator_step_slots"] == 3360,
        "all_final_checks_pass": all(result["final_checks"].values()),
        "all_iteration_checks_pass": all(all(row[key] for key in row if key != "iteration")
                                          for row in result["per_iteration_checks"]),
        "all_actor_event_counts": all(row["received"] == 2 and row["gae"] == 2
                                       and row["training"] == 2 for row in actor_events),
        "all_actor_training_checks_pass": all(row["all_training_checks_pass"] for row in actor_events),
        "reduced_grad_norms_equal_per_iteration": all(
            len({row["grad_norms"][iteration] for row in actor_events}) == 1
            for iteration in range(2)),
        "all_recorded_metrics_finite": all(
            all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in row["grad_norms"])
            for row in actor_events),
    }
    audit = {"schema_version": 1, "status": "pass" if all(checks.values()) else "fail",
             "scope": "independent audit of eight-GPU 7-Actor/1-Rollout full-horizon Runner evidence",
             "result_sha256": sha(args.result), "actor_events": actor_events, "checks": checks,
             "raw_result_status": result["status"], "total_chunk_samples": 672,
             "total_simulator_action_slots": 3360}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": audit["status"], "checks": len(checks)}))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
