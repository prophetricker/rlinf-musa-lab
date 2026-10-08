#!/usr/bin/env python3
"""Independently audit the recorded seven-GPU Runner evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    actor_world_size = int(result["config"]["cluster"]["component_placement"]["actor"].split("-")[-1]) + 1
    actor_events = []
    for rank in range(actor_world_size):
        path = args.events_dir / f"full240.actor.rank{rank}.events.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        actor_events.append({"rank": rank, "path": path.name, "sha256": sha(path),
                             "event_count": len(rows),
                             "received": sum(row["event"] == "official_trajectory_received" for row in rows),
                             "gae": sum(row["event"] == "official_gae_complete" for row in rows),
                             "training": sum(row["event"] == "official_training_complete" for row in rows),
                             "all_training_checks_pass": all(
                                 value for row in rows if row["event"] == "official_training_complete"
                                 for value in row["checks"].values()),
                             "grad_norms": [row["metrics"]["actor/grad_norm"] for row in rows
                                            if row["event"] == "official_training_complete"]})
    iterations = result["iterations"]
    checks = {
        "raw_result_pass": result["status"] == "pass",
        "six_actor_ranks": actor_world_size == 6 and len(actor_events) == 6,
        "seven_gpu_layout": result["config"]["cluster"]["component_placement"] == {
            "actor": "0-5", "rollout": "6", "env": "0"},
        "six_training_envs": result["config"]["env"]["train"]["total_num_envs"] == 6,
        "global_batch_preserved": result["config"]["actor"]["global_batch_size"] == 288,
        "two_iterations": result["iterations"] == 2,
        "chunk_sample_counts": result["global_received_samples_per_iteration"] == [288, 288],
        "separate_simulator_count": result["total_received_action_chunk_count"] == 576
            and result["total_simulator_step_slots"] == 2880,
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
            all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
                for value in row["grad_norms"])
            for row in actor_events),
    }
    audit = {"schema_version": 1, "status": "pass" if all(checks.values()) else "fail",
             "scope": "independent audit of seven-GPU 6-Actor/1-Rollout full-horizon Runner evidence",
             "result_sha256": sha(args.result), "actor_events": actor_events, "checks": checks,
             "raw_result_status": result["status"], "total_chunk_samples": 576,
             "total_simulator_action_slots": 2880}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": audit["status"], "checks": len(checks)}))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
