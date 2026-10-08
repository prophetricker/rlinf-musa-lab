#!/usr/bin/env python3
"""Audit completed training/save events without claiming the source run passed."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from audit_route2_runner_result import placement_ranks, sampling_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--min-distinct-task-ids", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be new")
    data = json.loads(args.result.read_text())
    lock = json.loads(args.source_lock.read_text())
    world = len(placement_ranks(data["actor_placement"]))
    iterations = data["iterations"]
    saved = data["saved_checkpoints"][-1]
    step = saved["global_step"]
    checks = {
        "source_run_remains_failed": data["status"] == "fail",
        "completed_save_stage": data["stage"] == f"runner_step_{step}/checkpoint_save_audited",
        "probe_matches": data["probe_sha256"] == hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "locked_sources_match": all(data["source_fingerprints"].get(row["path"]) == row["sha256"]
                                    for row in lock["production_files"]),
        "save_preserves_actor_state": saved["checkpoint_state"] == saved["post_save_actor_state"],
        "saved_rank_versions_and_steps": [row["rank"] for row in saved["checkpoint_state"]] == list(range(world))
            and all(row["version"] == step and row["optimizer_steps"] == step
                    and row["nonempty_adam_steps"] == [float(step)] for row in saved["checkpoint_state"]),
        "checkpoint_hash_manifest": len(saved["checkpoint_files"]) > world
            and saved["checkpoint_bytes"] == sum(row["bytes"] for row in saved["checkpoint_files"])
            and all(row["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    for row in saved["checkpoint_files"]),
    }
    prefix = args.result.name.removesuffix(".partial.json")
    load_events = lambda name: [json.loads(line) for line in (args.result.parent / name).read_text().splitlines()]
    env_events = load_events(f"{prefix}.env.rank0.events.jsonl")
    checks.update(sampling_checks({**data, "training_sampling_reports": env_events,
                                  "training_sampling_checks": {"observations_present": bool(env_events)}},
                                 world, data["steps_per_environment"], iterations, args.min_distinct_task_ids))
    ranks = []
    for rank in range(world):
        events = load_events(f"{prefix}.actor.rank{rank}.events.jsonl")
        by_event = lambda event: [row for row in events if row["event"] == event]
        receives = by_event("official_trajectory_received")
        gae = by_event("official_gae_complete")
        updates = by_event("official_training_complete")
        checks[f"rank_{rank}_events_complete"] = len(receives) == len(gae) == len(updates) == iterations
        checks[f"rank_{rank}_receive_gae_update_pass"] = (
            all(row["rank"] == rank and row["received_sample_count"] == data["policy_decisions_per_environment"]
                and row["stored_versions"] == [float(index)] and all(row["receive_checks"].values())
                for index, row in enumerate(receives))
            and all(row["finite"] for row in gae)
            and all(all(row["checks"].values()) and row["after"]["optimizer_steps"] == index + 1
                    for index, row in enumerate(updates)))
        ranks.append({"receives": receives, "updates": updates})
    for index in range(iterations):
        checks[f"iteration_{index}_distinct_data_reduced_gradient"] = (
            len({row["receives"][index]["policy_data_sha256"] for row in ranks}) == world
            and len({row["updates"][index]["metrics"]["actor/grad_norm"] for row in ranks}) == 1)
    result = {"status": "pass" if all(checks.values()) else "fail", "source_run_status": data["status"],
              "checks": checks, "raw_partial_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "actor_world_size": world, "checkpoint_bytes": saved["checkpoint_bytes"],
              "task_batches": [row["identities"] for row in env_events if row["event"] == "training_bootstrap"],
              "total_policy_chunks": world * data["policy_decisions_per_environment"] * iterations,
              "total_simulator_action_slots": world * data["steps_per_environment"] * iterations,
              "grad_norms": [row["metrics"]["actor/grad_norm"] for row in ranks[0]["updates"]],
              "scope": "completed training and state-preserving save events only; source Runner did not finish; final sync unverified; no learning claim"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "source_run_status": data["status"]}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
