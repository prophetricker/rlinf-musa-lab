#!/usr/bin/env python3
"""Cross-check final Runner evidence, exact source/probe hashes and sample units."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be new")
    data = json.loads(args.result.read_text())
    lock = json.loads(args.source_lock.read_text())
    cfg = data["config"]
    chunks = cfg["actor"]["model"]["num_action_chunks"]
    horizon = cfg["env"]["train"]["max_steps_per_rollout_epoch"]
    iterations = data["iterations"]
    decisions = horizon // chunks
    actors = data["actor_reports"]
    syncs = data["synchronizations"]
    checks = {"raw_status_pass": data["status"] == "pass",
              "probe_bytes_match": data["probe_sha256"] == hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "locked_production_files_match": all(data["source_fingerprints"].get(row["path"]) == row["sha256"]
                                                    for row in lock["production_files"]),
              "expected_two_ranks": [row["rank"] for row in actors] == [0, 1],
              "global_step_and_versions": data["runner_global_step"] == iterations
                  and data["rollout_final"]["version"] == iterations - 1
                  and all(row["final"]["version"] == iterations - 1 for row in actors),
              "all_raw_final_checks": all(data["final_checks"].values()),
              "all_raw_iteration_checks": all(all(value for key, value in row.items() if key != "iteration")
                                               for row in data["per_iteration_checks"]),
              "complete_iteration_versions": [row["runner_global_step"] for row in syncs] == list(range(iterations)),
              "all_full_state_synchronizations": all(all(row["checks"].values())
                  and row["actor_states"][0]["manifest"] == row["rollout_state"]["manifest"] for row in syncs),
              "all_rank_updates": all(row["final"]["optimizer_steps"] == iterations
                  and row["final"]["active_adam_steps"] == [float(iterations)] for row in actors),
              "correct_global_decision_counts": data["global_received_samples_per_iteration"] == [2 * decisions] * iterations}
    summaries = []
    for index in range(iterations):
        rows = [actor["iterations"][index] for actor in actors]
        checks[f"iteration_{index}_rank_data_and_updates"] = all(
            row["received_sample_count"] == decisions
            and row["stored_versions"] == [float(index)]
            and all(row["receive_checks"].values()) and row["gae"]["finite"]
            and all(row["training"]["checks"].values()) for row in rows)
        checks[f"iteration_{index}_distinct_data_and_reduced_norm"] = (
            len({row["policy_data_sha256"] for row in rows}) == 2
            and rows[0]["training"]["metrics"]["actor/grad_norm"] == rows[1]["training"]["metrics"]["actor/grad_norm"])
        valid_slots = [row.get("valid_simulator_action_slots", row["loss_mask"]["true_count"] if chunks == 1 else None)
                       for row in rows]
        summaries.append({"iteration": index, "received_policy_decisions": sum(row["received_sample_count"] for row in rows),
                          "simulator_step_slots": 2 * horizon, "valid_simulator_action_slots_per_rank": valid_slots,
                          "valid_chunk_samples_per_rank": [row["loss_mask"]["true_count"] for row in rows],
                          "reward_sum_per_rank": [row["rewards"]["sum"] for row in rows],
                          "boundary_flags_per_rank": [row["boundary_counts"] for row in rows],
                          "global_grad_norm": rows[0]["training"]["metrics"]["actor/grad_norm"]})
    total_decisions = 2 * decisions * iterations
    if chunks == 1:
        checks["single_action_transition_count"] = data["total_received_transition_count"] == total_decisions
    else:
        checks["separate_chunk_and_simulator_counts"] = data["total_received_action_chunk_count"] == total_decisions \
            and data["total_simulator_step_slots"] == total_decisions * chunks
    result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "raw_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "probe_sha256": data["probe_sha256"], "source_lock_sha256": hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
              "actions_per_policy_decision": chunks, "total_received_policy_decisions": total_decisions,
              "total_simulator_step_slots": 2 * horizon * iterations, "iterations": summaries,
              "elapsed_seconds": data["elapsed_seconds"], "final_weight_scope": data["final_weight_scope"],
              "scope": "independent cross-check of recorded execution evidence; no rerun or learning claim; boundary flags are not unique episode counts"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "total_received_policy_decisions": total_decisions}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
