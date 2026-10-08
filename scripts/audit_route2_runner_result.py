#!/usr/bin/env python3
"""Cross-check final Runner evidence, exact source/probe hashes and sample units."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def placement_ranks(value):
    ranks = []
    for part in value.split(","):
        bounds = [int(item) for item in part.split("-")]
        if len(bounds) == 1:
            ranks.extend(bounds)
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            ranks.extend(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError("invalid Actor placement")
    if not ranks or ranks != list(range(ranks[0], ranks[-1] + 1)) or ranks[0] < 0:
        raise ValueError("Actor placement must be a contiguous nonnegative range")
    return ranks


def sampling_checks(data, world_size, horizon, iterations, minimum=1):
    """Recompute coverage/identity checks from observations, not recorded booleans."""
    reports = data["training_sampling_reports"]
    starts = [row for row in reports if row["event"] == "training_bootstrap"]
    ends = [row for row in reports if row["event"] == "training_horizon_complete"]
    identities = lambda row: [(tuple(item["task_ids"]), tuple(item["trial_ids"]))
                              for item in row["identities"]]
    flattened = lambda row, key: [value for item in row["identities"] for value in item[key]]
    return {
        "sampling_events_alternate_and_cover_iterations": len(reports) == 2 * iterations
            and [row["event"] for row in reports] == ["training_bootstrap", "training_horizon_complete"] * iterations,
        "sampling_lane_counts": all(len(flattened(row, "task_ids")) == world_size
            and len(flattened(row, "trial_ids")) == world_size for row in reports),
        "sampling_task_trial_identity_stable": all(identities(a) == identities(b) for a, b in zip(starts, ends)),
        "sampling_description_counts": all(len(item["task_descriptions"]) == len(item["task_ids"])
            for row in starts for item in row["identities"]),
        "sampling_multiple_tasks": all(len(set(flattened(row, "task_ids"))) >= minimum for row in starts),
        "sampling_all_lanes_complete_horizon": all(
            [value for stage in row["elapsed_simulator_steps"] for value in stage] == [horizon] * world_size
            for row in ends),
        "sampling_reset_ids_advance": data["training_reset"]["specific_reset_id"] is not None or iterations == 1
            or all(identities(a) != identities(b) for a, b in zip(starts, starts[1:])),
        "sampling_raw_checks_pass": bool(data["training_sampling_checks"])
            and all(data["training_sampling_checks"].values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--resume-reference", type=Path)
    parser.add_argument("--reference-probe", type=Path,
                        help="frozen original probe for an interrupted-save reference")
    parser.add_argument("--resume-probe", type=Path,
                        help="frozen recovery probe when resuming from an interrupted save")
    parser.add_argument("--min-distinct-task-ids", type=int, default=1)
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
    start_step = data.get("start_global_step", 0)
    target_step = start_step + iterations
    explicit_final_sync = "target_global_step" in data
    final_version = target_step if explicit_final_sync else target_step - 1
    decisions = horizon // chunks
    actors = data["actor_reports"]
    syncs = data["synchronizations"]
    world_size = len(placement_ranks(cfg["cluster"]["component_placement"]["actor"]))
    total_envs = cfg["env"]["train"]["total_num_envs"]
    checks = {"raw_status_pass": data["status"] == "pass",
              "probe_bytes_match": data["probe_sha256"] == hashlib.sha256(
                  (args.resume_probe or args.probe).read_bytes()).hexdigest(),
              "locked_production_files_match": all(data["source_fingerprints"].get(row["path"]) == row["sha256"]
                                                    for row in lock["production_files"]),
              "expected_actor_ranks": [row["rank"] for row in actors] == list(range(world_size)),
              "one_environment_per_actor_rank": total_envs == world_size,
              "global_step_and_versions": data["runner_global_step"] == target_step
                  and data["rollout_final"]["version"] == final_version
                  and all(row["final"]["version"] == final_version for row in actors),
              "all_raw_final_checks": all(data["final_checks"].values()),
              "all_raw_iteration_checks": all(all(value for key, value in row.items() if key != "iteration")
                                               for row in data["per_iteration_checks"]),
              "complete_iteration_versions": [row["runner_global_step"] for row in syncs]
                  == list(range(start_step, target_step + int(explicit_final_sync))),
              "all_full_state_synchronizations": all(all(row["checks"].values())
                  and row["actor_states"][0]["manifest"] == row["rollout_state"]["manifest"] for row in syncs),
              "all_rank_updates": all(row["final"]["optimizer_steps"] == target_step
                  and row["final"]["active_adam_steps"] == [float(target_step)] for row in actors),
              "correct_global_decision_counts": data["global_received_samples_per_iteration"] == [total_envs * decisions] * iterations}
    if explicit_final_sync:
        checks["final_policy_matches_last_actor_sync"] = (
            syncs[-1]["actor_states"][0]["manifest"] == data["rollout_final"]["manifest"])
        checks["checkpoint_device_adapter_enabled"] = (
            data["transport_environment"].get("RLINF_MUSA_FSDP_OPTIM_DEVICE_HANDLE") == "1")
        checks["checkpoint_state_matches_update_and_version"] = all(
            row["optimizer_steps"] == target_step and row["nonempty_adam_steps"] == [float(target_step)]
            and row["version"] == target_step for row in data["checkpoint_state"])
        if data["save_final_requested"] or data.get("saved_checkpoints"):
            checks["save_preserved_training_state"] = data["checkpoint_state"] == data["post_save_actor_state"]
            checks["recorded_checkpoint_files"] = data["checkpoint_bytes"] == sum(
                row["bytes"] for row in data["checkpoint_files"]) > 0
        if data.get("saved_checkpoints"):
            saves = data["saved_checkpoints"]
            interval = data["save_interval"]
            expected_saves = [step for step in range(start_step + 1, target_step + 1)
                              if interval > 0 and (step % interval == 0 or step == target_step)]
            if data["save_final_requested"]:
                expected_saves.append(target_step)
            checks["inherited_save_schedule"] = [row["global_step"] for row in saves] == expected_saves
            checks["every_save_preserves_actor_state"] = all(
                row["checkpoint_state"] == row["post_save_actor_state"] for row in saves)
            checks["save_metadata_matches_final_state"] = saves[-1]["checkpoint_state"] == data["checkpoint_state"]
            checks["checkpoint_hash_manifest"] = all(
                bool(row["checkpoint_files"])
                and len({item["path"] for item in row["checkpoint_files"]}) == len(row["checkpoint_files"])
                and row["checkpoint_bytes"] == sum(item["bytes"] for item in row["checkpoint_files"])
                and all(item["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                        for item in row["checkpoint_files"]) for row in saves)
        if start_step:
            if args.resume_reference is None:
                parser.error("resumed Runner audit requires --resume-reference")
            reference = json.loads(args.resume_reference.read_text())
            interrupted = data.get("interrupted_save_reference")
            if interrupted:
                if args.reference_probe is None:
                    parser.error("interrupted save requires --reference-probe")
                original = reference
                saved = original["saved_checkpoints"][-1]
                checks["interrupted_reference_remains_incomplete"] = (
                    original["status"] == "fail"
                    and original["stage"] == f"runner_step_{start_step}/checkpoint_save_audited"
                    and saved["global_step"] == start_step
                    and saved["checkpoint_state"] == saved["post_save_actor_state"])
                checks["interrupted_original_probe_verified"] = (
                    original["probe_sha256"] == interrupted["original_probe_sha256"]
                    == hashlib.sha256(args.probe.read_bytes()).hexdigest())
                reference = {**original, "checkpoint_state": saved["checkpoint_state"]}
            checks["restore_reference_hash"] = data["resume_reference_sha256"] == hashlib.sha256(
                args.resume_reference.read_bytes()).hexdigest()
            checks["restored_rank_training_states_exact"] = (
                data["initial_actor_state"] == reference["checkpoint_state"]
                and len(data["restore_checks"]) == 11 * world_size and all(data["restore_checks"].values()))
            checks["resume_helpers_sources_equal"] = all(data[key] == reference[key] for key in (
                "source_fingerprints", "gr00t_source_fingerprints", "support_fingerprints"))
            if not interrupted:
                checks["resume_probe_equal"] = data["probe_sha256"] == reference["probe_sha256"]
            if "saved_checkpoints" in data:
                checks["checkpoint_bytes_verified_before_restore"] = data.get("checkpoint_file_hashes_verified_before_load") is True
    if "training_sampling_reports" in data:
        checks.update(sampling_checks(data, world_size, horizon, iterations, args.min_distinct_task_ids))
    summaries = []
    for index in range(iterations):
        rows = [actor["iterations"][index] for actor in actors]
        checks[f"iteration_{index}_rank_data_and_updates"] = all(
            row["received_sample_count"] == decisions
            and row["stored_versions"] == [float(start_step + index)]
            and all(row["receive_checks"].values()) and row["gae"]["finite"]
            and all(row["training"]["checks"].values()) for row in rows)
        checks[f"iteration_{index}_distinct_data_and_reduced_norm"] = (
            len({row["policy_data_sha256"] for row in rows}) == world_size
            and len({row["training"]["metrics"]["actor/grad_norm"] for row in rows}) == 1)
        valid_slots = [row.get("valid_simulator_action_slots", row["loss_mask"]["true_count"] if chunks == 1 else None)
                       for row in rows]
        summaries.append({"iteration": index, "received_policy_decisions": sum(row["received_sample_count"] for row in rows),
                          "simulator_step_slots": total_envs * horizon, "valid_simulator_action_slots_per_rank": valid_slots,
                          "valid_chunk_samples_per_rank": [row["loss_mask"]["true_count"] for row in rows],
                          "reward_sum_per_rank": [row["rewards"]["sum"] for row in rows],
                          "boundary_flags_per_rank": [row["boundary_counts"] for row in rows],
                          "global_grad_norm": rows[0]["training"]["metrics"]["actor/grad_norm"]})
    total_decisions = total_envs * decisions * iterations
    if chunks == 1:
        checks["single_action_transition_count"] = data["total_received_transition_count"] == total_decisions
    else:
        checks["separate_chunk_and_simulator_counts"] = data["total_received_action_chunk_count"] == total_decisions \
            and data["total_simulator_step_slots"] == total_decisions * chunks
    result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "raw_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "probe_sha256": data["probe_sha256"], "source_lock_sha256": hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
              "actions_per_policy_decision": chunks, "total_received_policy_decisions": total_decisions,
              "total_simulator_step_slots": total_envs * horizon * iterations, "iterations": summaries,
              "actor_world_size": world_size,
              "elapsed_seconds": data["elapsed_seconds"], "final_weight_scope": data["final_weight_scope"],
              "scope": "independent cross-check of recorded execution evidence; no rerun or learning claim; boundary flags are not unique episode counts"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "total_received_policy_decisions": total_decisions}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
