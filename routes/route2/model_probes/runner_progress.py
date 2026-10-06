#!/usr/bin/env python3
"""Read compact Runner evidence without loading Torch or starting a Ray job."""
import argparse
import json
from pathlib import Path
import time


def read_progress(output: Path):
    final = output.exists()
    partial = output.with_suffix(".partial.json")
    document = json.loads((output if final else partial).read_text()) if final or partial.exists() else {}
    # Probes start with status=fail and only write pass at final completion.
    status = document.get("status", "unknown") if final else "running"
    report = {"status": status, "final_exists": final,
              "stage": document.get("stage"), "error": document.get("error"),
              "elapsed_seconds": document.get("elapsed_seconds"),
              "synchronized_versions": [row["runner_global_step"]
                                        for row in document.get("synchronizations", [])],
              "total_received_transition_count": document.get("total_received_transition_count"),
              "total_received_action_chunk_count": document.get("total_received_action_chunk_count"),
              "total_simulator_step_slots": document.get("total_simulator_step_slots"),
              "actors": []}
    for rank in (0, 1):
        events = output.with_name(f"{output.stem}.actor.rank{rank}.events.jsonl")
        rows = []
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A concurrent append may expose its last partial line.
                    continue
        if not rows:
            continue
        last = rows[-1]
        receives = [row for row in rows if row["event"] == "official_trajectory_received"]
        updates = [row for row in rows if row["event"] == "official_training_complete"]
        report["actors"].append({
            "rank": rank, "last_event": last["event"],
            "last_event_age_seconds": round(time.time() - last["unix_time"], 1),
            "received_policy_decisions": sum(row["received_sample_count"] for row in receives),
            "completed_updates": len(updates),
            "receives": [{key: row.get(key) for key in
                          ("iteration", "stored_versions", "received_sample_count", "rewards",
                           "boundary_counts", "loss_mask", "receive_checks")} for row in receives],
            "updates": [{"checks": row["checks"],
                         "grad_norm": row["metrics"].get("actor/grad_norm"),
                         "active_adam_steps": row["after"]["active_adam_steps"]} for row in updates]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_progress(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
