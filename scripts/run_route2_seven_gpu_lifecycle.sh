#!/usr/bin/env bash
# Seven-GPU cross-task save, fresh-process restore, and full-horizon continuation.
set -euo pipefail

research_dir=${ROUTE2_RESEARCH_DIR:-/root/autodl-tmp/s4000-research}
label=${ROUTE2_SEVEN_LIFECYCLE_LABEL:-v1}
steps=${ROUTE2_SEVEN_STEPS:-240}
case "$label" in ''|*[!a-zA-Z0-9._-]*) exit 2 ;; esac
case "$steps" in ''|*[!0-9]*) exit 2 ;; esac
(( steps > 0 && steps % 5 == 0 )) || exit 2
global_batch=$((6 * steps / 5))
cd "$research_dir"

export PYTHONPATH="$research_dir/envs/route2-integration/lib/python3.10/site-packages:$research_dir/envs/route2-integration"
export MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=envs/route2/bin/python
save_output="route2/results/official-seven-gpu-save-$label.json"
resume_output="route2/results/official-seven-gpu-resume-$label.json"
for output in "$save_output" "$resume_output"; do
  [[ ! -e "$output" && ! -e "${output%.json}.partial.json" && ! -e "${output%.json}.log" ]]
done

common=(--rlinf-source route2/RLinf-official-actor
        --gr00t-source route2/Isaac-GR00T-official-actor
        --model-path route2/weights/Spatial-73f710e
        --iterations 1 --steps-per-env "$steps" --action-chunks 5
        --actor-placement 0-5 --rollout-placement 6 --env-placement 0
        --expected-accelerators 7 --total-envs 6 --global-batch-size "$global_batch"
        --specific-reset-id none --ordered-training-resets
        --min-distinct-task-ids 2)

"$python_bin" route2/model_probes/official_training_probe.py "${common[@]}" \
  --save-final --output "$save_output" > "${save_output%.json}.log" 2>&1
resume_dir=$("$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass"; print(d["checkpoint_dir"])' "$save_output")
"$python_bin" route2/model_probes/official_training_probe.py "${common[@]}" \
  --resume-dir "$resume_dir" --resume-reference "$save_output" \
  --output "$resume_output" > "${resume_output%.json}.log" 2>&1
"$python_bin" - "$save_output" "$resume_output" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    result = json.load(open(path))
    print({"file": path, "status": result["status"],
           "global_step": result["runner_global_step"],
           "distinct_task_ids": result["environment"][0]["distinct_task_ids"],
           "sync_versions": [row["runner_global_step"] for row in result["synchronizations"]],
           "restore_checks": len(result.get("restore_checks", {})),
           "checkpoint_bytes": result.get("checkpoint_bytes")})
PY
