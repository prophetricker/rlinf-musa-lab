#!/usr/bin/env bash
# Eight-GPU topology and full-horizon validation without checkpoint writes.
set -euo pipefail

research_dir=${ROUTE2_RESEARCH_DIR:-/root/autodl-tmp/s4000-research}
label=${ROUTE2_EIGHT_VALIDATION_LABEL:-v1}
steps=${ROUTE2_EIGHT_STEPS:-240}
iterations=${ROUTE2_EIGHT_ITERATIONS:-2}
case "$label" in ''|*[!a-zA-Z0-9._-]*) exit 2 ;; esac
case "$steps" in ''|*[!0-9]*) exit 2 ;; esac
case "$iterations" in ''|*[!0-9]*) exit 2 ;; esac
(( steps > 0 && steps % 5 == 0 && iterations > 0 )) || exit 2
global_batch=$((7 * steps / 5))
cd "$research_dir"

export PYTHONPATH="$research_dir/envs/route2-integration/lib/python3.10/site-packages:$research_dir/envs/route2-integration"
export MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=envs/route2/bin/python
output="route2/results/official-eight-gpu-validation-$label.json"
log="${output%.json}.log"
[[ ! -e "$output" && ! -e "${output%.json}.partial.json" && ! -e "$log" ]]

"$python_bin" route2/model_probes/official_training_probe.py \
  --rlinf-source route2/RLinf-official-actor \
  --gr00t-source route2/Isaac-GR00T-official-actor \
  --model-path route2/weights/Spatial-73f710e \
  --iterations "$iterations" --steps-per-env "$steps" --action-chunks 5 \
  --actor-placement 0-6 --rollout-placement 7 --env-placement 0 \
  --expected-accelerators 8 --total-envs 7 --global-batch-size "$global_batch" \
  --specific-reset-id none --ordered-training-resets \
  --min-distinct-task-ids 2 \
  --output "$output" > "$log" 2>&1

"$python_bin" scripts/audit_route2_eight_gpu_result.py \
  "$output" --events-dir "$(dirname "$output")" \
  --output "${output%.json}.audit.json"

"$python_bin" - "$output" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
print({"status": result["status"],
       "global_step": result["runner_global_step"],
       "actor_world_size": len(result["actor_reports"]),
       "received_chunks_per_iteration": result["global_received_samples_per_iteration"],
       "simulator_action_slots": result["total_simulator_step_slots"],
       "distinct_task_ids": result["environment"][0]["distinct_task_ids"]})
PY
