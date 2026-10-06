#!/usr/bin/env bash
# Run sequential GPU validations in the existing isolated S4000 research tree.
set -euo pipefail

phase=${1:-both}
case "$phase" in both|runner|eval) ;; *) exit 2 ;; esac
research_dir=${ROUTE2_RESEARCH_DIR:-/root/autodl-tmp/s4000-research}
label=${ROUTE2_VALIDATION_LABEL:-v1}
case "$label" in ''|*[!a-zA-Z0-9._-]*) exit 2 ;; esac
cd "$research_dir"

export PYTHONPATH="$research_dir/envs/route2-integration/lib/python3.10/site-packages:$research_dir/envs/route2-integration"
export MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=envs/route2/bin/python

if [[ "$phase" == both || "$phase" == runner ]]; then
  runner_output="route2/results/official-two-rank-chunk5-$label.json"
  [[ ! -e "$runner_output" && ! -e "${runner_output%.json}.partial.json" && ! -e "${runner_output%.json}.log" ]]
  "$python_bin" route2/model_probes/official_runner_probe.py \
    --rlinf-source route2/RLinf-official-actor \
    --gr00t-source route2/Isaac-GR00T-official-actor \
    --model-path route2/weights/Spatial-73f710e \
    --iterations 2 --steps-per-env 240 --action-chunks 5 \
    --output "$runner_output" > "${runner_output%.json}.log" 2>&1
  cat "$runner_output" | "$python_bin" -c 'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("status","total_received_action_chunk_count","total_simulator_step_slots","elapsed_seconds")})'
fi

if [[ "$phase" == both || "$phase" == eval ]]; then
  eval_output="route2/results/official-spatial-ten-task-pilot-$label.json"
  [[ ! -e "$eval_output" && ! -e "${eval_output%.json}.partial.json" && ! -e "${eval_output%.json}.log" ]]
  "$python_bin" route2/model_probes/official_eval_probe.py \
    --rlinf-source route2/RLinf-official-actor \
    --gr00t-source route2/Isaac-GR00T-official-actor \
    --model-path route2/weights/Spatial-73f710e \
    --trials-per-task 1 --output "$eval_output" > "${eval_output%.json}.log" 2>&1
  cat "$eval_output" | "$python_bin" -c 'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("status","metrics","checks","full_suite_coverage","elapsed_seconds")})'
fi
