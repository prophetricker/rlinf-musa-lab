#!/usr/bin/env bash
# Recover only a completed audited save; preserve the source run's failed status.
set -euo pipefail
research_dir=${ROUTE2_RESEARCH_DIR:-/root/autodl-tmp/s4000-research}
cd "$research_dir"
export PYTHONPATH="$research_dir/envs/route2-integration/lib/python3.10/site-packages:$research_dir/envs/route2-integration"
export MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
reference=route2/results/official-seven-gpu-save-v2.partial.json
output=route2/results/official-seven-gpu-resume-v3.json
[[ ! -e "$output" && ! -e "${output%.json}.partial.json" && ! -e "${output%.json}.log" ]]
resume_dir=$(envs/route2/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["stage"]=="runner_step_2/checkpoint_save_audited"; print(d["saved_checkpoints"][-1]["checkpoint_dir"])' "$reference")
envs/route2/bin/python route2/model_probes/official_training_probe.py \
  --rlinf-source route2/RLinf-official-actor --gr00t-source route2/Isaac-GR00T-official-actor \
  --model-path route2/weights/Spatial-73f710e --steps-per-env 240 --action-chunks 5 \
  --actor-placement 0-5 --rollout-placement 6 --env-placement 0 \
  --expected-accelerators 7 --total-envs 6 --global-batch-size 288 \
  --specific-reset-id none --ordered-training-resets --min-distinct-task-ids 2 \
  --iterations 1 --resume-dir "$resume_dir" --resume-reference "$reference" \
  --interrupted-save-probe route2/model_probes/official-training-seven-lifecycle-v2-probe.py \
  --output "$output" > "${output%.json}.log" 2>&1
