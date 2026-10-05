#!/usr/bin/env bash
# Portable fixed-budget learning recipe; historical runs used server-local scripts.
set -euo pipefail
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
probe_python="${ROUTE2_PYTHON:?Set ROUTE2_PYTHON to a matched Torch-MUSA Python}"
source_dir="${ROUTE2_SOURCE:-$package_root/worktrees/rlinf-route2}"
result_dir="${ROUTE2_RESULTS:-$package_root/results/learning}"
export PYTHONPATH="$source_dir:$package_root/probes:$package_root/routes/route2/learning${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_MUSA_TORCH_DETECTION_FALLBACK=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
case "${1:-}" in
  pendulum) task_name=pendulum; extra_args=(--env-id Pendulum-v1) ;;
  halfcheetah) task_name=halfcheetah; extra_args=(--env-id HalfCheetah-v5 --normalize-observations --batch-evaluation) ;;
  *) echo 'Usage: bash scripts/run_learning.sh {pendulum|halfcheetah}' >&2; exit 2 ;;
esac
mkdir -p "$result_dir"
source_commit="$("$probe_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rlinf"]["route2"]["upstream_commit"])' "$package_root/locks/sources.json")"
for seed in 7 17 27; do
  output="$result_dir/$task_name-seed$seed.json"
  if [[ -e "$output" ]]; then
    echo "Refuse to overwrite an existing run: $output; choose a new ROUTE2_RESULTS directory" >&2
    exit 2
  fi
  "$probe_python" "$package_root/routes/route2/learning/ppo_learning.py" \
    --source-commit "$source_commit" "${extra_args[@]}" --seed "$seed" --resume-check \
    --output "$output" > "$result_dir/$task_name-seed$seed.log" 2>&1
done
"$probe_python" "$package_root/routes/route2/learning/summarize_learning.py" \
  "$result_dir/$task_name-seed7.json" "$result_dir/$task_name-seed17.json" \
  "$result_dir/$task_name-seed27.json" --output "$result_dir/$task_name-summary.json"
