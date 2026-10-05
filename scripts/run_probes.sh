#!/usr/bin/env bash
# Portable entry point; historical GPU results used the original run_route2_* scripts.
set -euo pipefail
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
probe_python="${ROUTE2_PYTHON:?Set ROUTE2_PYTHON to a matched Torch-MUSA Python}"
source_dir="${ROUTE2_SOURCE:-$package_root/worktrees/rlinf-route2}"
result_dir="${ROUTE2_RESULTS:-$package_root/results}"
mkdir -p "$result_dir"
source_dir="$(cd "$source_dir" && pwd)"
result_dir="$(cd "$result_dir" && pwd)"
export PYTHONPATH="$source_dir:$source_dir/tests/unit_tests:$package_root/probes${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_MUSA_TORCH_DETECTION_FALLBACK=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
probe="$package_root/routes/route2/probes/current_rlinf_probe.py"
case "${1:-}" in
  imports)
    "$probe_python" "$probe" --source "$source_dir" --phase imports | tee "$result_dir/imports.jsonl"
    "$probe_python" "$probe" --source "$source_dir" --phase local --device cpu | tee "$result_dir/local-cpu.jsonl"
    ;;
  cpu)
    "$probe_python" "$package_root/routes/route2/probes/cpu_compat_checks.py" --source "$source_dir" | tee "$result_dir/cpu-regressions.jsonl"
    cd "$source_dir"
    "$probe_python" -m pytest tests/unit_tests/test_worker.py -q -k TestMUSADetectionFallback | tee "$result_dir/detection-tests.log"
    ;;
  channel)
    cd "$source_dir"
    timeout 240 "$probe_python" -m pytest --capture=no -vv -x --tb=short tests/unit_tests/test_channel.py -k 'TestRayChannelTransport or test_ray_get_preserves_dispatcher_consumer_routing' 2>&1 | tee "$result_dir/channel-tests.log"
    ;;
  worker|fsdp)
    phase=worker
    if [[ "$1" == fsdp ]]; then phase=native-fsdp; fi
    timeout 180 "$probe_python" "$probe" --source "$source_dir" --phase "$phase" --device musa 2>&1 | tee "$result_dir/$phase-musa.log"
    ;;
  loop)
    source_commit="$("$probe_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rlinf"]["route2"]["upstream_commit"])' "$package_root/locks/sources.json")"
    status=0
    timeout 240 "$probe_python" "$package_root/probes/mujoco_ppo_worker.py" --device musa --source-commit "$source_commit" --route-label route2 --horizon 32 --num-envs 4 --iterations 2 --epochs 2 --minibatch 64 --eval-episodes 1 --resume-check --output "$result_dir/loop.json" > "$result_dir/loop.log" 2>&1 || status=$?
    "$probe_python" "$package_root/scripts/summarize_log.py" "$result_dir/loop.log"
    exit "$status"
    ;;
  *)
    echo 'Usage: bash scripts/run_probes.sh {imports|cpu|channel|worker|fsdp|loop}' >&2
    exit 2
    ;;
esac
