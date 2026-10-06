#!/usr/bin/env python3
"""Verify singleton broadcast of CPU payloads in a GPU-visible real Worker.

The probe never allocates a GPU tensor or initializes a process group. A
diagnostic guard intercepts collective-group construction inside the real
Worker: singleton cases must bypass it, and one two-address control must reach
it. Identity/content and existing AsyncFuncWork wait/then behavior are checked
inside the Worker, before Ray serializes the compact evidence returned by RPC.
NodePlacement retains the node's GPU visibility; the probe records the actual
placement and requires MUSA and the default process group to remain uninitialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    sys.path.insert(0, str(args.source.resolve()))
    os.environ["PYTHONPATH"] = str(args.source.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    # This CPU scheduler probe does not need model-registry extensions.
    os.environ.pop("RLINF_EXT_MODULE", None)
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    started = time.monotonic()
    result = {"schema_version": 1, "status": "fail",
              "scope": "CPU payloads in one GPU-visible real NodePlacement Worker; singleton broadcast API; no distributed/GPU communication",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
        from rlinf.scheduler.collective import AsyncFuncWork
        from torch_musa.core._lazy_init import is_initialized as musa_is_initialized

        class GroupConstructionGuard(RuntimeError):
            pass

        class ProbeWorker(Worker):
            async def probe(self):
                report = {"rank": self._rank, "world_size": self._world_size,
                          "has_accelerator": self.has_accelerator,
                          "accelerator_type": str(self._accelerator_type),
                          "placement_environment": {name: os.environ.get(name) for name in (
                              "ACCELERATOR_TYPE", "LOCAL_ACCELERATOR_RANK", "LOCAL_HARDWARE_RANKS",
                              "VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
                          "musa_initialized_before": musa_is_initialized(),
                          "default_group_initialized_before": torch.distributed.is_initialized(),
                          "rows": [], "status": "fail"}
                if self._world_size != 1:
                    raise AssertionError("probe must use a single real NodePlacement Worker")
                if report["musa_initialized_before"] or report["default_group_initialized_before"]:
                    # Do not run AsyncFuncWork when MUSA was already active:
                    # it would record a GPU event. Preserve the real state for
                    # the coordinator to assess before any API cases execute.
                    report["error"] = "preflight requires uninitialized MUSA and default process group"
                    path = args.output.with_name(f"{args.output.stem}.rank{self._rank}.json")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    return report
                name = self._group_name
                group = [(name, [0])]
                guard_calls = []
                collective = self._collective
                had_override = "create_collective_group" in collective.__dict__
                previous_override = collective.__dict__.get("create_collective_group")

                def forbid_group_construction(*arguments, **keywords):
                    guard_calls.append([address.get_name() for address in arguments[0]])
                    raise GroupConstructionGuard("diagnostic guard: no process/collective group may be created")

                collective.create_collective_group = forbid_group_construction
                try:
                    def record(label, run):
                        row = {"case": label, "pass": False}
                        report["rows"].append(row)
                        calls_before = len(guard_calls)
                        try:
                            checks = run()
                            row.update(checks=checks,
                                       group_construction_attempts=len(guard_calls) - calls_before)
                            row["pass"] = all(checks.values()) and len(guard_calls) == calls_before
                        except Exception as error:
                            row.update(error_class=type(error).__name__, error=str(error),
                                       traceback=traceback.format_exc(),
                                       group_construction_attempts=len(guard_calls) - calls_before)

                    tensor = torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4)
                    payloads = {"bool": True, "none": None,
                                "object": {"mutable": [1, "two", {"three": 3}]},
                                "tensor": tensor, "tensor_list": [tensor, tensor + 1],
                                "tensor_dict": {"first": tensor, "second": tensor + 2}}
                    for label, payload in payloads.items():
                        for explicit_source in (False, True):
                            def check_sync(payload=payload, explicit_source=explicit_source):
                                keywords = {"groups": group}
                                if explicit_source:
                                    keywords["src"] = (name, 0)
                                output = self.broadcast(payload, **keywords)
                                checks = {"same_object_identity": output is payload}
                                if isinstance(payload, torch.Tensor):
                                    checks.update(same_storage=output.data_ptr() == payload.data_ptr(),
                                                  same_shape_dtype=output.shape == payload.shape and output.dtype == payload.dtype,
                                                  same_tensor_content=bool(torch.equal(output, payload)),
                                                  still_cpu=output.device.type == "cpu")
                                return checks

                            record(f"sync_{label}_{'explicit' if explicit_source else 'default'}_source", check_sync)

                    record("integer_rank_group", lambda: {
                        "same_object_identity": self.broadcast(tensor, groups=[(name, 0)]) is tensor})

                    for label in ("object", "tensor"):
                        payload = payloads[label]

                        def check_wait(payload=payload):
                            work = self.broadcast(payload, groups=group, src=(name, 0), async_op=True)
                            return {"real_async_func_work": isinstance(work, AsyncFuncWork),
                                    "done_immediately": work.done(),
                                    "wait_same_object": work.wait() is payload,
                                    "repeat_wait_same_object": work.wait() is payload,
                                    "no_chain_initially": work.get_next_work() is None,
                                    "last_work_is_self": work.get_last_work() is work}

                        record(f"async_wait_{label}", check_wait)
                        calls_before = len(guard_calls)
                        row = {"case": f"async_async_wait_{label}", "pass": False}
                        report["rows"].append(row)
                        try:
                            work = self.broadcast(payload, groups=group, async_op=True)
                            output = await work.async_wait()
                            repeated = await work.async_wait()
                            checks = {"real_async_func_work": isinstance(work, AsyncFuncWork),
                                      "done_immediately": work.done(),
                                      "async_wait_same_object": output is payload,
                                      "repeat_async_wait_same_object": repeated is payload,
                                      "sync_wait_after_async_same_object": work.wait() is payload}
                            row.update(checks=checks, group_construction_attempts=len(guard_calls) - calls_before)
                            row["pass"] = all(checks.values()) and len(guard_calls) == calls_before
                        except Exception as error:
                            row.update(error_class=type(error).__name__, error=str(error),
                                       traceback=traceback.format_exc(),
                                       group_construction_attempts=len(guard_calls) - calls_before)

                        def check_then(payload=payload):
                            markers = []
                            work = self.broadcast(payload, groups=group, async_op=True)
                            # AsyncFuncWork.__call__ ignores the Future argument;
                            # user callbacks receive only these explicit args.
                            tail = work.then(lambda marker: markers.append(marker), "then-ran")
                            tail_result = tail.wait()
                            repeated = tail.wait()
                            return {"real_async_tail": isinstance(tail, AsyncFuncWork),
                                    "callback_once_with_explicit_argument": markers == ["then-ran"],
                                    "callback_returns_none": tail_result is None and repeated is None,
                                    "tail_done": tail.done(), "source_done": work.done(),
                                    "source_still_same_object": work.wait() is payload,
                                    "chain_linked": work.get_next_work() is tail,
                                    "last_work_is_tail": work.get_last_work() is tail}

                        record(f"async_then_{label}", check_then)

                    invalid_cases = [
                        ("missing_groups", {"groups": None}, ValueError),
                        ("groups_not_list", {"groups": ((name, [0]),)}, TypeError),
                        ("empty_groups", {"groups": []}, ValueError),
                        ("invalid_group_entry", {"groups": [name]}, TypeError),
                        ("invalid_group_name_type", {"groups": [(9, [0])]}, TypeError),
                        ("empty_rank_list", {"groups": [(name, [])]}, ValueError),
                        ("invalid_rank_list_entry", {"groups": [(name, [0.5])]}, TypeError),
                        ("invalid_rank_type", {"groups": [(name, "0")]}, TypeError),
                        ("caller_nonmember", {"groups": [(name, [1])]}, ValueError),
                        ("invalid_source_name_type", {"groups": group, "src": (9, 0)}, TypeError),
                        ("invalid_source_rank_type", {"groups": group, "src": (name, "0")}, TypeError),
                        ("malformed_source_tuple", {"groups": group, "src": (name,)}, ValueError),
                        ("source_nonmember_rank", {"groups": group, "src": (name, 1)}, ValueError),
                        ("source_nonmember_group", {"groups": group, "src": ("OtherProbeGroup", 0)}, ValueError),
                    ]
                    for label, keywords, expected_error in invalid_cases:
                        for async_op in (False, True):
                            def check_invalid(keywords=keywords, expected_error=expected_error, async_op=async_op):
                                try:
                                    self.broadcast(tensor, async_op=async_op, **keywords)
                                except expected_error:
                                    return {"rejected_with_expected_exception": True}
                                return {"rejected_with_expected_exception": False}

                            record(f"reject_{label}_{'async' if async_op else 'sync'}", check_invalid)

                    report["singleton_group_construction_attempts"] = len(guard_calls)
                    # This control proves the optimization did not replace
                    # multi-address broadcast. A nonexistent rank is harmless:
                    # the guard stops dispatch before any group is constructed.
                    calls_before = len(guard_calls)
                    control = {"case": "two_address_dispatch_guard_control", "pass": False}
                    report["rows"].append(control)
                    try:
                        self.broadcast(tensor, groups=[(name, [0, 1])], src=(name, 0))
                    except GroupConstructionGuard:
                        control["pass"] = len(guard_calls) == calls_before + 1
                    except Exception as error:
                        control.update(error_class=type(error).__name__, error=str(error))
                    control["group_construction_attempts"] = len(guard_calls) - calls_before
                    report["construction_guard_calls"] = guard_calls
                    report["native_groups_constructed"] = 0
                    report["musa_initialized_after"] = musa_is_initialized()
                    report["default_group_initialized_after"] = torch.distributed.is_initialized()
                    report["checks"] = {
                        "all_api_cases_pass": all(row["pass"] for row in report["rows"]),
                        "singleton_never_requests_group": report["singleton_group_construction_attempts"] == 0,
                        "two_address_path_preserved": control["pass"],
                        "cpu_payload_tensor": tensor.device.type == "cpu",
                        "musa_remains_uninitialized": not report["musa_initialized_before"]
                            and not report["musa_initialized_after"],
                        "default_group_remains_uninitialized": not report["default_group_initialized_before"]
                            and not report["default_group_initialized_after"],
                    }
                    report["status"] = "pass" if all(report["checks"].values()) else "fail"
                finally:
                    if had_override:
                        collective.create_collective_group = previous_override
                    else:
                        del collective.create_collective_group
                path = args.output.with_name(f"{args.output.stem}.rank{self._rank}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
                return report

        cfg = OmegaConf.create({"num_nodes": 1, "component_placement": {"probe": "all"}})
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
        result["source_fingerprints"] = {str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.source / "rlinf/scheduler/worker/worker.py",
                         args.source / "rlinf/scheduler/collective/async_work.py",
                         args.source / "rlinf/scheduler/placement/node.py")}
        cluster = Cluster(cluster_cfg=cfg)
        workers = ProbeWorker.create_group().launch(cluster=cluster, name="SingletonBroadcastProbe",
            placement_strategy=NodePlacementStrategy(node_ranks=[0]))
        result["ranks"] = workers.probe().wait()
        if len(result["ranks"]) != 1:
            raise AssertionError("expected exactly one real NodePlacement Worker report")
        result["status"] = result["ranks"][0]["status"]
    except Exception as error:
        result.update(error_class=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if "ray" in locals():
            ray.shutdown()
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
