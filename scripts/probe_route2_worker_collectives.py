#!/usr/bin/env python3
"""Compare RLinf Ray Worker default and explicit MUSA communication groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("mesh", "mccl", "dual"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    sys.path.insert(0, str(args.source.resolve()))
    os.environ["PYTHONPATH"] = str(args.source.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    import ray
    import torch
    import torch_musa
    from omegaconf import OmegaConf
    from rlinf.scheduler import Cluster, Worker, PackedPlacementStrategy

    class ProbeWorker(Worker):
        def probe(self, mode):
            import datetime
            from rlinf.hybrid_engines.fsdp.utils import create_device_mesh
            import torch.distributed as dist
            result = {
                "rank": self._rank, "pid": os.getpid(), "mode": mode,
                "environment": {k: os.environ.get(k) for k in (
                    "MUSA_VISIBLE_DEVICES", "LOCAL_RANK", "NODE_LOCAL_RANK",
                    "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
                    "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG")},
                "current_device_before": torch.musa.current_device(),
                "visible_device_count": torch.musa.device_count(),
                "properties": str(torch.musa.get_device_properties(0)),
                "rows": [], "status": "fail",
            }
            def persist():
                path = args.output.with_name(f"{args.output.stem}.rank{self._rank}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
            persist()
            try:
                if mode != "mesh":
                    torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
                    dist.init_process_group(
                        backend="mccl" if mode == "mccl" else "cpu:gloo,musa:mccl",
                        timeout=datetime.timedelta(seconds=30),
                    )
                mesh = create_device_mesh(self._world_size)
                torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
                device = torch.device("musa", torch.musa.current_device())
                group = mesh.get_group()
                result["backend"] = str(dist.get_backend(group))
                result["device_backend"] = type(group._get_backend(device)).__name__
                result["device"] = str(device)
                for name in ("scalar_sum", "scalar_max", "sum", "max", "avg", "list_fp32", "base_fp32", "list_bf16", "base_bf16", "reduce_scatter"):
                    started = time.monotonic()
                    row = {"op": name, "pass": False}
                    result["rows"].append(row)
                    persist()
                    x = torch.full(() if name.startswith("scalar") else (16,), float(self._rank + 1), device=device,
                                   dtype=torch.bfloat16 if "bf16" in name else torch.float32)
                    if name in ("sum", "max", "avg", "scalar_sum", "scalar_max"):
                        reduction = name.removeprefix("scalar_")
                        op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "avg": dist.ReduceOp.AVG}[reduction]
                        dist.all_reduce(x, op=op, group=group)
                        expected = {"sum": 3, "max": 2, "avg": 1.5}[reduction]
                        assert bool((x == expected).all().item())
                    elif name.startswith("list"):
                        parts = [torch.empty_like(x) for _ in range(2)]
                        dist.all_gather(parts, x, group=group)
                        assert all(bool((value == rank + 1).all().item()) for rank, value in enumerate(parts))
                    elif name.startswith("base"):
                        y = torch.empty(32, device=device, dtype=x.dtype)
                        dist.all_gather_into_tensor(y, x, group=group)
                        assert bool((y[:16] == 1).all().item()) and bool((y[16:] == 2).all().item())
                    else:
                        y = torch.empty(8, device=device)
                        dist.reduce_scatter_tensor(y, x, group=group)
                        assert bool((y == 3).all().item())
                    torch.musa.synchronize()
                    row.update({"pass": True, "seconds": time.monotonic() - started})
                    persist()
                result["status"] = "pass"
            except Exception as error:
                result.update(error=str(error), error_class=type(error).__name__, traceback=traceback.format_exc())
            persist()
            return result

    result = {"scope": "RLinf two Ray Workers, isolated devices, small MUSA collectives only",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "status": "fail"}
    try:
        cluster = Cluster(cluster_cfg=OmegaConf.create({"num_nodes": 1, "component_placement": {"probe": "all"}}))
        workers = ProbeWorker.create_group().launch(cluster=cluster, name="CollectiveProbe",
                                                   placement_strategy=PackedPlacementStrategy(0, 1))
        result["ranks"] = workers.probe(args.mode).wait()
        result["status"] = "pass" if all(r["status"] == "pass" for r in result["ranks"]) else "fail"
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    finally:
        ray.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
