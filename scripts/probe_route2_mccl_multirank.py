#!/usr/bin/env python3
"""Validate the small MCCL collectives needed before two-GPU RLinf tests."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "-1")))
    parser.add_argument(
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", os.environ.get("MUSA_LOCAL_RANK", "-1"))),
    )
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "-1")))
    args = parser.parse_args()

    import torch
    import torch_musa

    if not torch.musa.is_available():
        raise RuntimeError("MUSA required; no substitute backend")
    rank = args.rank
    local_rank = args.local_rank if args.local_rank >= 0 else rank
    world_size = args.world_size
    if rank < 0 or world_size < 0:
        raise RuntimeError("run with torch.distributed.run/torchrun")
    if world_size != 2:
        raise RuntimeError(f"expected exactly two ranks, got {world_size}")
    device_count = torch.musa.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(f"local rank {local_rank} is outside {device_count} MUSA devices")
    torch.musa.set_device(local_rank)
    device = torch.device("musa", local_rank)

    report = {
        "scope": "two-rank MCCL small tensor collectives; no RLinf or model execution",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_count": device_count,
        "device_name": torch.musa.get_device_name(local_rank),
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "backend": "mccl",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }

    def row(name: str, start: float, observed: object) -> None:
        report["rows"].append({"op": name, "pass": True, "seconds": time.perf_counter() - start, "observed": observed})

    torch.distributed.init_process_group("mccl", timeout=datetime.timedelta(seconds=30))
    try:
        start = time.perf_counter()
        torch.distributed.barrier()
        row("barrier", start, True)

        start = time.perf_counter()
        broadcast = torch.tensor([17.0, 23.0], device=device) if rank == 0 else torch.zeros(2, device=device)
        torch.distributed.broadcast(broadcast, src=0)
        torch.testing.assert_close(broadcast.cpu(), torch.tensor([17.0, 23.0]), rtol=0, atol=0)
        row("broadcast", start, broadcast.cpu().tolist())

        start = time.perf_counter()
        reduced = torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        torch.testing.assert_close(reduced.cpu(), torch.tensor([3.0, 6.0]), rtol=0, atol=0)
        row("all_reduce_sum", start, reduced.cpu().tolist())

        start = time.perf_counter()
        gathered = [torch.zeros(2, device=device) for _ in range(world_size)]
        local = torch.tensor([float(rank), float(rank + 10)], device=device)
        torch.distributed.all_gather(gathered, local)
        observed = [tensor.cpu().tolist() for tensor in gathered]
        assert observed == [[0.0, 10.0], [1.0, 11.0]], observed
        row("all_gather", start, observed)

        start = time.perf_counter()
        received = torch.zeros(2, device=device)
        if rank == 0:
            sent = torch.tensor([31.0, 37.0], device=device)
            torch.distributed.send(sent, dst=1)
        else:
            torch.distributed.recv(received, src=0)
        if rank == 1:
            torch.testing.assert_close(received.cpu(), torch.tensor([31.0, 37.0]), rtol=0, atol=0)
        row("send_recv", start, received.cpu().tolist() if rank == 1 else True)

        start = time.perf_counter()
        torch.distributed.barrier()
        row("final_barrier", start, True)
    finally:
        torch.distributed.destroy_process_group()

    gathered_reports = [None for _ in range(world_size)]
    # Gloo is deliberately not used for the device tests; gather the compact
    # per-rank result through the filesystem after the process group closes.
    report["status"] = "pass"
    report["elapsed_seconds"] = sum(item["seconds"] for item in report["rows"])
    output = args.output.with_name(f"{args.output.stem}.rank{rank}{args.output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"], "rows": len(report["rows"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
