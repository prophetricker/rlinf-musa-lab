#!/usr/bin/env python3
"""Validate a two-rank FSDP FULL_SHARD update on MUSA with a tiny model."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_musa
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected two ranks, got {world_size}")
    if not torch.musa.is_available() or torch.musa.device_count() < 2:
        raise RuntimeError("two MUSA devices required")
    torch.musa.set_device(local_rank)
    device = torch.device("musa", local_rank)
    torch.distributed.init_process_group("mccl", timeout=datetime.timedelta(seconds=30))
    torch.manual_seed(20261006)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))

        def forward(self, inputs):
            return self.net(inputs)

    model = FSDP(
        TinyPolicy().to(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(16, device=device, dtype=torch.float32).reshape(2, 8) / 16
    targets = torch.full((2, 4), 0.25, device=device)
    before = {name: tensor_hash(param) for name, param in model.named_parameters()}
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = (outputs - targets).square().mean()
    loss.backward()
    grad_norm = float(model.clip_grad_norm_(1.0).detach().cpu())
    optimizer.step()
    torch.musa.synchronize()
    after = {name: tensor_hash(param) for name, param in model.named_parameters()}
    changed = sorted(name for name in before if before[name] != after[name])
    local_loss = loss.detach().clone()
    try:
        reduced_loss = local_loss.clone()
        torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
        reduced_loss /= world_size
        local_changed = torch.tensor([len(changed)], device=device, dtype=torch.int64)
        torch.distributed.all_reduce(local_changed, op=torch.distributed.ReduceOp.SUM)
        global_param_norm = torch.zeros((), device=device, dtype=torch.float32)
        for param in model.parameters():
            global_param_norm += param.detach().float().square().sum()
        torch.distributed.all_reduce(global_param_norm, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()
        report = {
            "status": "pass" if changed and torch.isfinite(reduced_loss).item() and grad_norm > 0 else "fail",
            "scope": "two-rank MUSA FSDP FULL_SHARD tiny policy; one optimizer update",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device_name": torch.musa.get_device_name(local_rank),
            "backend": str(torch.distributed.get_backend()),
            "torch": torch.__version__,
            "torch_musa": torch_musa.__version__,
            "strategy": str(ShardingStrategy.FULL_SHARD),
            "loss_local": float(local_loss.cpu()),
            "loss_mean_across_ranks": float(reduced_loss.cpu()),
            "grad_norm": grad_norm,
            "local_changed_parameter_count": len(changed),
            "global_changed_parameter_count": int(local_changed.cpu()),
            "global_parameter_squared_norm": float(global_param_norm.cpu()),
            "changed_parameter_names": changed,
            "elapsed_seconds": time.perf_counter() - start,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    finally:
        torch.distributed.destroy_process_group()
    output = args.output.with_name(f"{args.output.stem}.rank{rank}{args.output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"], "changed": len(changed)}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
