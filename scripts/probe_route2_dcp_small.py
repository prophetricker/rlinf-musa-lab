#!/usr/bin/env python3
"""Two-rank FSDP1 DCP round trip including a one-output value head.

Small same-process new-object checkpoint test; not a GR00T recovery result.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.support))
    import torch
    import torch_musa
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from musa_fsdp_optim_device import apply

    rank = int(os.environ["RANK"])
    torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("musa", torch.musa.current_device())
    dist.init_process_group("cpu:gloo,musa:mccl", timeout=datetime.timedelta(seconds=60))
    adapter = apply()
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    def create():
        torch.manual_seed(7123)
        model = FSDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                     torch.nn.Linear(8, 1)).to(device),
                     process_group=dist.group.WORLD, device_id=device,
                     use_orig_params=True, sharding_strategy=ShardingStrategy.FULL_SHARD)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

    def update(model, optim):
        x = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4) / 8
        optim.zero_grad(set_to_none=True)
        loss = (model(x) - 0.25).square().mean()
        loss.backward()
        model.clip_grad_norm_(1.0)
        optim.step()
        torch.musa.synchronize()
        return float(loss.detach().cpu())

    def sha(value):
        digest = hashlib.sha256()
        def visit(v):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().contiguous()
                digest.update(str((v.dtype, tuple(v.shape))).encode())
                digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    digest.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (tuple, list)):
                for child in v:
                    visit(child)
            else:
                digest.update(repr(v).encode())
        visit(value)
        return digest.hexdigest()

    def snapshot(model, optim):
        named = list(model.named_parameters())
        return {"model": sha([p for _, p in named]),
                "active_adam": sha({n: optim.state.get(p, {}) for n, p in named if p.numel()}),
                "param_groups": sha(optim.state_dict()["param_groups"])}

    model, optim = create()
    update(model, optim)
    saved = snapshot(model, optim)
    model_sd, optim_sd = get_state_dict(model, optimizers=optim, options=options)
    state_types = sorted({type(t).__name__ for t in model_sd.values()})
    dcp.save({"model": model_sd, "optim": optim_sd},
             storage_writer=dcp.FileSystemWriter(str(args.checkpoint)))
    expected_loss = update(model, optim)
    expected = snapshot(model, optim)
    restored_model, restored_optim = create()
    # Populate optimizer state on the new object before DCP reads into it.
    update(restored_model, restored_optim)
    m, o = get_state_dict(restored_model, optimizers=restored_optim, options=options)
    payload = {"model": m, "optim": o}
    dcp.load(payload, storage_reader=dcp.FileSystemReader(str(args.checkpoint)))
    set_state_dict(restored_model, optimizers=restored_optim,
                   model_state_dict=payload["model"], optim_state_dict=payload["optim"], options=options)
    restored = snapshot(restored_model, restored_optim)
    actual_loss = update(restored_model, restored_optim)
    actual = snapshot(restored_model, restored_optim)
    checks = {"restored_exact": saved == restored, "continuation_exact": expected == actual,
              "loss_exact": expected_loss == actual_loss, "sharded_tensor": state_types == ["ShardedTensor"]}
    report = {"status": "pass" if all(checks.values()) else "fail", "rank": rank,
              "checks": checks, "adapter": adapter, "state_types": state_types,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "saved": saved, "restored": restored, "expected": expected, "actual": actual,
              "loss_expected": expected_loss, "loss_actual": actual_loss,
              "scope": "two-rank composite PG; scalar output head; same-process new objects"}
    output = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "status": report["status"], "checks": checks}), flush=True)
    dist.destroy_process_group()
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
