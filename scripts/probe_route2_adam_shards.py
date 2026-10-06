#!/usr/bin/env python3
"""Exercise empty, scalar, offset, and large FP32 AdamW shards on MUSA."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    import torch
    import torch_musa
    result = {"status": "fail", "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "torch_musa": torch_musa.__version__, "rows": []}
    for name, size, offset in (("empty", 0, 0), ("scalar", None, 0), ("one", 1, 0),
                               ("offset", 17, 3), ("large", 1048577, 1)):
        row = {"name": name, "status": "fail"}
        result["rows"].append(row)
        try:
            storage = torch.ones((size or 1) + offset, device="musa", dtype=torch.float32)
            param = torch.nn.Parameter(storage[offset:offset + size] if size is not None else storage[0])
            optimizer = torch.optim.AdamW([param], lr=1e-8, betas=(.9, .95), foreach=False, fused=False)
            param.grad = torch.ones_like(param)
            row.update(shape=list(param.shape), storage_offset=param.storage_offset())
            optimizer.step()
            torch.musa.synchronize()
            row["optimizer_step_completed"] = True
            row["raw_empty_all"] = bool(torch.isfinite(param).all().item()) if not param.numel() else None
            assert not param.numel() or bool(torch.isfinite(param).all().item())
            assert float(optimizer.state[param]["step"]) == 1
            row["status"] = "pass"
        except Exception as error:
            row.update(error=str(error), traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "pass" if all(row["status"] == "pass" for row in result["rows"]) else "fail"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
