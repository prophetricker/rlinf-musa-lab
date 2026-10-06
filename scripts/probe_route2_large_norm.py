#!/usr/bin/env python3
"""Compare an S4000 large flat-shard norm with a bounded equivalent reduction."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "chunked"), required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    sys.path.insert(0, str(args.support))
    import torch
    import torch_musa
    from musa_fsdp_norm import chunked_l2_norm
    result = {"status": "fail", "mode": args.mode,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256((args.support / "musa_fsdp_norm.py").read_bytes()).hexdigest(),
              "rows": [], "torch": torch.__version__, "torch_musa": torch_musa.__version__,
              "launch_blocking": os.environ.get("MUSA_LAUNCH_BLOCKING")}
    try:
        for size in (1048577, 9437184, 16777216, 50331648, 75497472, 150994944):
            value = torch.full((size + 3,), .125, dtype=torch.float32, device="musa")[3:]
            torch.musa.synchronize()
            row = {"size": size, "offset": 3, "status": "running"}
            result["rows"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            started = time.monotonic()
            norm = torch.linalg.vector_norm(value, 2, dtype=torch.float32) if args.mode == "direct" else chunked_l2_norm([value])
            actual = float(norm.item())
            expected = math.sqrt(size) * .125
            relative = abs(actual - expected) / expected
            row.update(actual=actual, expected=expected, relative_error=relative,
                       seconds=time.monotonic() - started)
            assert math.isfinite(actual) and relative < 2e-5
            row["status"] = "pass"
            del value, norm
        result["status"] = "pass"
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
