#!/usr/bin/env python3
"""Check legacy L2 reduction accuracy and timing at actual GR00T shard sizes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    sys.path.insert(0, str(args.support))
    import torch
    import torch_musa
    from musa_fsdp_norm import legacy_l2_norm
    result = {"status": "running", "torch": torch.__version__, "torch_musa": torch_musa.__version__,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256((args.support / "musa_fsdp_norm.py").read_bytes()).hexdigest(),
              "rows": [], "relative_tolerance": 2e-5}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    torch.manual_seed(20261006)
    for sizes in ((1,), (17, 255, 4097), (65537, 1048577)):
        cpu = [torch.randn(size) for size in sizes]
        for scale in (1e-8, 1., 1e8):
            values = [value * scale for value in cpu]
            reference = float(torch.sqrt(sum((value.double() ** 2).sum() for value in values)))
            tensors = [value.to("musa") for value in values]
            actual = float(legacy_l2_norm(tensors).item())
            relative = abs(actual - reference) / reference
            row = {"kind": "random_cpu_fp64_reference", "sizes": list(sizes), "scale": scale,
                   "actual": actual, "reference": reference, "relative_error": relative,
                   "pass": math.isfinite(actual) and relative < 2e-5}
            result["rows"].append(row)
            persist()
            assert row["pass"]
            del tensors
    for size in (1048577, 9437184, 16777216, 50331648, 75497472, 150994944):
        # Exactly representable periodic data; double precision analytic L2
        # avoids a large CPU fixture or a second GPU implementation as oracle.
        pattern = (torch.arange(35, dtype=torch.float64) - 17) / 64
        quotient, remainder = divmod(size, 35)
        reference = math.sqrt(float((pattern ** 2).sum()) * quotient + float((pattern[:remainder] ** 2).sum()))
        values = pattern.float().to("musa").repeat(quotient + 1)[:size].clone()
        torch.musa.synchronize()
        row = {"kind": "periodic_analytic_reference", "size": size, "reference": reference, "timings": []}
        result["rows"].append(row)
        persist()
        for name, operation in (("linalg", lambda: torch.linalg.vector_norm(values, 2, dtype=torch.float32)),
                                ("legacy", lambda: legacy_l2_norm([values]))):
            for repeat in range(2):
                started = time.monotonic()
                actual = float(operation().item())
                relative = abs(actual - reference) / reference
                timing = {"operation": name, "repeat": repeat, "seconds": time.monotonic() - started,
                          "actual": actual, "relative_error": relative,
                          "pass": math.isfinite(actual) and relative < 2e-5}
                row["timings"].append(timing)
                persist()
                assert timing["pass"]
        del values
    result["status"] = "pass"
    persist()
    print(json.dumps({"status": "pass", "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
