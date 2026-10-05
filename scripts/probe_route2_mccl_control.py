#!/usr/bin/env python3
"""Check real single-rank MCCL object collectives required by DCP metadata."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    import torch_musa

    if not torch.musa.is_available():
        raise RuntimeError("MUSA required; no substitute backend")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.musa.set_device(0)
    report = {
        "scope": "single-rank MCCL control-plane APIs; not cross-rank validation",
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "world_size": 1,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    torch.distributed.init_process_group(
        "mccl", rank=0, world_size=1, timeout=datetime.timedelta(seconds=30)
    )
    payload = {"label": "dcp_control_probe", "cpu_tensor": torch.tensor([3, 5]), "tuple": (1, 2)}
    try:
        for name in ("all_gather_object", "gather_object", "broadcast_object_list", "scatter_object_list"):
            row = {"api": name, "pass": False}
            try:
                results = [None]
                if name == "all_gather_object":
                    torch.distributed.all_gather_object(results, payload)
                elif name == "gather_object":
                    torch.distributed.gather_object(payload, results, dst=0)
                elif name == "broadcast_object_list":
                    results = [payload]
                    torch.distributed.broadcast_object_list(results, src=0)
                else:
                    torch.distributed.scatter_object_list(results, [payload], src=0)
                received = results[0]
                assert received["label"] == payload["label"] and received["tuple"] == payload["tuple"]
                torch.testing.assert_close(received["cpu_tensor"], payload["cpu_tensor"], rtol=0, atol=0)
                row["pass"] = True
            except Exception as error:
                row.update(error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(limit=4))
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        torch.distributed.destroy_process_group()
    report["all_pass"] = all(row["pass"] for row in report["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
