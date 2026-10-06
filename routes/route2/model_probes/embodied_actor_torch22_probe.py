#!/usr/bin/env python3
"""Audit RLinf's Torch 2.2 FSDP1 opt-in for the official embodied Actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


IMPORT_CODE = """
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
print(EmbodiedFSDPActor.__module__)
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_import(source: Path, *, opt_in: bool) -> dict[str, object]:
    env = os.environ.copy()
    source_path = str(source.resolve())
    inherited_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, inherited_path) if path
    )
    if opt_in:
        env["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    else:
        env.pop("RLINF_EXPERIMENTAL_FSDP1_TORCH22", None)
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_CODE],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "opt_in": opt_in,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")

    fsdp_init = args.rlinf_source / "rlinf/hybrid_engines/fsdp/__init__.py"
    result = {
        "schema_version": 1,
        "scope": "EmbodiedFSDPActor import gate on Torch 2.2",
        "probe_sha256": sha256(Path(__file__)),
        "source": str(args.rlinf_source.resolve()),
        "fsdp_init_sha256": sha256(fsdp_init),
        "runtime": {"python": sys.version, "executable": sys.executable},
        "without_opt_in": run_import(args.rlinf_source, opt_in=False),
        "with_torch22_fsdp1_opt_in": run_import(args.rlinf_source, opt_in=True),
    }
    result["status"] = (
        "pass"
        if result["with_torch22_fsdp1_opt_in"]["passed"]
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
