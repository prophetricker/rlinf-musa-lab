#!/usr/bin/env python3
"""Read GPU memory and probe stages; no model, driver or environment changes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    started = time.monotonic()
    with args.output.open("x") as stream:
        while time.monotonic() - started <= args.seconds:
            raw = subprocess.check_output(["mthreads-gmi", "-q", "-d", "MEMORY", "--json"], text=True)
            data = json.loads(raw)
            stage_file = args.probe_output if args.probe_output.exists() else args.probe_output.with_suffix(".partial.json")
            stage = json.loads(stage_file.read_text()).get("stage") if stage_file.exists() else None
            stream.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                "timestamp": data["Timestamp"], "stage": stage,
                "memory": [{"index": gpu["Index"], **gpu["FB Memory Usage"]} for gpu in data["GPU"]]}) + "\n")
            stream.flush()
            if args.probe_output.exists():
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
