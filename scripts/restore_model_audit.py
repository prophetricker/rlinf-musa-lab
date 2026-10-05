#!/usr/bin/env python3
"""Restore hash-checked public model sources from the tracked audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="NVIDIA/Isaac-GR00T")
    parser.add_argument("--path", default="gr00t/model/action_head/cross_attention_dit.py")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "routes/route2/model_probes/audit_sources/manifest.json").read_text()
    )
    selected = [
        entry for entry in manifest["sources"]
        if entry["repo"] == args.repo and entry["path"] == args.path
    ]
    if len(selected) != 1:
        parser.error("Select exactly one repo/path present in the fixed source manifest")
    entry = selected[0]
    output = (args.output_root or root) / entry["file"]
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Existing source has unexpected content: {output}")
        print(json.dumps({"status": "already_verified", "source": str(output)}))
        return
    errors = []
    for url in (entry["url"], entry.get("transport_prefix", "") + entry["url"]):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Public source response did not match fixed SHA256")
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__ + ": " + str(error))
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        print(json.dumps({"status": "restored_verified", "commit": entry["commit"], "source": str(output), "sha256": entry["sha256"]}))
        return
    raise RuntimeError("Could not restore pinned source: " + "; ".join(errors))


if __name__ == "__main__":
    main()
