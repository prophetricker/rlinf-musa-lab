#!/usr/bin/env python3
"""Resume public pinned Spatial shards with bounded parallel Range requests.

The original downloader must be stopped first. Existing prefixes are preserved;
all completed shards require canonical LFS SHA256 and complete header validation.
No GPU, credentials, signed URLs in logs, package installs or tensor loading.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists() or args.workers not in range(1, 17) or args.chunk_mib not in range(1, 129):
        parser.error("New evidence path, workers1..16 and chunk-mib1..128 required")
    helper = args.support_dir / "download_spatial_weights.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != "663efd4dcbdd27e45070cb4442d6b7c2367634d89b1b26d5b15cb42727f5ebbb":
        raise ValueError("Pinned validation helper mismatch")
    sys.path.insert(0, str(args.support_dir.resolve()))
    from download_spatial_weights import SHARDS, REPO, REVISION, INDEX_HASH, file_hash, validate_file
    metadata = args.support_dir / "checkpoint_metadata"
    if file_hash(metadata / "model.safetensors.index.json") != INDEX_HASH:
        raise ValueError("Pinned index mismatch")
    index = json.loads((metadata / "model.safetensors.index.json").read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.destination).free < sum(s[0] for s in SHARDS.values()) * 2 + 512 * 1024**2:
        raise RuntimeError("Insufficient disk margin for range files and assembly")
    records = []
    for name, (size, _, _) in SHARDS.items():
        target = args.destination / name
        prefix = args.destination / (name + ".part")
        initial_offset = prefix.stat().st_size if prefix.exists() else 0
        if initial_offset > size:
            raise ValueError("Existing prefix exceeds expected shard")
        if not target.exists():
            chunks = args.destination / (name + f".ranges-{initial_offset}")
            chunks.mkdir(exist_ok=True)
            spans = [(start, min(start + args.chunk_mib * 1024**2, size) - 1)
                     for start in range(initial_offset, size, args.chunk_mib * 1024**2)]
            url = f"https://hf-mirror.com/{REPO}/resolve/{REVISION}/{name}"

            def transfer(span):
                start, end = span
                complete = chunks / f"{start}-{end}.complete"
                temporary = chunks / f"{start}-{end}.partial"
                expected = end - start + 1
                if complete.exists() and complete.stat().st_size == expected:
                    return span
                for attempt in range(3):
                    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, size):
                                raise ValueError("Server did not honor exact bounded Range")
                            total = 0
                            with temporary.open("wb") as stream:
                                while True:
                                    data = response.read(min(1024**2, expected - total + 1))
                                    if not data:
                                        break
                                    total += len(data)
                                    if total > expected:
                                        raise ValueError("Range exceeded expected length")
                                    stream.write(data)
                            if total != expected:
                                raise ValueError("Range ended early")
                        temporary.replace(complete)
                        return span
                    except Exception as error:
                        print(json.dumps({"file": name, "range": [start, end], "attempt": attempt + 1,
                                          "error_class": type(error).__name__}), flush=True)
                        if attempt == 2:
                            raise RuntimeError("Bounded public Range transfer failed; prefix/chunks preserved") from None
                        time.sleep(2)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(transfer, span) for span in spans]
                for i, future in enumerate(as_completed(futures), 1):
                    span = future.result()
                    print(json.dumps({"file": name, "completed_ranges": i, "total_ranges": len(spans), "range": list(span)}), flush=True)
            if prefix.exists() and prefix.stat().st_size != initial_offset:
                raise RuntimeError("Existing prefix changed during transfer; another downloader may be active")
            with prefix.open("ab") as stream:
                for start, end in spans:
                    with (chunks / f"{start}-{end}.complete").open("rb") as source:
                        shutil.copyfileobj(source, stream, length=8 * 1024**2)
            record = validate_file(prefix, name, metadata, index)
            prefix.rename(target)
            record["file"] = str(target.resolve())
            # Retain range cache until the coordinator checks successful delivery.
        else:
            record = validate_file(target, name, metadata, index)
        records.append({"initial_prefix_bytes": initial_offset, **record})
        print(json.dumps({"file": name, "complete_lfs_validation_pass": True}), flush=True)
    result = {"pass": True, "revision": REVISION, "files": records,
              "workers": args.workers, "chunk_mib": args.chunk_mib,
              "probe_sha256": file_hash(__file__), "validation_helper_sha256": file_hash(helper),
              "auth_used": False, "scope": "Full public weight download and canonical LFS/hash/header validation only; no model or tensor execution."}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"pass": False, "error_class": type(error).__name__, "scope": "download failed; prefixes and ranges preserved"}), file=sys.stderr)
        raise SystemExit(1)
