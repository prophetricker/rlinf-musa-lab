#!/usr/bin/env python3
"""Download only two public, revision-pinned Spatial safetensors shards.

Stdlib HTTP, bounded streaming and resumable .part files. Canonical/mirror URLs
are fixed; authorization, signed redirects and arbitrary URLs are never logged.
Verify exact file size and original raw header SHA256, all keys/dtype/shape/offset
against the frozen public header/index; record a full observed file SHA256.
No tensor load, install, Git operation, GPU use or checkpoint execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request


REPO = "RLinf/RLinf-Gr00t-SFT-Spatial"
REVISION = "73f710e70e7d571f8d828e51e0a428f5a1e0ac22"
INDEX_HASH = "bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746"
SHARDS = {
    "model-00001-of-00002.safetensors": (4999367032, 115184, "cea79e9d8782ab4350fd6e57aa9dea3393fc0dd08426d0230e06b9675b4a5fe3"),
    "model-00002-of-00002.safetensors": (2586705312, 17816, "31e150ad3f43b9a7779713cba80025504c72ebf8d67d28fb7f116f5ddae659f8"),
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_file(path, name, metadata_dir, index):
    expected_size, header_size, header_hash = SHARDS[name]
    if path.stat().st_size != expected_size:
        raise ValueError("File size does not match pinned shard size")
    with path.open("rb") as stream:
        observed_header_size = struct.unpack("<Q", stream.read(8))[0]
        if observed_header_size != header_size:
            raise ValueError("Header length does not match public pinned header")
        header_raw = stream.read(header_size)
    if hashlib.sha256(header_raw).hexdigest() != header_hash:
        raise ValueError("Original raw safetensors header hash mismatch")
    expected_raw = (metadata_dir / (name + ".header.json")).read_bytes()
    if hashlib.sha256(expected_raw).hexdigest() != header_hash:
        raise ValueError("Frozen local expected header hash mismatch")
    header = json.loads(header_raw)
    expected = json.loads(expected_raw)
    if header != expected:
        raise ValueError("Observed header values differ from frozen public header")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    index_names = {key for key, shard in index["weight_map"].items() if shard == name}
    if set(tensors) != index_names:
        raise ValueError("All tensor names must exactly match the frozen shard index")
    width = {"BF16": 2, "F32": 4}
    data_end = expected_size - 8 - header_size
    intervals = []
    for key, value in tensors.items():
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if dtype not in width or any(not isinstance(d, int) or d < 0 for d in shape):
            raise ValueError("Unexpected dtype/shape in fixed checkpoint")
        elements = 1
        for dim in shape:
            elements *= dim
        if offsets[1] - offsets[0] != elements * width[dtype]:
            raise ValueError("Tensor dtype/shape byte-size mismatch")
        intervals.append(tuple(offsets))
    intervals.sort()
    cursor = 0
    for begin, end in intervals:
        if begin != cursor or end < begin:
            raise ValueError("Tensor offsets are not a complete contiguous shard partition")
        cursor = end
    if cursor != data_end:
        raise ValueError("Tensor offsets do not cover the full pinned shard data")
    return {"file": str(path.resolve()), "bytes": expected_size, "header_bytes": header_size,
            "header_sha256": header_hash, "full_file_sha256": file_hash(path),
            "all_tensor_keys_dtype_shape_offsets_match": True, "tensor_count": len(tensors),
            "integrity_limit": "Full SHA256 is newly observed; expected public raw header/size/layout are pinned. No independent expected full-weight digest was supplied."}


def download_file(part, url, expected_size, attempts):
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            raise ValueError("Partial file is larger than pinned shard")
        if offset == expected_size:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "User-Agent": "SpatialPinnedProbe/1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                        raise ValueError("Range response does not match pinned shard/offset")
                elif status == 200:
                    # A public mirror may ignore Range. Restart the partial
                    # file in that case, preserving exact-size final validation.
                    offset = 0
                else:
                    raise ValueError(f"Unexpected public HTTP status {status}")
                mode = "ab" if offset else "wb"
                last_report = time.monotonic()
                with part.open(mode) as stream:
                    received = offset
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        if received + len(chunk) > expected_size:
                            raise ValueError("Public response exceeded pinned shard size")
                        stream.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report >= 10:
                            print(json.dumps({"partial": part.name, "bytes": received,
                                              "expected_bytes": expected_size}), flush=True)
                            last_report = time.monotonic()
                if received == expected_size:
                    return
                raise IOError("Public response ended before pinned shard size")
        except Exception as exc:
            # Do not stringify exceptions that may contain signed redirected
            # URLs. Only class/status are retained; original URL is fixed above.
            record = {"partial": part.name, "attempt": attempt, "error_class": type(exc).__name__}
            if isinstance(exc, urllib.error.HTTPError):
                record["http_status"] = exc.code
            print(json.dumps(record), flush=True)
            if attempt == attempts:
                raise RuntimeError("Public pinned shard transfer failed; resumable partial file preserved") from None
            time.sleep(min(attempt * 2, 10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).parent / "checkpoint_metadata")
    parser.add_argument("--transport", choices=("canonical", "hf-mirror"), default="canonical")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError("Evidence output already exists")
    if args.attempts < 1:
        raise ValueError("Attempts must be positive")
    index_path = args.metadata_dir / "model.safetensors.index.json"
    if file_hash(index_path) != INDEX_HASH:
        raise ValueError("Frozen index hash mismatch")
    index = json.loads(index_path.read_text())
    if set(index["weight_map"].values()) != set(SHARDS):
        raise ValueError("Unexpected shard index")
    args.destination.mkdir(parents=True, exist_ok=True)
    needed = sum(max(0, size - ((args.destination / name).stat().st_size if (args.destination / name).exists()
                                else (args.destination / (name + ".part")).stat().st_size
                                if (args.destination / (name + ".part")).exists() else 0))
                 for name, (size, _, _) in SHARDS.items())
    if shutil.disk_usage(args.destination).free < needed + 512 * 1024 * 1024:
        raise RuntimeError("Insufficient free disk for both pinned shards and transfer margin")
    base = "https://huggingface.co" if args.transport == "canonical" else "https://hf-mirror.com"
    records = []
    for name, (size, _, _) in SHARDS.items():
        target = args.destination / name
        url = f"{base}/{REPO}/resolve/{REVISION}/{name}"
        if not target.exists():
            part = args.destination / (name + ".part")
            download_file(part, url, size, args.attempts)
            result = validate_file(part, name, args.metadata_dir, index)
            part.rename(target)
            result["file"] = str(target.resolve())
        else:
            result = validate_file(target, name, args.metadata_dir, index)
        records.append({"canonical_url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                        "public_transport_url": url, **result})
        print(json.dumps({"file": name, "validated": True, "bytes": size}), flush=True)
    evidence = {"pass": True, "repo": REPO, "revision": REVISION, "transport": args.transport,
                "helper_sha256": file_hash(__file__), "index_sha256": INDEX_HASH,
                "files": records, "total_tensor_count": sum(r["tensor_count"] for r in records),
                "scope": "Only two public revision-pinned weight files. Complete size/header/allkeys/dtype/shape/offset audit; observed full hashes. No tensor loads or model execution.",
                "auth_used": False, "signed_redirects_recorded": False}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # User-facing failure class only; HTTP secrets/redirects stay unrecorded.
        print(json.dumps({"pass": False, "error_class": type(exc).__name__,
                          "note": "Validation or transfer failed; partial files are preserved; no success evidence emitted."}), file=sys.stderr)
        raise SystemExit(1)
