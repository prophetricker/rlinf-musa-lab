"""Read bounded public safetensors headers, never tensor payloads or credentials."""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "routes/route2/model_probes/weight_metadata"
REVISION = "73f710e70e7d571f8d828e51e0a428f5a1e0ac22"
BASE = f"https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial/resolve/{REVISION}"
LIMIT = 2 * 1024 * 1024


def read_exact(stream, length):
    parts = []
    while length:
        part = stream.read(length)
        if not part:
            raise ValueError("Incomplete safetensors header")
        parts.append(part)
        length -= len(part)
    return b"".join(parts)


def main():
    index = json.loads((OUTPUT / "model.safetensors.index.json").read_text())
    record = {
        "revision": REVISION,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded 2MiB Range requests; application reads/saves headers only, closes response immediately; no complete weight shards or tensors loaded",
        "shards": [],
    }
    for name in sorted(set(index["weight_map"].values())):
        request = urllib.request.Request(
            BASE + "/" + name,
            headers={"Range": f"bytes=0-{LIMIT + 7}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            content_range = response.headers.get("Content-Range")
            if response.status == 206 and not content_range.startswith("bytes 0-"):
                raise ValueError("Returned range did not start at file offset zero")
            prefix = read_exact(response, 8)
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= LIMIT:
                raise ValueError("Refusing an invalid or oversized header")
            header = read_exact(response, length)
        # The response is closed immediately after the header. Signed redirect URLs
        # and arbitrary HTTP headers are deliberately not recorded.
        parsed = json.loads(header)
        tensors = {k: v for k, v in parsed.items() if k != "__metadata__"}
        expected = {k for k, shard in index["weight_map"].items() if shard == name}
        if set(tensors) != expected:
            raise ValueError("Shard header names differ from the fixed index")
        path = OUTPUT / (name + ".header.json")
        path.write_bytes(header)
        row = {
            "file": name,
            "canonical_url": BASE + "/" + name,
            "response_status": response.status,
            "content_range": content_range,
            "header_bytes": length,
            "application_bytes_read": length + 8,
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "tensor_count": len(tensors),
            "tensor_names_match_index": True,
            "dtype_counts": {
                dtype: sum(v["dtype"] == dtype for v in tensors.values())
                for dtype in sorted({v["dtype"] for v in tensors.values()})
            },
        }
        record["shards"].append(row)
        print(json.dumps(row), flush=True)
    (OUTPUT / "header-acquisition.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
