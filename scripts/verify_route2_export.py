#!/usr/bin/env python3
"""Check a route 2 export's hashes, relative links and executable source syntax."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


def unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "EXPORT_MANIFEST.json").read_text(), object_pairs_hook=unique_keys)
    counts = {"files": 0, "python_ast": 0, "shell_syntax": 0, "relative_markdown_links": 0}
    forbidden = re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    )
    for entry in manifest["files"]:
        path = output / entry["path"]
        if not path.resolve().is_relative_to(output):
            raise ValueError("Manifest path escapes export directory")
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Export content differs from manifest: {entry['path']}")
        counts["files"] += 1
        if path.suffix in {".md", ".py", ".sh", ".json", ".jsonl", ".txt", ".patch"}:
            source = content.decode()
            if forbidden.search(source):
                raise ValueError(f"Credential pattern found in {entry['path']}; value omitted")
        if path.suffix == ".py":
            ast.parse(source, filename=str(path))
            counts["python_ast"] += 1
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
            counts["shell_syntax"] += 1
        elif path.suffix == ".json":
            json.loads(source, object_pairs_hook=unique_keys)
        elif path.suffix == ".jsonl":
            for line in source.splitlines():
                if line.strip():
                    json.loads(line, object_pairs_hook=unique_keys)
        elif path.suffix == ".md":
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source):
                if urlsplit(link).scheme or link.startswith("#"):
                    continue
                target = unquote(link.split("#", 1)[0].strip("<>"))
                if target:
                    if not (path.parent / target).exists():
                        raise ValueError(f"Missing relative link in {entry['path']}: {target}")
                    counts["relative_markdown_links"] += 1
    print(json.dumps({"pass": True, **counts, "research_commit": manifest["research_commit"]}))


if __name__ == "__main__":
    main()
