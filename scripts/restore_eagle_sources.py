#!/usr/bin/env python3
"""Restore a fixed, hash-checked GR00T tree and the experimental Eagle patch.

Creates a new directory only. Does not install packages or modify a checkout.
Use --tiny for the random diagnostic fixture, not pretrained weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
URL = "https://codeload.github.com/NVIDIA/Isaac-GR00T/tar.gz/" + COMMIT
ARCHIVE_HASH = "f98e0507e96281ffa289da9468e328b30de3733a00472f249f432e14050d1763"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional already-downloaded canonical archive")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--tiny", action="store_true")
    profile.add_argument("--spatial-full", action="store_true", help="Full Spatial geometry with explicit eager/FP32 config")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepared = root / "routes/route2/model_probes/eagle_integration/prepared"
    manifest_path = prepared / "spatial-full/source-preparation.json" if args.spatial_full else prepared / "source-preparation.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["gr00t_commit"] == COMMIT
    patch = prepared / "eagle-eager-lazy.patch"
    config_file = prepared / "spatial-full/spatial-eager-config.json" if args.spatial_full else prepared / "tiny-eagle-config.json"
    config_hash = manifest["full_config_sha256"] if args.spatial_full else manifest["tiny_config_sha256"]
    if digest(patch) != manifest["patch_sha256"] or digest(config_file) != config_hash:
        raise ValueError("Tracked patch/config hash mismatch")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output must be a new directory; existing files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".eagle-restore-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        archive = args.archive.resolve() if args.archive else scratch / "source.tar.gz"
        if not args.archive:
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if digest(archive) != ARCHIVE_HASH:
            raise ValueError("Archive does not match the fixed upstream SHA256")
        tree = scratch / "source"
        tree.mkdir()
        prefix = "Isaac-GR00T-" + COMMIT
        files = 0
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != prefix or ".." in parts or PurePosixPath(member.name).is_absolute():
                    raise ValueError("Unsafe archive path")
                destination = tree.joinpath(*parts[1:])
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    destination.chmod(member.mode & 0o777)
                    files += 1
                else:
                    raise ValueError("Archive links and special files are unsupported")
        for record in manifest["files"]:
            if digest(tree / record["path"]) != record["original_sha256"]:
                raise ValueError("Original source hash mismatch: " + record["path"])
        subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=tree, check=True)
        subprocess.run(["git", "apply", str(patch.resolve())], cwd=tree, check=True)
        if args.tiny or args.spatial_full:
            shutil.copyfile(config_file, tree / "gr00t/model/backbone/eagle2_hg_model/config.json")
        hashes = {}
        for record in manifest["files"]:
            expected = record["original_sha256"] if record.get("tiny_config_replacement") and not args.tiny else record["prepared_sha256"]
            actual = digest(tree / record["path"])
            if actual != expected:
                raise ValueError("Prepared source hash mismatch: " + record["path"])
            hashes[record["path"]] = actual
        tree.rename(output)
    print(json.dumps({"pass": True, "upstream_commit": COMMIT, "archive_url": URL,
                      "archive_sha256": ARCHIVE_HASH, "patch_sha256": manifest["patch_sha256"],
                      "tiny": args.tiny, "spatial_full": args.spatial_full,
                      "output": str(output), "files": files, "verified_files": hashes}))


if __name__ == "__main__":
    main()
