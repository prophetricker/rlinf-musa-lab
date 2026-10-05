#!/usr/bin/env python3
"""Reconstruct missing upstream worktrees from pinned refs and portable patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(*arguments: str) -> None:
    subprocess.run(["git", *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-patches", action="store_true")
    parser.add_argument(
        "--fsdp1-experimental",
        action="store_true",
        help="Also restore a separate opt-in FSDP1 source tree (requires --apply-patches).",
    )
    args = parser.parse_args()
    if args.fsdp1_experimental and not args.apply_patches:
        parser.error("--fsdp1-experimental requires --apply-patches")
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "locks/sources.json").read_text())
    patches = {
        "route1": root / "routes/route1/s4000-v03.patch",
        "route2": root / "routes/route2/patches/current-minimal-compat.patch",
    }
    for name, folder in [("rlinf", "RLinf"), ("torch_musa", "torch_musa")]:
        if name not in sources:
            continue
        source = sources[name]
        checkout = root / "external" / folder
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            run("clone", "--no-checkout", source["url"], str(checkout))
        for route, info in source.items():
            if not isinstance(info, dict) or "worktree" not in info:
                continue
            target = root / info["worktree"]
            if target.exists():
                print(f"Keep existing worktree: {target}")
                continue
            commit = info["upstream_commit"]
            available = (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "cat-file",
                        "-e",
                        commit + "^{commit}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if not available:
                run("-C", str(checkout), "fetch", "origin", commit)
            target.parent.mkdir(parents=True, exist_ok=True)
            branch = info["branch"]
            existing = (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "show-ref",
                        "--verify",
                        "--quiet",
                        "refs/heads/" + branch,
                    ]
                ).returncode
                == 0
            )
            if existing:
                run("-C", str(checkout), "worktree", "add", str(target), branch)
            else:
                run(
                    "-C",
                    str(checkout),
                    "worktree",
                    "add",
                    "-b",
                    branch,
                    str(target),
                    commit,
                )
            patch = patches.get(route)
            if args.apply_patches and patch and not existing:
                run("-C", str(target), "apply", "--check", str(patch))
                run("-C", str(target), "apply", str(patch))

    if args.fsdp1_experimental:
        info = sources["experiments"]["route2_fsdp1"]
        target = root / info["restore_worktree"]
        if target.exists():
            print(f"Keep existing experimental worktree: {target}")
            return
        incremental = root / info["patch"]
        if hashlib.sha256(incremental.read_bytes()).hexdigest() != info["patch_sha256"]:
            raise ValueError("Experimental FSDP1 patch does not match its source lock")
        target.parent.mkdir(parents=True, exist_ok=True)
        run(
            "-C", str(root / "external/RLinf"), "worktree", "add", "--detach",
            str(target), sources["rlinf"]["route2"]["upstream_commit"],
        )
        for patch in (patches["route2"], incremental):
            run("-C", str(target), "apply", "--check", str(patch))
            run("-C", str(target), "apply", str(patch))


if __name__ == "__main__":
    main()
