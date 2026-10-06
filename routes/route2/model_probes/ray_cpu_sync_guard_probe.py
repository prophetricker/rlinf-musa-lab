#!/usr/bin/env python3
"""CPU-only tests for explicit transport guards; does not import RLinf."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    spec = importlib.util.spec_from_file_location("ray_cpu", args.source / "rlinf/hybrid_engines/weight_syncer/ray_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from omegaconf import OmegaConf
    import torch
    cfg = OmegaConf.create({"type": "bucket", "transport": "ray_cpu", "bucket": {"bucket_device": "cpu"}})
    channel = SimpleNamespace(_use_ray_transport=True)
    rows = []
    assert module.validate_ray_cpu_sync(cfg, channel, 1, 1) is True
    rows.append("single_rank_cpu_bucket_accept")
    for name, overrides, handle, sizes in (
        ("missing_channel", {}, None, (1, 1)),
        ("collective_channel", {}, SimpleNamespace(_use_ray_transport=False), (1, 1)),
        ("multi_actor", {}, channel, (2, 1)),
        ("multi_rollout", {}, channel, (1, 2)),
        ("accelerator_bucket", {"bucket": {"bucket_device": "musa"}}, channel, (1, 1)),
        ("other_syncer", {"type": "patch"}, channel, (1, 1)),
        ("unknown_transport", {"transport": "typo"}, channel, (1, 1)),
        ("implicit_transport_with_channel", {"transport": "collective"}, channel, (1, 1)),
    ):
        altered = OmegaConf.merge(cfg, overrides)
        try:
            module.validate_ray_cpu_sync(altered, handle, *sizes)
        except ValueError:
            rows.append(name)
        else:
            raise AssertionError(f"guard failed: {name}")
    assert module.validate_ray_cpu_sync(OmegaConf.create({"type": "bucket"}), None, 2, 2) is False
    rows.append("collective_default_preserved")
    module.require_cpu_payload({"weights": [torch.ones(3), (torch.zeros(1),)]})
    rows.append("nested_cpu_payload_accept")
    try:
        module.require_cpu_payload({"nested": [torch.empty(1, device="meta")]})
    except ValueError:
        rows.append("nested_non_cpu_payload_reject")
    else:
        raise AssertionError("non-CPU payload accepted")
    result = {"status": "pass", "device": "cpu", "checks": rows, "count": len(rows),
              "scope": "transport configuration and nested payload guards; no Ray/GPU training"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
