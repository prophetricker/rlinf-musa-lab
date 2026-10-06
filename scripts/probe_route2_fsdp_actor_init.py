#!/usr/bin/env python3
"""Initialize the official GR00T Actor with two-rank FSDP FULL_SHARD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = args.rlinf_source.parent / "model_probes"
    if not support.exists():
        support = Path(__file__).resolve().parents[1] / "routes/route2/model_probes"
    paths = [
        str(args.rlinf_source.resolve()),
        str(args.gr00t_source.resolve()),
        str(support),
        str(Path(__file__).resolve().parent),
    ]
    sys_path = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = sys_path
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    result = {
        "status": "fail",
        "scope": "official GR00T Actor, two ranks, FSDP FULL_SHARD initialization only",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initialization": "sync_module_states=True",
        "transport_environment": {key: os.environ.get(key) for key in (
            "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG")},
    }
    started = time.monotonic()
    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from official_actor_config import make_config
        from rlinf.scheduler import Cluster, PackedPlacementStrategy
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

        class AuditActor(EmbodiedFSDPActor):
            def seed_probe_initialization(self, seed):
                import random
                import numpy as np
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.musa.manual_seed(seed)

            def snapshot(self):
                local_numel = sum(param.numel() for param in self.model.parameters())
                return {
                    "rank": self._rank,
                    "world_size": self._world_size,
                    "device": str(self.device),
                    "backend": str(torch.distributed.get_backend()),
                    "backend_detail": type(
                        torch.distributed.distributed_c10d._get_default_group()._get_backend(
                            torch.device("musa:0")
                        )
                    ).__name__,
                    "model_class": type(self.model).__name__,
                    "sharding_strategy": str(getattr(self.model, "sharding_strategy", "unknown")),
                    "local_parameter_numel": local_numel,
                    "global_parameter_numel_sum": None,
                    "optimizer_state_entries": len(self.optimizer.state),
                    "optimizer_groups": len(self.optimizer.param_groups),
                    "version": self.version,
                    "finite_local_parameters": all(
                        param.numel() == 0 or bool(torch.isfinite(param).all().item())
                        for param in self.model.parameters()
                    ),
                    "peak_allocated": int(torch.musa.max_memory_allocated()),
                    "peak_reserved": int(torch.musa.max_memory_reserved()),
                }

        cfg = make_config(args.rlinf_source, args.model_path, steps=8)
        cfg.cluster.component_placement = {"actor": "0-1", "rollout": "0-1", "env": "0"}
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.use_orig_params = True
        cluster = Cluster(cluster_cfg=cfg.cluster)
        result["runtime"] = {
            "torch": torch.__version__,
            "torch_musa": torch_musa.__version__,
            "cluster_accelerators": cluster.num_accelerators,
            "placement": {"actor": "0-1", "rollout": "0-1", "env": "0"},
        }
        if cluster.num_accelerators != 2:
            raise RuntimeError(f"expected two accelerators, got {cluster.num_accelerators}")
        actor = AuditActor.create_group(cfg).launch(
            cluster=cluster,
            name=cfg.actor.group_name,
            placement_strategy=PackedPlacementStrategy(0, 1),
        )
        actor.seed_probe_initialization(cfg.actor.seed).wait()
        actor.init_worker().wait()
        result["rank_snapshots"] = actor.snapshot().wait()
        result["global_parameter_numel_sum"] = sum(
            snapshot["local_parameter_numel"] for snapshot in result["rank_snapshots"]
        )
        result["status"] = "pass"
        result["elapsed_seconds"] = time.monotonic() - started
        ray.shutdown()
    except Exception as error:
        result.update(
            error_class=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - started,
        )
        try:
            if "ray" in locals():
                ray.shutdown()
        except Exception:
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
