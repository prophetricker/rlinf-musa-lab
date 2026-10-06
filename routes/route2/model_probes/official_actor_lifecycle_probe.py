#!/usr/bin/env python3
"""Run the real embodied Actor constructor and initialization through Ray.

The audit subclass only adds diagnostics; the constructor, model provider,
FSDP wrapping, optimizer creation and init_worker are upstream methods.
An initialization pass is not training, synchronization or recovery evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--phase", choices=("config", "init"), default="config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    for path in (args.rlinf_source / "rlinf", args.gr00t_source / "gr00t", args.model_path):
        if not path.is_dir():
            parser.error(f"missing directory: {path}")
    support = Path(__file__).resolve().parent
    paths = [str(args.rlinf_source.resolve()), str(args.gr00t_source.resolve()), str(support)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    result = {"schema_version": 1, "scope": "official EmbodiedFSDPActor lifecycle",
              "phase": args.phase, "status": "fail", "probe_sha256": sha(__file__),
              "source": str(args.rlinf_source.resolve()), "executable": sys.executable,
              "extension_sha256": sha(support / "actor_runtime_extension.py"),
              "config_builder_sha256": sha(support / "official_actor_config.py"),
              "fallback_sha256": sha(support / "gr00t_action_head_fallback.py")}
    stage = "configuration"
    started = time.monotonic()
    try:
        from official_actor_config import make_config
        from omegaconf import OmegaConf
        cfg = make_config(args.rlinf_source, args.model_path)
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        if args.phase == "init":
            import ray
            import torch
            import torch_musa  # noqa: F401
            from rlinf.scheduler import Cluster, PackedPlacementStrategy
            from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

            class AuditedActor(EmbodiedFSDPActor):
                def audit_init(self):
                    torch.musa.reset_peak_memory_stats()
                    init_started = time.monotonic()
                    self.init_worker()
                    from torch.distributed.fsdp import FullyShardedDataParallel
                    wrapped = list(FullyShardedDataParallel.fsdp_modules(self.model))
                    base = self.model.module
                    if not isinstance(self.model, FullyShardedDataParallel):
                        raise AssertionError("Actor model is not the real installed FSDP1")
                    if not self.optimizer.state:
                        raise AssertionError("Actor optimizer state was not initialized")
                    if not hasattr(base, "s4000_fallback_metadata"):
                        raise AssertionError("official model factory did not apply fallback")
                    trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
                    optimized = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
                    steps = [float(s["step"].item()) for s in self.optimizer.state.values()]
                    moments_zero = all(
                        not value.numel() or float(value.abs().max().item()) == 0.0
                        for state in self.optimizer.state.values()
                        for key, value in state.items() if key in ("exp_avg", "exp_avg_sq")
                    )
                    parameters_finite = all(
                        not p.numel() or bool(torch.isfinite(p).all().item())
                        for p in self.model.parameters()
                    )
                    if trainable != optimized or any(steps) or not moments_zero:
                        raise AssertionError("optimizer warmup changed the initial training state")
                    if not parameters_finite or self.optimizer_steps != 0:
                        raise AssertionError("invalid initial model or manager state")
                    return {
                        "actor_base_class": f"{EmbodiedFSDPActor.__module__}.{EmbodiedFSDPActor.__name__}",
                        "constructor": "inherited official constructor via create_group().launch()",
                        "model_class": type(base).__name__,
                        "strategy": type(self._strategy).__name__,
                        "fsdp_module_count": len(wrapped),
                        "fsdp_sharding": [str(m.sharding_strategy) for m in wrapped],
                        "backend": str(torch.distributed.get_backend()),
                        "world_size": self._world_size, "rank": self._rank,
                        "device": str(next(base.parameters()).device),
                        "optimizer_class": type(self.optimizer).__name__,
                        "optimizer_state_count": len(self.optimizer.state),
                        "optimizer_trainable_coverage_exact": trainable == optimized,
                        "optimizer_step_values": sorted(set(steps)),
                        "optimizer_moments_zero": moments_zero,
                        "optimizer_lrs": [float(g["lr"]) for g in self.optimizer.param_groups],
                        "parameters_finite": parameters_finite,
                        "optimizer_steps": self.optimizer_steps,
                        "parameter_count": sum(p.numel() for p in base.parameters()),
                        "trainable_count": sum(p.numel() for p in base.parameters() if p.requires_grad),
                        "fallback": base.s4000_fallback_metadata,
                        "peak_allocated": int(torch.musa.max_memory_allocated()),
                        "peak_reserved": int(torch.musa.max_memory_reserved()),
                        "init_elapsed_seconds": time.monotonic() - init_started,
                        "training_executed": False, "sync_executed": False,
                    }

            try:
                stage = "cluster"
                cluster = Cluster(cluster_cfg=cfg.cluster)
                if cluster.num_accelerators != 1:
                    raise RuntimeError("this scheduled probe requires exactly one accelerator")
                stage = "official_actor_constructor"
                actor = AuditedActor.create_group(cfg).launch(
                    cluster=cluster, name=cfg.actor.group_name,
                    placement_strategy=PackedPlacementStrategy(0, 0),
                )
                stage = "official_actor_init_worker"
                result["actor"] = actor.audit_init().wait()
                result["runtime"] = {"torch": torch.__version__,
                                     "torch_musa": torch_musa.__version__}
            finally:
                ray.shutdown()
        result["status"] = "pass"
    except Exception as error:
        result.update(stage=stage, error_class=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "phase": args.phase,
                      "stage": stage, "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
