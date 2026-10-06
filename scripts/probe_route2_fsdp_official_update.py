#!/usr/bin/env python3
"""Verify one official two-rank FULL_SHARD PPO update using a saved trajectory.

This is a functional fixture test, not fresh on-policy collection or learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def tensor_sha(value):
    import torch

    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_sha(value):
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(tensor_sha(item).encode())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def as_json(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: as_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--phase", choices=("update", "train", "recover"), default="update")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--diagnostic", action="store_true", help="synchronize and persist per-microbatch/optimizer boundaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.phase != "update" and args.checkpoint is None:
        parser.error("train/recover require a checkpoint path")
    if args.phase == "train" and args.checkpoint.exists():
        parser.error("train requires a new checkpoint directory")
    if args.phase == "recover" and args.reference is None:
        parser.error("recover requires the uninterrupted train reference")

    support = args.rlinf_source.parent / "model_probes"
    if not support.exists():
        support = Path(__file__).resolve().parents[1] / "routes/route2/model_probes"
    paths = [
        str(args.rlinf_source.resolve()),
        str(args.gr00t_source.resolve()),
        str(support),
        str(Path(__file__).resolve().parent),
    ]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    if os.environ.get("RLINF_MUSA_FSDP_INDEPENDENT_INIT", "0") != "0":
        parser.error("this probe requires the official synchronized initialization")

    result = {
        "schema_version": 1,
        "phase": args.phase,
        "diagnostic": args.diagnostic,
        "status": "fail",
        "scope": "official GR00T Actor, two ranks, FSDP FULL_SHARD, one saved-trajectory PPO update; not learning",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "transport_environment": {key: os.environ.get(key) for key in (
            "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG", "MUSA_LAUNCH_BLOCKING")},
        "initialization": "sync_module_states=True",
    }
    stage = "imports"
    started = time.monotonic()

    def progress(name):
        nonlocal stage
        stage = name
        result["stage"] = name
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(
            json.dumps(as_json(result), indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps({"stage": name, "elapsed_seconds": time.monotonic() - started}), flush=True)

    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from official_actor_config import make_config
        from rlinf.data.schema.embodied_types import Trajectory
        from rlinf.scheduler import Cluster, PackedPlacementStrategy
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        from rlinf.utils.utils import get_rng_state

        class AuditActor(EmbodiedFSDPActor):
            def event(self, name, **data):
                if not args.diagnostic:
                    return
                row = {"rank": self._rank, "event": name, "time": time.time(), **data}
                path = args.output.with_name(f"{args.output.stem}.rank{self._rank}.events.jsonl")
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

            def train_micro_batch(self, *arguments, **keywords):
                self.event("microbatch_enter", is_last=keywords["is_last"])
                value = super().train_micro_batch(*arguments, **keywords)
                if args.diagnostic:
                    torch.musa.synchronize()
                self.event("microbatch_exit")
                return value

            def optimizer_step(self):
                if args.diagnostic:
                    torch.musa.synchronize()
                    rows = [{"name": name, "shape": list(param.shape),
                             "stride": list(param.stride()), "offset": param.storage_offset(),
                             "grad_shape": list(param.grad.shape) if param.grad is not None else None,
                             "grad_stride": list(param.grad.stride()) if param.grad is not None else None,
                             "grad_offset": param.grad.storage_offset() if param.grad is not None else None,
                             "states": {key: {"shape": list(value.shape), "stride": list(value.stride()),
                                              "offset": value.storage_offset(), "device": str(value.device)}
                                        for key, value in self.optimizer.state.get(param, {}).items()
                                        if isinstance(value, torch.Tensor)}}
                            for name, param in self.model.named_parameters() if param.grad is not None]
                    path = args.output.with_name(f"{args.output.stem}.rank{self._rank}.gradients.json")
                    path.write_text(json.dumps(rows, indent=2) + "\n")
                self.event("optimizer_enter")
                value = super().optimizer_step()
                if args.diagnostic:
                    torch.musa.synchronize()
                self.event("optimizer_exit", grad_norm=float(value[0]))
                return value

            def seed_probe_initialization(self, seed):
                import random
                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.musa.manual_seed(seed)
                if args.diagnostic:
                    import faulthandler
                    faulthandler.dump_traceback_later(120, repeat=True)

            def snapshot(self):
                # Local fingerprints audit each rank's owned shard without
                # unsharding for a separate checkpoint/state-dict operation.
                local_values = [param.detach() for param in self.model.parameters()]
                finite = all(
                    not value.numel() or bool(torch.isfinite(value).all().item())
                    for value in local_values
                )
                if not finite:
                    raise AssertionError("non-finite official Actor weights")
                named = list(self.model.named_parameters())
                topology = [(name, str(param.dtype), list(param.shape), param.requires_grad)
                            for name, param in named]
                trainable = {name: param.detach() for name, param in named if param.requires_grad}
                optimizer_finite = all(
                    not isinstance(value, torch.Tensor) or not value.numel() or bool(torch.isfinite(value).all().item())
                    for state in self.optimizer.state.values() for value in state.values()
                )
                if not optimizer_finite:
                    raise AssertionError("non-finite Adam state")
                return {
                    "rank": self._rank,
                    "local_model_sha256": tree_sha(local_values),
                    "trainable_sha256": tree_sha(trainable),
                    "topology_sha256": tree_sha(topology),
                    "local_trainable_numel": sum(param.numel() for param in trainable.values()),
                    "local_parameter_numel": sum(value.numel() for value in local_values),
                    "state_count": len(local_values),
                    "optimizer_state_entries": len(self.optimizer.state),
                    "optimizer_sha256": tree_sha(self.optimizer.state_dict()),
                    "scheduler_sha256": tree_sha(self.lr_scheduler.state_dict()),
                    "rng_sha256": tree_sha(get_rng_state()),
                    "optimizer_steps": self.optimizer_steps,
                    "adam_steps": sorted(set(float(s["step"].item()) for s in self.optimizer.state.values())),
                    "nonempty_adam_steps": sorted(set(float(state["step"].item())
                        for param, state in self.optimizer.state.items() if param.numel())),
                    "empty_adam_state_entries": sum(not param.numel() for param in self.optimizer.state),
                    "version": self.version,
                    "finite": finite,
                    "optimizer_finite": optimizer_finite,
                    "peak_allocated": int(torch.musa.max_memory_allocated()),
                    "peak_reserved": int(torch.musa.max_memory_reserved()),
                }

            def batch_summary(self):
                batch = self.rollout_batch
                return {
                    "rank": self._rank,
                    "sample_indices": self.probe_sample_indices,
                    "full_gae_sha256": self.probe_full_gae_sha256,
                    "shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
                    "batch_sha256": tree_sha(batch),
                    "advantages_finite": bool(torch.isfinite(batch["advantages"]).all().item()),
                    "returns_finite": bool(torch.isfinite(batch["returns"]).all().item()),
                }

            def load_trajectory_for_probe(self, trajectory):
                # A Ray Channel is a queue, so two FSDP ranks would race for a
                # single item. Broadcast the same CPU trajectory as an RPC
                # argument and apply RLinf's normal receive-side processing in
                # every rank.
                self.rollout_batch = self._process_received_rollout_batch(
                    Trajectory.to_batch([trajectory])
                )

            def partition_training_fixture(self):
                # Compute official GAE and normalization on the complete
                # temporal sequence first, then assign four unique samples to
                # each rank. Boundary fields retain one bootstrap row because
                # official run_training trims it before flattening/shuffling.
                steps = self.rollout_batch["prev_logprobs"].shape[0]
                assert steps == 8 and self._world_size == 2
                assert bool(torch.isfinite(self.rollout_batch["advantages"]).all().item())
                assert bool(torch.isfinite(self.rollout_batch["returns"]).all().item())
                self.probe_full_gae_sha256 = tree_sha({key: self.rollout_batch.get(key) for key in (
                    "advantages", "returns", "loss_mask", "loss_mask_sum")})
                start = self._rank * 4
                stop = start + 4

                def partition(value, path=()):
                    if isinstance(value, dict):
                        return {key: partition(nested, (*path, key)) for key, nested in value.items()}
                    if isinstance(value, torch.Tensor):
                        boundary = path[-1] in {"dones", "terminations", "truncations", "prev_values"}
                        assert value.shape[0] == steps + int(boundary), (path, value.shape)
                        assert value.shape[1] == 1, (path, value.shape)
                        return value[start:stop + int(boundary)].clone()
                    if value is None:
                        return None
                    raise TypeError(f"unexpected training field {path}: {type(value)}")

                self.rollout_batch = partition(self.rollout_batch)
                self.probe_sample_indices = list(range(start, stop))
                return self.batch_summary()

        cfg = make_config(args.rlinf_source, args.model_path, steps=8)
        cfg.cluster.component_placement = {"actor": "0-1", "rollout": "0-1", "env": "0"}
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.use_orig_params = True
        cfg.actor.global_batch_size = 8
        cfg.actor.micro_batch_size = 1
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        result["config_sha256"] = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
        result["source_fingerprints"] = {
            str(path.relative_to(args.rlinf_source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.rlinf_source / "rlinf").rglob("*.py")
        }
        cluster = Cluster(cluster_cfg=cfg.cluster)
        if cluster.num_accelerators != 2:
            raise RuntimeError(f"expected two accelerators, got {cluster.num_accelerators}")
        result["runtime"] = {
            "torch": torch.__version__,
            "torch_musa": torch_musa.__version__,
            "cluster_accelerators": cluster.num_accelerators,
            "backend": "MCCL via RLinf Worker mesh",
        }

        progress("official_actor_initialization")
        actor = AuditActor.create_group(cfg).launch(
            cluster=cluster,
            name=cfg.actor.group_name,
            placement_strategy=PackedPlacementStrategy(0, 1),
        )
        actor.seed_probe_initialization(cfg.actor.seed).wait()
        actor.init_worker().wait()
        exact_keys = ("local_model_sha256", "trainable_sha256", "topology_sha256",
                      "optimizer_sha256", "scheduler_sha256", "rng_sha256",
                      "optimizer_steps", "adam_steps", "nonempty_adam_steps", "version")
        reference = None
        if args.phase == "recover":
            progress("fresh_process_load_checkpoint")
            actor.load_checkpoint(str(args.checkpoint)).wait()
            reference = json.loads(args.reference.read_text())
        before = actor.snapshot().wait()
        result["before"] = before
        if reference is not None:
            result["restore_comparison"] = [{key: old[key] == restored[key] for key in exact_keys}
                for old, restored in zip(reference["checkpoint_state"], before, strict=True)]
            if not all(all(row.values()) for row in result["restore_comparison"]):
                raise AssertionError("fresh-process checkpoint restoration differs on a rank")

        progress("trajectory_load")
        trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=False)
        if not isinstance(trajectory, Trajectory):
            raise TypeError(f"expected Trajectory, got {type(trajectory)!r}")
        result["trajectory_shapes"] = {
            "actions": list(trajectory.actions.shape),
            "rewards": list(trajectory.rewards.shape),
            "prev_logprobs": list(trajectory.prev_logprobs.shape),
        }
        progress("load_trajectory")
        actor.load_trajectory_for_probe(trajectory).wait()
        progress("compute_gae")
        gae = actor.compute_advantages_and_returns().wait()
        result["gae"] = as_json(gae)
        result["batch"] = actor.partition_training_fixture().wait()
        if result["batch"][0]["full_gae_sha256"] != result["batch"][1]["full_gae_sha256"]:
            raise AssertionError("ranks computed different full-trajectory GAE")
        indices = [index for row in result["batch"] for index in row["sample_indices"]]
        if sorted(indices) != list(range(8)):
            raise AssertionError("training fixture duplicated or omitted samples")
        result["unique_global_samples"] = len(indices)
        progress("official_fsdp_ppo_update")
        metrics = actor.run_training().wait()
        result["training_metrics"] = as_json(metrics)
        after = actor.snapshot().wait()
        result["after"] = after
        result["checks"] = [{
            "rank": old["rank"],
            "rank_identity": old["rank"] == new["rank"],
            "same_topology": old["topology_sha256"] == new["topology_sha256"],
            "changed_trainable": old["trainable_sha256"] != new["trainable_sha256"],
            "changed_model": old["local_model_sha256"] != new["local_model_sha256"],
            "changed_optimizer": old["optimizer_sha256"] != new["optimizer_sha256"],
            "one_optimizer_step": new["optimizer_steps"] - old["optimizer_steps"] == 1,
            "adam_steps": new["nonempty_adam_steps"],
            "finite_nonzero_grad_norm": math.isfinite(float(metric["actor/grad_norm"])) and float(metric["actor/grad_norm"]) > 0,
            "finite_metrics": all(math.isfinite(float(value)) for value in metric.values()),
        } for old, new, metric in zip(before, after, metrics, strict=True)]
        if len(result["checks"]) != 2 or not all(
            all(row[key] for key in ("rank_identity", "same_topology", "changed_trainable", "changed_model", "changed_optimizer", "one_optimizer_step", "finite_nonzero_grad_norm", "finite_metrics"))
            and row["adam_steps"] == [before[row["rank"]]["optimizer_steps"] + 1.0] for row in result["checks"]
        ):
            raise AssertionError("both ranks must change with one finite, nonzero-gradient Adam update")
        if float(metrics[0]["actor/grad_norm"]) != float(metrics[1]["actor/grad_norm"]):
            raise AssertionError("reduced gradient norm differs between ranks")
        actor.set_global_step(before[0]["version"] + 1).wait()
        result["versioned_after"] = actor.snapshot().wait()
        if args.phase == "train":
            progress("save_two_rank_dcp_checkpoint")
            actor.save_checkpoint(str(args.checkpoint), 1).wait()
            result["checkpoint"] = str(args.checkpoint)
            result["checkpoint_state"] = actor.snapshot().wait()
            progress("uninterrupted_continuation")
            actor.load_trajectory_for_probe(trajectory).wait()
            actor.compute_advantages_and_returns().wait()
            actor.partition_training_fixture().wait()
            continuation_metrics = as_json(actor.run_training().wait())
            actor.set_global_step(2).wait()
            result["continuation"] = {"metrics": continuation_metrics, "after": actor.snapshot().wait()}
            if not all(row["optimizer_steps"] == 2 and row["nonempty_adam_steps"] == [2.0]
                       for row in result["continuation"]["after"]):
                raise AssertionError("continuation must perform the second optimizer update")
        if reference is not None:
            result["continuation_comparison"] = [{key: old[key] == restored[key] for key in exact_keys}
                for old, restored in zip(reference["continuation"]["after"], result["versioned_after"], strict=True)]
            result["continuation_metrics_exact"] = reference["continuation"]["metrics"] == as_json(metrics)
            if not all(all(row.values()) for row in result["continuation_comparison"]) or not result["continuation_metrics_exact"]:
                raise AssertionError("fresh-process continuation differs from uninterrupted path")
        result["status"] = "pass"
        result["elapsed_seconds"] = time.monotonic() - started
        ray.shutdown()
    except Exception as error:
        result.update(
            stage=stage,
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
    args.output.write_text(json.dumps(as_json(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "stage": result.get("stage", stage), "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
