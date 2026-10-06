#!/usr/bin/env python3
"""Exercise inherited RLinf Actor/Rollout methods on one S4000.

The subclasses add only diagnostics and CPU-safe RPC boundaries. Training,
trajectory receive, GAE, weight synchronization, and checkpoint methods are
inherited. Recovery is invoked in a separate driver process. Zero rewards and
finite parameter changes establish execution, not learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
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
            digest.update(pickle.dumps(item, protocol=4))
    visit(value)
    return digest.hexdigest()


def model_hashes(state):
    return {name: tensor_sha(tensor) for name, tensor in sorted(state.items())}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--phase", choices=("sync", "train", "recover", "eval"), required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--continuation-trajectory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 2 or args.iterations < 1:
        parser.error("output must be new; steps >= 2 and iterations >= 1")
    if args.phase in ("train", "recover") and (
        args.checkpoint is None or args.continuation_trajectory is None
    ):
        parser.error("train/recover require checkpoint and continuation trajectory paths")
    if args.phase == "recover" and args.reference is None:
        parser.error("recover requires reference JSON from a completed train phase")
    if args.phase == "train" and (args.checkpoint.exists() or args.continuation_trajectory.exists()):
        parser.error("train requires new checkpoint and continuation trajectory paths")
    paths = [str(args.rlinf_source.resolve()), str(args.gr00t_source.resolve()),
             str(Path(__file__).resolve().parent)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    result = {"schema_version": 1, "status": "fail", "phase": args.phase,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "iterations": [], "synchronizations": [], "scope": "one Actor and one Rollout; NO_SHARD"}
    stage = "imports"
    started = time.monotonic()
    env = None

    def progress(name):
        nonlocal stage
        stage = name
        result["stage"] = name
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(json.dumps(as_json(result), indent=2) + "\n")
        print(json.dumps({"stage": name, "elapsed_seconds": time.monotonic() - started}), flush=True)

    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from official_actor_config import make_config
        from rlinf.scheduler import Channel, Cluster, PackedPlacementStrategy
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
        from rlinf.data.schema.embodied_types import Trajectory, TrajectoryStep
        from rlinf.utils.nested_dict_process import put_tensor_device
        from rlinf.utils.utils import get_rng_state, set_rng_state

        class AuditActor(EmbodiedFSDPActor):
            def snapshot(self):
                state = self.get_rollout_state_dict()
                hashes = model_hashes(state)
                finite = all(bool(torch.isfinite(t).all().item()) for t in state.values())
                if not finite:
                    raise AssertionError("non-finite Actor weights")
                steps = sorted(set(float(s["step"].item()) for s in self.optimizer.state.values()))
                return {"model_hashes": hashes, "model_sha256": tree_sha(hashes),
                        "optimizer_sha256": tree_sha(self.optimizer.state_dict()),
                        "scheduler_sha256": tree_sha(self.lr_scheduler.state_dict()),
                        "rng_sha256": tree_sha(get_rng_state()),
                        "optimizer_steps": self.optimizer_steps, "adam_steps": steps,
                        "version": self.version, "sync_names": self.param_names_need_sync,
                        "state_count": len(hashes), "finite": finite,
                        "peak_allocated": int(torch.musa.max_memory_allocated()),
                        "peak_reserved": int(torch.musa.max_memory_reserved()),
                        "default_backend": str(torch.distributed.get_backend()),
                        "device_backend": type(torch.distributed.distributed_c10d._get_default_group()._get_backend(torch.device("musa:0"))).__name__}

            def score(self, details):
                rng = get_rng_state()
                try:
                    self.model.eval()
                    with torch.no_grad():
                        output = self.model(
                            forward_inputs=put_tensor_device(details["forward_inputs"], self.device),
                            prev_logprobs=details["prev_logprobs"].to(self.device),
                            compute_values=True, compute_entropy=False,
                        )
                    return {k: v.detach().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)}
                finally:
                    set_rng_state(rng)

            def batch_audit(self):
                batch = self.rollout_batch
                for key in ("advantages", "returns"):
                    if not bool(torch.isfinite(batch[key]).all()):
                        raise AssertionError(f"non-finite {key}")
                return {"shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                        "rewards": batch["rewards"].tolist(),
                        "values": batch["prev_values"].tolist(),
                        "advantages": batch["advantages"].tolist(),
                        "returns": batch["returns"].tolist(), "batch_sha256": tree_sha(batch)}

        class AuditRollout(MultiStepRolloutWorker):
            def snapshot(self):
                hashes = model_hashes(self.hf_model.state_dict())
                return {"model_hashes": hashes, "model_sha256": tree_sha(hashes),
                        "version": self.version, "global_step": self.global_step,
                        "fallback": self.hf_model.s4000_fallback_metadata}

            def predict_cpu(self, obs, mode):
                actions, details = self.predict(obs, mode=mode)
                return actions.detach().cpu(), put_tensor_device(details, "cpu")

            def score(self, details):
                rng = get_rng_state()
                try:
                    with torch.no_grad():
                        output = self.hf_model(
                            forward_inputs=put_tensor_device(details["forward_inputs"], self.device),
                            prev_logprobs=details["prev_logprobs"].to(self.device),
                            compute_values=True, compute_entropy=False,
                        )
                    return {k: v.detach().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)}
                finally:
                    set_rng_state(rng)

        cfg = make_config(args.rlinf_source, args.model_path, steps=args.steps)
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
        progress("cluster")
        cluster = Cluster(cluster_cfg=cfg.cluster)
        if cluster.num_accelerators != 1:
            raise RuntimeError("probe requires exactly one accelerator")
        placement = PackedPlacementStrategy(0, 0)
        rollout = None
        if args.phase != "recover":
            progress("official_rollout_initialization")
            rollout = AuditRollout.create_group(cfg).launch(
                cluster=cluster, name=cfg.rollout.group_name, placement_strategy=placement)
            rollout.init_worker().wait()
        progress("official_actor_initialization")
        actor = AuditActor.create_group(cfg).launch(
            cluster=cluster, name=cfg.actor.group_name, placement_strategy=placement)
        actor.init_worker().wait()
        if args.phase == "eval" and args.checkpoint is not None:
            actor.load_checkpoint(str(args.checkpoint)).wait()
        trajectories = Channel.create("ProbeTrajectories", maxsize=1, transport="ray")
        weights = Channel.create("ProbeWeights", maxsize=2, transport="ray")

        def train(trajectory, label):
            progress(f"{label}/receive")
            receive = actor.recv_rollout_trajectories(trajectories)
            trajectories.put(trajectory)
            receive.wait()
            metrics = actor.compute_advantages_and_returns().wait()
            audit = actor.batch_audit().wait()[0]
            before = actor.snapshot().wait()[0]
            progress(f"{label}/official_ppo")
            update_started = time.monotonic()
            training = as_json(actor.run_training().wait()[0])
            after = actor.snapshot().wait()[0]
            if not all(torch.isfinite(torch.tensor(v)).all().item() for v in training.values()):
                raise AssertionError("non-finite training metric")
            if training["actor/grad_norm"] <= 0:
                raise AssertionError("training produced no nonzero gradients")
            changed = [name for name in before["model_hashes"]
                       if before["model_hashes"][name] != after["model_hashes"][name]]
            if not changed or after["adam_steps"] != [before["optimizer_steps"] + 1.0]:
                raise AssertionError("optimizer did not perform a real update")
            actor.set_global_step(before["version"] + 1).wait()
            return {"label": label, "gae_metrics": as_json(metrics), "batch": audit,
                    "training_metrics": training, "changed_parameter_names": changed,
                    "elapsed_seconds": time.monotonic() - update_started,
                    "after": actor.snapshot().wait()[0]}

        if args.phase == "recover":
            progress("fresh_process_load_checkpoint")
            actor.load_checkpoint(str(args.checkpoint)).wait()
            restored = actor.snapshot().wait()[0]
            reference = json.loads(args.reference.read_text())
            saved = reference["checkpoint_state"]
            keys = ("model_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256",
                    "optimizer_steps", "adam_steps", "version")
            result["restore_comparison"] = {k: restored[k] == saved[k] for k in keys}
            if not all(result["restore_comparison"].values()):
                raise AssertionError("restored training state differs from saved state")
            trajectory = torch.load(args.continuation_trajectory, map_location="cpu", weights_only=False)
            continuation = train(trajectory, "restored_continuation")
            result["continuation"] = continuation
            expected = reference["continuation"]
            result["continuation_comparison"] = {k: continuation["after"][k] == expected["after"][k] for k in keys}
            result["continuation_comparison"]["training_metrics"] = continuation["training_metrics"] == expected["training_metrics"]
            if not all(result["continuation_comparison"].values()):
                raise AssertionError("fresh-process continuation differs from uninterrupted path")
        else:
            from rlinf.envs.sim.libero.libero_env import LiberoEnv
            def new_env(reset_id):
                env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
                env_cfg.specific_reset_id = reset_id
                env_cfg.is_eval = args.phase == "eval"
                return LiberoEnv(env_cfg, num_envs=1, seed_offset=0,
                                 total_num_processes=1, worker_info=None)

            env = new_env(0)
            obs, _ = env.reset()
            fixed = None
            def sync(version):
                nonlocal fixed
                progress(f"synchronize_version_{version}")
                recv = rollout.sync_model_from_actor(weights)
                send = actor.sync_model_to_rollout(weights)
                send.wait()
                recv.wait()
                progress(f"synchronize_version_{version}/hashes")
                a = actor.snapshot().wait()[0]
                r = rollout.snapshot().wait()[0]
                if a["model_hashes"] != r["model_hashes"] or a["version"] != r["version"]:
                    raise AssertionError("Actor/Rollout weights or versions differ")
                result["latest_weight_hash_check"] = {"version": version, "exact": True,
                                                      "state_count": a["state_count"]}
                rollout.reload_model().wait()
                if fixed is None:
                    _, fixed = rollout.predict_cpu(obs, "train").wait()[0]
                progress(f"synchronize_version_{version}/fixed_scores")
                left = actor.score(fixed).wait()[0]
                right = rollout.score(fixed).wait()[0]
                if not all(bool(torch.isfinite(v).all()) for v in [*left.values(), *right.values()]):
                    raise AssertionError("non-finite fixed-input policy outputs")
                comparisons = {k: {"equal": bool(torch.equal(left[k], right[k])),
                                   "max_abs": float((left[k] - right[k]).abs().max())} for k in left}
                if not all(row["equal"] for row in comparisons.values()):
                    raise AssertionError(f"fixed-input Actor/Rollout outputs differ: {comparisons}")
                rollout.offload_model().wait()
                result["synchronizations"].append({"version": version,
                    "all_state_hashes_equal": True, "state_count": a["state_count"],
                    "selected_sync_count": len(a["sync_names"]),
                    "model_sha256": a["model_sha256"], "fixed_output_comparison": comparisons,
                    "selected_old_new_logprob_max_abs": float((left["logprobs"] - left["prev_logprobs"]).abs().max()),
                    "selected_old_new_action_ratio": (left["logprobs"] - left["prev_logprobs"]).sum(dim=-1).exp().tolist()})

            def collect(label):
                nonlocal obs
                progress(f"{label}/libero_rollout")
                obs, _ = env.reset()
                rollout.reload_model().wait()
                rows, steps = [], []
                for index in range(args.steps):
                    action, details = rollout.predict_cpu(obs, "train").wait()[0]
                    action = action.reshape(1, 7)
                    if not all(bool(torch.isfinite(v).all()) for v in (action, details["prev_logprobs"], details["prev_values"])):
                        raise AssertionError("non-finite rollout policy outputs")
                    next_obs, reward, terminated, truncated, info = env.step(action, auto_reset=False)
                    boundary = torch.zeros((1, 1), dtype=torch.bool) if index == 0 else None
                    step = TrajectoryStep(
                        actions=action.reshape(1, 1, 7), prev_logprobs=details["prev_logprobs"],
                        prev_values=details["prev_values"], forward_inputs=details["forward_inputs"],
                        versions=torch.full_like(details["prev_logprobs"], float(len(result["iterations"]))),
                        rewards=reward.reshape(1, 1), dones=(terminated | truncated).reshape(1, 1),
                        terminations=terminated.reshape(1, 1), truncations=truncated.reshape(1, 1),
                        initial_dones=boundary, initial_terminations=boundary,
                        initial_truncations=boundary,
                    )
                    steps.append(step)
                    rows.append({"index": index, "action": action.tolist(), "reward": reward.tolist(),
                                 "terminated": terminated.tolist(), "truncated": truncated.tolist()})
                    obs = next_obs
                    if bool((terminated | truncated).any()):
                        raise RuntimeError("early completion requires separately configured batch size")
                _, bootstrap = rollout.predict_cpu(obs, "train").wait()[0]
                steps[-1].final_prev_values = bootstrap["prev_values"]
                rollout.offload_model().wait()
                return Trajectory.from_steps(steps, max_episode_length=240), rows

            sync(actor.snapshot().wait()[0]["version"])
            if args.phase == "train":
                for index in range(args.iterations):
                    trajectory, rows = collect(f"iteration_{index + 1}")
                    row = train(trajectory, f"iteration_{index + 1}")
                    row["rollout"] = rows
                    result["iterations"].append(row)
                    sync(index + 1)
                trajectory, rows = collect("continuation_fixture")
                torch.save(trajectory, args.continuation_trajectory)
                progress("save_full_training_checkpoint")
                actor.save_checkpoint(str(args.checkpoint), args.iterations).wait()
                result["checkpoint_state"] = actor.snapshot().wait()[0]
                result["checkpoint"] = str(args.checkpoint)
                result["continuation"] = train(trajectory, "uninterrupted_continuation")
                result["continuation"]["rollout"] = rows
            elif args.phase == "eval":
                env.close()
                env = None
                result["episodes"] = []
                rollout.reload_model().wait()
                for reset_id in args.episode_ids:
                    env = new_env(reset_id)
                    obs, reset_info = env.reset()
                    identity = {"task_ids": env.task_ids.tolist(), "trial_ids": env.trial_ids.tolist(),
                                "task_descriptions": list(obs["task_descriptions"])}
                    episode_started = time.monotonic()
                    rows = []
                    for index in range(240):
                        action, details = rollout.predict_cpu(obs, "eval").wait()[0]
                        action = action.reshape(1, 7)
                        if not bool(torch.isfinite(action).all()):
                            raise AssertionError("non-finite evaluation action")
                        obs, reward, terminated, truncated, info = env.step(action, auto_reset=False)
                        rows.append({"index": index, "action": action.tolist(), "reward": reward.tolist(),
                                     "terminated": terminated.tolist(), "truncated": truncated.tolist()})
                        if index % 30 == 0:
                            progress(f"evaluation_{reset_id}/step_{index + 1}")
                        if bool((terminated | truncated).any()):
                            break
                    if not bool((terminated | truncated).any()):
                        raise AssertionError("episode ended without success or timeout boundary")
                    result["episodes"].append({"reset_id": reset_id, "rows": rows,
                        **identity,
                        "success": bool(terminated.any()), "timeout": bool(truncated.any()),
                        "steps": len(rows), "elapsed_seconds": time.monotonic() - episode_started})
                    env.close()
                    env = None
                rollout.offload_model().wait()
            result["actor_final"] = actor.snapshot().wait()[0]
        result["status"] = "pass"
    except Exception as error:
        result.update(stage=stage, error_class=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        if "ray" in locals():
            ray.shutdown()
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(as_json(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "stage": stage, "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
