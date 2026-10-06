#!/usr/bin/env python3
"""Audit two FULL_SHARD Actor ranks synchronizing one offloaded Rollout.

The sync phase checks complete CPU state reconstruction and fixed-input scores.
The train phase additionally collects a fresh eight-step LIBERO trajectory,
computes inherited GAE, splits eight unique samples across the Actor ranks,
executes one inherited PPO update, and checks synchronization at version one.
This establishes execution and policy consistency, not learning.
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

from official_actor_rollout_probe import as_json, tensor_sha, tree_sha


def state_manifest(state):
    """Hash ordinary full-state CPU tensors without sending their bytes by RPC."""
    import torch

    rows = {}
    for name, value in sorted(state.items()):
        if type(value) is not torch.Tensor or value.device.type != "cpu":
            raise TypeError(f"expected full CPU tensor for {name}, got {type(value)!r}")
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(f"non-finite state tensor: {name}")
        rows[name] = {"sha256": tensor_sha(value), "shape": list(value.shape),
                      "dtype": str(value.dtype)}
    return rows


def host_memory():
    rows = Path("/proc/self/status").read_text().splitlines()
    return {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in rows if line.startswith(("VmRSS:", "VmHWM:"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--phase", choices=("sync", "train"), default="sync")
    parser.add_argument("--fixture", type=Path,
                        help="optional saved CPU trajectory used only for fixed-input score checks")
    parser.add_argument("--trajectory-output", type=Path,
                        help="new path to preserve the fresh version-zero training trajectory")
    parser.add_argument("--norm-mode", choices=("legacy", "upstream"), default="legacy")
    parser.add_argument("--expected-state-count", type=int, default=907)
    parser.add_argument("--expected-selected-count", type=int, default=322)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        parser.error("output and partial output must be new")
    if args.trajectory_output is not None and args.trajectory_output.exists():
        parser.error("trajectory output must be new")
    if args.phase != "train" and args.trajectory_output is not None:
        parser.error("trajectory output requires phase=train")
    if os.environ.get("RLINF_MUSA_FSDP_INDEPENDENT_INIT", "0") != "0":
        parser.error("this probe requires official synchronized initialization")
    if os.environ.get("MCCL_P2P_DISABLE", "1") != "1":
        parser.error("validated Worker placement requires MCCL_P2P_DISABLE=1")

    paths = [str(args.rlinf_source.resolve()), str(args.gr00t_source.resolve()),
             str(Path(__file__).resolve().parent)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["RLINF_EXT_MODULE"] = "actor_runtime_extension"
    os.environ["RLINF_EXPERIMENTAL_FSDP1_TORCH22"] = "1"
    os.environ["RLINF_MUSA_FSDP_LEGACY_NORM"] = str(int(args.norm_mode == "legacy"))
    os.environ["MCCL_P2P_DISABLE"] = "1"
    result = {"schema_version": 1, "status": "fail", "phase": args.phase,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "two FULL_SHARD Actor ranks; one Rollout on GPU1; CPU Bucket transport",
              "initialization": "sync_module_states=True",
              "norm_mode": args.norm_mode, "synchronizations": [],
              "transport_environment": {name: os.environ.get(name) for name in (
                  "MCCL_P2P_DISABLE", "MCCL_SHM_DISABLE", "MCCL_DEBUG", "MUSA_LAUNCH_BLOCKING")}}
    started = time.monotonic()
    stage = "imports"
    env = None

    def progress(name):
        nonlocal stage
        stage = name
        result["stage"] = name
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(
            json.dumps(as_json(result), indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"stage": name, "elapsed_seconds": time.monotonic() - started}), flush=True)

    try:
        import ray
        import torch
        import torch_musa
        from omegaconf import OmegaConf
        from official_actor_config import make_config
        from rlinf.data.schema.embodied_types import Trajectory, TrajectoryStep
        from rlinf.scheduler import Channel, Cluster, PackedPlacementStrategy
        from rlinf.utils.nested_dict_process import put_tensor_device
        from rlinf.utils.utils import get_rng_state, set_rng_state
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

        def seed(seed_value):
            import random
            import numpy as np

            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.musa.manual_seed(seed_value)
            if args.diagnostic:
                import faulthandler
                faulthandler.dump_traceback_later(120, repeat=True)

        class AuditActor(EmbodiedFSDPActor):
            def seed_probe(self, seed_value):
                seed(seed_value)

            def snapshot_full(self):
                # Both FSDP ranks must enter this collective. CPU full-state
                # export is rank-zero-only; an empty rank-one dict is expected.
                state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
                manifest = state_manifest(state)
                return {"rank": self._rank, "manifest": manifest,
                        "model_sha256": tree_sha(manifest), "state_count": len(manifest),
                        "selected_sync_names": list(self.param_names_need_sync),
                        "version": self.version,
                        "host_memory_bytes": host_memory(),
                        "peak_allocated": int(torch.musa.max_memory_allocated()),
                        "peak_reserved": int(torch.musa.max_memory_reserved())}

            def snapshot_local(self):
                trainable = {name: param.detach() for name, param in self.model.named_parameters()
                             if param.requires_grad}
                finite = all(not param.numel() or bool(torch.isfinite(param).all().item())
                             for param in self.model.parameters())
                optimizer_finite = all(not isinstance(value, torch.Tensor) or not value.numel()
                    or bool(torch.isfinite(value).all().item())
                    for state in self.optimizer.state.values() for value in state.values())
                return {"rank": self._rank, "trainable_sha256": tree_sha(trainable),
                        "local_trainable_numel": sum(value.numel() for value in trainable.values()),
                        "optimizer_sha256": tree_sha(self.optimizer.state_dict()),
                        "optimizer_steps": self.optimizer_steps,
                        "nonempty_adam_steps": sorted(set(float(state["step"].item())
                            for param, state in self.optimizer.state.items() if param.numel())),
                        "finite": finite, "optimizer_finite": optimizer_finite,
                        "version": self.version,
                        "host_memory_bytes": host_memory(),
                        "peak_allocated": int(torch.musa.max_memory_allocated()),
                        "peak_reserved": int(torch.musa.max_memory_reserved())}

            def score_cpu(self, details):
                rng = get_rng_state()
                training = self.model.training
                try:
                    self.model.eval()
                    with torch.no_grad():
                        output = self.model(
                            forward_inputs=put_tensor_device(details["forward_inputs"], self.device),
                            prev_logprobs=details["prev_logprobs"].to(self.device),
                            compute_values=True, compute_entropy=False)
                    return {key: value.detach().cpu() for key, value in output.items()
                            if isinstance(value, torch.Tensor)}
                finally:
                    self.model.train(training)
                    set_rng_state(rng)

            def load_trajectory_for_probe(self, trajectory):
                # RPC copies a CPU trajectory to both ranks. A single Ray
                # Channel item would be consumed by only one of the ranks.
                self.rollout_batch = self._process_received_rollout_batch(
                    Trajectory.to_batch([trajectory]))

            def partition_training_samples(self):
                # Official GAE sees the complete temporal sequence first.
                # Each rank then trains four unique samples; boundary tensors
                # retain their bootstrap row for official flattening/trimming.
                steps = self.rollout_batch["prev_logprobs"].shape[0]
                assert steps == 8 and self._world_size == 2
                assert bool(torch.isfinite(self.rollout_batch["advantages"]).all().item())
                assert bool(torch.isfinite(self.rollout_batch["returns"]).all().item())
                gae_sha = tree_sha({key: self.rollout_batch.get(key) for key in (
                    "advantages", "returns", "loss_mask", "loss_mask_sum")})
                start, stop = self._rank * 4, self._rank * 4 + 4

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
                return {"rank": self._rank, "sample_indices": list(range(start, stop)),
                        "full_gae_sha256": gae_sha, "batch_sha256": tree_sha(self.rollout_batch),
                        "shapes": {key: list(value.shape) for key, value in self.rollout_batch.items()
                                   if isinstance(value, torch.Tensor)}}

        class AuditRollout(MultiStepRolloutWorker):
            def seed_probe(self, seed_value):
                seed(seed_value)

            def snapshot_cpu(self):
                state = self.hf_model.state_dict()
                manifest = state_manifest(state)
                return {"rank": self._rank, "manifest": manifest,
                        "model_sha256": tree_sha(manifest), "state_count": len(manifest),
                        "version": self.version, "global_step": self.global_step,
                        "fallback": self.hf_model.s4000_fallback_metadata,
                        "host_memory_bytes": host_memory(),
                        "peak_allocated": int(torch.musa.max_memory_allocated()),
                        "peak_reserved": int(torch.musa.max_memory_reserved())}

            def predict_cpu(self, obs):
                actions, details = self.predict(obs, mode="train")
                return actions.detach().cpu(), put_tensor_device(details, "cpu")

            def score_cpu(self, details):
                rng = get_rng_state()
                try:
                    with torch.no_grad():
                        output = self.hf_model(
                            forward_inputs=put_tensor_device(details["forward_inputs"], self.device),
                            prev_logprobs=details["prev_logprobs"].to(self.device),
                            compute_values=True, compute_entropy=False)
                    return {key: value.detach().cpu() for key, value in output.items()
                            if isinstance(value, torch.Tensor)}
                finally:
                    set_rng_state(rng)

        cfg = make_config(args.rlinf_source, args.model_path, steps=8)
        cfg.cluster.component_placement = {"actor": "0-1", "rollout": "1", "env": "0"}
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.use_orig_params = True
        cfg.actor.fsdp_config.torch22_state_dict_backend = "sharded_tensor"
        cfg.actor.global_batch_size = 8
        cfg.actor.micro_batch_size = 1
        cfg.rollout.enable_offload = True
        cfg.weight_syncer.actor_state_mode = "full_cpu_rank0"
        result["config"] = OmegaConf.to_container(cfg, resolve=True)
        result["config_sha256"] = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
        result["source_fingerprints"] = {
            str(path.relative_to(args.rlinf_source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.rlinf_source / "rlinf").rglob("*.py")}
        result["support_fingerprints"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("official_actor_config.py", "official_actor_rollout_probe.py",
                         "actor_runtime_extension.py", "musa_fsdp_norm.py")}
        progress("cluster")
        cluster = Cluster(cluster_cfg=cfg.cluster)
        if cluster.num_accelerators != 2:
            raise RuntimeError(f"expected two accelerators, got {cluster.num_accelerators}")
        result["runtime"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__,
                             "accelerators": cluster.num_accelerators}

        progress("official_rollout_initialization")
        rollout = AuditRollout.create_group(cfg).launch(cluster=cluster, name=cfg.rollout.group_name,
            placement_strategy=PackedPlacementStrategy(1, 1))
        rollout.seed_probe(cfg.actor.seed).wait()
        rollout.init_worker().wait()
        progress("official_actor_initialization")
        actor = AuditActor.create_group(cfg).launch(cluster=cluster, name=cfg.actor.group_name,
            placement_strategy=PackedPlacementStrategy(0, 1))
        actor.seed_probe(cfg.actor.seed).wait()
        actor.init_worker().wait()
        weights = Channel.create("TwoRankProbeWeights", maxsize=2, transport="ray")

        fixed = None
        if args.fixture is not None:
            fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
            if not isinstance(fixture, Trajectory):
                raise TypeError(f"expected a saved Trajectory, got {type(fixture)!r}")

            def first(value):
                if isinstance(value, dict):
                    return {key: first(nested) for key, nested in value.items()}
                if isinstance(value, torch.Tensor):
                    return value[0].clone()
                if value is None:
                    return None
                raise TypeError(f"unexpected fixed-input field: {type(value)}")

            fixed = {"forward_inputs": first(fixture.forward_inputs),
                     "prev_logprobs": fixture.prev_logprobs[0].clone()}
            result["fixed_input_source"] = {"kind": "saved trajectory; scores recomputed under current weights",
                "path": str(args.fixture.resolve()), "sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "stored_versions": sorted(set(fixture.versions.reshape(-1).tolist()))
                    if fixture.versions is not None else None}

        if args.fixture is None or args.phase == "train":
            from rlinf.envs.sim.libero.libero_env import LiberoEnv
            progress("libero_environment")
            env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
            env_cfg.specific_reset_id = 0
            env_cfg.is_eval = False
            env = LiberoEnv(env_cfg, num_envs=1, seed_offset=0,
                            total_num_processes=1, worker_info=None)
            obs, _ = env.reset()
            result["environment"] = {"task_ids": env.task_ids.tolist(), "trial_ids": env.trial_ids.tolist(),
                                     "task_descriptions": list(obs["task_descriptions"])}

        def synchronize(version):
            nonlocal fixed
            progress(f"synchronize_version_{version}")
            recv = rollout.sync_model_from_actor(weights)
            send = actor.sync_model_to_rollout(weights)
            send.wait()
            recv.wait()
            progress(f"synchronize_version_{version}/full_state_audit")
            actor_states = actor.snapshot_full().wait()
            rollout_state = rollout.snapshot_cpu().wait()[0]
            if [row["rank"] for row in actor_states] != [0, 1]:
                raise AssertionError("missing Actor rank in full-state reconstruction")
            leader = actor_states[0]
            if leader["state_count"] != args.expected_state_count:
                raise AssertionError(f"unexpected full-state count: {leader['state_count']}")
            for row in actor_states:
                if row["version"] != version:
                    raise AssertionError("Actor version differs from synchronization version")
                if row["manifest"] and row["manifest"] != leader["manifest"]:
                    raise AssertionError("nonempty Actor full-state reconstructions differ")
                if row["selected_sync_names"] != leader["selected_sync_names"]:
                    raise AssertionError("Actor ranks disagree on selected synchronization names")
            selected = leader["selected_sync_names"]
            if len(selected) != args.expected_selected_count or len(set(selected)) != len(selected):
                raise AssertionError("unexpected or duplicated selected synchronization names")
            if not set(selected) <= set(leader["manifest"]):
                raise AssertionError("selected sync names missing from full reconstruction")
            if leader["manifest"] != rollout_state["manifest"] or rollout_state["version"] != version:
                raise AssertionError("Actor/Rollout state hashes, shapes, dtypes, or versions differ")
            row = {"version": version, "actor_full_states": actor_states, "rollout_state": rollout_state,
                   "all_state_hashes_shapes_dtypes_equal": True,
                   "selected_names_covered": True, "selected_sync_count": len(selected)}
            result["synchronizations"].append(row)
            if fixed is None:
                progress(f"synchronize_version_{version}/fixed_input_collection")
                rollout.reload_model().wait()
                _, fixed = rollout.predict_cpu(obs).wait()[0]
                rollout.offload_model().wait()
                result["fixed_input_source"] = {"kind": "fresh version-zero LIBERO policy sample", "version": version}
            row["fixed_input_sha256"] = tree_sha(fixed)
            progress(f"synchronize_version_{version}/actor_fixed_scores")
            # FSDP ranks score together while Rollout remains on CPU. After
            # their collectives finish, score the single Rollout on GPU1.
            left = actor.score_cpu(fixed).wait()
            progress(f"synchronize_version_{version}/rollout_fixed_scores")
            rollout.reload_model().wait()
            right = rollout.score_cpu(fixed).wait()[0]
            rollout.offload_model().wait()
            comparisons = []
            for rank, output in enumerate(left):
                if set(output) != set(right):
                    raise AssertionError("Actor/Rollout output keys differ")
                if not all(bool(torch.isfinite(value).all()) for value in [*output.values(), *right.values()]):
                    raise AssertionError("non-finite fixed-input policy output")
                comparisons.append({"rank": rank, "outputs": {key: {
                    "equal": bool(torch.equal(output[key], right[key])),
                    "shape": list(output[key].shape),
                    "max_abs": float((output[key] - right[key]).abs().max())} for key in output}})
            row["fixed_output_comparisons"] = comparisons
            if not all(item["equal"] for comp in comparisons for item in comp["outputs"].values()):
                raise AssertionError(f"fixed-input Actor/Rollout outputs differ: {comparisons}")
            row["selected_old_new_logprob_max_abs"] = float((left[0]["logprobs"] - left[0]["prev_logprobs"]).abs().max())
            row["selected_old_new_action_ratio"] = (
                left[0]["logprobs"] - left[0]["prev_logprobs"]).sum(dim=-1).exp().tolist()

        synchronize(0)
        if args.phase == "train":
            progress("fresh_version_zero_libero_rollout")
            obs, _ = env.reset()
            rollout.reload_model().wait()
            steps, rows = [], []
            for index in range(8):
                action, details = rollout.predict_cpu(obs).wait()[0]
                action = action.reshape(1, 7)
                if not all(bool(torch.isfinite(value).all()) for value in (
                    action, details["prev_logprobs"], details["prev_values"])):
                    raise AssertionError("non-finite fresh rollout output")
                obs, reward, terminated, truncated, _ = env.step(action, auto_reset=False)
                boundary = torch.zeros((1, 1), dtype=torch.bool) if index == 0 else None
                steps.append(TrajectoryStep(actions=action.reshape(1, 1, 7),
                    prev_logprobs=details["prev_logprobs"], prev_values=details["prev_values"],
                    forward_inputs=details["forward_inputs"],
                    versions=torch.zeros_like(details["prev_logprobs"]), rewards=reward.reshape(1, 1),
                    dones=(terminated | truncated).reshape(1, 1), terminations=terminated.reshape(1, 1),
                    truncations=truncated.reshape(1, 1), initial_dones=boundary,
                    initial_terminations=boundary, initial_truncations=boundary))
                rows.append({"index": index, "action": action.tolist(), "reward": reward.tolist(),
                             "terminated": terminated.tolist(), "truncated": truncated.tolist()})
                if bool((terminated | truncated).any()):
                    raise RuntimeError("early boundary requires a separately configured batch")
                progress(f"fresh_version_zero_libero_rollout/step_{index + 1}")
            _, bootstrap = rollout.predict_cpu(obs).wait()[0]
            steps[-1].final_prev_values = bootstrap["prev_values"]
            rollout.offload_model().wait()
            trajectory = Trajectory.from_steps(steps, max_episode_length=240)
            if not bool((trajectory.versions == 0).all()):
                raise AssertionError("fresh trajectory must originate from version zero")
            result["fresh_trajectory"] = {"version": 0, "rollout": rows,
                "trajectory_sha256": tree_sha(trajectory.to_batch([trajectory])),
                "actions_shape": list(trajectory.actions.shape),
                "prev_logprobs_shape": list(trajectory.prev_logprobs.shape)}
            if args.trajectory_output is not None:
                args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(trajectory, args.trajectory_output)
                result["fresh_trajectory"].update(path=str(args.trajectory_output.resolve()),
                    file_sha256=hashlib.sha256(args.trajectory_output.read_bytes()).hexdigest())
            progress("official_full_trajectory_gae")
            actor.load_trajectory_for_probe(trajectory).wait()
            result["gae_metrics"] = as_json(actor.compute_advantages_and_returns().wait())
            batches = actor.partition_training_samples().wait()
            result["rank_batches"] = batches
            indices = [index for batch in batches for index in batch["sample_indices"]]
            if sorted(indices) != list(range(8)) or batches[0]["full_gae_sha256"] != batches[1]["full_gae_sha256"]:
                raise AssertionError("rank partition duplicated/omitted samples or full GAE differs")
            result["unique_global_samples"] = len(indices)
            before = actor.snapshot_local().wait()
            result["local_before"] = before
            progress("official_two_rank_ppo_update")
            metrics = actor.run_training().wait()
            after = actor.snapshot_local().wait()
            result["training_metrics"] = as_json(metrics)
            result["local_after"] = after
            checks = [{"rank": old["rank"],
                "changed_trainable_shard": old["trainable_sha256"] != new["trainable_sha256"],
                "changed_optimizer": old["optimizer_sha256"] != new["optimizer_sha256"],
                "one_optimizer_step": new["optimizer_steps"] - old["optimizer_steps"] == 1,
                "expected_adam_step": new["nonempty_adam_steps"] == [old["optimizer_steps"] + 1.0],
                "finite_parameters_and_optimizer": new["finite"] and new["optimizer_finite"],
                "finite_nonzero_grad_norm": math.isfinite(float(metric["actor/grad_norm"])) and float(metric["actor/grad_norm"]) > 0,
                "finite_metrics": all(math.isfinite(float(value)) for value in metric.values())}
                for old, new, metric in zip(before, after, metrics, strict=True)]
            result["update_checks"] = checks
            if len(checks) != 2 or not all(all(value for key, value in check.items() if key != "rank") for check in checks):
                raise AssertionError("both Actor shards must pass a real finite one-step update")
            if float(metrics[0]["actor/grad_norm"]) != float(metrics[1]["actor/grad_norm"]):
                raise AssertionError("reduced gradient norms differ between Actor ranks")
            actor.set_global_step(1).wait()
            synchronize(1)
            previous = result["synchronizations"][0]["actor_full_states"][0]["manifest"]
            current = result["synchronizations"][1]["actor_full_states"][0]["manifest"]
            changed = [name for name in previous if previous[name] != current[name]]
            selected = set(result["synchronizations"][1]["actor_full_states"][0]["selected_sync_names"])
            result["changed_full_state_names"] = changed
            result["all_changed_states_selected_for_sync"] = bool(changed) and set(changed) <= selected
            if not result["all_changed_states_selected_for_sync"]:
                raise AssertionError("changed state omitted from selected weight synchronization")
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
    args.output.write_text(json.dumps(as_json(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "stage": stage, "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
