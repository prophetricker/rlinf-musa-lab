"""Probe current RLinf imports and an upstream MLP/PPO update on one Worker.

Run imports independently, then let the coordinator schedule ``--phase worker``.
This exercises the current scheduler and PPO components with synthetic batches;
it does not run EmbodiedFSDPActor, GR00T, an environment, or a learning benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def emit(name: str, status: str, **details: Any) -> None:
    """Write one compact, machine-readable probe result."""
    print(
        json.dumps({"probe": name, "status": status, **details}, ensure_ascii=False),
        flush=True,
    )


def import_probe(module_name: str) -> None:
    """Import a real module, preserving the concrete first failure."""
    try:
        module = importlib.import_module(module_name)
        emit(module_name, "pass", file=getattr(module, "__file__", None))
    except Exception as error:  # noqa: BLE001 - preserve arbitrary vendor import failures
        emit(module_name, "blocked", error_type=type(error).__name__, error=str(error))


def imports() -> None:
    """Audit APIs without launching Workers or allocating MUSA tensors."""
    import torch

    for package in [
        "torch",
        "torch_musa",
        "ray",
        "omegaconf",
        "transformers",
        "accelerate",
    ]:
        try:
            emit(
                "package",
                "info",
                package=package,
                version=importlib.metadata.version(package),
            )
        except importlib.metadata.PackageNotFoundError:
            emit("package", "missing", package=package)
    for module in [
        "torch_musa",
        "torch.distributed.tensor",
        "torch.distributed._tensor",
        "torch.distributed.device_mesh",
        "torch.distributed.checkpoint.state_dict",
        "torch.distributed.checkpoint.stateful",
        "rlinf.scheduler",
        "rlinf.utils.utils",
        "rlinf.algorithms.advantages",
        "rlinf.algorithms.losses",
        "rlinf.models.embodiment.mlp_policy.mlp_policy",
        "rlinf.hybrid_engines.fsdp",
        "rlinf.workers.actor.embodied_fsdp_actor_worker",
        "rlinf.models.embodiment.gr00t.gr00t_n1d5.musa_patches",
        "flash_attn.flash_attn_interface",
    ]:
        import_probe(module)
    from torch.distributed.fsdp import FullyShardedDataParallel

    emit(
        "fsdp1_native",
        "info",
        file=importlib.import_module(FullyShardedDataParallel.__module__).__file__,
    )
    emit(
        "torch_api",
        "info",
        torch_version=torch.__version__,
        torch_path=torch.__file__,
        scaled_dot_product_attention=hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ),
        device_mesh=hasattr(torch.distributed, "device_mesh"),
    )


def native_fsdp_update(device: str) -> dict[str, Any]:
    """Test Torch's FSDP1 separately from the current RLinf FSDP package."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    backend = "mccl" if device == "musa" else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(17, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6)
        ).to(device)
        wrapped = FullyShardedDataParallel(
            model,
            device_id=torch.device(device),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        inputs = torch.randn(8, 17).to(device)
        targets = torch.randn(8, 6).to(device)
        before = [param.detach().cpu().clone() for param in wrapped.parameters()]
        loss = (wrapped(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        deltas = [
            float((param.detach().cpu() - old).abs().max())
            for param, old in zip(wrapped.parameters(), before, strict=True)
        ]
        if max(deltas) <= 0:
            raise AssertionError(
                "The native FSDP1 optimizer did not change any parameter"
            )
        return {
            "device": device,
            "backend": backend,
            "strategy": "NO_SHARD",
            "world_size": 1,
            "max_parameter_delta": max(deltas),
            "loss": float(loss.detach().cpu()),
            "rlinf_fsdp_package_used": False,
        }
    finally:
        torch.distributed.destroy_process_group()


def ppo_update(device: str) -> dict[str, Any]:
    """Use upstream MLPPolicy, GAE and PPO loss for a real optimizer update."""
    import torch

    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    from rlinf.algorithms.losses import compute_ppo_actor_loss
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    torch.manual_seed(42)
    policy = MLPPolicy(17, 6, 1, True, False).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    states = torch.randn(32, 17).to(device)
    actions = (0.2 * torch.randn(32, 6)).to(device)
    with torch.no_grad():
        previous = policy(forward_inputs={"states": states, "action": actions})
        old_logprobs = previous["logprobs"].sum(-1).detach()
    rewards = torch.linspace(0.01, 0.5, 32).reshape(4, 8).to(device)
    values = torch.zeros(5, 8).to(device)
    dones = torch.zeros(5, 8, dtype=torch.bool).to(device)
    dones[-1] = True
    advantages, returns = compute_gae_advantages_and_returns(
        rewards,
        values=values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    cpu_advantages, cpu_returns = compute_gae_advantages_and_returns(
        rewards.cpu(),
        values=values.cpu(),
        dones=dones.cpu(),
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    torch.testing.assert_close(advantages.cpu(), cpu_advantages)
    torch.testing.assert_close(returns.cpu(), cpu_returns)
    before = {
        name: param.detach().cpu().clone() for name, param in policy.named_parameters()
    }
    current = policy(forward_inputs={"states": states, "action": actions})
    actor_loss, metrics = compute_ppo_actor_loss(
        current["logprobs"].sum(-1).float(),
        old_logprobs.float(),
        0.2,
        0.2,
        advantages.reshape(-1).float(),
    )
    value_loss = (current["values"].reshape(-1) - returns.reshape(-1)).square().mean()
    loss = actor_loss + 0.5 * value_loss
    loss.backward()
    gradients = [param.grad for param in policy.parameters() if param.grad is not None]
    if not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise AssertionError("PPO backward produced non-finite gradients")
    optimizer.step()
    if device == "musa":
        torch.musa.synchronize()
    deltas = {
        name: float((param.detach().cpu() - before[name]).abs().max())
        for name, param in policy.named_parameters()
    }
    if max(deltas.values()) <= 0:
        raise AssertionError("The optimizer did not change any policy parameter")
    with torch.no_grad():
        expected = policy(forward_inputs={"states": states, "action": actions})
    checkpoint = io.BytesIO()
    torch.save(
        {"model": policy.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, map_location="cpu")
    resumed = MLPPolicy(17, 6, 1, True, False).to(device)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    with torch.no_grad():
        actual = resumed(forward_inputs={"states": states, "action": actions})
    for key in expected:
        torch.testing.assert_close(actual[key].cpu(), expected[key].cpu())

    for model, opt in [(policy, optimizer), (resumed, resumed_optimizer)]:
        opt.zero_grad(set_to_none=True)
        result = model(forward_inputs={"states": states, "action": actions})
        continued_actor_loss, _ = compute_ppo_actor_loss(
            result["logprobs"].sum(-1).float(),
            old_logprobs.float(),
            0.2,
            0.2,
            advantages.reshape(-1).float(),
        )
        continued_value_loss = (
            (result["values"].reshape(-1) - returns.reshape(-1)).square().mean()
        )
        (continued_actor_loss + 0.5 * continued_value_loss).backward()
        opt.step()
    for original, restored in zip(
        policy.parameters(), resumed.parameters(), strict=True
    ):
        torch.testing.assert_close(original.detach().cpu(), restored.detach().cpu())
    for original_state, restored_state in zip(
        optimizer.state.values(), resumed_optimizer.state.values(), strict=True
    ):
        for key, value in original_state.items():
            torch.testing.assert_close(value.cpu(), restored_state[key].cpu())
    return {
        "device": str(next(policy.parameters()).device),
        "loss": float(loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "max_parameter_delta": max(deltas.values()),
        "gradient_tensors": len(gradients),
        "gae_cpu_parity": True,
        "checkpoint_output_parity": True,
        "checkpoint_continued_update_parity": True,
        "policy_path": importlib.import_module(MLPPolicy.__module__).__file__,
        "optimizer_state_entries": len(resumed_optimizer.state),
        "metric_keys": sorted(metrics),
    }


def worker_probe(device: str, native_fsdp: bool = False) -> None:
    """Launch one real RLinf Worker and run the upstream PPO update there."""
    if device == "musa":
        import torch_musa  # noqa: F401
    import ray

    from rlinf.scheduler import (
        Cluster,
        NodePlacementStrategy,
        PackedPlacementStrategy,
        Worker,
    )

    class PPOProbeWorker(Worker):
        """A limited nn.Module extension; no claim of FSDP support."""

        def update(self) -> dict[str, Any]:
            """Update the upstream MLP with current RLinf's PPO components."""
            result = native_fsdp_update(device) if native_fsdp else ppo_update(device)
            result.update(
                rank=self._rank,
                world_size=self._world_size,
                accelerator_type=str(self.accelerator_type),
                scheduler_device_type=self.torch_device_type,
            )
            self.log_info("Completed the current-RLinf MLP/PPO compatibility probe")
            return result

    try:
        cluster = Cluster(num_nodes=1)
        emit(
            "cluster",
            "pass",
            num_accelerators=cluster.num_accelerators,
            worker_accelerator=str(Worker.accelerator_type),
            worker_device=Worker.torch_device_type,
        )
        if device == "musa":
            if cluster.num_accelerators < 1:
                raise RuntimeError(
                    "RLinf did not detect the S4000; refusing a CPU substitute"
                )
            placement = PackedPlacementStrategy(0, 0)
        else:
            placement = NodePlacementStrategy([0])
        group = PPOProbeWorker.create_group().launch(
            cluster=cluster,
            name="route2_current_ppo_probe",
            placement_strategy=placement,
        )
        for result in group.update().wait():
            emit(
                "native_torch_fsdp1_worker"
                if native_fsdp
                else "current_rlinf_worker_ppo",
                "pass",
                **result,
            )
    finally:
        ray.shutdown()


def main() -> None:
    """Dispatch probes only after explicit coordinator selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["imports", "local", "worker", "native-fsdp"],
        default="imports",
    )
    parser.add_argument("--device", choices=["cpu", "musa"], default="cpu")
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = (
        str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    emit("source", "info", source=str(source), phase=args.phase, device=args.device)
    started = time.monotonic()
    try:
        if args.phase == "imports":
            imports()
        elif args.phase == "local":
            if args.device == "musa":
                import torch_musa  # noqa: F401
            emit("current_rlinf_local_ppo", "pass", **ppo_update(args.device))
        else:
            worker_probe(args.device, native_fsdp=args.phase == "native-fsdp")
    except Exception as error:  # noqa: BLE001 - report the selected integration phase
        emit("phase", "blocked", error_type=type(error).__name__, error=str(error))
        sys.exit(1)
    finally:
        emit("elapsed", "info", seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
