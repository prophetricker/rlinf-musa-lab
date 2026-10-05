# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probe actual RLinf FSDP1 imports and a scheduled single-rank MUSA update.

The imports phase does not create a Cluster, process group, or GPU tensor.
The strategy phase must be scheduled by the GPU coordinator. It exercises a
tiny FP32 MSE update, not PPO, checkpointing, an official Actor, or a simulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "c70606f08cdca259b8dec03d4430926b5b8fac9d"
BASELINE_COMMIT = "3e7329c629293f1b0f9c331b5bb24eccc3e3f142"
OPT_IN = "RLINF_EXPERIMENTAL_FSDP1_TORCH22"
RTOL = 3e-4
ATOL = 3e-6


def emit(probe: str, status: str, **details: Any) -> None:
    """Emit stage evidence before entering operations that can abort Workers."""
    print(json.dumps({"probe": probe, "status": status, **details}), flush=True)


def source_evidence(source: Path) -> dict[str, Any]:
    """Record fixed references and current source hashes without remote URLs."""
    files = [
        "rlinf/hybrid_engines/fsdp/__init__.py",
        "rlinf/hybrid_engines/fsdp/strategy/base.py",
        "rlinf/hybrid_engines/fsdp/strategy/checkpoint.py",
        "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
        "rlinf/hybrid_engines/fsdp/utils.py",
    ]
    result = {
        "source": str(source),
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "expected_baseline_commit": BASELINE_COMMIT,
        "file_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in files
        },
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        result["actual_head_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(source), "diff", "--binary", "HEAD", "--", *files],
            stderr=subprocess.DEVNULL,
        )
        result["fsdp_worktree_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        result["fsdp_worktree_has_diff"] = bool(diff)
    except (OSError, subprocess.CalledProcessError):
        result["actual_head_commit"] = None
    return result


def imports() -> None:
    """Import real strategy modules and inspect the installed API signatures."""
    import torch
    import torch_musa  # noqa: F401
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision

    emit(
        "runtime",
        "info",
        torch_runtime_version=torch.__version__,
        torch_path=torch.__file__,
        experimental_opt_in=os.environ.get(OPT_IN) == "1",
        distribution_versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "torch_musa", "ray", "omegaconf", "accelerate"]
        },
    )
    for name in [
        "rlinf.hybrid_engines.fsdp",
        "rlinf.hybrid_engines.fsdp.utils",
        "rlinf.hybrid_engines.fsdp.strategy.checkpoint",
        "rlinf.hybrid_engines.fsdp.strategy.base",
        "rlinf.hybrid_engines.fsdp.strategy.fsdp",
    ]:
        emit(name, "started")
        module = importlib.import_module(name)
        emit(name, "pass", file=module.__file__)

    import torch.distributed.checkpoint as dcp

    import rlinf.hybrid_engines.fsdp as backend
    from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model

    if backend.FSDP is not FullyShardedDataParallel:
        raise AssertionError("RLinf does not export the real installed FSDP1 type")
    strategy = FSDPStrategyBase.create(
        OmegaConf.create({"fsdp_config": {"strategy": "fsdp"}}), world_size=1
    )
    if not isinstance(strategy, FSDPStrategy):
        raise AssertionError("The RLinf factory did not create FSDPStrategy")

    if not hasattr(backend, "fully_shard"):
        if hasattr(backend, "FSDPModule"):
            raise AssertionError("Unavailable FSDP2 type was supplied by a substitute")
        for name, operation in [
            ("require_fsdp2", backend.require_fsdp2),
            (
                "factory_fsdp2",
                lambda: FSDPStrategyBase.create(
                    OmegaConf.create({"fsdp_config": {"strategy": "fsdp2"}}),
                    world_size=1,
                ),
            ),
            (
                "apply_fsdp2_to_model",
                lambda: apply_fsdp2_to_model(
                    torch.nn.Linear(2, 2), {}, None, None, None, False
                ),
            ),
        ]:
            try:
                operation()
            except ImportError as error:
                if "FSDP2 is unavailable" not in str(error):
                    raise
                emit(name, "pass", rejected=True, reason=str(error))
            else:
                raise AssertionError(f"{name} silently accepted missing FSDP2 APIs")

    emit(
        "fsdp1_imports",
        "pass",
        real_fsdp1_type=True,
        strategy_class=f"{type(strategy).__module__}.{type(strategy).__name__}",
        signatures={
            "FSDP1": str(inspect.signature(FullyShardedDataParallel)),
            "MixedPrecision": str(inspect.signature(MixedPrecision)),
            "dcp.save": str(inspect.signature(dcp.save)),
            "dcp.load": str(inspect.signature(dcp.load)),
        },
        checkpoint_executed=False,
    )


def snapshot(model: Any, gradients: bool = False) -> dict[str, Any]:
    """Copy parameter or gradient tensors to CPU with comparable names."""
    result = {}
    for name, param in model.named_parameters():
        value = param.grad if gradients else param
        if value is None:
            raise AssertionError(f"Missing gradient for {name}")
        name = name.removeprefix("_fsdp_wrapped_module.")
        result[name] = value.detach().cpu().clone()
    return result


def compare(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    """Require finite values, identical shapes, and stated FP32 CPU parity."""
    import torch

    if actual.keys() != expected.keys():
        raise AssertionError("FSDP and CPU parameter names differ")
    maximum_error = 0.0
    for name in expected:
        if not bool(torch.isfinite(actual[name]).all()):
            raise AssertionError(f"Non-finite values in {name}")
        torch.testing.assert_close(actual[name], expected[name], rtol=RTOL, atol=ATOL)
        maximum_error = max(
            maximum_error, float((actual[name] - expected[name]).abs().max())
        )
    return maximum_error


def strategy_update(rank: int, world_size: int) -> dict[str, Any]:
    """Execute actual RLinf FSDP1 methods and compare one step with CPU."""
    import torch
    import torch_musa  # noqa: F401
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel

    from rlinf.hybrid_engines.fsdp import FSDP
    from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import (
        create_device_mesh,
        gradient_reduction_group,
    )
    from rlinf.scheduler import Cluster, Worker

    if FSDP is not FullyShardedDataParallel:
        raise AssertionError("RLinf does not export the real installed FSDP1 type")
    if rank != 0 or world_size != 1 or Worker.torch_device_type != "musa":
        raise AssertionError("This probe requires one actual MUSA Worker")
    if torch.distributed.is_initialized():
        raise AssertionError("The probe requires an uninitialized default group")
    device = torch.device(f"musa:{os.environ['LOCAL_RANK']}")
    torch.musa.set_device(device)
    torch.manual_seed(42)
    reference = torch.nn.Sequential(
        torch.nn.Linear(17, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6)
    )
    initial = snapshot(reference)
    model = copy.deepcopy(reference).to(device)
    inputs = torch.randn(8, 17)
    targets = torch.randn(8, 6)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    reference_loss = (reference(inputs) - targets).square().mean()
    reference_loss.backward()
    reference_raw_gradients = snapshot(reference, gradients=True)
    reference_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0))
    reference_clipped_gradients = snapshot(reference, gradients=True)
    reference_optimizer.step()

    config = OmegaConf.create(
        {
            "model": {"model_type": "mlp_policy", "is_lora": False},
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "disable": True,
                "use_orig_params": True,
                "cpu_offload": False,
                "forward_prefetch": False,
                "backward_prefetch": None,
                "limit_all_gathers": True,
                "enable_gradient_accumulation": False,
                "mixed_precision": {
                    "param_dtype": "fp32",
                    "reduce_dtype": "fp32",
                    "buffer_dtype": "fp32",
                },
            },
            "optim": {"clip_grad": 1.0},
        }
    )
    timeout = Cluster.get_collective_timeout()
    emit("mccl_default_group", "started", timeout_seconds=timeout.total_seconds())
    torch.distributed.init_process_group(
        backend="mccl", rank=rank, world_size=world_size, timeout=timeout
    )
    try:
        emit("rlinf_create_device_mesh", "started")
        mesh = create_device_mesh(world_size)
        dp_group = gradient_reduction_group(mesh)
        strategy = FSDPStrategyBase.create(
            config, world_size=world_size, dp_group=dp_group
        )
        if not isinstance(strategy, FSDPStrategy):
            raise AssertionError("The actual strategy is not RLinf FSDPStrategy")
        emit("rlinf_wrap_model", "started")
        wrapped = strategy.wrap_model(model, mesh)
        if not isinstance(wrapped, FSDP):
            raise AssertionError("wrap_model did not create a real FSDP1 model")
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        emit("rlinf_forward_backward", "started")
        with strategy.before_micro_batch(wrapped, is_last_micro_batch=True):
            loss = (wrapped(inputs.to(device)) - targets.to(device)).square().mean()
            loss.backward()
        torch.testing.assert_close(
            loss.detach().cpu(), reference_loss.detach(), rtol=RTOL, atol=ATOL
        )
        raw_gradient_error = compare(
            snapshot(wrapped, gradients=True), reference_raw_gradients
        )
        emit("rlinf_clip_grad_norm", "started")
        gradient_norm = strategy.clip_grad_norm_(wrapped)
        if not math.isfinite(gradient_norm):
            raise AssertionError("RLinf clipping returned a non-finite norm")
        if not math.isclose(gradient_norm, reference_norm, rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError("RLinf gradient norm differs from the CPU reference")
        clipped = snapshot(wrapped, gradients=True)
        clipped_gradient_error = compare(clipped, reference_clipped_gradients)
        clipped_norm = math.sqrt(
            sum(float(value.square().sum()) for value in clipped.values())
        )
        if clipped_norm > 1.0 + ATOL:
            raise AssertionError("RLinf did not clip gradients to the configured norm")
        emit("adamw_step", "started")
        optimizer.step()
        torch.musa.synchronize()
        final_parameters = snapshot(wrapped)
        parameter_error = compare(final_parameters, snapshot(reference))
        maximum_delta = max(
            float((value - initial[name]).abs().max())
            for name, value in final_parameters.items()
        )
        if maximum_delta <= 0:
            raise AssertionError("AdamW did not change any FSDP parameter")
        return {
            "device": str(device),
            "backend": str(torch.distributed.get_backend(dp_group)),
            "world_size": world_size,
            "rank": rank,
            "scheduler_device_type": Worker.torch_device_type,
            "strategy_class": f"{type(strategy).__module__}.{type(strategy).__name__}",
            "strategy_source": inspect.getfile(type(strategy)),
            "real_fsdp1_type": True,
            "sharding_strategy": "NO_SHARD",
            "mesh_device_type": mesh.device_type,
            "mesh_dim_names": list(mesh.mesh_dim_names),
            "collective_timeout_seconds": timeout.total_seconds(),
            "default_group_backend_explicit": True,
            "precision": "fp32",
            "loss": float(loss.detach().cpu()),
            "cpu_reference_loss": float(reference_loss.detach()),
            "gradient_norm": gradient_norm,
            "clipped_gradient_norm": clipped_norm,
            "cpu_reference_gradient_norm": reference_norm,
            "gradient_tensors": len(clipped),
            "raw_gradient_max_abs_error": raw_gradient_error,
            "clipped_gradient_max_abs_error": clipped_gradient_error,
            "parameter_max_abs_error": parameter_error,
            "max_parameter_delta": maximum_delta,
            "parity_rtol": RTOL,
            "parity_atol": ATOL,
            "cpu_reference_update_parity": True,
            "official_actor_used": False,
            "checkpoint_executed": False,
            "offload_executed": False,
            "ppo_executed": False,
            "simulator_used": False,
        }
    finally:
        torch.distributed.destroy_process_group()


def worker_probe() -> None:
    """Launch one real Worker, preserving the explicit driver opt-in."""
    import ray
    import torch_musa  # noqa: F401

    from rlinf.scheduler import Cluster, PackedPlacementStrategy, Worker

    experimental_enabled = os.environ.get(OPT_IN) == "1"

    class FSDPProbeWorker(Worker):
        """A strategy-level probe, separate from the official embodied Actor."""

        def update(self) -> dict[str, Any]:
            """Run the scheduled update with real FSDP1 and CPU parity."""
            if experimental_enabled:
                os.environ[OPT_IN] = "1"
            return strategy_update(self._rank, self._world_size)

    try:
        cluster = Cluster(num_nodes=1)
        if cluster.num_accelerators < 1:
            raise RuntimeError("RLinf did not detect MUSA; refusing a CPU substitute")
        group = FSDPProbeWorker.create_group().launch(
            cluster=cluster,
            name="route2_fsdp1_strategy_probe",
            placement_strategy=PackedPlacementStrategy(0, 0),
        )
        for result in group.update().wait():
            emit("rlinf_fsdp1_strategy_update", "pass", **result)
    finally:
        ray.shutdown()


def main() -> None:
    """Select read-only imports or a coordinator-scheduled GPU update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=["imports", "strategy"], default="imports")
    parser.add_argument("--enable-torch22", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = (
        str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    if args.enable_torch22:
        os.environ[OPT_IN] = "1"
    emit("source", "info", phase=args.phase, **source_evidence(source))
    started = time.monotonic()
    try:
        if args.phase == "imports":
            imports()
        else:
            worker_probe()
    except Exception as error:  # noqa: BLE001 - preserve vendor integration failures
        emit(
            "phase",
            "blocked",
            phase=args.phase,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        sys.exit(1)
    finally:
        emit("elapsed", "info", seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
