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

"""Probe real RLinf FSDP1 checkpoint restoration into new training objects.

Run API imports without a Worker first. The coordinator then schedules one
format per invocation: local_shard before dcp. The GPU phase creates a real
single-rank MUSA FSDPStrategy, saves after update 1, rebuilds FSDP/AdamW/StepLR,
and compares restored states/RNG and update 2 with uninterrupted training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BASELINE_COMMIT = "f9b74d95f97ad4234311580ab32c7b586e1dc7af"
UPSTREAM_COMMIT = "c70606f08cdca259b8dec03d4430926b5b8fac9d"
OPT_IN = "RLINF_EXPERIMENTAL_FSDP1_TORCH22"
CLIP_GRAD = 0.25
LOSS_SCALE = 16.0
RTOL = 3e-4
ATOL = 3e-6


def emit(probe: str, status: str, **details: Any) -> None:
    """Write a compact evidence record that survives later Worker failures."""
    print(json.dumps({"probe": probe, "status": status, **details}), flush=True)


def stage(name: str, checkpoint_format: str) -> None:
    """Identify the next operation before entering real runtime APIs."""
    emit("checkpoint_stage", "started", stage=name, format=checkpoint_format)


def source_evidence(source: Path) -> dict[str, Any]:
    """Hash the actual implementation independently of expected Git refs."""
    files = [
        "rlinf/hybrid_engines/fsdp/__init__.py",
        "rlinf/hybrid_engines/fsdp/strategy/base.py",
        "rlinf/hybrid_engines/fsdp/strategy/checkpoint.py",
        "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
        "rlinf/hybrid_engines/fsdp/utils.py",
        "rlinf/utils/utils.py",
    ]
    result = {
        "source": str(source),
        "expected_baseline_commit": BASELINE_COMMIT,
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "source_file_sha256": {
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
    except (OSError, subprocess.CalledProcessError):
        result["actual_head_commit"] = None
    return result


def imports() -> None:
    """Read installed DCP signatures without process groups or GPU tensors."""
    import torch
    import torch.distributed.checkpoint as dcp
    import torch_musa  # noqa: F401
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy

    emit(
        "checkpoint_imports",
        "pass",
        torch_runtime_version=torch.__version__,
        distribution_versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "torch_musa", "ray", "omegaconf"]
        },
        strategy_source=inspect.getfile(FSDPStrategy),
        stateful_checkpoint_source=inspect.getfile(Checkpoint),
        signatures={
            "dcp.save": str(inspect.signature(dcp.save)),
            "dcp.load": str(inspect.signature(dcp.load)),
            "FileSystemWriter": str(inspect.signature(dcp.FileSystemWriter)),
            "StateDictOptions": str(inspect.signature(StateDictOptions)),
            "get_state_dict": str(inspect.signature(get_state_dict)),
            "set_state_dict": str(inspect.signature(set_state_dict)),
        },
        checkpoint_executed=False,
    )


def cpu_copy(value: Any) -> Any:
    """Detach every tensor so expected states cannot alias live training state."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def assert_tree(actual: Any, expected: Any, exact: bool = True) -> float:
    """Compare complete state structure, tensor finiteness and scalar values."""
    import numpy as np
    import torch

    if isinstance(expected, torch.Tensor):
        actual = actual.detach().cpu()
        if actual.dtype.is_floating_point and not bool(torch.isfinite(actual).all()):
            raise AssertionError("A state tensor contains non-finite values")
        torch.testing.assert_close(
            actual, expected, rtol=0 if exact else RTOL, atol=0 if exact else ATOL
        )
        if not actual.numel():
            return 0.0
        return float(
            (actual.to(torch.float64) - expected.to(torch.float64)).abs().max()
        )
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return 0.0
    if isinstance(expected, Mapping):
        if actual.keys() != expected.keys():
            raise AssertionError("Checkpoint mapping keys differ")
        return max(
            [assert_tree(actual[key], expected[key], exact) for key in expected]
            or [0.0]
        )
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("Checkpoint sequence type/length differs")
        return max(
            [
                assert_tree(actual[index], expected[index], exact)
                for index in range(len(expected))
            ]
            or [0.0]
        )
    if actual != expected:
        raise AssertionError(f"Checkpoint scalar differs: {actual!r} != {expected!r}")
    return 0.0


def parameters(model: Any, gradients: bool = False) -> dict[str, Any]:
    """Keep canonical names for new FSDP wrappers and the CPU reference."""
    result = {}
    for name, param in model.named_parameters():
        value = param.grad if gradients else param
        if value is None:
            raise AssertionError(f"Missing gradient for {name}")
        result[name.removeprefix("_fsdp_wrapped_module.")] = cpu_copy(value)
    return result


def training_state(model: Any, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    """Snapshot model parameters, every Adam state and scheduler/param groups."""
    return {
        "model": parameters(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "scheduler": cpu_copy(scheduler.state_dict()),
    }


def make_model() -> Any:
    """Build a new tiny model with four FP32 parameter tensors."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(17, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6)
    )


def optimizer_and_scheduler(model: Any) -> tuple[Any, Any]:
    """Create distinct AdamW and StepLR objects for each training branch."""
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler


def update(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    inputs: Any,
    targets: Any,
    device: Any,
    strategy: Any = None,
) -> dict[str, float]:
    """Require active clipping, execute AdamW and advance the real scheduler."""
    from contextlib import nullcontext

    import torch

    optimizer.zero_grad(set_to_none=True)
    context = (
        strategy.before_micro_batch(model, is_last_micro_batch=True)
        if strategy is not None
        else nullcontext()
    )
    with context:
        loss = (
            LOSS_SCALE * (model(inputs.to(device)) - targets.to(device)).square().mean()
        )
        loss.backward()
    raw = parameters(model, gradients=True)
    if not all(bool(torch.isfinite(value).all()) for value in raw.values()):
        raise AssertionError("Non-finite gradient before clipping")
    norm = float(
        strategy.clip_grad_norm_(model)
        if strategy is not None
        else torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    )
    clipped = parameters(model, gradients=True)
    clipped_norm = math.sqrt(
        sum(float(value.square().sum()) for value in clipped.values())
    )
    if not math.isfinite(norm) or norm <= CLIP_GRAD:
        raise AssertionError("The chosen batch did not require actual gradient scaling")
    if clipped_norm > CLIP_GRAD + ATOL:
        raise AssertionError("Gradient norm exceeds the configured clipping bound")
    if not any(bool((raw[name] != clipped[name]).any()) for name in raw):
        raise AssertionError("clip_grad_norm_ did not actively scale any gradient")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "musa":
        torch.musa.synchronize()
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_norm": norm,
        "clipped_gradient_norm": clipped_norm,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def assert_adam_state(optimizer: Any, step: int, zero_momentum: bool = False) -> None:
    """Require all four Adam counters and both momentum tensors to be present."""
    state = optimizer.state_dict()["state"]
    if len(state) != 4:
        raise AssertionError("Adam state did not cover all four parameter tensors")
    for value in state.values():
        if float(value["step"]) != step:
            raise AssertionError(
                "Adam step counter differs from expected training step"
            )
        for key in ["exp_avg", "exp_avg_sq"]:
            if bool(value[key].eq(0).all()) != zero_momentum:
                raise AssertionError(f"Unexpected Adam momentum contents: {key}")


def rng_samples() -> dict[str, Any]:
    """Draw each actual RNG stream without comparing different device streams."""
    import numpy as np
    import torch

    return {
        "python": [random.random() for _ in range(4)],
        "numpy": np.random.rand(4),
        "torch_cpu": torch.rand(4),
        "musa": torch.rand(4, device="musa").cpu(),
    }


def checkpoint_update(
    rank: int, world_size: int, checkpoint_format: str, checkpoint_dir: str
) -> dict[str, Any]:
    """Save actual strategy state and restore into independently built objects."""
    import numpy as np
    import torch
    import torch_musa  # noqa: F401
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel

    from rlinf.hybrid_engines.fsdp import FSDP
    from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
    from rlinf.hybrid_engines.fsdp.utils import (
        create_device_mesh,
        gradient_reduction_group,
    )
    from rlinf.scheduler import Cluster, Worker
    from rlinf.utils.utils import get_rng_state, warmup_optimizer_state

    if FSDP is not FullyShardedDataParallel:
        raise AssertionError("The RLinf FSDP export is not the real FSDP1 type")
    if rank != 0 or world_size != 1 or Worker.torch_device_type != "musa":
        raise AssertionError("This phase requires one real MUSA Worker")
    path = Path(checkpoint_dir)
    if path.exists():
        raise FileExistsError("Refusing to reuse a checkpoint directory")
    path.mkdir(parents=True)
    device = torch.device(f"musa:{os.environ['LOCAL_RANK']}")
    torch.musa.set_device(device)
    torch.manual_seed(42)
    torch.musa.manual_seed(43)
    np.random.seed(44)
    random.seed(45)
    reference = make_model()
    original_model = copy.deepcopy(reference).to(device)
    inputs = [torch.randn(8, 17), torch.randn(8, 17)]
    targets = [torch.randn(8, 6), torch.randn(8, 6)]
    reference_optimizer, reference_scheduler = optimizer_and_scheduler(reference)
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
            "optim": {"clip_grad": CLIP_GRAD},
        }
    )
    if torch.distributed.is_initialized():
        raise AssertionError("Expected an uninitialized default process group")
    timeout = Cluster.get_collective_timeout()
    stage("mccl_default_group", checkpoint_format)
    torch.distributed.init_process_group(
        backend="mccl", rank=rank, world_size=world_size, timeout=timeout
    )
    try:
        mesh = create_device_mesh(world_size)
        strategy = FSDPStrategyBase.create(
            config, world_size=world_size, dp_group=gradient_reduction_group(mesh)
        )
        stage("wrap_original", checkpoint_format)
        original = strategy.wrap_model(original_model, mesh)
        if not isinstance(original, FSDP):
            raise AssertionError("The original model is not real FSDP1")
        optimizer, scheduler = optimizer_and_scheduler(original)
        stage("update1_active_clip", checkpoint_format)
        update1 = update(
            original, optimizer, scheduler, inputs[0], targets[0], device, strategy
        )
        update(
            reference,
            reference_optimizer,
            reference_scheduler,
            inputs[0],
            targets[0],
            torch.device("cpu"),
        )
        cpu_update1_error = assert_tree(
            parameters(original), parameters(reference), exact=False
        )
        assert_adam_state(optimizer, step=1)
        saved = training_state(original, optimizer, scheduler)
        saved_rng = cpu_copy(get_rng_state())
        if set(saved_rng) != {"cpu", "numpy", "random", "musa"}:
            raise AssertionError("Checkpoint RNG coverage is incomplete")
        stage("strategy_save_checkpoint", checkpoint_format)
        strategy.save_checkpoint(
            original,
            optimizer,
            scheduler,
            str(path),
            save_full_model_weights=False,
            checkpoint_format=checkpoint_format,
        )
        assert_tree(training_state(original, optimizer, scheduler), saved)
        assert_tree(get_rng_state(), saved_rng)
        expected_samples = rng_samples()
        stage("uninterrupted_update2", checkpoint_format)
        continuous = update(
            original, optimizer, scheduler, inputs[1], targets[1], device, strategy
        )
        update(
            reference,
            reference_optimizer,
            reference_scheduler,
            inputs[1],
            targets[1],
            torch.device("cpu"),
        )
        cpu_update2_error = assert_tree(
            parameters(original), parameters(reference), exact=False
        )
        expected = training_state(original, optimizer, scheduler)
        expected_end_rng = cpu_copy(get_rng_state())
        assert_adam_state(optimizer, step=2)

        stage("rebuild_new_fsdp_optimizer_scheduler", checkpoint_format)
        restored_strategy = FSDPStrategyBase.create(
            config, world_size=world_size, dp_group=gradient_reduction_group(mesh)
        )
        restored = restored_strategy.wrap_model(make_model().to(device), mesh)
        if not isinstance(restored, FSDP) or restored is original:
            raise AssertionError("The restored model was not a new FSDP object")
        if not {id(value) for value in restored.parameters()}.isdisjoint(
            {id(value) for value in original.parameters()}
        ):
            raise AssertionError(
                "The restored FSDP model shares live parameter objects"
            )
        restored_optimizer, restored_scheduler = optimizer_and_scheduler(restored)
        if restored_optimizer.state or any(
            value.grad is not None for value in restored.parameters()
        ):
            raise AssertionError(
                "Optimizer warmup requires fresh state and None gradients"
            )
        before_warmup = parameters(restored)
        stage("warmup_fresh_optimizer", checkpoint_format)
        warmup_optimizer_state(restored_optimizer)
        assert_tree(parameters(restored), before_warmup)
        assert_adam_state(restored_optimizer, step=0, zero_momentum=True)
        stage("strategy_load_checkpoint", checkpoint_format)
        restored_strategy.load_checkpoint(
            restored,
            restored_optimizer,
            restored_scheduler,
            str(path),
            checkpoint_format=checkpoint_format,
        )
        assert_tree(
            training_state(restored, restored_optimizer, restored_scheduler), saved
        )
        assert_tree(get_rng_state(), saved_rng)
        assert_adam_state(restored_optimizer, step=1)
        assert_tree(rng_samples(), expected_samples)
        stage("restored_update2_active_clip", checkpoint_format)
        resumed = update(
            restored,
            restored_optimizer,
            restored_scheduler,
            inputs[1],
            targets[1],
            device,
            restored_strategy,
        )
        final = training_state(restored, restored_optimizer, restored_scheduler)
        model_error = assert_tree(final["model"], expected["model"])
        optimizer_error = assert_tree(final["optimizer"], expected["optimizer"])
        assert_tree(final["scheduler"], expected["scheduler"])
        assert_tree(resumed, continuous)
        assert_tree(get_rng_state(), expected_end_rng)
        assert_adam_state(restored_optimizer, step=2)
        files = [
            {"path": str(file.relative_to(path)), "bytes": file.stat().st_size}
            for file in sorted(path.rglob("*"))
            if file.is_file()
        ]
        if not files:
            raise AssertionError("No checkpoint files were created")
        return {
            "format": checkpoint_format,
            "checkpoint_dir": str(path),
            "files": files,
            "device": str(device),
            "backend": str(torch.distributed.get_backend()),
            "world_size": world_size,
            "rank": rank,
            "strategy_class": f"{type(strategy).__module__}.{type(strategy).__name__}",
            "precision": "fp32",
            "sharding_strategy": "NO_SHARD",
            "fresh_objects": True,
            "fresh_process": False,
            "warmup_parameter_no_op": True,
            "warmup_step_zero": True,
            "active_gradient_scaling": True,
            "clip_grad": CLIP_GRAD,
            "loss_scale": LOSS_SCALE,
            "update1": update1,
            "uninterrupted_update2": continuous,
            "restored_update2": resumed,
            "cpu_update1_parameter_max_abs_error": cpu_update1_error,
            "cpu_update2_parameter_max_abs_error": cpu_update2_error,
            "continued_model_max_abs_error": model_error,
            "continued_optimizer_max_abs_error": optimizer_error,
            "restored_model_optimizer_scheduler_exact": True,
            "continued_model_optimizer_scheduler_exact": True,
            "rng_streams": ["python", "numpy", "torch_cpu", "musa"],
            "rng_state_and_next_samples_exact": True,
            "cpu_reference_rtol": RTOL,
            "cpu_reference_atol": ATOL,
            "gpu_housekeeping_skipped": False,
            "full_model_weights_exported": False,
            "official_actor_used": False,
            "offload_executed": False,
            "ppo_executed": False,
            "simulator_used": False,
        }
    finally:
        torch.distributed.destroy_process_group()


def worker_probe(checkpoint_format: str, checkpoint_dir: Path) -> None:
    """Schedule only one real MUSA Worker and one checkpoint format."""
    import ray
    import torch_musa  # noqa: F401

    from rlinf.scheduler import Cluster, PackedPlacementStrategy, Worker

    experimental_enabled = os.environ.get(OPT_IN) == "1"

    class CheckpointProbeWorker(Worker):
        """A backend checkpoint probe, separate from the official Actor."""

        def run(self) -> dict[str, Any]:
            """Return evidence after all checkpoint comparisons pass."""
            if experimental_enabled:
                os.environ[OPT_IN] = "1"
            return checkpoint_update(
                self._rank, self._world_size, checkpoint_format, str(checkpoint_dir)
            )

    try:
        cluster = Cluster(num_nodes=1)
        if cluster.num_accelerators < 1:
            raise RuntimeError("RLinf did not detect MUSA; refusing a CPU substitute")
        group = CheckpointProbeWorker.create_group().launch(
            cluster=cluster,
            name=f"route2_fsdp1_checkpoint_{checkpoint_format}",
            placement_strategy=PackedPlacementStrategy(0, 0),
        )
        for result in group.run().wait():
            emit("rlinf_fsdp1_checkpoint", "pass", **result)
    finally:
        ray.shutdown()


def main() -> None:
    """Dispatch read-only imports or an explicitly scheduled checkpoint run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=["imports", "checkpoint"], default="imports")
    parser.add_argument("--format", choices=["local_shard", "dcp"])
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--enable-torch22", action="store_true")
    args = parser.parse_args()
    if args.phase == "checkpoint" and (
        args.format is None or args.checkpoint_dir is None
    ):
        parser.error("--format and --checkpoint-dir are required for checkpoint")
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = (
        str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    if args.enable_torch22:
        os.environ[OPT_IN] = "1"
    emit(
        "source",
        "info",
        phase=args.phase,
        format=args.format,
        **source_evidence(source),
    )
    started = time.monotonic()
    try:
        if args.phase == "imports":
            imports()
        else:
            worker_probe(args.format, args.checkpoint_dir.resolve())
    except Exception as error:  # noqa: BLE001 - retain concrete vendor failures
        emit(
            "phase",
            "blocked",
            phase=args.phase,
            format=args.format,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        sys.exit(1)
    finally:
        emit("elapsed", "info", seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
