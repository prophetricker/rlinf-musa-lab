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

"""Probe real MUSA FSDP1 checkpoint continuation across two independent commands.

The coordinator runs save to completion, then restore with the same format,
checkpoint directory, reference file and source. Each command creates a separate
driver and real RLinf Worker. The CPU reference artifact is an assertion oracle;
only the actual strategy checkpoint supplies restored training state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import checkpoint_probe as core

SCHEMA_VERSION = 1
VERIFIED_HELPER_SHA256 = (
    "da07c5591fe6d50a2eb8983cd706f117762720cfa4b37d44c8527820c8772a79"
)
SOURCE_FILES = [
    "rlinf/hybrid_engines/fsdp/__init__.py",
    "rlinf/hybrid_engines/fsdp/strategy/base.py",
    "rlinf/hybrid_engines/fsdp/strategy/checkpoint.py",
    "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
    "rlinf/hybrid_engines/fsdp/utils.py",
    "rlinf/utils/utils.py",
]


def configuration() -> dict[str, Any]:
    """Keep the tested FSDP1 settings without changing the verified helper."""
    return {
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
        "optim": {"clip_grad": core.CLIP_GRAD},
    }


def file_hash(path: Path) -> str:
    """Read file bytes in blocks so checkpoints do not require a second copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_evidence(source: Path) -> dict[str, Any]:
    """Record source/helper/probe/config hashes without invoking Git."""
    helper_hash = file_hash(Path(core.__file__).resolve())
    if helper_hash != VERIFIED_HELPER_SHA256:
        raise AssertionError("checkpoint_probe.py differs from the verified helper")
    config_bytes = json.dumps(
        configuration(), sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "source": str(source),
        "verified_checkpoint_commit_prefix": "c46875bb",
        "source_file_sha256": {name: file_hash(source / name) for name in SOURCE_FILES},
        "helper_sha256": helper_hash,
        "process_probe_sha256": file_hash(Path(__file__).resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def process_identity() -> dict[str, Any]:
    """Identify Linux processes by boot, PID and kernel start time, not PID alone."""
    pid = os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid": pid,
        "start_ticks": int(fields[19]),
        "clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK")),
    }


def same_process(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Distinguish a reused PID from the actual saved process."""
    return all(left[key] == right[key] for key in ["boot_id", "pid", "start_ticks"])


def assert_finished_driver(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Require save to have exited before another command starts restoration."""
    if (
        saved["hostname"] != current["hostname"]
        or saved["boot_id"] != current["boot_id"]
    ):
        raise AssertionError("This probe requires the same Linux host and boot")
    if same_process(saved, current):
        raise AssertionError("Save and restore must use different driver processes")
    try:
        fields = (
            Path(f"/proc/{saved['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        )
    except FileNotFoundError:
        return
    if int(fields[19]) == saved["start_ticks"] and fields[0] != "Z":
        raise AssertionError(
            "The save driver is still running; wait for its command to exit"
        )


def checkpoint_files(path: Path) -> list[dict[str, Any]]:
    """Inventory complete checkpoint bytes separately from the CPU oracle."""
    files = [
        {
            "path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": file_hash(file),
        }
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]
    if not files:
        raise AssertionError("The strategy did not create checkpoint files")
    return files


def manifest_path(reference_file: Path) -> Path:
    """Keep a completion manifest beside the separate CPU reference file."""
    return reference_file.with_name(reference_file.name + ".manifest.json")


def assert_cpu_tensors(value: Any) -> None:
    """Ensure the reference cannot reintroduce device tensors before real load."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise AssertionError("The reference artifact contains a non-CPU tensor")
    elif isinstance(value, dict):
        for item in value.values():
            assert_cpu_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_cpu_tensors(item)


def write_reference(reference_file: Path, reference: dict[str, Any]) -> dict[str, Any]:
    """Publish only a completed CPU oracle and completed strategy checkpoint."""
    import torch

    assert_cpu_tensors(reference)
    completion_file = manifest_path(reference_file)
    if reference_file.exists() or completion_file.exists():
        raise FileExistsError("Refusing to overwrite a reference artifact")
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = reference_file.with_name(reference_file.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        torch.save(reference, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if reference_file.exists():
        raise FileExistsError("The reference artifact appeared while saving")
    os.replace(temporary, reference_file)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "format": reference["format"],
        "reference_file": str(reference_file),
        "reference_sha256": file_hash(reference_file),
        "reference_bytes": reference_file.stat().st_size,
        "checkpoint_files": reference["checkpoint_files"],
        "provenance": reference["provenance"],
        "runtime": reference["runtime"],
        "save_driver": reference["save_driver"],
        "save_worker": reference["save_worker"],
    }
    temporary_manifest = completion_file.with_name(
        completion_file.name + f".tmp.{os.getpid()}"
    )
    with temporary_manifest.open("x", encoding="utf-8") as stream:
        json.dump(completion, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if completion_file.exists():
        raise FileExistsError("The completion manifest appeared while saving")
    os.replace(temporary_manifest, completion_file)
    return completion


def check_provenance(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require the same source implementation, helper, process probe and config."""
    for key in [
        "source_file_sha256",
        "helper_sha256",
        "process_probe_sha256",
        "config_sha256",
    ]:
        if actual[key] != expected[key]:
            raise AssertionError(f"Save/restore provenance differs: {key}")


def read_completion(
    checkpoint_format: str,
    checkpoint_dir: Path,
    reference_file: Path,
    provenance: dict[str, Any],
    driver: dict[str, Any],
) -> dict[str, Any]:
    """Validate finished output and a different driver before scheduling GPU work."""
    completion = json.loads(manifest_path(reference_file).read_text())
    if (
        completion["schema_version"] != SCHEMA_VERSION
        or completion["format"] != checkpoint_format
    ):
        raise AssertionError("Reference schema or checkpoint format differs")
    check_provenance(provenance, completion["provenance"])
    assert_finished_driver(completion["save_driver"], driver)
    if file_hash(reference_file) != completion["reference_sha256"]:
        raise AssertionError("The completed reference artifact has changed")
    if checkpoint_files(checkpoint_dir) != completion["checkpoint_files"]:
        raise AssertionError("Checkpoint files differ from the completed save")
    return completion


def runtime_evidence(device: Any) -> dict[str, Any]:
    """Constrain exact continuation to the matching runtime and device model."""
    import torch
    import torch_musa

    return {
        "torch_runtime_version": torch.__version__,
        "torch_musa_runtime_version": torch_musa.__version__,
        "distribution_versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torch_musa", "ray", "omegaconf"]
        },
        "device": str(device),
        "device_model": torch.musa.get_device_name(device.index),
        "world_size": 1,
        "precision": "fp32",
        "sharding_strategy": "NO_SHARD",
    }


class Stages:
    """Keep the concrete last stage when an exception returns from a Worker."""

    def __init__(self, phase: str, checkpoint_format: str):
        self.phase = phase
        self.checkpoint_format = checkpoint_format
        self.last_stage = "worker_start"

    def mark(self, name: str) -> None:
        """Emit before calling an operation that may fail or abort the Worker."""
        self.last_stage = name
        core.emit(
            "checkpoint_process_stage",
            "started",
            phase=self.phase,
            format=self.checkpoint_format,
            stage=name,
        )


def setup_device(rank: int, world_size: int) -> Any:
    """Require an actual single-rank MUSA Worker and choose its assigned device."""
    import torch
    import torch_musa  # noqa: F401

    from rlinf.scheduler import Worker

    if rank != 0 or world_size != 1 or Worker.torch_device_type != "musa":
        raise AssertionError("This probe requires one real MUSA Worker")
    device = torch.device(f"musa:{os.environ['LOCAL_RANK']}")
    torch.musa.set_device(device)
    return device


@contextmanager
def build_training_context(model: Any, device: Any, stages: Stages):
    """Use real MCCL, device mesh, FSDPStrategy and fresh optimizer/scheduler."""
    import torch
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel

    from rlinf.hybrid_engines.fsdp import FSDP
    from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import (
        create_device_mesh,
        gradient_reduction_group,
    )
    from rlinf.scheduler import Cluster

    if FSDP is not FullyShardedDataParallel:
        raise AssertionError("The RLinf FSDP export is not the real installed FSDP1")
    if torch.distributed.is_initialized():
        raise AssertionError("Expected an uninitialized default process group")
    stages.mark("mccl_default_group")
    torch.distributed.init_process_group(
        backend="mccl", rank=0, world_size=1, timeout=Cluster.get_collective_timeout()
    )
    try:
        stages.mark("real_mesh_factory_wrap")
        mesh = create_device_mesh(1)
        strategy = FSDPStrategyBase.create(
            OmegaConf.create(configuration()),
            world_size=1,
            dp_group=gradient_reduction_group(mesh),
        )
        if not isinstance(strategy, FSDPStrategy):
            raise AssertionError("The factory did not create real RLinf FSDPStrategy")
        wrapped = strategy.wrap_model(model.to(device), mesh)
        if not isinstance(wrapped, FSDP):
            raise AssertionError("The wrapped model is not real FSDP1")
        optimizer, scheduler = core.optimizer_and_scheduler(wrapped)
        yield strategy, wrapped, optimizer, scheduler
    finally:
        torch.distributed.destroy_process_group()


def save_phase(
    rank: int,
    world_size: int,
    checkpoint_format: str,
    checkpoint_dir: Path,
    reference_file: Path,
    driver: dict[str, Any],
    provenance: dict[str, Any],
    stages: Stages,
) -> dict[str, Any]:
    """Save after step 1 and persist a separate CPU oracle from continuous step 2."""
    import numpy as np
    import torch

    from rlinf.utils.utils import get_rng_state

    device = setup_device(rank, world_size)
    if (
        checkpoint_dir.exists()
        or reference_file.exists()
        or manifest_path(reference_file).exists()
    ):
        raise FileExistsError("Save requires new checkpoint and reference paths")
    checkpoint_dir.mkdir(parents=True)
    worker = process_identity()
    runtime = runtime_evidence(device)
    torch.manual_seed(42)
    torch.musa.manual_seed(43)
    np.random.seed(44)
    random.seed(45)
    cpu_reference = core.make_model()
    original_model = copy.deepcopy(cpu_reference)
    inputs = [torch.randn(8, 17), torch.randn(8, 17)]
    targets = [torch.randn(8, 6), torch.randn(8, 6)]
    cpu_optimizer, cpu_scheduler = core.optimizer_and_scheduler(cpu_reference)
    with build_training_context(original_model, device, stages) as context:
        strategy, model, optimizer, scheduler = context
        stages.mark("update1_active_clip")
        update1 = core.update(
            model, optimizer, scheduler, inputs[0], targets[0], device, strategy
        )
        core.update(
            cpu_reference,
            cpu_optimizer,
            cpu_scheduler,
            inputs[0],
            targets[0],
            torch.device("cpu"),
        )
        cpu_error1 = core.assert_tree(
            core.parameters(model), core.parameters(cpu_reference), exact=False
        )
        core.assert_adam_state(optimizer, step=1)
        saved = core.training_state(model, optimizer, scheduler)
        saved_rng = core.cpu_copy(get_rng_state())
        if set(saved_rng) != {"cpu", "numpy", "random", "musa"}:
            raise AssertionError("Missing one of the four checkpoint RNG streams")
        stages.mark("strategy_save_checkpoint")
        strategy.save_checkpoint(
            model,
            optimizer,
            scheduler,
            str(checkpoint_dir),
            save_full_model_weights=False,
            checkpoint_format=checkpoint_format,
        )
        core.assert_tree(core.training_state(model, optimizer, scheduler), saved)
        core.assert_tree(get_rng_state(), saved_rng)
        stages.mark("draw_expected_rng_samples")
        expected_samples = core.rng_samples()
        sampled_rng = core.cpu_copy(get_rng_state())
        stages.mark("uninterrupted_update2_active_clip")
        update2 = core.update(
            model, optimizer, scheduler, inputs[1], targets[1], device, strategy
        )
        core.update(
            cpu_reference,
            cpu_optimizer,
            cpu_scheduler,
            inputs[1],
            targets[1],
            torch.device("cpu"),
        )
        cpu_error2 = core.assert_tree(
            core.parameters(model), core.parameters(cpu_reference), exact=False
        )
        core.assert_adam_state(optimizer, step=2)
        reference = {
            "schema_version": SCHEMA_VERSION,
            "format": checkpoint_format,
            "provenance": provenance,
            "runtime": runtime,
            "save_driver": driver,
            "save_worker": worker,
            "checkpoint_files": checkpoint_files(checkpoint_dir),
            "saved_state": saved,
            "saved_rng": saved_rng,
            "next_inputs": core.cpu_copy(inputs[1]),
            "next_targets": core.cpu_copy(targets[1]),
            "expected_next_rng_samples": core.cpu_copy(expected_samples),
            "expected_sampled_rng": sampled_rng,
            "expected_update2_state": core.training_state(model, optimizer, scheduler),
            "expected_update2_rng": core.cpu_copy(get_rng_state()),
            "update1": update1,
            "expected_update2": update2,
            "cpu_update1_parameter_max_abs_error": cpu_error1,
            "cpu_update2_parameter_max_abs_error": cpu_error2,
        }
        stages.mark("publish_cpu_reference_and_completion")
        completion = write_reference(reference_file, reference)
        return {
            "phase": "save",
            "format": checkpoint_format,
            "checkpoint_dir": str(checkpoint_dir),
            "reference_file": str(reference_file),
            "reference_sha256": completion["reference_sha256"],
            "reference_bytes": completion["reference_bytes"],
            "completion_manifest_sha256": file_hash(manifest_path(reference_file)),
            "checkpoint_files": reference["checkpoint_files"],
            "save_driver": driver,
            "save_worker": worker,
            "runtime": runtime,
            "update1": update1,
            "uninterrupted_update2": update2,
            "cpu_update1_parameter_max_abs_error": cpu_error1,
            "cpu_update2_parameter_max_abs_error": cpu_error2,
            "checkpoint_and_reference_complete": True,
            "active_gradient_scaling": True,
            "fresh_process_resume_verified": False,
            "gpu_housekeeping_skipped": False,
        }


def restore_phase(
    rank: int,
    world_size: int,
    checkpoint_format: str,
    checkpoint_dir: Path,
    reference_file: Path,
    driver: dict[str, Any],
    provenance: dict[str, Any],
    stages: Stages,
) -> dict[str, Any]:
    """Read the CPU oracle before real load, then require exact independent resume."""
    import torch

    from rlinf.utils.utils import get_rng_state, warmup_optimizer_state

    stages.mark("read_complete_cpu_reference")
    completion = read_completion(
        checkpoint_format, checkpoint_dir, reference_file, provenance, driver
    )
    reference = torch.load(reference_file, map_location="cpu", weights_only=False)
    assert_cpu_tensors(reference)
    if (
        reference["schema_version"] != SCHEMA_VERSION
        or reference["format"] != checkpoint_format
    ):
        raise AssertionError("CPU reference schema or format differs")
    for key in [
        "provenance",
        "runtime",
        "save_driver",
        "save_worker",
        "checkpoint_files",
    ]:
        core.assert_tree(reference[key], completion[key])
    worker = process_identity()
    if worker["boot_id"] != reference["save_worker"]["boot_id"] or same_process(
        worker, reference["save_worker"]
    ):
        raise AssertionError(
            "Save and restore must use separate Worker processes on one boot"
        )
    device = setup_device(rank, world_size)
    runtime = runtime_evidence(device)
    core.assert_tree(runtime, reference["runtime"])
    # All reference deserialization and random object construction precede load.
    with build_training_context(core.make_model(), device, stages) as context:
        strategy, model, optimizer, scheduler = context
        if optimizer.state or any(
            value.grad is not None for value in model.parameters()
        ):
            raise AssertionError("Warmup requires fresh empty state and None gradients")
        before_warmup = core.parameters(model)
        stages.mark("warmup_fresh_optimizer")
        warmup_optimizer_state(optimizer)
        core.assert_tree(core.parameters(model), before_warmup)
        core.assert_adam_state(optimizer, step=0, zero_momentum=True)
        stages.mark("strategy_load_checkpoint")
        strategy.load_checkpoint(
            model,
            optimizer,
            scheduler,
            str(checkpoint_dir),
            checkpoint_format=checkpoint_format,
        )
        stages.mark("assert_step1_state_and_rng_exact")
        core.assert_tree(
            core.training_state(model, optimizer, scheduler), reference["saved_state"]
        )
        core.assert_tree(get_rng_state(), reference["saved_rng"])
        core.assert_adam_state(optimizer, step=1)
        stages.mark("assert_next_rng_samples_exact")
        core.assert_tree(core.rng_samples(), reference["expected_next_rng_samples"])
        core.assert_tree(get_rng_state(), reference["expected_sampled_rng"])
        stages.mark("restored_update2_active_clip")
        update2 = core.update(
            model,
            optimizer,
            scheduler,
            reference["next_inputs"],
            reference["next_targets"],
            device,
            strategy,
        )
        final = core.training_state(model, optimizer, scheduler)
        stages.mark("assert_continuation_state_and_rng_exact")
        expected = reference["expected_update2_state"]
        model_error = core.assert_tree(final["model"], expected["model"])
        optimizer_error = core.assert_tree(final["optimizer"], expected["optimizer"])
        core.assert_tree(final["scheduler"], expected["scheduler"])
        core.assert_tree(update2, reference["expected_update2"])
        core.assert_tree(get_rng_state(), reference["expected_update2_rng"])
        core.assert_adam_state(optimizer, step=2)
        if checkpoint_files(checkpoint_dir) != reference["checkpoint_files"]:
            raise AssertionError("Checkpoint files changed during restoration")
        return {
            "phase": "restore",
            "format": checkpoint_format,
            "checkpoint_dir": str(checkpoint_dir),
            "reference_file": str(reference_file),
            "reference_sha256": completion["reference_sha256"],
            "completion_manifest_sha256": file_hash(manifest_path(reference_file)),
            "checkpoint_files": reference["checkpoint_files"],
            "save_driver": reference["save_driver"],
            "restore_driver": driver,
            "save_worker": reference["save_worker"],
            "restore_worker": worker,
            "runtime": runtime,
            "backend": str(torch.distributed.get_backend()),
            "fresh_objects": True,
            "fresh_process": True,
            "save_driver_finished": True,
            "driver_and_worker_process_identity_differ": True,
            "warmup_parameter_no_op": True,
            "warmup_step_zero": True,
            "active_gradient_scaling": True,
            "clip_grad": core.CLIP_GRAD,
            "loss_scale": core.LOSS_SCALE,
            "restored_step1_state_exact": True,
            "continued_model_optimizer_scheduler_exact": True,
            "continued_model_max_abs_error": model_error,
            "continued_optimizer_max_abs_error": optimizer_error,
            "uninterrupted_update2": reference["expected_update2"],
            "restored_update2": update2,
            "rng_streams": ["python", "numpy", "torch_cpu", "musa"],
            "rng_states_samples_and_continuation_exact": True,
            "gpu_housekeeping_skipped": False,
            "full_model_weights_exported": False,
            "official_actor_used": False,
            "offload_executed": False,
            "ppo_executed": False,
            "simulator_used": False,
        }


def worker_probe(
    phase: str,
    checkpoint_format: str,
    checkpoint_dir: Path,
    reference_file: Path,
    source: Path,
    driver: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Run one phase in one real Worker; return concrete failures to the driver."""
    import ray
    import torch_musa  # noqa: F401

    from rlinf.scheduler import Cluster, PackedPlacementStrategy, Worker

    experimental_enabled = os.environ.get(core.OPT_IN) == "1"

    class CheckpointProcessWorker(Worker):
        """Create no shared model state between the two commands."""

        def run(self) -> dict[str, Any]:
            """Return stage and traceback on failure without reporting success."""
            stages = Stages(phase, checkpoint_format)
            try:
                if experimental_enabled:
                    os.environ[core.OPT_IN] = "1"
                actual_source = source_evidence(source)
                check_provenance(actual_source, provenance)
                operation = save_phase if phase == "save" else restore_phase
                return {
                    "status": "pass",
                    **operation(
                        self._rank,
                        self._world_size,
                        checkpoint_format,
                        checkpoint_dir,
                        reference_file,
                        driver,
                        actual_source,
                        stages,
                    ),
                }
            except Exception as error:  # noqa: BLE001 - preserve actual runtime failure
                return {
                    "status": "blocked",
                    "phase": phase,
                    "format": checkpoint_format,
                    "last_stage": stages.last_stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }

    try:
        cluster = Cluster(num_nodes=1)
        if cluster.num_accelerators < 1:
            raise RuntimeError("RLinf did not detect MUSA; refusing a CPU substitute")
        group = CheckpointProcessWorker.create_group().launch(
            cluster=cluster,
            name=f"route2_checkpoint_process_{checkpoint_format}_{phase}",
            placement_strategy=PackedPlacementStrategy(0, 0),
        )
        for result in group.run().wait():
            status = result.pop("status")
            core.emit("rlinf_fsdp1_checkpoint_process", status, **result)
            if status != "pass":
                raise RuntimeError(
                    f"Worker failed during {result['last_stage']}: {result['error']}"
                )
    finally:
        ray.shutdown()


def main() -> None:
    """Require a separate explicit save or restore command for each format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=["save", "restore"], required=True)
    parser.add_argument("--format", choices=["local_shard", "dcp"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--enable-torch22", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    reference_file = args.reference_file.resolve()
    if checkpoint_dir == reference_file or checkpoint_dir in reference_file.parents:
        parser.error("--reference-file must be outside --checkpoint-dir")
    probe_directory = Path(__file__).resolve().parent
    sys.path[:0] = [str(source), str(probe_directory)]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(probe_directory), os.environ.get("PYTHONPATH", "")]
    )
    if args.enable_torch22:
        os.environ[core.OPT_IN] = "1"
    started = time.monotonic()
    try:
        provenance = source_evidence(source)
        driver = process_identity()
        core.emit(
            "source",
            "info",
            phase=args.phase,
            format=args.format,
            driver=driver,
            **provenance,
        )
        if args.phase == "restore":
            read_completion(
                args.format, checkpoint_dir, reference_file, provenance, driver
            )
        elif (
            checkpoint_dir.exists()
            or reference_file.exists()
            or manifest_path(reference_file).exists()
        ):
            raise FileExistsError("Save requires new checkpoint and reference paths")
        worker_probe(
            args.phase,
            args.format,
            checkpoint_dir,
            reference_file,
            source,
            driver,
            provenance,
        )
    except Exception as error:  # noqa: BLE001 - report setup and child-stage failures
        core.emit(
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
        core.emit("elapsed", "info", seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
