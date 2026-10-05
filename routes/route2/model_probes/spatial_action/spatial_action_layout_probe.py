#!/usr/bin/env python3
# Copyright 2026 RLinf research contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check fixed Spatial action geometry with real Attention and two-layer DiT.

Random shared quantized weights; native CPU -> FP32-accumulation CPU -> optional
MUSA. No model downloads, optimizer, global patch or invented DiT mask routing.
Strict and analytic-zero semantic gates are reported independently. Exit 0
requires the strict gate, all contracts and no runtime error.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

SUPPORT_HASHES = {
    "attention_probe.py": "704b68ef1e7bcde4b0578519f6d9c2c0a51298fd8a5e27527754718ce693b775",
    "fp32_attention_processor.py": "0edcb656696bcd3a47e014c3d549cf029a268c4a2f3f9694bea9993bd9d53a6f",
    "gr00t_action_interface_probe.py": "c1a0e36336cbefc5499ffa8bb7c1f38bfe2384f64505c55f7a976d8dad3645f4",
}
CONFIG_HASH = "6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526"
DIT_HASH = "1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62"
ATTENTION_HASH = "afe47ff6d33c865af745e28eed4bd8048b66fd52de7c2b3bad37867e826776c9"
HF_REVISION = "73f710e70e7d571f8d828e51e0a428f5a1e0ac22"
GR00T_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guarded(path, expected):
    observed = file_hash(path)
    if observed != expected:
        raise RuntimeError(
            f"Hash mismatch: {path}: expected {expected}, observed {observed}"
        )
    return {"path": str(Path(path).resolve()), "sha256": observed}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_support(directory):
    sources = {
        name: guarded(directory / name, sha) for name, sha in SUPPORT_HASHES.items()
    }
    # Load exact dependencies before the old helper's ordinary imports. Do not
    # depend on cwd, import search order or an unversioned copy of these files.
    import_file("attention_probe", directory / "attention_probe.py")
    import_file("fp32_attention_processor", directory / "fp32_attention_processor.py")
    helper = import_file(
        "spatial_pinned_action_helper", directory / "gr00t_action_interface_probe.py"
    )
    return helper, sources


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate config key: {key}")
        result[key] = value
    return result


def json_safe(value):
    """Serialize real constructor dtype metadata without changing numerics."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "dtype":
        return str(value)
    return value


def spatial_config(path):
    evidence = guarded(path, CONFIG_HASH)
    config = json.loads(path.read_text(), object_pairs_hook=unique_object)
    head = config["action_head_cfg"]
    diffusion = head["diffusion_model_cfg"]
    expected = {
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "cross_attention_dim": 2048,
        "output_dim": 1024,
        "num_layers": 16,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": True,
        "positional_embeddings": None,
    }
    if any(diffusion[key] != value for key, value in expected.items()):
        raise ValueError(
            "Pinned Spatial diffusion geometry differs from declared contract"
        )
    if (
        head["input_embedding_dim"],
        head["backbone_embedding_dim"],
        head["action_horizon"],
        head["num_target_vision_tokens"],
    ) != (1536, 2048, 16, 32):
        raise ValueError("Pinned Spatial action layout differs from declared contract")
    vl = head["vl_self_attention_cfg"]
    if (vl["num_attention_heads"], vl["attention_head_dim"]) != (32, 64):
        raise ValueError("Pinned VL attention geometry differs from declared contract")
    return config, {
        **evidence,
        "hf_revision": HF_REVISION,
        "checkpoint_weights_loaded": False,
    }


def cases_from_config(config):
    head = config["action_head_cfg"]
    diffusion = head["diffusion_model_cfg"]
    action = {
        "query_dim": head["input_embedding_dim"],
        "heads": diffusion["num_attention_heads"],
        "dim_head": diffusion["attention_head_dim"],
        "dropout": 0.0,
        "bias": True,
        "out_bias": True,
        "scale_qk": True,
    }
    cross = {**action, "cross_attention_dim": head["backbone_embedding_dim"]}
    vl = {**action, "query_dim": 2048, "dim_head": 64}
    dit = {**diffusion, "num_layers": 2, "dropout": 0.0, "final_dropout": False}
    cases = [
        {"name": "action_self", "config": action, "mask": None},
        {"name": "action_cross", "config": cross, "mask": None},
        {"name": "action_cross_rightpad_bool", "config": cross, "mask": "bool"},
        {"name": "action_cross_rightpad_additive", "config": cross, "mask": "additive"},
        {
            "name": "action_cross_rightpad_4d_bool",
            "config": cross,
            "mask": "bool",
            "mask4d": True,
        },
        {
            "name": "action_cross_rightpad_4d_additive",
            "config": cross,
            "mask": "additive",
            "mask4d": True,
        },
        {
            "name": "action_norms_4d_rightpad_additive",
            "mask": "additive",
            "mask4d": True,
            "spatial": True,
            "config": {
                **cross,
                "norm_num_groups": 32,
                "spatial_norm_dim": 32,
                "cross_attention_norm": "layer_norm",
                "qk_norm": "layer_norm",
                "residual_connection": True,
                "rescale_output_factor": 1.5,
            },
        },
        {"name": "vl_self", "config": vl, "mask": None, "vl": True},
        {
            "name": "vl_self_rightpad_4d_bool",
            "config": vl,
            "mask": "bool",
            "mask4d": True,
            "vl": True,
        },
        {
            "name": "gr00t_spatial_dit_two_layers",
            "config": dit,
            "mask": None,
            "phase": "dit",
        },
    ]
    for case in cases:
        case.setdefault("phase", "attention")
        case["seed_offset"] = (
            2
            if case.get("spatial")
            else 3
            if case.get("vl")
            else 4
            if case["phase"] == "dit"
            else 0
            if case["name"] == "action_self"
            else 1
        )
    return cases


def make_data(torch, case, dtype, seed, lengths):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = len(lengths)
    query = 570 if case.get("vl") else 49
    width = 2048 if case.get("vl") else 1536

    def random(shape):
        return (torch.randn(shape, generator=gen) * 0.5).to(dtype)

    # Explicitly retain state->future->action concatenation order. These are
    # synthetic already-embedded leaves, not the official encoders/Embedding.
    if case.get("vl"):
        hidden = random((batch, query, width))
    else:
        hidden = torch.cat(
            [
                random((batch, 1, width)),
                random((batch, 32, width)),
                random((batch, 16, width)),
            ],
            dim=1,
        )
    if case.get("spatial"):
        hidden = hidden.transpose(1, 2).reshape(batch, width, 1, query).contiguous()
    inputs = {"hidden": hidden}
    if case["phase"] == "dit" or case["config"].get("cross_attention_dim"):
        inputs["encoder"] = random((batch, 570, 2048))
    if case.get("spatial"):
        inputs["temb"] = random((batch, 32, 1, 7))
    # Real right padding from an unpadded all-ones token mask; lengths are a
    # declared representative fixture, not a measured LIBERO sample.
    allowed = torch.stack(
        [
            torch.nn.functional.pad(
                torch.ones(length, dtype=torch.bool), (0, 570 - length), value=False
            )
            for length in lengths
        ]
    )
    mask = None
    if case["mask"]:
        mask = allowed[:, None, :]
        if case.get("mask4d"):
            # Source prepare_attention_mask repeats on dim0 if dim0 < B*H.
            # Thus raw [B,H,Q,K] is NOT its accepted entry representation.
            # [B*H,1,Q,K] survives prepare and its real view gives [B,H,Q,K].
            mask = (
                mask[:, None].expand(batch, case["config"]["heads"], query, 570).clone()
            )
            # A separate head/query-varying contract fixture, on top of right
            # padding, verifies that no head/query axis was flattened wrongly.
            head_index = torch.arange(case["config"]["heads"])[None, :, None, None]
            query_index = torch.arange(query)[None, None, :, None]
            key_index = torch.arange(570)[None, None, None, :]
            slot = (head_index + query_index) % max(1, min(lengths) - 1) + 1
            mask = mask & (key_index != slot)
            mask = mask.reshape(
                batch * case["config"]["heads"], 1, query, 570
            ).contiguous()
        if case["mask"] == "additive":
            mask = torch.zeros(mask.shape, dtype=dtype).masked_fill(
                ~mask, float("-inf")
            )
    out_shape = (batch, 49, 1024) if case["phase"] == "dit" else tuple(hidden.shape)
    upstream = torch.randn(out_shape, generator=gen).to(dtype) / math.sqrt(
        math.prod(out_shape[1:])
    )
    if case.get("vl") and mask is not None:
        # Padded queries can still have outputs. The independent masked VL
        # contract asserts only visible queries; no global zero-query policy.
        upstream = upstream * allowed[:, :, None].to(dtype)
    timestep = torch.tensor([3 + 8 * i for i in range(batch)], dtype=torch.int64)
    return inputs, mask, allowed, timestep, upstream


def execute(
    torch,
    helper,
    template,
    inputs,
    mask,
    timestep,
    upstream,
    case,
    kind,
    device,
    Attention,
    native_type,
    stage,
    dit_mask=None,
):
    stage("clone_and_transfer")
    model = copy.deepcopy(template).to(device).eval()
    traces = []
    bindings = helper.bind_processors(model, Attention, native_type, kind, traces)
    xs = {
        name: value.to(device).detach().requires_grad_()
        for name, value in inputs.items()
    }
    local_mask = None if mask is None else mask.to(device)
    local_dit_mask = None if dit_mask is None else dit_mask.to(device)
    stage("forward")
    if case["phase"] == "dit":
        output = model(
            hidden_states=xs["hidden"],
            encoder_hidden_states=xs["encoder"],
            timestep=timestep.to(device),
            encoder_attention_mask=local_dit_mask,
        )
    else:
        output = model(
            xs["hidden"],
            encoder_hidden_states=xs.get("encoder"),
            attention_mask=local_mask,
            temb=xs.get("temb"),
        )
    stage("input_and_all_parameter_vjp")
    (output * upstream.to(device)).sum().backward()
    helper.synchronize(torch, device)
    stage("snapshot")
    input_grads = {
        name: None if x.grad is None else x.grad.detach().cpu().double()
        for name, x in xs.items()
    }
    parameter_grads = {
        name: None if x.grad is None else x.grad.detach().cpu().double()
        for name, x in model.named_parameters()
        if x.requires_grad
    }
    cpu_output = output.detach().cpu().double()
    diagnostics = helper.analytic_zero_diagnostics(torch, model, Attention)
    finite = all(
        x is not None and bool(torch.isfinite(x).all())
        for x in [cpu_output, *input_grads.values(), *parameter_grads.values()]
    )
    result = {
        "output": cpu_output,
        "input_grads": input_grads,
        "parameter_grads": parameter_grads,
        "metadata": {
            "finite": finite,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "missing_gradients": {
                "inputs": [n for n, g in input_grads.items() if g is None],
                "parameters": [n for n, g in parameter_grads.items() if g is None],
            },
            "processor_bindings": bindings,
            "kernel_traces": traces,
            "analytic_zero_parameter_diagnostics": diagnostics,
            "state_sha256": {
                n: helper.tensor_hash(torch, x) for n, x in model.state_dict().items()
            },
            "parameter_runtime": {
                n: {
                    "device": str(x.device),
                    "dtype": str(x.dtype),
                    "requires_grad": x.requires_grad,
                    "shape": list(x.shape),
                }
                for n, x in model.named_parameters()
            },
        },
    }
    del model, xs, output
    gc.collect()
    return result


def exact_snapshot(torch, actual, expected, valid_queries=None):
    a, b = actual["output"], expected["output"]
    if valid_queries is not None:
        a, b = a[valid_queries], b[valid_queries]
    fields = {"output": bool(torch.equal(a, b))}
    for group in ("input_grads", "parameter_grads"):
        fields[group + "_names_match"] = set(actual[group]) == set(expected[group])
        fields[group + "_all_exact"] = fields[group + "_names_match"] and all(
            actual[group][n] is not None
            and expected[group][n] is not None
            and bool(torch.equal(actual[group][n], expected[group][n]))
            for n in actual[group]
        )
    return {**fields, "pass": all(fields.values())}


def padding_contract(
    torch,
    helper,
    template,
    inputs,
    mask,
    allowed,
    timestep,
    upstream,
    case,
    kind,
    device,
    Attention,
    native_type,
    stage,
    base,
):
    changed_inputs = {n: x.clone() for n, x in inputs.items()}
    name = "hidden" if case.get("vl") else "encoder"
    blocked = (~allowed)[:, :, None].expand_as(changed_inputs[name])
    pattern = (torch.arange(changed_inputs[name].shape[-1]) % 2).to(
        changed_inputs[name].dtype
    )
    changed_inputs[name] += (~allowed)[:, :, None].to(pattern.dtype) * pattern * 3.0
    changed = execute(
        torch,
        helper,
        template,
        changed_inputs,
        mask,
        timestep,
        upstream,
        case,
        kind,
        device,
        Attention,
        native_type,
        stage,
    )
    exact = exact_snapshot(torch, changed, base, allowed if case.get("vl") else None)
    grad = base["input_grads"].get(name)
    values = None if grad is None else grad[blocked]
    finite = values is not None and bool(torch.isfinite(values).all())
    zero = {
        "finite": finite,
        "max_abs": float(values.abs().max()) if finite else None,
        "pass": finite and bool((values == 0).all()),
        "input": name,
    }
    # Positive control prevents an accidentally ineffectual test fixture.
    # Reuse the same model/input/upstream; change only mask visibility.
    unmasked = execute(
        torch,
        helper,
        template,
        inputs,
        None,
        timestep,
        upstream,
        case,
        kind,
        device,
        Attention,
        native_type,
        stage,
    )
    a, b = base["output"], unmasked["output"]
    if case.get("vl"):
        a, b = a[allowed], b[allowed]
    delta = float((a - b).abs().max())
    sensitivity = {
        "visible_output_max_abs_change": delta,
        "pass": math.isfinite(delta) and delta > 0,
    }
    visible_inputs = {n: x.clone() for n, x in inputs.items()}
    # Change a guaranteed visible token in a nonconstant feature direction,
    # so source per-token cross LayerNorm cannot erase the perturbation.
    visible_inputs[name][:, 0, ::2] += 2.0
    visible = execute(
        torch,
        helper,
        template,
        visible_inputs,
        mask,
        timestep,
        upstream,
        case,
        kind,
        device,
        Attention,
        native_type,
        stage,
    )
    a, b = base["output"], visible["output"]
    if case.get("vl"):
        a, b = a[allowed], b[allowed]
    visible_delta = float((a - b).abs().max())
    visible_control = {
        "visible_token_index": 0,
        "output_max_abs_change": visible_delta,
        "pass": math.isfinite(visible_delta) and visible_delta > 0,
    }
    return {
        "blocked_token_perturbation_exact": exact,
        "blocked_input_gradient_strict_zero": zero,
        "mask_removal_positive_control": sensitivity,
        "visible_token_perturbation_positive_control": visible_control,
        "query_scope": "visible VL query outputs; padded-query upstream is zero"
        if case.get("vl")
        else "all 49 action queries",
        "pass": exact["pass"]
        and zero["pass"]
        and sensitivity["pass"]
        and visible_control["pass"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--device", choices=("cpu", "musa", "musa:0"), default="cpu")
    parser.add_argument("--phase", choices=("attention", "dit", "all"), default="all")
    parser.add_argument("--dtypes", default="fp32")
    parser.add_argument("--cases", default="all")
    parser.add_argument(
        "--lengths",
        default="513",
        help="Representative unpadded lengths, one per batch item, each 1..569",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--support-dir", type=Path, default=base)
    parser.add_argument(
        "--bundle-manifest",
        type=Path,
        default=Path(__file__).resolve().with_name("source-manifest.json"),
    )
    parser.add_argument(
        "--spatial-config",
        type=Path,
        default=base / "backbone_sources/checkpoint-config.json",
    )
    parser.add_argument(
        "--gr00t-source",
        type=Path,
        default=base
        / "audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dtype_names = args.dtypes.split(",")
    if (
        not dtype_names
        or any(n not in ("fp32", "bf16") for n in dtype_names)
        or len(set(dtype_names)) != len(dtype_names)
    ):
        parser.error("--dtypes must contain unique fp32 and/or bf16")
    try:
        lengths = [int(n) for n in args.lengths.split(",")]
    except ValueError:
        parser.error("--lengths must be comma-separated integers")
    if not lengths or any(n < 1 or n >= 570 for n in lengths):
        parser.error(
            "--lengths must retain at least one visible and one padded key in each 570-token row"
        )
    if args.output.exists():
        parser.error("Output already exists; select a new evidence path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "probe": "spatial_action_layout",
        "platform": platform.platform(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "scope": "Random-weight real modules at Spatial widths, reduced DiT depth; eval/dropout0; no checkpoint weights/optimizer/PPO/simulator",
        "reference": "Identical quantized state/input/upstream; actual native CPU AttnProcessor2_0, not FP64 model",
        "gates": {
            "fp32": {"atol": 2e-4, "rtol": 2e-4, "relative_l2_limit": 2e-4},
            "bf16": {"atol": 0.025, "rtol": 0.04, "relative_l2_limit": 0.04},
        },
        "layout": {
            "query_segments": [
                {"name": "state", "start": 0, "length": 1},
                {"name": "future", "start": 1, "length": 32},
                {"name": "action", "start": 33, "length": 16},
            ],
            "action_queries": 49,
            "keys": 570,
            "unpadded_lengths": lengths,
            "lengths_are_actual_observed_batch": False,
        },
        "source_sha256": {"runner": file_hash(Path(__file__))},
        "events": [],
        "cases": [],
        "rows": [],
        "strict_interfaces_pass": False,
        "semantic_interfaces_pass": False,
    }

    def stage(name, **fields):
        event = {"stage": name, "status": "started", **fields}
        result["last_stage"] = event
        result["events"].append(event)
        print(
            json.dumps({"probe": "spatial_action_stage", **event}, ensure_ascii=False),
            flush=True,
        )

    def persist():
        args.output.write_text(
            json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )

    try:
        stage("guard_support_and_config")
        bundle = json.loads(
            args.bundle_manifest.read_text(), object_pairs_hook=unique_object
        )
        guarded(Path(__file__), bundle["runner_sha256"])
        if (
            bundle["support_sha256"] != SUPPORT_HASHES
            or bundle["spatial_config_sha256"] != CONFIG_HASH
            or bundle["gr00t_dit_sha256"] != DIT_HASH
        ):
            raise RuntimeError(
                "Bundle manifest differs from the runner's fixed source pins"
            )
        result["bundle_manifest"] = {
            "path": str(args.bundle_manifest.resolve()),
            "sha256": file_hash(args.bundle_manifest),
        }
        helper, result["support_sources"] = load_support(args.support_dir)
        config, result["spatial_config_source"] = spatial_config(args.spatial_config)
        result["official_config"] = config
        cases = [
            case
            for case in cases_from_config(config)
            if args.phase == "all" or case["phase"] == args.phase
        ]
        if args.cases != "all":
            requested = set(args.cases.split(","))
            cases = [case for case in cases if case["name"] in requested]
            if requested != {case["name"] for case in cases}:
                raise ValueError("Unknown case or case outside selected phase")
        result["selected_cases"] = [case["name"] for case in cases]
        stage("runtime_imports_and_source_guard")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import diffusers
        import torch
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0

        if diffusers.__version__ != "0.30.2":
            raise RuntimeError(
                f"Expected Diffusers 0.30.2, observed {diffusers.__version__}"
            )
        result["native_attention_source"] = guarded(
            inspect.getsourcefile(Attention), ATTENTION_HASH
        )
        result["runtime"] = {
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "diffusers_version": diffusers.__version__,
            "diffusers_path": diffusers.__file__,
        }
        result["distribution_versions"] = {}
        for name in (
            "torch",
            "torch_musa",
            "diffusers",
            "numpy",
            "huggingface_hub",
            "safetensors",
            "packaging",
        ):
            try:
                result["distribution_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["distribution_versions"][name] = None
        if args.device.startswith("musa"):
            import torch_musa

            if not torch.musa.is_available():
                raise RuntimeError(
                    "MUSA requested but unavailable; no CPU substitution"
                )
            result["runtime"].update(
                torch_musa_version=getattr(torch_musa, "__version__", None),
                musa_device=torch.musa.get_device_name(0),
            )
        torch.set_num_threads(1)
        DiT = None
        if any(case["phase"] == "dit" for case in cases):
            result["gr00t_source"] = {
                **guarded(args.gr00t_source, DIT_HASH),
                "commit": GR00T_COMMIT,
                "direct_file_import": True,
            }
            DiT = import_file("spatial_fixed_gr00t_dit", args.gr00t_source).DiT
        equivalent_peers = {}
        for case in cases:
            for dtype_name in dtype_names:
                stage("case_prepare", case=case["name"], dtype=dtype_name)
                dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
                torch.manual_seed(args.seed + case["seed_offset"])
                template = (
                    (
                        DiT(**case["config"])
                        if case["phase"] == "dit"
                        else Attention(**case["config"])
                    )
                    .to(dtype)
                    .eval()
                )
                inputs, mask, allowed, timestep, upstream = make_data(
                    torch, case, dtype, args.seed + case["seed_offset"], lengths
                )
                initial = {
                    n: helper.tensor_manifest(torch, x)
                    for n, x in template.state_dict().items()
                }
                case_record = {
                    "name": case["name"],
                    "phase": case["phase"],
                    "dtype": dtype_name,
                    "constructor_config": case["config"],
                    "effective_config": dict(template.config)
                    if case["phase"] == "dit"
                    else case["config"],
                    "overrides": {
                        "num_layers": 2,
                        "dropout": 0.0,
                        "final_dropout": False,
                    }
                    if case["phase"] == "dit"
                    else {"dropout": 0.0},
                    "parameter_count": sum(x.numel() for x in template.parameters()),
                    "initial_state": initial,
                    "inputs": {
                        n: helper.tensor_manifest(torch, x) for n, x in inputs.items()
                    },
                    "upstream": helper.tensor_manifest(torch, upstream),
                    "timestep": helper.tensor_manifest(torch, timestep),
                    "mask": None
                    if mask is None
                    else helper.tensor_manifest(torch, mask),
                    "rightpad_allowed": helper.tensor_manifest(torch, allowed),
                    "mask_semantics": "independent Attention rightpad contract"
                    if mask is not None
                    else "actual no-mask source baseline",
                    "training": False,
                    "dropout_training_validated": False,
                }
                if mask is not None:
                    prepared = template.prepare_attention_mask(
                        mask, 570, len(lengths)
                    ).view(len(lengths), template.heads, -1, 570)
                    case_record["source_prepared_mask"] = helper.tensor_manifest(
                        torch, prepared
                    )
                    visible_mask = (
                        prepared
                        if prepared.dtype == torch.bool
                        else prepared != float("-inf")
                    )
                    mask_checks = {
                        "every_head_query_nonempty": bool(
                            visible_mask.any(dim=-1).all()
                        ),
                        "all_rightpad_keys_blocked": not bool(
                            (visible_mask & (~allowed)[:, None, None, :]).any()
                        ),
                    }
                    if case.get("mask4d"):
                        mask_checks.update(
                            head_axis_varies=not bool(
                                torch.equal(visible_mask[:, 0], visible_mask[:, 1])
                            ),
                            query_axis_varies=not bool(
                                torch.equal(
                                    visible_mask[:, :, 0], visible_mask[:, :, 1]
                                )
                            ),
                        )
                        case_record["additional_mask_fixture"] = (
                            "One visible key is additionally blocked per head/query; this is an API-contract fixture, not observed LIBERO padding"
                        )
                    case_record["mask_preparation_checks"] = mask_checks
                    if not all(mask_checks.values()):
                        raise RuntimeError(
                            "Representative mask fixture lacks required visibility/axis variation"
                        )
                    expected_queries = 570 if case.get("vl") else 49
                    if prepared.shape[2] not in (1, expected_queries):
                        raise RuntimeError(
                            "Source mask preparation changed the query dimension"
                        )
                result["cases"].append(case_record)
                peers = {}
                kinds = [("native_cpu", "cpu"), ("cpu_fallback", "cpu")]
                if args.device.startswith("musa"):
                    kinds.append(("musa_fallback", args.device))
                for kind, device in kinds:
                    row = {
                        "case": case["name"],
                        "dtype": dtype_name,
                        "kind": kind,
                        "device": device,
                        "pass": False,
                        "semantic_pass": False,
                    }
                    start = time.monotonic()

                    def local_stage(name, kind=kind):
                        stage(name, case=case["name"], dtype=dtype_name, kind=kind)

                    try:
                        actual = execute(
                            torch,
                            helper,
                            template,
                            inputs,
                            mask,
                            timestep,
                            upstream,
                            case,
                            kind,
                            device,
                            Attention,
                            AttnProcessor2_0,
                            local_stage,
                        )
                        row["execution"] = actual["metadata"]
                        row["state_hashes_match"] = actual["metadata"][
                            "state_sha256"
                        ] == {n: m["sha256"] for n, m in initial.items()}
                        checks = [
                            row["state_hashes_match"],
                            actual["metadata"]["finite"],
                        ]
                        semantic_checks = checks.copy()
                        if mask is not None and kind != "native_cpu":
                            traces = actual["metadata"]["kernel_traces"]
                            expected_mask_shape = case_record["source_prepared_mask"][
                                "shape"
                            ]
                            row["actual_kernel_mask_shape_matches_source"] = bool(
                                traces
                            ) and all(
                                trace["mask_shape"] == expected_mask_shape
                                for trace in traces
                            )
                            checks.append(
                                row["actual_kernel_mask_shape_matches_source"]
                            )
                            semantic_checks.append(
                                row["actual_kernel_mask_shape_matches_source"]
                            )
                        for peer_name in ("native_cpu", "cpu_fallback"):
                            required = kind != "native_cpu" and (
                                peer_name == "native_cpu" or kind == "musa_fallback"
                            )
                            if required and peer_name not in peers:
                                row[peer_name + "_comparison"] = {
                                    "pass": False,
                                    "semantic_pass": False,
                                    "missing_peer": True,
                                }
                                checks.append(False)
                                semantic_checks.append(False)
                            elif required:
                                comparison = helper.semantic_compare(
                                    torch, actual, peers[peer_name], dtype_name
                                )
                                row[peer_name + "_comparison"] = comparison
                                checks.append(comparison["pass"])
                                semantic_checks.append(comparison["semantic_pass"])
                        if case["name"] in (
                            "action_cross_rightpad_bool",
                            "action_cross_rightpad_additive",
                            "action_cross_rightpad_4d_bool",
                            "action_cross_rightpad_4d_additive",
                        ):
                            pair_key = (dtype_name, kind, bool(case.get("mask4d")))
                            if case["mask"] == "bool":
                                equivalent_peers[pair_key] = actual
                            else:
                                expected_pair = equivalent_peers.pop(pair_key, None)
                                pair = (
                                    {"pass": False, "missing_peer": True}
                                    if expected_pair is None
                                    else exact_snapshot(torch, actual, expected_pair)
                                )
                                row["bool_additive_equivalence_exact"] = pair
                                checks.append(pair["pass"])
                                semantic_checks.append(pair["pass"])
                        if mask is not None:
                            local_stage("rightpad_perturbation_and_gradient_contract")
                            contract = padding_contract(
                                torch,
                                helper,
                                template,
                                inputs,
                                mask,
                                allowed,
                                timestep,
                                upstream,
                                case,
                                kind,
                                device,
                                Attention,
                                AttnProcessor2_0,
                                local_stage,
                                actual,
                            )
                            row["rightpad_contract"] = contract
                            checks.append(contract["pass"])
                            semantic_checks.append(contract["pass"])
                        if case["phase"] == "dit":
                            local_stage("dit_mask_argument_ignored_diagnostic")
                            argument_mask = allowed[:, None, :]
                            with_argument = execute(
                                torch,
                                helper,
                                template,
                                inputs,
                                None,
                                timestep,
                                upstream,
                                case,
                                kind,
                                device,
                                Attention,
                                AttnProcessor2_0,
                                local_stage,
                                dit_mask=argument_mask,
                            )
                            ignored = exact_snapshot(torch, with_argument, actual)
                            row["dit_mask_argument_ignored"] = {
                                **ignored,
                                "conclusion": "source ignores encoder_attention_mask; this is not a padding-mask support gate",
                            }
                            checks.append(ignored["pass"])
                            semantic_checks.append(ignored["pass"])
                        peers[kind] = actual
                        row["pass"] = all(checks)
                        row["semantic_pass"] = all(semantic_checks)
                    except Exception as exc:
                        row.update(
                            error=type(exc).__name__ + ": " + str(exc),
                            traceback=traceback.format_exc(),
                            last_stage=result["last_stage"],
                        )
                    row["elapsed_seconds"] = round(time.monotonic() - start, 3)
                    result["rows"].append(row)
                    print(
                        json.dumps(
                            {
                                "probe": "spatial_action_row",
                                **{
                                    n: row[n]
                                    for n in (
                                        "case",
                                        "dtype",
                                        "kind",
                                        "pass",
                                        "semantic_pass",
                                        "error",
                                        "elapsed_seconds",
                                    )
                                    if n in row
                                },
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    persist()
                del peers, template
                gc.collect()
        expected_rows = (
            len(cases) * len(dtype_names) * (3 if args.device.startswith("musa") else 2)
        )
        result["expected_rows"] = expected_rows
        result["strict_interfaces_pass"] = len(result["rows"]) == expected_rows and all(
            row["pass"] for row in result["rows"]
        )
        result["semantic_interfaces_pass"] = len(
            result["rows"]
        ) == expected_rows and all(row["semantic_pass"] for row in result["rows"])
    except Exception as exc:
        result.update(
            setup_error=type(exc).__name__ + ": " + str(exc),
            traceback=traceback.format_exc(),
        )
    result["status"] = "pass" if result["strict_interfaces_pass"] else "fail"
    result["exit_code"] = 0 if result["strict_interfaces_pass"] else 1
    persist()
    print(
        json.dumps(
            {
                "probe": "spatial_action_complete",
                "status": result["status"],
                "strict_interfaces_pass": result["strict_interfaces_pass"],
                "semantic_interfaces_pass": result["semantic_interfaces_pass"],
                "exit_code": result["exit_code"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
