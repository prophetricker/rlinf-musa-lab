#!/usr/bin/env python3
"""Probe real Diffusers Attention and fixed GR00T DiT with random weights.

CPU native AttnProcessor2_0 is the baseline. Compare a CPU FP32-accumulation
processor and, only when explicitly selected, the same processor on MUSA.
All inputs and parameters receive deterministic VJP checks, without downloads,
package installation, global patches, model checkpoints or optimizer updates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

from attention_probe import error_metrics, synchronize
from fp32_attention_processor import FP32AttentionProcessor


DIFFUSERS_VERSION = "0.30.2"
ATTENTION_SOURCE_SHA256 = "afe47ff6d33c865af745e28eed4bd8048b66fd52de7c2b3bad37867e826776c9"
DIT_SOURCE_SHA256 = "1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62"
DIT_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(torch, tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensor_manifest(torch, tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": tensor_hash(torch, tensor),
            "finite": bool(torch.isfinite(tensor).all())}


def checked_source(path, expected):
    observed = file_hash(path)
    if observed != expected:
        raise RuntimeError(f"Source mismatch for {path}: expected {expected}, observed {observed}")
    return {"path": str(Path(path).resolve()), "sha256": observed}


def load_dit(path):
    source = checked_source(path, DIT_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location("route2_fixed_gr00t_cross_attention_dit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DiT, {**source, "commit": DIT_COMMIT, "direct_file_import": True}


def metrics(torch, actual, expected, dtype_name):
    result = error_metrics(torch, actual, expected, dtype_name)
    result["actual_l2"] = float(actual.norm()) if result["finite"] else None
    result["expected_l2"] = float(expected.norm()) if result["finite"] else None
    return result


def compare(torch, actual, expected, dtype_name):
    result = {"forward": metrics(torch, actual["output"], expected["output"], dtype_name),
              "input_vjp": {}, "parameter_vjp": {}}
    for group, field in (("input_grads", "input_vjp"), ("parameter_grads", "parameter_vjp")):
        keys_match = set(actual[group]) == set(expected[group])
        result[field + "_names_match"] = keys_match
        for name in sorted(set(actual[group]) | set(expected[group])):
            a, b = actual[group].get(name), expected[group].get(name)
            result[field][name] = ({"pass": False, "finite": False,
                                    "missing_actual": a is None, "missing_expected": b is None}
                                   if a is None or b is None else metrics(torch, a, b, dtype_name))
    result["pass"] = (result["forward"]["pass"]
                      and result["input_vjp_names_match"] and result["parameter_vjp_names_match"]
                      and all(v["pass"] for v in result["input_vjp"].values())
                      and all(v["pass"] for v in result["parameter_vjp"].values()))
    return result


def semantic_compare(torch, actual, expected, dtype_name):
    """Separate meaningful VJPs from analytically zero softmax-bias VJPs.

    The original per-parameter gate remains in ``compare``. A zero reference
    gradient makes relative-L2 unstable, so this second view reports whether
    every nonzero-reference parameter, output, and input gradient passed while
    retaining the exact-zero bias diagnostics separately.
    """
    result = compare(torch, actual, expected, dtype_name)
    zero_names = set(actual["metadata"].get("analytic_zero_parameter_diagnostics", {}))
    meaningful = {
        name: metric
        for name, metric in result["parameter_vjp"].items()
        if name not in zero_names
    }
    result["analytic_zero_parameter_names"] = sorted(zero_names)
    result["meaningful_parameter_vjp"] = meaningful
    result["semantic_pass"] = (
        result["forward"]["pass"]
        and result["input_vjp_names_match"]
        and result["parameter_vjp_names_match"]
        and all(metric["pass"] for metric in result["input_vjp"].values())
        and all(metric["pass"] for metric in meaningful.values())
    )
    return result


def bind_processors(model, Attention, native_type, kind, traces):
    bindings = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if kind == "native_cpu":
                module.set_processor(native_type())
            else:
                module.set_processor(FP32AttentionProcessor(
                    trace=lambda data, name=name: traces.append({"module": name, **data})))
            bindings[name] = type(module.get_processor()).__name__
    if not bindings:
        raise RuntimeError("No real Diffusers Attention modules found")
    return bindings


def analytic_zero_diagnostics(torch, model, Attention):
    diagnostics = {}
    for name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        prefix = name + "." if name else ""
        target, reason = None, None
        if module.norm_k is None and module.to_k is not None and module.to_k.bias is not None:
            target = prefix + "to_k.bias"
            reason = "Without key normalization, shared key-projection bias shifts every allowed softmax logit equally"
        elif module.norm_k is not None and getattr(module.norm_k, "bias", None) is not None:
            target = prefix + "norm_k.bias"
            reason = "Per-head key-normalization output bias shifts every allowed softmax logit equally"
        if target is None:
            continue
        grad = dict(model.named_parameters())[target].grad
        finite = grad is not None and bool(torch.isfinite(grad).all())
        diagnostics[target] = {
            "reason": reason, "finite": finite,
            "gradient_l2": float(grad.detach().cpu().double().norm()) if finite else None,
            "gradient_max_abs": float(grad.detach().cpu().double().abs().max()) if finite else None,
            "expected_analytic_gradient": 0.0,
            "gate_note": "Diagnostic only; included in unchanged per-parameter allclose/relative-L2 gate",
        }
    return diagnostics


def execute(torch, template, inputs, mask, timestep, upstream, case, kind, device, Attention, native_type):
    model = copy.deepcopy(template).to(device).eval()
    traces = []
    bindings = bind_processors(model, Attention, native_type, kind, traces)
    xs = {name: value.to(device).detach().requires_grad_() for name, value in inputs.items()}
    local_mask = None if mask is None else mask.to(device)
    if case["phase"] == "dit":
        output = model(hidden_states=xs["hidden"], encoder_hidden_states=xs["encoder"],
                       timestep=timestep.to(device), encoder_attention_mask=None)
    else:
        output = model(xs["hidden"], encoder_hidden_states=xs.get("encoder"),
                       attention_mask=local_mask, temb=xs.get("temb"))
    (output * upstream.to(device)).sum().backward()
    synchronize(torch, device)
    input_grads = {name: None if value.grad is None else value.grad.detach().cpu().double()
                   for name, value in xs.items()}
    parameter_grads = {name: None if value.grad is None else value.grad.detach().cpu().double()
                       for name, value in model.named_parameters() if value.requires_grad}
    analytic_zero_parameters = analytic_zero_diagnostics(torch, model, Attention)
    missing = {"inputs": [n for n, v in input_grads.items() if v is None],
               "parameters": [n for n, v in parameter_grads.items() if v is None]}
    all_values = [output.detach().cpu().double(), *input_grads.values(), *parameter_grads.values()]
    finite = all(v is not None and bool(torch.isfinite(v).all()) for v in all_values)
    actual_state = {name: tensor_manifest(torch, value) for name, value in model.state_dict().items()}
    result = {"output": all_values[0], "input_grads": input_grads, "parameter_grads": parameter_grads,
              "metadata": {"finite": finite, "missing_gradients": missing,
                           "output_shape": list(output.shape), "output_dtype": str(output.dtype),
                           "processor_bindings": bindings, "kernel_traces": traces,
                           "analytic_zero_parameter_diagnostics": analytic_zero_parameters,
                           "state_sha256": {name: value["sha256"] for name, value in actual_state.items()},
                           "parameter_runtime": {name: {"device": str(v.device), "dtype": str(v.dtype),
                                                        "requires_grad": v.requires_grad}
                                                 for name, v in model.named_parameters()}}}
    return result


def case_definitions():
    common = dict(query_dim=256, heads=4, dim_head=64, dropout=0.0, bias=True,
                  out_bias=True, scale_qk=True)
    cross = {**common, "cross_attention_dim": 1536}
    cases = [dict(name="attention_self", phase="attention", config=common, mask=None),
             dict(name="attention_cross", phase="attention", config=cross, mask=None),
             dict(name="processor_mask_bool", phase="attention", config=cross, mask="bool", contract=True),
             dict(name="processor_mask_additive", phase="attention", config=cross, mask="additive", contract=True),
             dict(name="processor_norms_4d_residual", phase="attention", mask=None,
                  config={**cross, "norm_num_groups": 32, "spatial_norm_dim": 32,
                          "cross_attention_norm": "layer_norm", "qk_norm": "layer_norm",
                          "residual_connection": True, "rescale_output_factor": 1.5}, spatial=True)]
    cases.append(dict(name="gr00t_dit_two_layers", phase="dit", mask=None,
                      config=dict(num_attention_heads=4, attention_head_dim=64, num_layers=2,
                                  cross_attention_dim=1536, interleave_self_attention=True,
                                  dropout=0.0, final_dropout=False, norm_type="ada_norm",
                                  activation_fn="gelu-approximate")))
    # Keep each case reproducible across phase/case filtering. Bool/additive
    # masks intentionally share the cross-attention model and random inputs.
    offsets = {"attention_self": 0, "attention_cross": 1,
               "processor_mask_bool": 1, "processor_mask_additive": 1,
               "processor_norms_4d_residual": 2, "gr00t_dit_two_layers": 3}
    for case in cases:
        case["seed_offset"] = offsets[case["name"]]
    return cases


def make_case_inputs(torch, case, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def random(shape):
        return (torch.randn(shape, generator=gen) * 0.5).to(dtype)
    hidden_shape = (2, 256, 1, 17) if case.get("spatial") else (2, 17, 256)
    inputs = {"hidden": random(hidden_shape)}
    if case["phase"] == "dit" or case["config"].get("cross_attention_dim") is not None:
        inputs["encoder"] = random((2, 31, 1536))
    if case.get("spatial"):
        inputs["temb"] = random((2, 32, 1, 5))
    allowed, mask = None, None
    if case["mask"]:
        allowed = torch.arange(31)[None, :] < torch.tensor([28, 27])[:, None]
        mask = allowed[:, None, :]
        if case["mask"] == "additive":
            mask = torch.zeros(mask.shape, dtype=dtype).masked_fill(~mask, float("-inf"))
    shape = (2, 17, 26) if case["phase"] == "dit" else hidden_shape
    # Quantize one shared upstream vector before all CPU/device evaluations.
    upstream = torch.randn(shape, generator=gen).to(dtype) / math.sqrt(math.prod(shape[1:]))
    timestep = torch.tensor([3, 11], dtype=torch.int64)
    return inputs, mask, allowed, timestep, upstream


def contract_checks(torch, actual, changed, allowed, dtype_name):
    result = {"blocked_encoder_perturbation": compare(torch, changed, actual, dtype_name)}
    grad = actual["input_grads"].get("encoder")
    if grad is None:
        result["blocked_encoder_gradient"] = {"pass": False, "missing": True}
    else:
        values = grad[(~allowed)[:, :, None].expand_as(grad)]
        finite = bool(torch.isfinite(values).all())
        result["blocked_encoder_gradient"] = {
            "finite": finite, "max_abs": float(values.abs().max()) if finite else None,
            "pass": finite and bool((values == 0).all())}
    result["pass"] = all(result[name]["pass"] for name in
                         ("blocked_encoder_perturbation", "blocked_encoder_gradient"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa", "musa:0"), default="cpu")
    parser.add_argument("--phase", choices=("attention", "dit", "all"), default="attention")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--gr00t-source", type=Path, default=Path(__file__).resolve().parent /
                        "audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(x not in {"fp32", "bf16"} for x in args.dtypes.split(",")):
        parser.error("--dtypes must contain fp32 and/or bf16")
    cases = [case for case in case_definitions() if args.phase == "all" or case["phase"] == args.phase]
    if args.cases != "all":
        requested = set(args.cases.split(","))
        cases = [case for case in cases if case["name"] in requested]
        if requested != {case["name"] for case in cases}:
            parser.error("Unknown case or case not included in selected phase")
    result = {"scope": "Random-weight real Attention/GR00T DiT interface gate; not checkpoint, training or learning",
              "platform": platform.platform(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "gates": {"fp32": {"atol": 2e-4, "rtol": 2e-4, "relative_l2_limit": 2e-4},
                        "bf16": {"atol": 0.025, "rtol": 0.04, "relative_l2_limit": 0.04}},
              "reference": "CPU native Diffusers AttnProcessor2_0 on identical quantized state/inputs/VJP; not CPU FP64",
              "source_sha256": {name: file_hash(Path(__file__).resolve().parent / name)
                                for name in ("gr00t_action_interface_probe.py", "fp32_attention_processor.py", "attention_probe.py")},
              "rows": [], "cases": []}
    success = True
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        import diffusers
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0
        if diffusers.__version__ != DIFFUSERS_VERSION:
            raise RuntimeError(f"Expected Diffusers {DIFFUSERS_VERSION}, got {diffusers.__version__}")
        result["native_attention_source"] = checked_source(inspect.getsourcefile(Attention), ATTENTION_SOURCE_SHA256)
        result["runtime"] = {"torch_version": torch.__version__, "torch_path": torch.__file__,
                             "diffusers_version": diffusers.__version__, "diffusers_path": diffusers.__file__}
        result["distribution_versions"] = {}
        for name in ("torch", "torch_musa", "diffusers", "huggingface_hub", "numpy", "safetensors", "packaging", "Pillow", "requests", "filelock", "regex", "importlib_metadata"):
            try:
                result["distribution_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["distribution_versions"][name] = None
        if args.device.startswith("musa"):
            import torch_musa
            if not torch.musa.is_available():
                raise RuntimeError("MUSA requested but unavailable; no silent CPU fallback")
            result["runtime"]["torch_musa_version"] = getattr(torch_musa, "__version__", None)
            result["runtime"]["musa_device"] = torch.musa.get_device_name(0)
        torch.set_num_threads(1)
        DiT = None
        if any(case["phase"] == "dit" for case in cases):
            DiT, result["gr00t_dit_source"] = load_dit(args.gr00t_source)
        for case in cases:
            for dtype_name in args.dtypes.split(","):
                dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
                case_seed = args.seed + case["seed_offset"]
                torch.manual_seed(case_seed)
                template = (DiT(**case["config"]) if case["phase"] == "dit" else Attention(**case["config"])).to(dtype).eval()
                # Serialize all constructor defaults when the real DiT offers ConfigMixin.
                effective_config = dict(template.config) if case["phase"] == "dit" else case["config"]
                inputs, mask, allowed, timestep, upstream = make_case_inputs(torch, case, dtype, case_seed)
                state = {name: tensor_manifest(torch, value) for name, value in template.state_dict().items()}
                expected_state = {name: value["sha256"] for name, value in state.items()}
                case_record = {"name": case["name"], "phase": case["phase"], "dtype": dtype_name,
                               "initialization_seed": case_seed, "input_seed": case_seed,
                               "constructor_config": case["config"],
                               "effective_config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in effective_config.items()},
                               "state": state,
                               "trainable_parameters": {name: {"shape": list(v.shape), "dtype": str(v.dtype),
                                                                "numel": v.numel(), "requires_grad": v.requires_grad}
                                                        for name, v in template.named_parameters()},
                               "inputs": {name: tensor_manifest(torch, v) for name, v in inputs.items()},
                               "upstream": tensor_manifest(torch, upstream),
                               "mask": None if mask is None else {"shape": list(mask.shape), "dtype": str(mask.dtype),
                                                                  "sha256": tensor_hash(torch, mask), "nonempty_rows": bool(allowed.any(-1).all())},
                               "timestep": tensor_manifest(torch, timestep) if case["phase"] == "dit" else None,
                               "dit_encoder_attention_mask": None if case["phase"] == "dit" else "not a DiT case"}
                result["cases"].append(case_record)
                baseline, cpu_fallback = None, None
                evaluations = [("native_cpu", "cpu"), ("fp32_eager", "cpu")]
                if args.device.startswith("musa"):
                    evaluations.append(("fp32_eager", args.device))
                for kind, device in evaluations:
                    row = {"case": case["name"], "phase": case["phase"], "dtype": dtype_name,
                           "kind": kind, "device": device, "pass": False}
                    try:
                        actual = execute(torch, template, inputs, mask, timestep, upstream, case, kind, device, Attention, AttnProcessor2_0)
                        row.update(actual["metadata"])
                        row["initial_state_matches"] = row["state_sha256"] == expected_state
                        if kind == "native_cpu":
                            baseline = actual
                            row["native_cpu_baseline"] = True
                            row["pass"] = row["finite"] and row["initial_state_matches"]
                        else:
                            if baseline is None:
                                raise RuntimeError("Native CPU baseline failed; reference unavailable")
                            row["native_cpu_comparison"] = compare(torch, actual, baseline, dtype_name)
                            row["native_cpu_semantic_comparison"] = semantic_compare(
                                torch, actual, baseline, dtype_name
                            )
                            row["pass"] = row["finite"] and row["initial_state_matches"] and row["native_cpu_comparison"]["pass"]
                            if device == "cpu":
                                cpu_fallback = actual
                            elif cpu_fallback is not None:
                                row["cpu_fallback_comparison"] = compare(torch, actual, cpu_fallback, dtype_name)
                                row["cpu_fallback_semantic_comparison"] = semantic_compare(
                                    torch, actual, cpu_fallback, dtype_name
                                )
                                row["pass"] &= row["cpu_fallback_comparison"]["pass"]
                        if case.get("contract"):
                            changed_inputs = dict(inputs)
                            gen = torch.Generator(device="cpu").manual_seed(case_seed + 1000)
                            replacement = (torch.randn(inputs["encoder"].shape, generator=gen) * 8 + 3).to(dtype)
                            changed_inputs["encoder"] = torch.where(allowed[:, :, None], inputs["encoder"], replacement)
                            changed = execute(torch, template, changed_inputs, mask, timestep, upstream, case, kind, device, Attention, AttnProcessor2_0)
                            row["mask_contract"] = contract_checks(torch, actual, changed, allowed, dtype_name)
                            row["pass"] &= row["mask_contract"]["pass"]
                    except Exception as exc:
                        row.update({"pass": False, "error": type(exc).__name__ + ": " + str(exc),
                                    "traceback": traceback.format_exc(limit=4)})
                    result["rows"].append(row)
                    success &= row["pass"]
                    failing_parameters = []
                    for comparison in ("native_cpu_comparison", "cpu_fallback_comparison"):
                        failing_parameters.extend(name for name, v in row.get(comparison, {}).get("parameter_vjp", {}).items() if not v["pass"])
                    print(json.dumps({"case": case["name"], "dtype": dtype_name, "kind": kind,
                                      "device": device, "pass": row["pass"], "error": row.get("error"),
                                      "failing_parameter_vjp": sorted(set(failing_parameters))}, ensure_ascii=False), flush=True)
    except Exception as exc:
        success = False
        result["setup_error"] = type(exc).__name__ + ": " + str(exc)
        result["setup_traceback"] = traceback.format_exc(limit=5)
        print(json.dumps({"setup_error": result["setup_error"]}, ensure_ascii=False), flush=True)
    result["required_interfaces_pass"] = success and bool(result["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0 if result["required_interfaces_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
