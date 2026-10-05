#!/usr/bin/env python3
"""Tiny real Transformers Qwen3/SigLIP frozen-feature interface probe.

No pretrained weights, Eagle/RADIO import, installation, global attention patch,
optimizer update, or availability shim. Installed Transformers source and small
official checkpoint/config files must match pinned hashes. CPU source eager is
the reference; CPU/MUSA fallback promotes only the attention kernel to FP32.
An independent synthetic Linear checks leaf-feature/weight/bias VJPs on identical
CPU-reference features. It is not the checkpoint's eagle_linear (an Identity).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import traceback

from fp32_backbone_attention import BackboneAttentionKernel, bind_source_kernel


VERSION = "4.51.3"
TRANSFORMERS_COMMIT = "5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
CHECKPOINT_COMMIT = "73f710e70e7d571f8d828e51e0a428f5a1e0ac22"
SOURCE_HASHES = {
    "qwen3": "704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2",
    "siglip": "e8972dfaa3903191b936a76f0887ccf2e6648aa26cbde6a148b24a0c5790ecd2",
    "eagle_config": "e6957b163b6c62b685f69fd9d601e183509d109246290ad579d461692d925dba",
    "checkpoint_config": "6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(path, expected):
    digest = file_hash(path)
    if digest != expected:
        raise RuntimeError(f"Source hash mismatch: {path}: expected {expected}, observed {digest}")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def tensor_hash(torch, tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def tensor_manifest(torch, tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": tensor_hash(torch, tensor),
            "finite": bool(torch.isfinite(tensor).all())}


def state_manifest(torch, model):
    tensors = {name: tensor_manifest(torch, value) for name, value in model.state_dict().items()}
    digest = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "tensors": tensors}


def metrics(torch, actual, expected, dtype_name):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "pass": False, "max_abs": None, "relative_l2": None}
    atol, rtol, l2_limit = (2e-4, 2e-4, 2e-4) if dtype_name == "fp32" else (0.025, 0.04, 0.04)
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1e-12))
    return {"finite": True, "max_abs": float(delta.abs().max()), "relative_l2": relative,
            "atol": atol, "rtol": rtol, "relative_l2_limit": l2_limit,
            "actual_l2": float(actual.norm()), "expected_l2": float(expected.norm()),
            "pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)) and relative <= l2_limit}


def synchronize(torch, device):
    if device.startswith("musa"):
        torch.musa.synchronize()


def instantiate(torch, imports, family, dtype, seed):
    torch.manual_seed(seed)
    if family == "qwen3":
        config = imports[family].Qwen3Config(
            vocab_size=256, hidden_size=512, intermediate_size=1536,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=128, max_position_embeddings=40960,
            rms_norm_eps=1e-6, rope_theta=1000000.0, rope_scaling=None,
            attention_bias=False, attention_dropout=0.0, hidden_act="silu",
            use_cache=False, use_sliding_window=False, sliding_window=None,
            max_window_layers=2, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            _attn_implementation="eager")
        model = imports[family].Qwen3Model(config)
        width = 512
    else:
        config = imports[family].SiglipVisionConfig(
            hidden_size=288, intermediate_size=1076, num_hidden_layers=2,
            num_attention_heads=4, image_size=56, patch_size=14, num_channels=3,
            hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
            attention_dropout=0.0, vision_use_head=False,
            _attn_implementation="eager")
        model = imports[family].SiglipVisionModel(config)
        width = 288
    model.to(dtype=dtype).eval().requires_grad_(False)
    torch.manual_seed(seed + 1)
    projection = torch.nn.Linear(width, 64).to(dtype=dtype).eval()
    return model, projection


def make_inputs(torch, family, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed + 2)
    if family == "qwen3":
        ids = torch.randint(3, 256, (2, 13), generator=gen)
        lengths = torch.tensor([9, 11])
        mask = torch.arange(13)[None, :] < lengths[:, None]
        ids = ids.masked_fill(~mask, 0)
        padded = ids.clone()
        new_ids = torch.randint(3, 256, ids.shape, generator=gen)
        padded[~mask] = new_ids[~mask]
        future = ids.clone()
        future[:, 5:] = new_ids[:, 5:]
        return {"input_ids": ids, "attention_mask": mask.long()}, mask, {
            "rightpad_perturbed": padded, "future_perturbed": future}
    pixels = (torch.randn(2, 3, 56, 56, generator=gen) * 0.5).to(dtype)
    valid = torch.ones(2, 16, dtype=torch.bool)
    return {"pixel_values": pixels}, valid, {}


def bind_model(model, family, module_source, fallback, traces, label):
    cls = module_source.Qwen3Attention if family == "qwen3" else module_source.SiglipAttention
    bindings = {}
    for name, module in model.named_modules():
        if isinstance(module, cls):
            kernel = BackboneAttentionKernel(
                family, module_source.eager_attention_forward,
                repeat_kv=getattr(module_source, "repeat_kv", None), fallback=fallback,
                trace=lambda row, name=name: traces.append({"call": label[0], "module": name, **row}))
            bindings[name] = bind_source_kernel(module, kernel)
    if len(bindings) != model.config.num_hidden_layers:
        raise RuntimeError("Unexpected real attention module count")
    return bindings


def forward_features(model, family, inputs):
    result = model(**inputs, output_attentions=False, output_hidden_states=True)
    if family == "qwen3" and result.past_key_values is not None:
        raise RuntimeError("Cache must remain disabled")
    return result.last_hidden_state


def execute_features(torch, template, family, inputs, valid, perturb, kind, device, module_source):
    model = copy.deepcopy(template).to(device).eval().requires_grad_(False)
    traces, label = [], ["main"]
    bindings = bind_model(model, family, module_source, kind != "native_cpu", traces, label)
    local = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        output = forward_features(model, family, local)
        invariance = {}
        if family == "qwen3":
            for case_name, changed_ids in perturb.items():
                label[0] = case_name
                changed = forward_features(model, family, {**local, "input_ids": changed_ids.to(device)})
                selector = valid if case_name == "rightpad_perturbed" else (
                    torch.arange(output.shape[1])[None, :].expand_as(valid) < 5)
                delta = (changed[selector.to(device)] - output[selector.to(device)]).float()
                invariance[case_name] = {
                    "selected_tokens": int(selector.sum()), "max_abs": float(delta.abs().max()),
                    "exact_zero": bool((delta == 0).all()), "pass": bool((delta == 0).all())}
    synchronize(torch, device)
    no_grad = all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    finite = bool(torch.isfinite(output).all())
    blocked_ok = all(row["blocked_probability_max_abs"] in (None, 0.0) for row in traces)
    metadata = {
        "bindings": bindings, "attention_traces": traces,
        "state_after_device_transfer": state_manifest(torch, model),
        "output": tensor_manifest(torch, output), "output_requires_grad": output.requires_grad,
        "frozen_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_backbone_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all_backbone_parameter_grads_none": no_grad,
        "perturbation_invariance": invariance,
        "blocked_attention_probabilities_exact_zero": blocked_ok,
        "scope_pass": finite and no_grad and not output.requires_grad and blocked_ok
                      and all(v["pass"] for v in invariance.values()),
    }
    return {"output": output.detach().cpu(), "metadata": metadata}


def execute_projection(torch, template, features, upstream, device):
    model = copy.deepcopy(template).to(device).eval()
    model.requires_grad_(True)
    leaf = features.detach().to(device).requires_grad_(True)
    output = model(leaf)
    (output * upstream.to(device)).sum().backward()
    synchronize(torch, device)
    gradients = {"feature_leaf": leaf.grad.detach().cpu(),
                 **{name: p.grad.detach().cpu() for name, p in model.named_parameters()}}
    finite = bool(torch.isfinite(output).all()) and all(bool(torch.isfinite(v).all()) for v in gradients.values())
    return {"output": output.detach().cpu(), "gradients": gradients,
            "metadata": {"input": tensor_manifest(torch, leaf),
                         "upstream": tensor_manifest(torch, upstream),
                         "state_after_device_transfer": state_manifest(torch, model),
                         "output": tensor_manifest(torch, output),
                         "gradients": {name: tensor_manifest(torch, v) for name, v in gradients.items()},
                         "finite": finite, "independent_detached_feature_leaf": leaf.is_leaf}}


def compare_projection(torch, actual, expected, dtype_name):
    output = metrics(torch, actual["output"], expected["output"], dtype_name)
    gradients = {name: metrics(torch, value, expected["gradients"][name], dtype_name)
                 for name, value in actual["gradients"].items()}
    return {"forward": output, "vjp": gradients,
            "pass": output["pass"] and all(v["pass"] for v in gradients.values())}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    result.add_argument("--families", default="qwen3,siglip")
    result.add_argument("--dtypes", default="fp32,bf16")
    result.add_argument("--seed", type=int, default=314159)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--eagle-config", type=Path, default=Path(__file__).parent / "audit_sources/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json")
    result.add_argument("--checkpoint-config", type=Path, default=Path(__file__).parent / "backbone_sources/checkpoint-config.json")
    return result


def main():
    args = parser().parse_args()
    families, dtypes = args.families.split(","), args.dtypes.split(",")
    if any(f not in ("qwen3", "siglip") for f in families) or any(d not in ("fp32", "bf16") for d in dtypes):
        raise ValueError("Unsupported family/dtype")
    import torch
    import transformers
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.siglip import modeling_siglip
    if transformers.__version__ != VERSION:
        raise RuntimeError(f"Required Transformers {VERSION}, got {transformers.__version__}")
    sources = {name: check_source(inspect.getfile(module), SOURCE_HASHES[name])
               for name, module in (("qwen3", modeling_qwen3), ("siglip", modeling_siglip))}
    sources["eagle_config"] = check_source(args.eagle_config, SOURCE_HASHES["eagle_config"])
    sources["checkpoint_config"] = check_source(args.checkpoint_config, SOURCE_HASHES["checkpoint_config"])
    if args.device == "musa":
        import torch_musa
        if not torch.musa.is_available():
            raise RuntimeError("Explicit MUSA run requires an available MUSA device")
    imports = {"qwen3": modeling_qwen3, "siglip": modeling_siglip}
    packages = {}
    for name in ("torch", "torch_musa", "transformers", "huggingface_hub", "tokenizers", "regex", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils import is_flash_attn_2_available
    registry_before = tuple(ALL_ATTENTION_FUNCTIONS.keys())
    fa_before = is_flash_attn_2_available()
    class_forward_before = {f: (imports[f].Qwen3Attention if f == "qwen3" else imports[f].SiglipAttention).forward for f in families}
    evidence = {"probe": str(Path(__file__).resolve()), "probe_sha256": file_hash(__file__),
                "kernel_sha256": file_hash(Path(__file__).with_name("fp32_backbone_attention.py")),
                "platform": platform.platform(), "python": sys.version,
                "torch_import_path": torch.__file__, "packages": packages, "sources": sources,
                "transformers_commit": TRANSFORMERS_COMMIT, "checkpoint_commit": CHECKPOINT_COMMIT,
                "seed": args.seed, "device": args.device, "cases": [], "rows": [],
                "scope": "Tiny random real Qwen3Model/SiglipVisionModel, frozen feature forward; independent synthetic Linear VJP on identical CPU reference features. No Eagle/RADIO/checkpoint/training result.",
                "projection_scope": "Synthetic width->64 Linear; official Spatial project_to_dim=null means eagle_linear Identity, so this is not its parameter-training result.",
                "siglip_scope": "vision_use_head=False; Eagle uses hidden features, ignored pooling computation is outside this probe. No image processor/PIL/torchvision decode test."}
    overall = True
    for family_index, family in enumerate(families):
        for dtype_name in dtypes:
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            seed = args.seed + family_index * 100
            template, projection = instantiate(torch, imports, family, dtype, seed)
            inputs, valid, perturb = make_inputs(torch, family, dtype, seed)
            case = {"family": family, "dtype": dtype_name, "seed": seed,
                    "config": template.config.to_dict(), "backbone_state": state_manifest(torch, template),
                    "projection_state": state_manifest(torch, projection),
                    "inputs": {name: tensor_manifest(torch, tensor) for name, tensor in inputs.items()},
                    "valid_token_selector": tensor_manifest(torch, valid),
                    "perturbations": {name: tensor_manifest(torch, tensor) for name, tensor in perturb.items()}}
            case["config_sha256"] = hashlib.sha256(json.dumps(case["config"], sort_keys=True).encode()).hexdigest()
            evidence["cases"].append(case)
            reference = None
            cpu_fallback = None
            projection_reference = None
            projection_inputs = upstream = None
            evaluations = [("native_cpu", "cpu"), ("fallback_cpu", "cpu")]
            if args.device == "musa":
                evaluations.append(("fallback_musa", "musa"))
            for kind, device in evaluations:
                row = {"family": family, "dtype": dtype_name, "kind": kind, "device": device,
                       "reference": "CPU pinned Transformers source eager on identical quantized state/inputs"}
                try:
                    if kind != "native_cpu" and reference is None:
                        raise RuntimeError("CPU source reference failed; cannot compare")
                    actual = execute_features(torch, template, family, inputs, valid, perturb, kind, device, imports[family])
                    if reference is None:
                        reference = actual
                        projection_inputs = reference["output"][valid].detach().contiguous()
                        gen = torch.Generator(device="cpu").manual_seed(seed + 3)
                        upstream = (torch.randn(projection_inputs.shape[0], 64, generator=gen)
                                    / math.sqrt(projection_inputs.shape[0] * 64)).to(dtype)
                    adapter = execute_projection(torch, projection, projection_inputs, upstream, device)
                    if projection_reference is None:
                        projection_reference = adapter
                    row.update(actual["metadata"])
                    row["valid_token_forward"] = metrics(torch, actual["output"][valid], reference["output"][valid], dtype_name)
                    row["all_token_forward_diagnostic"] = metrics(torch, actual["output"], reference["output"], dtype_name)
                    row["projection"] = {**adapter["metadata"], "comparison": compare_projection(torch, adapter, projection_reference, dtype_name)}
                    row["backbone_state_matches_cpu_template"] = row["state_after_device_transfer"]["sha256"] == case["backbone_state"]["sha256"]
                    row["projection_state_matches_cpu_template"] = adapter["metadata"]["state_after_device_transfer"]["sha256"] == case["projection_state"]["sha256"]
                    row["pass"] = (row["scope_pass"] and row["valid_token_forward"]["pass"]
                                   and row["projection"]["comparison"]["pass"]
                                   and row["backbone_state_matches_cpu_template"]
                                   and row["projection_state_matches_cpu_template"])
                    if kind == "fallback_cpu":
                        cpu_fallback = actual
                    if kind == "fallback_musa" and cpu_fallback is not None:
                        row["cpu_fallback_peer_forward"] = metrics(torch, actual["output"][valid], cpu_fallback["output"][valid], dtype_name)
                        row["pass"] = row["pass"] and row["cpu_fallback_peer_forward"]["pass"]
                except Exception as exc:
                    row.update({"pass": False, "error": type(exc).__name__ + ": " + str(exc),
                                "traceback": traceback.format_exc(limit=5)})
                evidence["rows"].append(row)
                overall = overall and row["pass"]
                print(json.dumps({k: row[k] for k in ("family", "dtype", "kind", "pass", "error") if k in row}), flush=True)
    registry_after = tuple(ALL_ATTENTION_FUNCTIONS.keys())
    class_unchanged = all((imports[f].Qwen3Attention if f == "qwen3" else imports[f].SiglipAttention).forward is value
                          for f, value in class_forward_before.items())
    evidence["no_global_patch"] = {"registry_before": registry_before, "registry_after": registry_after,
                                   "registry_unchanged": registry_before == registry_after,
                                   "source_class_forward_unchanged": class_unchanged,
                                   "flash_attention_available_before": fa_before,
                                   "flash_attention_available_after": is_flash_attn_2_available()}
    evidence["pass"] = (overall and registry_before == registry_after and class_unchanged
                        and fa_before == is_flash_attn_2_available())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
