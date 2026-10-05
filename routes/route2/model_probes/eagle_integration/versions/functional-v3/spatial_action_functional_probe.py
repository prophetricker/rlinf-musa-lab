#!/usr/bin/env python3
"""Validate whether Spatial backbone drift changes the GR00T action function.

The probe loads the real Spatial action-head checkpoint and, when requested, the
real 585-tensor Eagle backbone. CPU and MUSA runs use identical fixed inputs,
noise and timesteps. It records action denoising, finite gradients and one
AdamW update. Numerical gates remain diagnostic: a finite functional result is
not relabelled as an exact CPU/MUSA match.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path


CONFIG_SHA256 = "6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526"
INDEX_SHA256 = "bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746"
TF_HASHES = {
    "qwen3": "704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2",
    "siglip": "e8972dfaa3903191b936a76f0887ccf2e6648aa26cbde6a148b24a0c5790ecd2",
}
ACTION_SOURCE_HASHES = {
    "gr00t/model/action_head/flow_matching_action_head.py": "8a8e6cf7ec63e2a335559990c4ab62bbb81e487d82ea4a969f452a93e0dbdd69",
    "gr00t/model/action_head/action_encoder.py": "c5ec2088a521e6c3a477bfdcbf191ffc470a3edc8787ebf0ac25d2da4f380cc4",
    "gr00t/model/action_head/cross_attention_dit.py": "1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metric(torch, actual, expected):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "pass": False}
    delta = actual - expected
    denom = expected.norm().clamp_min(1e-12)
    return {
        "finite": True,
        "shape_actual": list(actual.shape),
        "shape_expected": list(expected.shape),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / denom),
        "allclose_atol_rtol_2e-4": bool(torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)),
        "relative_l2_le_2e-4": float(delta.norm() / denom) <= 2e-4,
    }


def tensor_map_comparison(torch, actual, expected):
    if set(actual) != set(expected):
        raise RuntimeError("Snapshot parameter names differ")
    rows, delta_sq, ref_sq, actual_sq, dot = {}, 0.0, 0.0, 0.0, 0.0
    for name in sorted(expected):
        a, b = actual[name].double(), expected[name].double()
        delta = a - b
        delta_sq += float(delta.square().sum())
        ref_sq += float(b.square().sum())
        actual_sq += float(a.square().sum())
        dot += float((a * b).sum())
        rows[name] = metric(torch, a, b)
        rows[name]["reference_l2"] = float(b.norm())
    return {"parameters": rows,
            "global_relative_l2": math.sqrt(delta_sq / max(ref_sq, 1e-24)),
            "global_cosine": dot / max(math.sqrt(ref_sq * actual_sq), 1e-24),
            "max_abs": max(v["max_abs"] for v in rows.values()),
            "all_parameter_mixed_allclose": all(v["allclose_atol_rtol_2e-4"] for v in rows.values()),
            "snapshot_scope": "Every action-head parameter; category-specific tensors restricted to the active embodiment slice. Other slices checked for exact zero gradients."}


def bind_action_attention(head, kind):
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fp32_attention_processor import FP32AttentionProcessor
    bindings = {}
    for name, module in head.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor2_0() if kind == "native" else FP32AttentionProcessor())
            bindings[name] = type(module.get_processor()).__name__
    if len(bindings) != 20:
        raise RuntimeError(f"Expected all 16 DiT + 4 VL attention bindings: {len(bindings)}")
    return bindings


def tensor_summary(torch, value):
    value = value.detach().cpu().double()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "l2": float(value.norm()),
        "max_abs": float(value.abs().max()),
    }


def sync(torch, device):
    if device.startswith("musa"):
        torch.musa.synchronize()


def config_and_inputs(torch, config_path: Path, seed: int, embodiment_id: int):
    if sha256(config_path) != CONFIG_SHA256:
        raise RuntimeError("Pinned Spatial checkpoint config hash mismatch")
    config = json.loads(config_path.read_text())
    head_config = config["action_head_cfg"]
    expected = (32, 16, 32, 1536, 2048, 1024)
    observed = (
        head_config["action_dim"],
        head_config["action_horizon"],
        head_config["num_target_vision_tokens"],
        head_config["input_embedding_dim"],
        head_config["backbone_embedding_dim"],
        head_config["hidden_size"],
    )
    if observed != expected:
        raise RuntimeError(f"Unexpected Spatial action geometry: {observed}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    features_cpu = torch.randn((1, 570, 2048), generator=gen) * 0.02
    # This is the fixed action sample used by the training path. The model's
    # own forward samples both noise and beta time, so the probe supplies them.
    state = torch.randn((1, 1, 64), generator=gen) * 0.1
    action = torch.randn((1, 16, 32), generator=gen) * 0.1
    noise = torch.randn((1, 16, 32), generator=gen)
    timestep = torch.tensor([371], dtype=torch.long)
    initial_action_noise = torch.randn((1, 16, 32), generator=gen)
    # LIBERO has seven live action coordinates, with the remaining 25 padded.
    mask = torch.zeros((1, 16, 32), dtype=torch.float32)
    mask[:, :, :7] = 1
    state[:, :, 8:] = 0
    action[:, :, 7:] = 0
    embodiment_id = torch.tensor([embodiment_id], dtype=torch.long)
    backbone_mask = torch.zeros((1, 570), dtype=torch.long)
    backbone_mask[:, :550] = 1
    return {
        "features_cpu": features_cpu.float(),
        "state": state.float(),
        "action": action.float(),
        "noise": noise.float(),
        "timestep": timestep,
        "initial_action_noise": initial_action_noise.float(),
        "action_mask": mask,
        "embodiment_id": embodiment_id,
        "backbone_mask": backbone_mask,
    }


def load_action_head(torch, source_tree: Path, config_path: Path, weights: Path, index: dict):
    for relative, expected in ACTION_SOURCE_HASHES.items():
        if sha256(source_tree / relative) != expected:
            raise RuntimeError(f"Pinned action source mismatch: {relative}")
    sys.path.insert(0, str(source_tree.resolve()))
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHead,
        FlowmatchingActionHeadConfig,
    )
    action_cfg = json.loads(config_path.read_text())["action_head_cfg"]
    head = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**action_cfg))
    from safetensors import safe_open

    action_keys = sorted(key for key in index["weight_map"] if key.startswith("action_head."))
    if len(action_keys) != 314:
        raise RuntimeError(f"Expected 314 action-head tensors, found {len(action_keys)}")
    shards = sorted({index["weight_map"][key] for key in action_keys})
    state = {}
    with contextlib.ExitStack() as stack:
        handles = {
            name: stack.enter_context(safe_open(weights / name, framework="pt", device="cpu"))
            for name in shards
        }
        for key in action_keys:
            state[key.removeprefix("action_head.")] = handles[index["weight_map"][key]].get_tensor(key)
    if not all(value.dtype == torch.float32 for value in state.values()):
        raise RuntimeError("Expected all 314 action checkpoint tensors in FP32")
    loaded = head.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"Action-head strict load failed: {loaded}")
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise RuntimeError("Action-head checkpoint contains non-finite parameters")
    if not all(torch.equal(value, state[name]) for name, value in head.state_dict().items()):
        raise RuntimeError("Loaded action-head values differ from checkpoint")
    return head, {"tensor_count": len(action_keys), "shards": shards, "strict": True,
                  "missing_keys": [], "unexpected_keys": [], "all_values_checkpoint_exact": True}


def fixed_action_features(head, data, device):
    torch = __import__("torch")
    features = data["features"].to(device)
    state = data["state"].to(device)
    action = data["action"].to(device)
    noise = data["noise"].to(device)
    timestep = data["timestep"].to(device)
    embodiment = data["embodiment_id"].to(device)
    action_mask = data["action_mask"].to(device)
    vl = head.vlln(features)
    vl = head.vl_self_attention(vl)
    state_features = head.state_encoder(state, embodiment)
    t = timestep[:, None, None]
    noisy = (1 - t.float() / head.num_timestep_buckets) * noise + (
        t.float() / head.num_timestep_buckets
    ) * action
    action_features = head.action_encoder(noisy, timestep, embodiment)
    positions = torch.arange(action_features.shape[1], device=device)
    action_features = action_features + head.position_embedding(positions).unsqueeze(0)
    future = head.future_tokens.weight.unsqueeze(0).expand(vl.shape[0], -1, -1)
    joined = torch.cat((state_features, future, action_features), dim=1)
    model_output = head.model(
        hidden_states=joined,
        encoder_hidden_states=vl,
        encoder_attention_mask=data["backbone_mask"].to(device),
        timestep=timestep,
        return_all_hidden_states=False,
    )
    pred = head.action_decoder(model_output, embodiment)[:, -action.shape[1] :]
    velocity = action - noise
    loss = ((pred - velocity).pow(2) * action_mask).sum() / action_mask.sum()
    return pred, loss


def fixed_denoise(torch, head, data, device):
    features = data["features"].to(device)
    state = data["state"].to(device)
    embodiment = data["embodiment_id"].to(device)
    actions = data["initial_action_noise"].to(device).clone()
    vl = head.vlln(features)
    vl = head.vl_self_attention(vl)
    state_features = head.state_encoder(state, embodiment)
    for step in range(4):
        timestep = torch.full((features.shape[0],), step * 250, dtype=torch.long, device=device)
        action_features = head.action_encoder(actions, timestep, embodiment)
        positions = torch.arange(action_features.shape[1], device=device)
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        future = head.future_tokens.weight.unsqueeze(0).expand(vl.shape[0], -1, -1)
        joined = torch.cat((state_features, future, action_features), dim=1)
        output = head.model(
            hidden_states=joined,
            encoder_hidden_states=vl,
            timestep=timestep,
        )
        velocity = head.action_decoder(output, embodiment)[:, -actions.shape[1] :]
        actions = actions + 0.25 * velocity
    return actions


def source_formula_controls(torch, head, data, device):
    """Replay unmodified upstream methods with a private fixed-noise provider."""
    from transformers.feature_extraction_utils import BatchFeature

    class FixedNoiseTorch:
        def __init__(self, noise):
            self.noise = noise
            self.calls = 0

        def __getattr__(self, name):
            return getattr(torch, name)

        def randn(self, *args, **kwargs):
            size = kwargs.get("size", args[0] if args else None)
            if tuple(size) != tuple(self.noise.shape):
                raise RuntimeError("Unexpected source random draw")
            self.calls += 1
            return self.noise.to(device=kwargs["device"], dtype=kwargs["dtype"]).clone()

    def original_call(name, noise):
        method = getattr(type(head), name)
        method = inspect.unwrap(method)
        proxy = FixedNoiseTorch(noise)
        source_function = types.FunctionType(method.__code__, {**method.__globals__, "torch": proxy},
                                             method.__name__, method.__defaults__, method.__closure__)
        output = source_function(head,
                                 BatchFeature(data={"backbone_features": data["features"].to(device).clone(),
                                                    "backbone_attention_mask": data["backbone_mask"].to(device)}),
                                 BatchFeature(data={key: data[key].to(device) for key in
                                                    ("state", "action", "action_mask", "embodiment_id")}))
        if proxy.calls != 1:
            raise RuntimeError("Expected exactly one controlled upstream random draw")
        return output

    try:
        # Instance-only override; no source class, Torch or attention globals change.
        head.sample_time = lambda batch_size, device, dtype: (data["timestep"].to(device).to(dtype) / head.num_timestep_buckets)
        with torch.no_grad():
            manual_pred, manual_loss = fixed_action_features(head, data, device)
            source_loss = original_call("forward", data["noise"])["loss"]
            manual_actions = fixed_denoise(torch, head, data, device)
            source_actions = original_call("get_action", data["initial_action_noise"])["action_pred"]
            without_features = fixed_denoise(torch, head, {**data, "features": torch.zeros_like(data["features"])}, device)
        source_loss_equal = bool(torch.equal(manual_loss, source_loss))
        source_action_equal = bool(torch.equal(manual_actions, source_actions))
        positive = metric(torch, without_features[:, :, :7], manual_actions[:, :, :7])
        return {"source_forward_loss_exact": source_loss_equal,
                "source_get_action_exact": source_action_equal,
                "zero_features_positive_control": positive,
                "pass": source_loss_equal and source_action_equal and positive["max_abs"] > 2e-4}
    finally:
        del head.sample_time


def run_head(torch, template, data, device, label, attention_kind):
    print(json.dumps({"stage": label, "attention": attention_kind}), flush=True)
    started = time.monotonic()
    head = copy.deepcopy(template).to(device).eval()
    bindings = bind_action_attention(head, attention_kind)
    if device.startswith("musa"):
        torch.musa.reset_peak_memory_stats()
    controls = source_formula_controls(torch, head, data, device)
    # Eval disables dropout while preserving autograd for the training check.
    for param in head.parameters():
        param.requires_grad_(True)
    with torch.no_grad():
        actions = fixed_denoise(torch, head, data, device)
    sync(torch, device)
    head.zero_grad(set_to_none=True)
    pred, loss = fixed_action_features(head, data, device)
    loss.backward()
    sync(torch, device)
    gradients = []
    missing = []
    nonzero = 0
    grad_l2_sq = 0.0
    grad_snapshot = {}
    category_names = {prefix + "." + suffix for prefix in
                      ("state_encoder.layer1", "state_encoder.layer2", "action_encoder.W1", "action_encoder.W2",
                       "action_encoder.W3", "action_decoder.layer1", "action_decoder.layer2") for suffix in ("W", "b")}
    category = int(data["embodiment_id"][0])
    inactive_zero = True
    for name, param in head.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        grad = param.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"Non-finite gradient in {label}:{name}")
        norm = float(grad.float().norm())
        grad_l2_sq += norm * norm
        if float(grad.abs().max()) > 0:
            nonzero += 1
        if name in category_names:
            inactive_zero = inactive_zero and bool((grad[:category] == 0).all()) and bool((grad[category + 1:] == 0).all())
            grad_snapshot[name] = grad[category].detach().cpu().clone()
        else:
            grad_snapshot[name] = grad.detach().cpu().clone()
        if name in {
            "model.transformer_blocks.0.attn1.to_q.weight",
            "model.transformer_blocks.15.ff.net.2.weight",
            "action_decoder.layer2.W",
        }:
            gradients.append({"name": name, "l2": norm, "max_abs": float(grad.abs().max())})
    before = {name: p.detach().clone() for name, p in head.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-6, foreach=False)
    optimizer.step()
    sync(torch, device)
    del optimizer
    deltas = []
    update_snapshot = {}
    updated_finite = True
    for name, param in head.named_parameters():
        if name not in before:
            continue
        delta = (param.detach() - before[name]).float()
        updated_finite = updated_finite and bool(torch.isfinite(param).all())
        deltas.append((float(delta.abs().max()), float(delta.norm()), int((delta != 0).sum())))
        update_snapshot[name] = (delta[category] if name in category_names else delta).detach().cpu().clone()
    result = {
        "label": label,
        "device": str(device),
        "attention_kind": attention_kind,
        "attention_bindings": bindings,
        "source_formula_controls": controls,
        "inactive_category_gradients_exact_zero": inactive_zero,
        "updated_parameters_finite": updated_finite,
        "action_prediction": tensor_summary(torch, actions),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradient_parameter_count": len([p for p in head.parameters() if p.requires_grad]),
        "gradient_missing_count": len(missing),
        "gradient_missing_names": missing[:8],
        "gradient_nonzero_parameter_count": nonzero,
        "gradient_global_l2": math.sqrt(grad_l2_sq),
        "representative_gradients": gradients,
        "update_max_abs": max(row[0] for row in deltas),
        "update_l2": math.sqrt(sum(row[1] ** 2 for row in deltas)),
        "update_nonzero_elements": sum(row[2] for row in deltas),
        "functional_pass": bool(
            torch.isfinite(loss)
            and not missing
            and nonzero > 0
            and inactive_zero and updated_finite and controls["pass"]
            and max(row[0] for row in deltas) > 0
            and bool(torch.isfinite(actions).all())
            and list(actions.shape) == [1, 16, 32]
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes_after_head_transfer": torch.musa.max_memory_allocated() if device.startswith("musa") else None,
        "peak_reserved_bytes_after_head_transfer": torch.musa.max_memory_reserved() if device.startswith("musa") else None,
    }
    del head, before
    gc.collect()
    if device.startswith("musa"):
        torch.musa.empty_cache()
    print(json.dumps({"stage": label + "_complete", "functional_pass": result["functional_pass"]}), flush=True)
    return result, actions.detach().cpu(), {"prediction": pred.detach().cpu(), "gradients": grad_snapshot, "update": update_snapshot}


def load_backbone(torch, source_tree, weights, metadata_dir, index, config_path, device, data):
    print(json.dumps({"stage": "backbone_" + device}), flush=True)
    from eagle_composite_probe import register_real_classes
    from fp32_backbone_attention import BackboneAttentionKernel, bind_source_kernel
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.siglip import modeling_siglip
    from transformers.utils import is_flash_attn_2_available

    modules = {"qwen3": modeling_qwen3, "siglip": modeling_siglip}
    for family, module in modules.items():
        if sha256(Path(inspect.getfile(module))) != TF_HASHES[family]:
            raise RuntimeError(f"Transformers {family} source hash mismatch")
    before_attention = {key: ALL_ATTENTION_FUNCTIONS[key] for key in ALL_ATTENTION_FUNCTIONS}
    before_flash = is_flash_attn_2_available()
    from gr00t.model.backbone.eagle_backbone import EagleBackbone
    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

    registration = {}
    with register_real_classes(AutoConfig, AutoModel, CONFIG_MAPPING, Eagle2_5_VLConfig,
                               Eagle2_5_VLForConditionalGeneration, registration):
        model = EagleBackbone(tune_llm=False, tune_visual=False, select_layer=12,
                              project_to_dim=None, use_flash_attention=False,
                              load_bf16=False).eval()
        expected = sorted(key.removeprefix("backbone.") for key in index["weight_map"] if key.startswith("backbone."))
        shards = sorted({index["weight_map"]["backbone." + key] for key in expected})
        state = {}
        with contextlib.ExitStack() as stack:
            handles = {name: stack.enter_context(safe_open(weights / name, framework="pt", device="cpu")) for name in shards}
            for key in expected:
                full = "backbone." + key
                value = handles[index["weight_map"][full]].get_tensor(full)
                if value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(model.state_dict()[key].shape):
                    raise RuntimeError(f"Backbone checkpoint contract mismatch: {key}")
                state[key] = value
            loaded = model.load_state_dict({key: value.float() for key, value in state.items()}, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"Backbone strict load failed: {loaded}")
        local = model.to(device).eval()
        bindings = []
        if device.startswith("musa"):
            for name, module in local.named_modules():
                family = "qwen3" if isinstance(module, modeling_qwen3.Qwen3Attention) else (
                    "siglip" if isinstance(module, modeling_siglip.SiglipAttention) else None)
                if family:
                    source = modules[family]
                    kernel = BackboneAttentionKernel(
                        family, source.eager_attention_forward,
                        repeat_kv=getattr(source, "repeat_kv", None), fallback=True)
                    bindings.append(bind_source_kernel(module, kernel))
        batch = {
            "eagle_input_ids": data["ids"].to(device),
            "eagle_attention_mask": data["input_mask"].to(device),
            "eagle_pixel_values": data["pixels"].to(device),
            "eagle_image_flags": data["image_flags"].to(device),
            "eagle_image_sizes": data["image_sizes"].to(device),
        }
        with torch.no_grad():
            output = local(local.prepare_input(batch))
        sync(torch, device)
        features = output.backbone_features.detach().cpu()
        mask = output.backbone_attention_mask.detach().cpu()
        result = {
            "device": str(device),
            "features": tensor_summary(torch, features),
            "mask_exact": bool(torch.equal(mask, data["backbone_mask"])),
            "features_finite": bool(torch.isfinite(features).all()),
            "attention_bindings": len(bindings),
            "attention_globals_unchanged": (
                list(ALL_ATTENTION_FUNCTIONS) == list(before_attention)
                and all(ALL_ATTENTION_FUNCTIONS[k] is v for k, v in before_attention.items())
                and before_flash == is_flash_attn_2_available()
            ),
        }
        del local, model, state, output
        gc.collect()
        if device.startswith("musa"):
            torch.musa.empty_cache()
    # The context manager restores temporary Transformers registrations only
    # after the model has run; record that post-condition after leaving it.
    result["registration_restored"] = registration.get("model_registries_restored", False)
    return result, features


def backbone_inputs(torch, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(3, 10000, (1, 570), generator=gen)
    ids[:, 3:515] = 151669
    mask = torch.zeros((1, 570), dtype=torch.long)
    mask[:, :550] = 1
    ids[mask == 0] = 0
    return {
        "ids": ids,
        "input_mask": mask,
        "pixels": torch.randn((2, 3, 224, 224), generator=gen) * 0.5,
        "image_flags": torch.ones((2, 1), dtype=torch.long),
        "image_sizes": torch.full((2, 2), 224, dtype=torch.long),
        "backbone_mask": mask,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).parent / "prepared/spatial-full/source-preparation.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).parent / "checkpoint_metadata")
    parser.add_argument("--checkpoint-config", type=Path, default=Path(__file__).parent.parent / "backbone_sources/checkpoint-config.json")
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--with-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=271829)
    parser.add_argument("--backbone-seed", type=int, default=271829)
    parser.add_argument("--embodiment-id", type=int, choices=range(32), default=31)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.cpu_threads < 1:
        parser.error("Output must be new and cpu-threads positive")
    result = {"pass": False, "scope": __doc__, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "started": time.time()}
    try:
        import torch
        import transformers
        if transformers.__version__ != "4.51.3":
            raise RuntimeError(f"Requires Transformers 4.51.3, got {transformers.__version__}")
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(args.cpu_threads)
        if args.device == "musa":
            import torch_musa
            if not torch.musa.is_available():
                raise RuntimeError("MUSA is not available")
        index_path = args.metadata_dir / "model.safetensors.index.json"
        if sha256(index_path) != INDEX_SHA256:
            # The index hash is checked for identity but calculated once here
            # so an upstream metadata refresh cannot silently change the test.
            raise RuntimeError(f"Pinned checkpoint index hash mismatch: {sha256(index_path)}")
        index = json.loads(index_path.read_text())["weight_map"]
        index_wrapper = {"weight_map": index}
        from eagle_composite_probe import verify_sources, tensor_info
        from download_spatial_weights import validate_file
        preparation, sources = verify_sources(args)
        weight_validation = [validate_file(args.weights / name, name, args.metadata_dir, index_wrapper)
                             for name in sorted(set(index.values()))]
        data = config_and_inputs(torch, args.checkpoint_config, args.seed, args.embodiment_id)
        action_template, action_load = load_action_head(torch, args.source_tree, args.checkpoint_config, args.weights, index_wrapper)
        result.update(
            versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "safetensors")},
            action_head_load=action_load,
            action_config_sha256=sha256(args.checkpoint_config),
            probe_sha256=sha256(Path(__file__)),
            sources={**sources, **{relative: {"sha256": digest} for relative, digest in ACTION_SOURCE_HASHES.items()}},
            source_preparation=preparation,
            weight_validation=weight_validation,
            fixture={key: tensor_info(torch, value) for key, value in data.items()},
            cpu_threads=torch.get_num_threads(),
            seed=args.seed,
            embodiment_id=args.embodiment_id,
            inference_noise="Standard normal fixed on CPU and copied exactly to every device",
            gradient_comparison_scope="Complete active-embodiment parameter slices and all shared action-head parameters",
            original_backbone_strict_gate_overridden=False,
        )
        if args.device == "musa":
            result["versions"]["torch_musa"] = importlib.metadata.version("torch_musa")
        result["support_sha256"] = {"fp32_attention_processor.py": sha256(Path(__file__).parent.parent / "fp32_attention_processor.py")}
        backbone_pair = None
        if args.with_backbone:
            bdata = backbone_inputs(torch, args.backbone_seed)
            result["backbone_inputs"] = {key: tensor_info(torch, value) for key, value in bdata.items()}
            cpu_record, cpu_features = load_backbone(torch, args.source_tree, args.weights, args.metadata_dir, index_wrapper, args.checkpoint_config, "cpu", bdata)
            musa_record, musa_features = load_backbone(torch, args.source_tree, args.weights, args.metadata_dir, index_wrapper, args.checkpoint_config, args.device, bdata)
            result["backbone_cpu"] = cpu_record
            result["backbone_device"] = musa_record
            result["backbone_feature_device_delta"] = metric(torch, musa_features[bdata["backbone_mask"].bool()], cpu_features[bdata["backbone_mask"].bool()])
            data["features_cpu"] = cpu_features
            data["features_device"] = musa_features
            backbone_pair = True
        else:
            data["features_device"] = data["features_cpu"].clone()

        # CPU head sees both feature variants; the MUSA head is run on the
        # MUSA feature variant. This separates head/device effects from the
        # propagation of the backbone feature difference.
        cpu_data = {**data, "features": data["features_cpu"]}
        device_data = {**data, "features": data["features_device"]}
        cpu_cpu, cpu_actions, cpu_snapshot = run_head(torch, action_template, cpu_data, "cpu", "cpu_head_cpu_features", "native")
        cpu_fallback, cpu_fallback_actions, cpu_fallback_snapshot = run_head(torch, action_template, cpu_data, "cpu", "cpu_fallback_head_cpu_features", "fallback")
        cpu_dev, cpu_dev_actions, cpu_dev_snapshot = run_head(torch, action_template, device_data, "cpu", "cpu_head_device_features", "native")
        result["action_runs"] = [cpu_cpu, cpu_fallback, cpu_dev]
        result["action_comparisons"] = {"cpu_fallback_vs_native": {
            "inference": metric(torch, cpu_fallback_actions, cpu_actions),
            "gradients": tensor_map_comparison(torch, cpu_fallback_snapshot["gradients"], cpu_snapshot["gradients"]),
            "update": tensor_map_comparison(torch, cpu_fallback_snapshot["update"], cpu_snapshot["update"])}}
        if args.device == "musa":
            musa_cpu, musa_cpu_actions, musa_cpu_snapshot = run_head(torch, action_template, cpu_data, "musa", "musa_head_cpu_features", "fallback")
            musa_dev, musa_dev_actions, musa_dev_snapshot = run_head(torch, action_template, device_data, "musa", "musa_head_device_features", "fallback")
            result["action_runs"] += [musa_cpu, musa_dev]
            result["action_comparisons"].update({
                "head_device_on_same_features": {"inference": metric(torch, musa_cpu_actions, cpu_actions), "loss": abs(musa_cpu["loss"] - cpu_cpu["loss"])},
                "backbone_feature_propagation_on_cpu_head": {"inference": metric(torch, cpu_dev_actions, cpu_actions), "loss": abs(cpu_dev["loss"] - cpu_cpu["loss"]), "gradient_global_l2_abs": abs(cpu_dev["gradient_global_l2"] - cpu_cpu["gradient_global_l2"])},
                "backbone_feature_propagation_on_musa_head": {"inference": metric(torch, musa_dev_actions, musa_cpu_actions), "loss": abs(musa_dev["loss"] - musa_cpu["loss"]), "gradient_global_l2_abs": abs(musa_dev["gradient_global_l2"] - musa_cpu["gradient_global_l2"])},
                "end_to_end_cpu_cpu_vs_musa_device": {"inference": metric(torch, musa_dev_actions, cpu_actions), "loss": abs(musa_dev["loss"] - cpu_cpu["loss"]), "gradient_global_l2_abs": abs(musa_dev["gradient_global_l2"] - cpu_cpu["gradient_global_l2"])},
            })
            snapshot_pairs = {
                "head_device_on_same_features": (musa_cpu_snapshot, cpu_snapshot),
                "backbone_feature_propagation_on_cpu_head": (cpu_dev_snapshot, cpu_snapshot),
                "backbone_feature_propagation_on_musa_head": (musa_dev_snapshot, musa_cpu_snapshot),
                "end_to_end_cpu_cpu_vs_musa_device": (musa_dev_snapshot, cpu_snapshot),
            }
            for name, (actual, expected) in snapshot_pairs.items():
                result["action_comparisons"][name]["prediction"] = metric(torch, actual["prediction"], expected["prediction"])
                result["action_comparisons"][name]["gradients"] = tensor_map_comparison(torch, actual["gradients"], expected["gradients"])
                result["action_comparisons"][name]["update"] = tensor_map_comparison(torch, actual["update"], expected["update"])
            result["numerical_action_gate_pass"] = all(
                comparison["inference"]["allclose_atol_rtol_2e-4"]
                and comparison["inference"]["relative_l2_le_2e-4"]
                for comparison in result["action_comparisons"].values())
            result["functional_pass"] = all(row["functional_pass"] for row in result["action_runs"])
        else:
            result["functional_pass"] = all(row["functional_pass"] for row in result["action_runs"])
        result["backbone_pair_used"] = backbone_pair
        backbone_contract = not args.with_backbone or all(
            all(record[field] for field in ("mask_exact", "features_finite", "registration_restored", "attention_globals_unchanged"))
            for record in (result["backbone_cpu"], result["backbone_device"]))
        result["functional_completion"] = bool(result["functional_pass"] and backbone_contract)
        # Overall exit is completion of this functional experiment; original
        # numerical gates are independently retained, including failed features.
        result["pass"] = result["functional_completion"]
    except Exception as error:
        result.update(error_class=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=12))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": result["pass"], "functional_pass": result.get("functional_pass"), "error": result.get("error")}, indent=2), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
