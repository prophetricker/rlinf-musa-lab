#!/usr/bin/env python3
"""Construct the fixed, minimally patched real EagleBackbone with a tiny config.

CPU source eager vs CPU/MUSA FP32 attention kernels. Frozen real SigLIP,
connector, image token replacement, Qwen3, and Identity feature wrapper.
No weights, optimizer, fake modules, availability override, or hand-written
multimodal forward. Temporary real AutoConfig/AutoModel registration is restored.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import sys
import traceback

from fp32_backbone_attention import BackboneAttentionKernel, bind_source_kernel

TF_VERSION = "4.51.3"
TF_HASHES = {
    "qwen3": "704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2",
    "siglip": "e8972dfaa3903191b936a76f0887ccf2e6648aa26cbde6a148b24a0c5790ecd2",
}
KERNEL_HASH = "db580ebeea14b7f7b174927e2d589ddb720f4f31a1991f4ddecc512d12be7cf1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_json_values(torch, value):
    """Serialize the actual constructed config without mutating its dtype."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: config_json_values(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_json_values(torch, item) for item in value]
    return value


def tensor_info(torch, value):
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(), "finite": bool(torch.isfinite(value).all())}


def state_info(torch, model):
    tensors = {name: tensor_info(torch, value) for name, value in model.state_dict().items()}
    return {"sha256": hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(), "tensors": tensors}


def metrics(torch, actual, expected, dtype):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        return {"pass": False, "finite": False}
    atol, rtol, limit = (2e-4, 2e-4, 2e-4) if dtype == "fp32" else (0.025, 0.04, 0.04)
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1e-12))
    return {"finite": True, "max_abs": float(delta.abs().max()), "relative_l2": relative,
            "atol": atol, "rtol": rtol, "relative_l2_limit": limit,
            "pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)) and relative <= limit}


def verify_sources(args):
    manifest = json.loads(args.source_manifest.read_text())
    sources = {}
    for record in manifest["files"]:
        path = args.source_tree / record["path"]
        digest = file_hash(path)
        if digest != record["prepared_sha256"]:
            raise RuntimeError(f"Prepared actual source hash mismatch: {path}")
        sources[record["path"]] = {"path": str(path.resolve()), "sha256": digest,
                                    "original_sha256": record["original_sha256"]}
    if file_hash(Path(__file__).with_name("fp32_backbone_attention.py")) != KERNEL_HASH:
        raise RuntimeError("Frozen backbone kernel hash mismatch")
    return manifest, sources


@contextlib.contextmanager
def register_real_classes(AutoConfig, AutoModel, CONFIG_MAPPING, config_cls, model_cls, record):
    mappings = (CONFIG_MAPPING, AutoModel._model_mapping)
    before = [dict(mapping._extra_content) for mapping in mappings]
    try:
        AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)
        AutoModel.register(config_cls, model_cls, exist_ok=True)
        record["temporary_model_registry_change"] = True
        record["config_class"] = config_cls.__module__ + "." + config_cls.__name__
        record["model_class"] = model_cls.__module__ + "." + model_cls.__name__
        yield
    finally:
        for mapping, snapshot in zip(mappings, before):
            mapping._extra_content.clear()
            mapping._extra_content.update(snapshot)
        record["model_registries_restored"] = all(
            mapping._extra_content.keys() == snapshot.keys()
            and all(mapping._extra_content[key] is value for key, value in snapshot.items())
            for mapping, snapshot in zip(mappings, before))


def inputs(torch, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    ids = torch.randint(3, 255, (2, 24), generator=gen)
    ids[:, 3:19] = 255
    mask = torch.arange(24)[None, :] < torch.tensor([22, 23])[:, None]
    ids[~mask] = 0
    pixels = (torch.randn(3, 3, 56, 56, generator=gen) * 0.5).to(dtype)
    flags = torch.tensor([[1], [1], [0]], dtype=torch.long)
    image_sizes = torch.tensor([[56, 56], [56, 56], [56, 56]], dtype=torch.long)
    data = {"eagle_input_ids": ids, "eagle_attention_mask": mask.long(),
            "eagle_pixel_values": pixels, "eagle_image_flags": flags,
            "eagle_image_sizes": image_sizes}
    pad_ids = ids.clone()
    random_ids = torch.randint(3, 255, ids.shape, generator=gen)
    pad_ids[~mask] = random_ids[~mask]
    future_ids = ids.clone()
    future_ids[:, 21:] = random_ids[:, 21:]
    rejected = pixels.clone()
    rejected[2] = rejected[2] + 0.5
    selected = pixels.clone()
    selected[:2] = selected[:2] + 0.25
    return data, mask, {
        "padding_text": {"eagle_input_ids": pad_ids},
        "future_text": {"eagle_input_ids": future_ids},
        "rejected_image": {"eagle_pixel_values": rejected},
        "selected_image": {"eagle_pixel_values": selected},
    }


def bind_attention(model, modules, fallback, traces, label):
    bindings = {}
    for name, module in model.named_modules():
        family = "qwen3" if isinstance(module, modules["qwen3"].Qwen3Attention) else (
            "siglip" if isinstance(module, modules["siglip"].SiglipAttention) else None)
        if family is None:
            continue
        src = modules[family]
        kernel = BackboneAttentionKernel(family, src.eager_attention_forward,
            repeat_kv=getattr(src, "repeat_kv", None), fallback=fallback,
            trace=lambda row, name=name: traces.append({"call": label[0], "module": name, **row}))
        bindings[name] = bind_source_kernel(module, kernel)
    if len(bindings) != 4:
        raise RuntimeError("Expected two real Qwen3 and two real SigLIP attention modules")
    return bindings


def execute(torch, template, batch, valid, perturbations, kind, device, modules):
    model = copy.deepcopy(template).to(device).eval()
    initial_state = state_info(torch, model)
    traces, label, captures = [], ["main"], {}
    bindings = bind_attention(model, modules, kind != "native_cpu", traces, label)
    eagle = model.eagle_model
    capture = lambda: captures.setdefault(label[0], {})

    def connector_hook(module, positional, output):
        capture()["connector_output"] = output.detach().cpu().clone()

    def language_pre_hook(module, positional, keywords):
        capture()["language_inputs_embeds"] = keywords["inputs_embeds"].detach().cpu().clone()
        capture()["language_attention_mask"] = keywords["attention_mask"].detach().cpu().clone()

    def vision_hook(module, positional, output):
        capture()["vision_hidden"] = output.last_hidden_state.detach().cpu().clone()
        capture()["vision_pooler"] = output.pooler_output.detach().cpu().clone()

    def eagle_hook(module, positional, output):
        capture()["selected_hidden_state"] = output.hidden_states[model.select_layer].detach().cpu().clone()
        capture()["logits"] = output.logits.detach().cpu().clone()
        capture()["hidden_states_count"] = len(output.hidden_states)
        capture()["cache_none"] = output.past_key_values is None

    handles = [eagle.mlp1.register_forward_hook(connector_hook),
               eagle.language_model.register_forward_pre_hook(language_pre_hook, with_kwargs=True),
               eagle.vision_model.register_forward_hook(vision_hook),
               eagle.register_forward_hook(eagle_hook)]
    local_batch = {name: value.to(device) for name, value in batch.items()}
    printed = io.StringIO()
    outputs = {}
    try:
        with torch.no_grad(), contextlib.redirect_stdout(printed):
            for case_name, updates in [("main", {})] + list(perturbations.items()):
                label[0] = case_name
                changed = {**local_batch, **{name: value.to(device) for name, value in updates.items()}}
                prepared = model.prepare_input(changed)
                result = model(prepared)
                outputs[case_name] = result.backbone_features.detach().cpu().clone()
                capture()["output_requires_grad"] = result.backbone_features.requires_grad
                capture()["backbone_attention_mask"] = result.backbone_attention_mask.detach().cpu().clone()
        if device == "musa":
            torch.musa.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    main = captures["main"]
    ids, flags = batch["eagle_input_ids"], batch["eagle_image_flags"].view(-1)
    image_selector = ids == eagle.image_token_index
    connected = main["connector_output"][flags == 1].reshape(-1, 512)
    with torch.no_grad():
        token_embeddings = eagle.language_model.get_input_embeddings()(batch["eagle_input_ids"].to(device)).detach().cpu()
    fused = main["language_inputs_embeds"]
    substitution = {
        "selected_image_token_count": int(image_selector.sum()),
        "connected_selected_image_token_count": connected.shape[0],
        "all_image_token_embeddings_match_real_connector_exactly": torch.equal(fused[image_selector], connected),
        "all_nonimage_token_embeddings_match_real_lookup_exactly": torch.equal(fused[~image_selector], token_embeddings[~image_selector]),
        "source_truncation_warning_absent": "warning:" not in printed.getvalue(),
        "extra_image_computed_then_filtered": main["connector_output"].shape[0] == 3 and int((flags == 1).sum()) == 2,
    }
    invariance = {}
    for name in ("padding_text", "future_text", "rejected_image"):
        selector = valid if name != "future_text" else torch.arange(24)[None, :].expand_as(valid) < 21
        delta = (outputs[name][selector] - outputs["main"][selector]).float()
        invariance[name] = {"max_abs": float(delta.abs().max()), "exact_zero": bool((delta == 0).all()),
                            "selected_tokens": int(selector.sum())}
    image_delta = (outputs["selected_image"] - outputs["main"]).float()
    image_control = {"changed_valid_output_l2": float(image_delta[valid].norm()),
                     "preimage_prefix3_max_abs": float(image_delta[:, :3].abs().max()),
                     "preimage_prefix3_exact_zero": bool((image_delta[:, :3] == 0).all()),
                     "selected_image_changes_valid_output": bool((image_delta[valid] != 0).any())}
    final_state = state_info(torch, model)
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters())
    all_finite = all(bool(torch.isfinite(value).all()) for value in outputs.values()) and all(
        tensor_info(torch, value)["finite"] for cap in captures.values() for value in cap.values() if torch.is_tensor(value))
    flags_ok = all(cap["cache_none"] and cap["hidden_states_count"] == 3 and not cap["output_requires_grad"]
                   and torch.equal(cap["backbone_attention_mask"], batch["eagle_attention_mask"])
                   and torch.equal(cap["language_attention_mask"], batch["eagle_attention_mask"])
                   for cap in captures.values())
    identity_ok = all(torch.equal(outputs[name], cap["selected_hidden_state"]) for name, cap in captures.items())
    binding_ok = all(b["same_source_code_object"] and b["private_globals"] and b["instance_only"] for b in bindings.values())
    blocked_ok = all(t["blocked_probability_max_abs"] in (None, 0.0) for t in traces)
    substitution_ok = (substitution["selected_image_token_count"] == substitution["connected_selected_image_token_count"] == 32
                       and all(substitution[key] for key in substitution if key not in (
                           "selected_image_token_count", "connected_selected_image_token_count")))
    metadata = {"bindings": bindings, "binding_invariants_pass": binding_ok,
                "attention_traces": traces, "initial_state": initial_state, "final_state_sha256": final_state["sha256"],
                "state_unchanged": initial_state["sha256"] == final_state["sha256"],
                "all_parameters_frozen_grad_none": frozen, "all_outputs_and_captures_finite": all_finite,
                "identity_matches_source_selected_hidden_state": identity_ok,
                "mask_cache_hiddenstate_contracts_pass": flags_ok,
                "substitution": substitution, "perturbation_invariance": invariance,
                "image_positive_control": image_control, "blocked_probabilities_exact_zero": blocked_ok,
                "captures": {name: {key: tensor_info(torch, value) if torch.is_tensor(value) else value
                                     for key, value in cap.items()} for name, cap in captures.items()},
                "scope_pass": all_finite and frozen and binding_ok and blocked_ok and identity_ok and flags_ok
                              and substitution_ok and initial_state["sha256"] == final_state["sha256"]
                              and all(v["exact_zero"] for v in invariance.values())
                              and image_control["selected_image_changes_valid_output"] and image_control["preimage_prefix3_exact_zero"]}
    return {"output": outputs["main"], "logits": main["logits"],
            "connector_output": main["connector_output"], "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).parent / "prepared/source-preparation.json")
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evidence output already exists: {args.output}")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    preparation, sources = verify_sources(args)
    if any(d not in ("fp32", "bf16") for d in args.dtypes.split(",")):
        raise ValueError("Unsupported dtype")
    import torch
    import transformers
    torch.set_num_threads(args.cpu_threads)
    if transformers.__version__ != TF_VERSION:
        raise RuntimeError("Requires pinned Transformers4.51.3")
    if args.device == "musa":
        import torch_musa
        if not torch.musa.is_available():
            raise RuntimeError("Explicit MUSA execution requires an available device")
    from transformers import AutoConfig, AutoModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.siglip import modeling_siglip
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils import is_flash_attn_2_available
    modules = {"qwen3": modeling_qwen3, "siglip": modeling_siglip}
    for family, module in modules.items():
        path = inspect.getfile(module)
        if file_hash(path) != TF_HASHES[family]:
            raise RuntimeError(f"Installed source hash mismatch: {family}")
        sources["transformers_" + family] = {"path": path, "sha256": file_hash(path)}
    before_registry = {key: ALL_ATTENTION_FUNCTIONS[key] for key in ALL_ATTENTION_FUNCTIONS}
    before_fa = is_flash_attn_2_available()
    before_forwards = {family: (module.Qwen3Attention if family == "qwen3" else module.SiglipAttention).forward
                       for family, module in modules.items()}
    if "gr00t" in sys.modules:
        raise RuntimeError("Use a fresh process to import the exact isolated source tree")
    sys.path.insert(0, str(args.source_tree.resolve()))
    from gr00t.model.backbone.eagle_backbone import EagleBackbone
    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
    for cls in (EagleBackbone, Eagle2_5_VLConfig, Eagle2_5_VLForConditionalGeneration):
        actual = Path(inspect.getfile(cls)).resolve()
        if not actual.is_relative_to(args.source_tree.resolve()):
            raise RuntimeError("Actual imported class is outside the prepared tree")
    registration = {}
    evidence = {"probe": str(Path(__file__).resolve()), "probe_sha256": file_hash(__file__),
                "kernel_sha256": KERNEL_HASH, "sources": sources, "source_preparation": preparation,
                "source_manifest_sha256": file_hash(args.source_manifest), "device": args.device,
                "seed": args.seed, "cpu_threads": torch.get_num_threads(),
                "rows": [], "cases": [], "temporary_model_registration": registration,
                "packages": {}, "torch_import_path": torch.__file__,
                "scope": "Real fixed EagleBackbone constructor/forward with minimally patched real Eagle/SigLIP/Qwen3, frozen connector and Identity; tiny random config, no weights/backward/training.",
                "vision_pooler_scope": "Real default vision pooling head executes for all three images. Eagle ignores its output; checked finite/layout only, not numerical parity."}
    for name in ("torch", "torch_musa", "transformers", "huggingface_hub", "tokenizers", "safetensors", "peft", "accelerate", "einops", "timm"):
        try:
            evidence["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            evidence["packages"][name] = None
    overall = True
    with register_real_classes(AutoConfig, AutoModel, CONFIG_MAPPING, Eagle2_5_VLConfig,
                               Eagle2_5_VLForConditionalGeneration, registration):
        for dtype_name in args.dtypes.split(","):
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            torch.manual_seed(args.seed)
            template = EagleBackbone(tune_llm=False, tune_visual=False, select_layer=2,
                                     project_to_dim=None, use_flash_attention=False,
                                     load_bf16=False, eagle_path=None).to(dtype=dtype).eval()
            eagle = template.eagle_model
            contract = {"actual_wrapper_class": type(template).__module__ + "." + type(template).__name__,
                        "actual_eagle_class": type(eagle).__module__ + "." + type(eagle).__name__,
                        "wrapper_select_layer": template.select_layer,
                        "initial_config_llm_layers": eagle.language_model.config.num_hidden_layers,
                        "retained_llm_layers": len(eagle.language_model.model.layers),
                        "vision_layers": len(eagle.vision_model.vision_model.encoder.layers),
                        "vision_pooler_enabled": eagle.vision_model.vision_model.use_head,
                        "connector_type": type(eagle.mlp1[0]).__name__,
                        "connector_shape": list(eagle.mlp1[0].weight.shape),
                        "feature_projector_type": type(template.eagle_linear).__name__,
                        "num_image_tokens_per_image": eagle.num_image_token,
                        "use_pixel_shuffle": eagle.use_pixel_shuffle,
                        "eager_settings": [eagle.config._attn_implementation, eagle.vision_model.config._attn_implementation,
                                           eagle.language_model.config._attn_implementation],
                        "parameter_count": sum(p.numel() for p in template.parameters()),
                        "trainable_parameter_count": sum(p.numel() for p in template.parameters() if p.requires_grad)}
            if not (contract["retained_llm_layers"] == 2 and contract["initial_config_llm_layers"] == 3
                    and contract["connector_shape"] == [512, 288] and contract["feature_projector_type"] == "Identity"
                    and contract["num_image_tokens_per_image"] == 16 and contract["vision_pooler_enabled"]
                    and contract["eager_settings"] == ["eager"] * 3 and contract["trainable_parameter_count"] == 0):
                raise RuntimeError("Actual constructed composite does not match declared tiny contract")
            batch, valid, perturb = inputs(torch, dtype, args.seed)
            state = state_info(torch, template)
            evidence["cases"].append({"dtype": dtype_name, "contract": contract, "state": state,
                                      "inputs": {name: tensor_info(torch, value) for name, value in batch.items()},
                                      "config": config_json_values(torch, eagle.config.to_dict()),
                                      "source_methods": {"wrapper_forward": inspect.getfile(type(template)),
                                                         "eagle_forward_extract_feature": inspect.getfile(type(eagle))}})
            reference = cpu_fallback = None
            evaluations = [("native_cpu", "cpu"), ("fallback_cpu", "cpu")]
            if args.device == "musa":
                evaluations.append(("fallback_musa", "musa"))
            for kind, device in evaluations:
                row = {"dtype": dtype_name, "kind": kind, "device": device}
                try:
                    if kind != "native_cpu" and reference is None:
                        raise RuntimeError("CPU source reference failed")
                    actual = execute(torch, template, batch, valid, perturb, kind, device, modules)
                    if reference is None:
                        reference = actual
                    comparison = {"valid_features": metrics(torch, actual["output"][valid], reference["output"][valid], dtype_name),
                                  "valid_logits": metrics(torch, actual["logits"][valid], reference["logits"][valid], dtype_name),
                                  "connector_output": metrics(torch, actual["connector_output"], reference["connector_output"], dtype_name)}
                    row.update(actual["metadata"])
                    row["comparison_to_cpu_source_eager"] = comparison
                    row["state_matches_cpu_template"] = row["initial_state"]["sha256"] == state["sha256"]
                    row["pass"] = row["scope_pass"] and row["state_matches_cpu_template"] and all(v["pass"] for v in comparison.values())
                    if kind == "fallback_cpu":
                        cpu_fallback = actual
                    elif kind == "fallback_musa" and cpu_fallback is not None:
                        row["cpu_fallback_peer_features"] = metrics(torch, actual["output"][valid], cpu_fallback["output"][valid], dtype_name)
                        row["pass"] = row["pass"] and row["cpu_fallback_peer_features"]["pass"]
                except Exception as exc:
                    row.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc(limit=6))
                    row["pass"] = False
                evidence["rows"].append(row)
                overall = overall and row["pass"]
                print(json.dumps({key: row[key] for key in ("dtype", "kind", "pass", "error") if key in row}), flush=True)
    registry_unchanged = (list(ALL_ATTENTION_FUNCTIONS) == list(before_registry)
                          and all(ALL_ATTENTION_FUNCTIONS[key] is value for key, value in before_registry.items()))
    classes_unchanged = all((modules[f].Qwen3Attention if f == "qwen3" else modules[f].SiglipAttention).forward is value
                            for f, value in before_forwards.items())
    unused_imports = {name: name not in sys.modules for name in (
        "gr00t.model.gr00t_n1", "gr00t.model.policy", "gr00t.model.backbone.eagle2_hg_model.radio_model", "peft")}
    evidence["no_attention_global_patch"] = {"registry_entries_unchanged": registry_unchanged,
        "source_class_forward_unchanged": classes_unchanged,
        "flash_attention_available_before": before_fa, "flash_attention_available_after": is_flash_attn_2_available()}
    evidence["unused_functionality_modules_not_loaded"] = unused_imports
    evidence["pass"] = (overall and registration["model_registries_restored"] and registry_unchanged and classes_unchanged
                        and before_fa == is_flash_attn_2_available() and all(unused_imports.values()))
    payload = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as evidence_file:
        evidence_file.write(payload)
    return 0 if evidence["pass"] else 1


def run_with_setup_evidence():
    try:
        return main()
    except Exception as exc:
        # Preserve setup/constructor failures as failures even before row
        # execution begins. Never overwrite existing evidence. The registration
        # context's finally still restores mappings on constructor exceptions.
        failure_parser = argparse.ArgumentParser(add_help=False)
        failure_parser.add_argument("--output", type=Path)
        known, _ = failure_parser.parse_known_args()
        if known.output is not None and not known.output.exists():
            failure = {"pass": False, "setup_or_serialization_error": type(exc).__name__ + ": " + str(exc),
                       "traceback": traceback.format_exc(limit=8),
                       "probe_sha256": file_hash(__file__), "kernel_sha256": KERNEL_HASH,
                       "python": sys.executable, "argv": sys.argv, "rows": []}
            known.output.parent.mkdir(parents=True, exist_ok=True)
            with known.output.open("x") as evidence_file:
                evidence_file.write(json.dumps(failure, indent=2) + "\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run_with_setup_evidence())
