#!/usr/bin/env python3
"""Strictly load the complete Spatial Eagle backbone and compare frozen forward.

Full source geometry, B1/S570, two synthetic raw 224px images. Checkpoint BF16
backbone weights are converted to FP32. Action tensors are not loaded/executed.
No tokenizer/image processor, LIBERO, backward, optimizer, rollout or learning.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

from eagle_composite_probe import (
    TF_HASHES, metrics, register_real_classes, tensor_info, verify_sources,
)
from fp32_backbone_attention import BackboneAttentionKernel, bind_source_kernel
from download_spatial_weights import INDEX_HASH, file_hash, validate_file


def stage(name):
    print(json.dumps({"stage": name}), flush=True)


def state_record(torch, model):
    records = {key: tensor_info(torch, value) for key, value in model.state_dict().items()}
    return {"sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "all_finite": all(value["finite"] for value in records.values()), "tensors": records}


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=root / "prepared/spatial-full/source-preparation.json")
    parser.add_argument("--metadata-dir", type=Path, default=root / "checkpoint_metadata")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.cpu_threads < 1:
        parser.error("Output must be new and cpu-threads positive")
    result = {"pass": False, "probe_sha256": file_hash(__file__), "rows": [],
              "scope": __doc__, "device": args.device, "dtype": "fp32", "weight_load": None}
    try:
        prep, sources = verify_sources(args)
        index_path = args.metadata_dir / "model.safetensors.index.json"
        if file_hash(index_path) != INDEX_HASH:
            raise ValueError("Pinned index hash mismatch")
        index = json.loads(index_path.read_text())
        names = sorted(key for key in index["weight_map"] if key.startswith("backbone."))
        if len(names) != 585:
            raise ValueError("Expected all 585 backbone tensors")
        required_shards = sorted({index["weight_map"][key] for key in names})
        stage("validate_complete_required_shards")
        weight_files = [validate_file(args.weights / name, name, args.metadata_dir, index)
                        for name in required_shards]
        result.update(source_preparation=prep, sources=sources, weight_files=weight_files,
                      required_backbone_shards=required_shards,
                      unused_action_tensor_count=len(index["weight_map"]) - len(names))
        import torch
        import transformers
        from safetensors import safe_open
        if transformers.__version__ != "4.51.3":
            raise ValueError("Requires pinned Transformers")
        torch.set_num_threads(args.cpu_threads)
        torch.set_default_dtype(torch.float32)
        if args.device == "musa":
            import torch_musa
            if not torch.musa.is_available():
                raise RuntimeError("MUSA not available")
        from transformers import AutoConfig, AutoModel
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.qwen3 import modeling_qwen3
        from transformers.models.siglip import modeling_siglip
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.utils import is_flash_attn_2_available
        modules = {"qwen3": modeling_qwen3, "siglip": modeling_siglip}
        for family, module in modules.items():
            if file_hash(inspect.getfile(module)) != TF_HASHES[family]:
                raise ValueError("Installed Transformers source mismatch")
        attention_before = {key: ALL_ATTENTION_FUNCTIONS[key] for key in ALL_ATTENTION_FUNCTIONS}
        fa_before = is_flash_attn_2_available()
        class_before = {f: (m.Qwen3Attention if f == "qwen3" else m.SiglipAttention).forward
                        for f, m in modules.items()}
        if "gr00t" in sys.modules:
            raise RuntimeError("Use a fresh process")
        sys.path.insert(0, str(args.source_tree.resolve()))
        from gr00t.model.backbone.eagle_backbone import EagleBackbone
        from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
        from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
        registration = {}
        with register_real_classes(AutoConfig, AutoModel, CONFIG_MAPPING,
                                   Eagle2_5_VLConfig, Eagle2_5_VLForConditionalGeneration, registration):
            stage("actual_full_cpu_constructor")
            torch.manual_seed(271828)
            model = EagleBackbone(tune_llm=False, tune_visual=False, select_layer=12,
                                  project_to_dim=None, use_flash_attention=False, load_bf16=False).eval()
            eagle = model.eagle_model
            contract = {"language_layers": len(eagle.language_model.model.layers),
                        "vision_layers": len(eagle.vision_model.vision_model.encoder.layers),
                        "connector_shape": list(eagle.mlp1[0].weight.shape),
                        "image_tokens_per_image": eagle.num_image_token,
                        "projector": type(model.eagle_linear).__name__,
                        "eager": [eagle.config._attn_implementation,
                                  eagle.vision_model.config._attn_implementation,
                                  eagle.language_model.config._attn_implementation],
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if contract != {"language_layers": 12, "vision_layers": 27, "connector_shape": [2048, 1152],
                            "image_tokens_per_image": 256, "projector": "Identity", "eager": ["eager"] * 3, "trainable": 0}:
                raise ValueError("Full source constructor contract mismatch")
            result["contract"] = contract
            actual_keys = set(model.state_dict())
            expected_keys = {key.removeprefix("backbone.") for key in names}
            if actual_keys != expected_keys:
                raise ValueError("Actual source and checkpoint backbone key sets differ")
            stage("real_strict_load_state_dict")
            loaded_records = {}
            with contextlib.ExitStack() as stack:
                handles = {name: stack.enter_context(safe_open(args.weights / name, framework="pt", device="cpu"))
                           for name in required_shards}
                state = {key.removeprefix("backbone."): handles[index["weight_map"][key]].get_tensor(key) for key in names}
                source_state = model.state_dict()
                aliases = {}
                for key, value in source_state.items():
                    aliases.setdefault((value.data_ptr(), value.storage_offset(), tuple(value.shape)), []).append(key)
                    if state[key].shape != value.shape or state[key].dtype != torch.bfloat16 or value.dtype != torch.float32:
                        raise ValueError("Checkpoint/constructor shape or expected dtype mismatch: " + key)
                tied_groups = [keys for keys in aliases.values() if len(keys) > 1]
                for keys in tied_groups:
                    if not all(torch.equal(state[keys[0]], state[key]) for key in keys[1:]):
                        raise ValueError("Aliased model parameters have inconsistent checkpoint values")
                loaded = model.load_state_dict(state, strict=True)
                if loaded.missing_keys or loaded.unexpected_keys:
                    raise ValueError("Strict load returned missing/unexpected keys")
                for key, value in model.state_dict().items():
                    expected = tensor_info(torch, state[key].float())
                    actual = tensor_info(torch, value)
                    if expected != actual or not actual["finite"]:
                        raise ValueError("Loaded FP32 tensor differs from checkpoint conversion: " + key)
                    loaded_records[key] = {"source_dtype": "torch.bfloat16", **actual}
                del state, source_state
            result["weight_load"] = {"strict": True, "missing_keys": [], "unexpected_keys": [],
                                     "tensor_count": len(loaded_records), "tied_groups": tied_groups,
                                     "all_loaded_values_exact_bf16_to_fp32": True, "tensors": loaded_records}
            initial = state_record(torch, model)
            result["loaded_state_sha256"] = initial["sha256"]
            gen = torch.Generator().manual_seed(271829)
            ids = torch.randint(3, 10000, (1, 570), generator=gen)
            ids[:, 3:515] = eagle.image_token_index
            mask = (torch.arange(570)[None, :] < 550).long()
            ids[mask == 0] = 0
            batch = {"eagle_input_ids": ids, "eagle_attention_mask": mask,
                     "eagle_pixel_values": torch.randn(2, 3, 224, 224, generator=gen) * 0.5,
                     "eagle_image_flags": torch.ones(2, 1, dtype=torch.long),
                     "eagle_image_sizes": torch.full((2, 2), 224, dtype=torch.long)}
            result["inputs"] = {key: tensor_info(torch, value) for key, value in batch.items()}
            reference = peer = None
            kinds = [("native_cpu", "cpu"), ("fallback_cpu", "cpu")]
            if args.device == "musa":
                kinds.append(("fallback_musa", "musa"))
            for kind, device in kinds:
                stage(kind)
                started = time.monotonic()
                local = copy.deepcopy(model).to(device).eval()
                if device == "musa":
                    torch.musa.reset_peak_memory_stats()
                traces, bindings, captures = [], {}, {}
                for name, module in local.named_modules():
                    family = "qwen3" if isinstance(module, modeling_qwen3.Qwen3Attention) else (
                        "siglip" if isinstance(module, modeling_siglip.SiglipAttention) else None)
                    if family:
                        src = modules[family]
                        kernel = BackboneAttentionKernel(family, src.eager_attention_forward,
                            repeat_kv=getattr(src, "repeat_kv", None), fallback=kind != "native_cpu",
                            trace=lambda row, name=name: traces.append({"module": name, **row}))
                        bindings[name] = bind_source_kernel(module, kernel)
                if len(bindings) != 39:
                    raise ValueError("Expected 12 Qwen3 + 27 Siglip attention bindings")
                def connector_hook(module, inputs, output):
                    captures["connector"] = output.detach().cpu().clone()
                def language_hook(module, inputs, keywords):
                    captures["fused"] = keywords["inputs_embeds"].detach().cpu().clone()
                def eagle_hook(module, inputs, output):
                    captures["logits"] = output.logits.detach().cpu().clone()
                    captures["selected"] = output.hidden_states[12].detach().cpu().clone()
                    captures["cache_none"] = output.past_key_values is None
                def vision_hook(module, inputs, output):
                    captures["vision_pooler"] = output.pooler_output.detach().cpu().clone()
                hooks = [local.eagle_model.mlp1.register_forward_hook(connector_hook),
                         local.eagle_model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True),
                         local.eagle_model.register_forward_hook(eagle_hook),
                         local.eagle_model.vision_model.register_forward_hook(vision_hook)]
                try:
                    with torch.no_grad():
                        out = local(local.prepare_input({key: value.to(device) for key, value in batch.items()}))
                    if device == "musa":
                        torch.musa.synchronize()
                finally:
                    for hook in hooks:
                        hook.remove()
                features = out.backbone_features.detach().cpu()
                if reference is None:
                    reference = {"features": features.clone(), "logits": captures["logits"], "connector": captures["connector"]}
                comparison = {"features": metrics(torch, features[mask.bool()], reference["features"][mask.bool()], "fp32"),
                              "logits": metrics(torch, captures["logits"][mask.bool()], reference["logits"][mask.bool()], "fp32"),
                              "connector": metrics(torch, captures["connector"], reference["connector"], "fp32")}
                image_slots = ids == eagle.image_token_index
                source_contract = {"image_substitution_exact": torch.equal(captures["fused"][image_slots], captures["connector"].reshape(-1, 2048)),
                                   "identity_exact": torch.equal(features, captures["selected"]),
                                   "mask_exact": torch.equal(out.backbone_attention_mask.detach().cpu(), mask),
                                   "cache_none": captures["cache_none"],
                                   "no_grad": not out.backbone_features.requires_grad and all(p.grad is None and not p.requires_grad for p in local.parameters()),
                                   "bindings_instance_only_source_code": all(v["same_source_code_object"] and v["private_globals"] and v["instance_only"] for v in bindings.values()),
                                   "blocked_probabilities_zero": all(t["blocked_probability_max_abs"] in (None, 0.0) for t in traces)}
                source_contract["features_and_all_captures_finite"] = bool(torch.isfinite(features).all()) and all(
                    bool(torch.isfinite(v).all()) for v in captures.values() if torch.is_tensor(v))
                after = state_record(torch, local)
                source_contract["loaded_state_unchanged"] = after["sha256"] == initial["sha256"] and after["all_finite"]
                row = {"kind": kind, "device": device, "comparison": comparison, "source_contract": source_contract,
                       "features": tensor_info(torch, features), "attention_traces": traces,
                       "elapsed_seconds_including_clone_hash": time.monotonic() - started,
                       "peak_allocated_bytes": torch.musa.max_memory_allocated() if device == "musa" else None,
                       "peak_reserved_bytes": torch.musa.max_memory_reserved() if device == "musa" else None}
                if kind == "fallback_cpu":
                    peer = features.clone()
                elif kind == "fallback_musa":
                    row["comparison_to_cpu_fallback"] = metrics(torch, features[mask.bool()], peer[mask.bool()], "fp32")
                row["pass"] = all(source_contract.values()) and all(v["pass"] for v in comparison.values()) and row.get("comparison_to_cpu_fallback", {"pass": True})["pass"]
                result["rows"].append(row)
                print(json.dumps({"kind": kind, "pass": row["pass"], "comparison": comparison}), flush=True)
                del local, out, features, captures, after
                gc.collect()
                if device == "musa":
                    torch.musa.empty_cache()
        globals_ok = (list(ALL_ATTENTION_FUNCTIONS) == list(attention_before)
                      and all(ALL_ATTENTION_FUNCTIONS[k] is v for k, v in attention_before.items())
                      and fa_before == is_flash_attn_2_available()
                      and all((modules[f].Qwen3Attention if f == "qwen3" else modules[f].SiglipAttention).forward is v for f, v in class_before.items()))
        result.update(temporary_model_registration=registration, attention_globals_unchanged=globals_ok,
                      cpu_threads=torch.get_num_threads())
        result["pass"] = all(row["pass"] for row in result["rows"]) and len(result["rows"]) == len(kinds) and globals_ok and registration["model_registries_restored"]
    except Exception as error:
        result.update(error_class=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
