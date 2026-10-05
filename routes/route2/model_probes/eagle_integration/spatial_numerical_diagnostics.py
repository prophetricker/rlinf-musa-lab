"""Read-only FP32 diagnosis on real loaded Spatial weights and source modules.

Uses the source CPU forward's exact hidden/mask/RoPE values for independent
layer and operator replays. CPU FP64 formulas are diagnostic references, not
alternative production forwards. Original acceptance thresholds stay unchanged.
No captured tensors or weights are serialized into the public result.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path

from eagle_composite_probe import tensor_info
from fp32_backbone_attention import BackboneAttentionKernel, bind_source_kernel


LAYERS = (3, 7, 11)  # Source zero-based indices, not human ordinal numbers.
VISION_LAYERS = (0, 12, 24, 26)
COMPONENTS = (
    "input_layernorm", "self_attn", "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm",
    "self_attn.o_proj", "post_attention_layernorm", "mlp",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def tree_map(torch, value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, tuple):
        return tuple(tree_map(torch, x, device) for x in value)
    if isinstance(value, list):
        return [tree_map(torch, x, device) for x in value]
    if isinstance(value, dict):
        return {k: tree_map(torch, x, device) for k, x in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Unsupported replay argument type: " + type(value).__name__)


def tensor_layout(torch, value):
    return {**tensor_info(torch, value), "stride": list(value.stride()),
            "contiguous": value.is_contiguous()}


def argument_records(torch, value):
    if torch.is_tensor(value):
        return tensor_layout(torch, value)
    if isinstance(value, (list, tuple)):
        return [argument_records(torch, x) for x in value]
    if isinstance(value, dict):
        return {k: argument_records(torch, x) for k, x in value.items()}
    return value


def transfers_exact(torch, before, after):
    if torch.is_tensor(before):
        return (torch.is_tensor(after) and before.dtype == after.dtype
                and torch.equal(before.cpu(), after.cpu()))
    if isinstance(before, (tuple, list)):
        return type(before) is type(after) and len(before) == len(after) and all(
            transfers_exact(torch, x, y) for x, y in zip(before, after))
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            transfers_exact(torch, before[k], after[k]) for k in before)
    return before == after


@contextlib.contextmanager
def kernel_mode(language, source, fallback, observer=None, family="qwen3"):
    """Temporary instance bindings only; restore the precise previous functions."""
    saved, records = [], []
    try:
        for module in language.modules():
            attention_type = source.Qwen3Attention if family == "qwen3" else source.SiglipAttention
            if not isinstance(module, attention_type):
                continue
            saved.append((module, "forward" in module.__dict__, module.forward))
            kernel = BackboneAttentionKernel(family, source.eager_attention_forward,
                                             repeat_kv=getattr(source, "repeat_kv", None), fallback=fallback)
            if observer is not None:
                original_kernel = kernel
                def observed(attention, query, key, value, mask, scaling,
                             dropout=0.0, _kernel=original_kernel, **kwargs):
                    observer.update(q=query.detach().cpu().clone(),
                                    k=key.detach().cpu().clone(),
                                    v=value.detach().cpu().clone(),
                                    mask=None if mask is None else mask.detach().cpu().clone(),
                                    scaling=scaling)
                    return _kernel(attention, query, key, value, mask,
                                   scaling=scaling, dropout=dropout, **kwargs)
                kernel = observed
            records.append(bind_source_kernel(module, kernel))
        yield records
    finally:
        for module, had_instance, forward in saved:
            if had_instance:
                module.forward = forward
            else:
                del module.forward


class SpatialNumericalDiagnostics:
    def __init__(self, torch, compare):
        self.torch, self.compare = torch, compare
        self.snapshots = {}

    def attach(self, model, kind):
        if kind not in ("native_cpu", "fallback_musa"):
            return []
        torch = self.torch
        snapshot = {"layers": {}, "components": {}, "final_norm": {}, "language": {},
                    "vision": {}, "connector": {}}
        self.snapshots[kind] = snapshot
        handles = []
        def pre(target):
            def capture(module, inputs, kwargs):
                target["args"] = tree_map(torch, inputs, "cpu")
                target["kwargs"] = tree_map(torch, kwargs, "cpu")
            return capture
        def post(target):
            def capture(module, inputs, output):
                value = output if torch.is_tensor(output) else output[0]
                target["output"] = value.detach().cpu().clone()
            return capture
        language = model.eagle_model.language_model
        handles.append(language.register_forward_pre_hook(pre(snapshot["language"]), with_kwargs=True))
        handles.append(language.model.norm.register_forward_pre_hook(pre(snapshot["final_norm"]), with_kwargs=True))
        handles.append(language.model.norm.register_forward_hook(post(snapshot["final_norm"])))
        vision = model.eagle_model.vision_model.vision_model
        handles.append(model.eagle_model.mlp1.register_forward_pre_hook(pre(snapshot["connector"]), with_kwargs=True))
        handles.append(model.eagle_model.mlp1.register_forward_hook(post(snapshot["connector"])))
        if kind == "native_cpu":
            for index in LAYERS:
                layer = language.model.layers[index]
                entry = snapshot["layers"].setdefault(index, {})
                handles.append(layer.register_forward_pre_hook(pre(entry), with_kwargs=True))
                handles.append(layer.register_forward_hook(post(entry)))
                for name in COMPONENTS:
                    target = snapshot["components"].setdefault((index, name), {})
                    module = layer.get_submodule(name)
                    handles.append(module.register_forward_pre_hook(pre(target), with_kwargs=True))
                    handles.append(module.register_forward_hook(post(target)))
            for index in VISION_LAYERS:
                layer = vision.encoder.layers[index]
                entry = snapshot["vision"].setdefault(index, {})
                handles.append(layer.register_forward_pre_hook(pre(entry), with_kwargs=True))
                handles.append(layer.register_forward_hook(post(entry)))
        return handles

    def run(self, cpu_model, gpu_model, mask):
        torch = self.torch
        from transformers.models.qwen3 import modeling_qwen3 as source
        from transformers.models.siglip import modeling_siglip as vision_source
        cpu, gpu = self.snapshots["native_cpu"], self.snapshots["fallback_musa"]
        cpu_language = cpu_model.eagle_model.language_model
        gpu_language = gpu_model.eagle_model.language_model
        valid = mask.bool()
        result = {
            "scope": __doc__, "diagnostic_only": True,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selected_layer_indices_zero_based": list(LAYERS),
            "thresholds_changed": False,
            "layer_replays": [], "operator_replays": [],
            "operator_metric_scope": "All source-shaped elements, including padded query positions; diagnostics only",
        }
        def metric(x, y):
            return self.compare(torch, x, y)
        def valid_metric(x, y):
            return metric(x[valid], y[valid])
        language_kwargs = cpu["language"]["kwargs"]
        if (cpu["language"]["args"] or language_kwargs.get("past_key_values") is not None
                or language_kwargs.get("use_cache") not in (None, False)
                or cpu_language.config.use_cache is not False):
            raise ValueError("Requires actual keyword-only language input, no cache")
        if language_kwargs["inputs_embeds"].shape != (1, 570, 2048):
            raise ValueError("Requires exact full Spatial language input geometry")
        result["fused_input_comparison"] = valid_metric(
            gpu["language"]["kwargs"]["inputs_embeds"], language_kwargs["inputs_embeds"])
        # Same CPU vision/connector/fused input, source MUSA mask and RoPE generation.
        language_outputs = {}
        with torch.no_grad():
            started = time.monotonic()
            torch.musa.reset_peak_memory_stats()
            for fallback in (False, True):
                mode = "fallback" if fallback else "source_eager"
                print(json.dumps({"diagnostic_stage": "cpu_fused_to_musa_language", "mode": mode}), flush=True)
                kwargs = tree_map(torch, language_kwargs, "musa")
                exact = transfers_exact(torch, language_kwargs,
                                        {k: kwargs[k] for k in language_kwargs})
                with kernel_mode(gpu_language, source, fallback):
                    output = gpu_language.model(**kwargs)
                language_outputs[mode] = output.last_hidden_state.detach().cpu().clone()
                result.setdefault("cpu_fused_to_musa_language", {})[mode] = {
                    "input_transfer_exact": exact,
                    "comparison_to_full_cpu_features": valid_metric(
                        language_outputs[mode], cpu["final_norm"]["output"]),
                    "cache_none": output.past_key_values is None,
                }
                del output, kwargs
            result["cpu_fused_to_musa_language"]["source_vs_fallback"] = valid_metric(
                language_outputs["fallback"], language_outputs["source_eager"])
            result["vision_same_input"] = self.vision_replays(cpu_model, gpu_model, vision_source)
            # Source RoPE buffers are nonpersistent; audit them independently.
            cpu_rope, gpu_rope = cpu_language.model.rotary_emb, gpu_language.model.rotary_emb
            rope_hidden = cpu["layers"][LAYERS[0]]["args"][0]
            rope_ids = cpu["layers"][LAYERS[0]]["kwargs"]["position_ids"]
            cpu_position = cpu_rope(rope_hidden, rope_ids)
            gpu_position = gpu_rope(rope_hidden.to("musa"), rope_ids.to("musa"))
            result["rope"] = {
                "position_ids": tensor_layout(torch, rope_ids),
                "source_position_ids_are_arange": torch.equal(rope_ids, torch.arange(570)[None, :]),
                "cpu_inv_freq": tensor_layout(torch, cpu_rope.inv_freq),
                "musa_inv_freq": tensor_layout(torch, gpu_rope.inv_freq),
                "inv_freq_exact": torch.equal(cpu_rope.inv_freq.cpu(), gpu_rope.inv_freq.cpu()),
                "cos": metric(gpu_position[0], cpu_position[0]),
                "sin": metric(gpu_position[1], cpu_position[1]),
                "captured_cpu_tuple_matches_regenerated": all(torch.equal(x, y) for x, y in zip(
                    cpu["layers"][LAYERS[0]]["kwargs"]["position_embeddings"], cpu_position)),
            }
            for index in LAYERS:
                print(json.dumps({"diagnostic_stage": "same_input_layer_and_operators", "index": index}), flush=True)
                entry = cpu["layers"][index]
                args, kwargs = entry["args"], entry["kwargs"]
                if kwargs.get("past_key_value") is not None or kwargs.get("use_cache") is not False:
                    raise ValueError("Layer replay cannot contain a cache")
                if len(args) != 1 or kwargs["attention_mask"].shape != (1, 1, 570, 570):
                    raise ValueError("Layer replay input/mask contract changed")
                outputs = {}
                for device, language in (("cpu", cpu_language), ("musa", gpu_language)):
                    layer = language.model.layers[index]
                    local_args, local_kwargs = tree_map(torch, args, device), tree_map(torch, kwargs, device)
                    for fallback in (False, True):
                        mode = "fallback" if fallback else "source_eager"
                        with kernel_mode(layer, source, fallback):
                            output = layer(*local_args, **local_kwargs)[0].detach().cpu()
                        outputs[(device, mode)] = output.clone()
                        result["layer_replays"].append({
                            "index": index, "device": device, "mode": mode,
                            "arguments": argument_records(torch, {"args": local_args, "kwargs": local_kwargs}),
                            "input_transfer_exact": transfers_exact(torch, (args, kwargs), (local_args, local_kwargs)),
                            "comparison_to_captured_cpu_output": valid_metric(output, entry["output"]),
                        })
                result.setdefault("layer_source_vs_fallback", {})[str(index)] = {
                    device: valid_metric(outputs[(device, "fallback")], outputs[(device, "source_eager")])
                    for device in ("cpu", "musa")}
                for name in COMPONENTS:
                    record = self.operator(cpu_language.model.layers[index].get_submodule(name),
                                           gpu_language.model.layers[index].get_submodule(name),
                                           cpu["components"][(index, name)], source)
                    result["operator_replays"].append({"index": index, "name": name, **record})
            result["final_norm"] = self.norm_cross(cpu_language.model.norm, gpu_language.model.norm,
                                                  cpu["final_norm"], gpu["final_norm"], valid)
        result["controls"] = {
            "all_layer_transfers_exact": all(row["input_transfer_exact"] for row in result["layer_replays"]),
            "all_operator_transfers_exact": all(row["input_transfer_exact"] for row in result["operator_replays"]),
            "all_cpu_layer_replays_exact": all(row["comparison_to_captured_cpu_output"]["max_abs"] == 0.0
                                               for row in result["layer_replays"] if row["device"] == "cpu"),
            "all_cpu_operator_replays_exact": all(row["cpu_replay_to_capture"]["max_abs"] == 0.0
                                                  for row in result["operator_replays"]),
            "cpu_fused_transfer_exact": all(result["cpu_fused_to_musa_language"][name]["input_transfer_exact"]
                                             for name in ("source_eager", "fallback")),
            "norm_source_replays_exact": (result["final_norm"]["source_cpu_capture_replay_exact"]
                                          and result["final_norm"]["source_musa_capture_replay_exact"]),
            "rope_cpu_source_control": (result["rope"]["source_position_ids_are_arange"]
                                         and result["rope"]["captured_cpu_tuple_matches_regenerated"]),
            "rope_buffer_transfer_exact": result["rope"]["inv_freq_exact"],
            "kernel_qkv_strides_preserved": all(row["kernel_stages_same_cpu_qkv"]["qkv_transfer_stride_preserved"]
                                                for row in result["operator_replays"] if row["name"] == "self_attn"),
            "kernel_blocked_probabilities_zero": all(value == 0.0 for row in result["operator_replays"]
                if row["name"] == "self_attn" for value in row["kernel_stages_same_cpu_qkv"]["blocked_probability_max_abs"].values()),
            "vision_cpu_source_replays_exact": all(row["cpu_replay_to_capture"]["max_abs"] == 0.0
                                                   for row in result["vision_same_input"]["layers"]),
            "connector_cpu_source_replay_exact": result["vision_same_input"]["connector"]["cpu_replay_to_capture"]["max_abs"] == 0.0,
        }
        result["controls_pass"] = all(result["controls"].values())
        torch.musa.synchronize()
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_allocated_bytes"] = torch.musa.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.musa.max_memory_reserved()
        result["completed"] = True
        return result

    def vision_replays(self, cpu_model, gpu_model, source):
        torch = self.torch
        cpu_snapshot = self.snapshots["native_cpu"]
        cpu_vision = cpu_model.eagle_model.vision_model.vision_model
        gpu_vision = gpu_model.eagle_model.vision_model.vision_model
        records = []
        for index in VISION_LAYERS:
            entry = cpu_snapshot["vision"][index]
            args, kwargs = tree_map(torch, entry["args"], "cpu"), tree_map(torch, entry["kwargs"], "cpu")
            local_args, local_kwargs = tree_map(torch, entry["args"], "musa"), tree_map(torch, entry["kwargs"], "musa")
            with kernel_mode(cpu_vision.encoder.layers[index], source, False, family="siglip"):
                cpu_out = cpu_vision.encoder.layers[index](*args, **kwargs)
            musa_outputs = {}
            for fallback in (False, True):
                with kernel_mode(gpu_vision.encoder.layers[index], source, fallback, family="siglip"):
                    output = gpu_vision.encoder.layers[index](*local_args, **local_kwargs)
                value = (output[0] if isinstance(output, tuple) else output).detach().cpu()
                musa_outputs["fallback" if fallback else "source_eager"] = value
            cpu_value = (cpu_out[0] if isinstance(cpu_out, tuple) else cpu_out).detach().cpu()
            gpu_value = musa_outputs["fallback"]
            records.append({"index": index,
                            "arguments": argument_records(torch, {"args": args, "kwargs": kwargs}),
                            "input_transfer_exact": transfers_exact(torch, (args, kwargs), (local_args, local_kwargs)),
                            "cpu_replay_to_capture": self.compare(torch, cpu_value, entry["output"]),
                            "musa_to_cpu_same_input": self.compare(torch, gpu_value, cpu_value),
                            "musa_to_capture": self.compare(torch, gpu_value, entry["output"]),
                            "musa_source_vs_fallback": self.compare(torch, musa_outputs["fallback"], musa_outputs["source_eager"])})
        connector = cpu_snapshot["connector"]
        cargs, ckwargs = tree_map(torch, connector["args"], "cpu"), tree_map(torch, connector["kwargs"], "cpu")
        gargs, gkwargs = tree_map(torch, connector["args"], "musa"), tree_map(torch, connector["kwargs"], "musa")
        cpu_connector = cpu_model.eagle_model.mlp1(*cargs, **ckwargs).detach().cpu()
        musa_connector = gpu_model.eagle_model.mlp1(*gargs, **gkwargs).detach().cpu()
        return {"layers": records,
                "connector": {"input_transfer_exact": transfers_exact(torch, (cargs, ckwargs), (gargs, gkwargs)),
                              "cpu_replay_to_capture": self.compare(torch, cpu_connector, connector["output"]),
                              "musa_to_cpu_same_input": self.compare(torch, musa_connector, cpu_connector),
                              "musa_to_capture": self.compare(torch, musa_connector, connector["output"])}}

    def operator(self, cpu_module, gpu_module, entry, source):
        torch = self.torch
        args, kwargs = entry["args"], entry["kwargs"]
        cpu_args, cpu_kwargs = tree_map(torch, args, "cpu"), tree_map(torch, kwargs, "cpu")
        gpu_args, gpu_kwargs = tree_map(torch, args, "musa"), tree_map(torch, kwargs, "musa")
        observations, outputs = {}, {}
        attention = isinstance(cpu_module, source.Qwen3Attention)
        for device, module, a, k in (("cpu", cpu_module, cpu_args, cpu_kwargs), ("musa", gpu_module, gpu_args, gpu_kwargs)):
            modes = (False, True) if attention else (None,)
            for fallback in modes:
                mode = "source_eager" if fallback is False else "fallback" if fallback is True else "source"
                observer = {}
                context = kernel_mode(module, source, fallback, observer) if attention else contextlib.nullcontext()
                with context:
                    out = module(*a, **k)
                value = out if torch.is_tensor(out) else out[0]
                outputs[(device, mode)] = value.detach().cpu().clone()
                if attention and device == "cpu" and not fallback:
                    observations = observer
        base_mode = "source_eager" if attention else "source"
        record = {"arguments": argument_records(torch, {"args": args, "kwargs": kwargs}),
                  "input_transfer_exact": transfers_exact(torch, (args, kwargs), (gpu_args, gpu_kwargs)),
                  "cpu_replay_to_capture": self.compare(torch, outputs[("cpu", base_mode)], entry["output"]),
                  "musa_to_cpu_same_input": self.compare(torch, outputs[("musa", base_mode)], outputs[("cpu", base_mode)])}
        if attention:
            record["source_vs_fallback"] = {device: self.compare(torch, outputs[(device, "fallback")], outputs[(device, "source_eager")])
                                             for device in ("cpu", "musa")}
            record["kernel_stages_same_cpu_qkv"] = self.attention_stages(cpu_module, observations, source)
            return record
        x = args[0] if args else kwargs["hidden_states"]
        if isinstance(cpu_module, torch.nn.Linear):
            reference = torch.nn.functional.linear(x.double(), cpu_module.weight.double(),
                None if cpu_module.bias is None else cpu_module.bias.double())
            record["fp64_reference_formula"] = "linear on captured FP32 inputs/weights converted exactly to FP64"
        elif isinstance(cpu_module, source.Qwen3RMSNorm):
            reference = self.norm_formula(x, cpu_module)
            record["fp64_reference_formula"] = "x * rsqrt(mean(x**2)+eps) * weight, all arithmetic CPU FP64"
        elif isinstance(cpu_module, source.Qwen3MLP):
            if cpu_module.config.hidden_act != "silu":
                raise ValueError("Only actual SiLU MLP reference supported")
            gate = torch.nn.functional.linear(x.double(), cpu_module.gate_proj.weight.double())
            up = torch.nn.functional.linear(x.double(), cpu_module.up_proj.weight.double())
            reference = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, cpu_module.down_proj.weight.double())
            record["fp64_reference_formula"] = "explicit SiLU/gate/up/down formula, all arithmetic CPU FP64"
            fixed_gate, fixed_up = cpu_module.gate_proj(x), cpu_module.up_proj(x)
            product_cpu = cpu_module.act_fn(fixed_gate) * fixed_up
            product_musa = gpu_module.act_fn(fixed_gate.to("musa")) * fixed_up.to("musa")
            product_fp64 = torch.nn.functional.silu(fixed_gate.double()) * fixed_up.double()
            record["silu_product_same_cpu_projected_inputs"] = {
                "musa_to_cpu": self.compare(torch, product_musa, product_cpu),
                "cpu_to_fp64": self.compare(torch, product_cpu, product_fp64),
                "musa_to_fp64": self.compare(torch, product_musa, product_fp64),
            }
        else:
            raise ValueError("Unexpected diagnostic component")
        record["cpu_to_fp64"] = self.compare(torch, outputs[("cpu", "source")], reference)
        record["musa_to_fp64"] = self.compare(torch, outputs[("musa", "source")], reference)
        return record

    def attention_stages(self, attention, observation, source):
        torch = self.torch
        q, k, v, mask = (observation[name] for name in ("q", "k", "v", "mask"))
        scale = observation["scaling"]
        stages = {}
        raw = {}
        for device, dtype in (("cpu", torch.float32), ("musa", torch.float32), ("fp64_cpu", torch.float64)):
            actual_device = "cpu" if device == "fp64_cpu" else device
            lq, lk, lv = (x.to(device=actual_device, dtype=dtype) for x in (q, k, v))
            lk, lv = source.repeat_kv(lk, attention.num_key_value_groups), source.repeat_kv(lv, attention.num_key_value_groups)
            scores = torch.matmul(lq, lk.transpose(-1, -2)) * scale
            masked = scores + mask.to(device=actual_device, dtype=dtype)
            probabilities = torch.softmax(masked, dim=-1, dtype=dtype)
            output = torch.matmul(probabilities, lv)
            raw[device] = {"scores_unmasked": scores.cpu(), "probabilities": probabilities.cpu(), "pv": output.cpu()}
        blocked = (mask == torch.finfo(mask.dtype).min) | (mask == float("-inf"))
        for name in ("scores_unmasked", "probabilities", "pv"):
            stages.setdefault("chain_from_same_qkv", {})[name] = {"musa_to_cpu": self.compare(torch, raw["musa"][name], raw["cpu"][name]),
                            "cpu_to_fp64": self.compare(torch, raw["cpu"][name], raw["fp64_cpu"][name]),
                            "musa_to_fp64": self.compare(torch, raw["musa"][name], raw["fp64_cpu"][name])}
        fixed_scores = raw["cpu"]["scores_unmasked"] + mask
        fixed_probabilities = raw["cpu"]["probabilities"]
        fixed_values = source.repeat_kv(v, attention.num_key_value_groups)
        stages["isolated_same_cpu_stage_inputs"] = {}
        for name in ("softmax", "pv"):
            values = {}
            for device, dtype in (("cpu", torch.float32), ("musa", torch.float32), ("fp64_cpu", torch.float64)):
                actual_device = "cpu" if device == "fp64_cpu" else device
                if name == "softmax":
                    value = torch.softmax(fixed_scores.to(device=actual_device, dtype=dtype), dim=-1, dtype=dtype)
                else:
                    value = torch.matmul(fixed_probabilities.to(device=actual_device, dtype=dtype),
                                         fixed_values.to(device=actual_device, dtype=dtype))
                values[device] = value.cpu()
            stages["isolated_same_cpu_stage_inputs"][name] = {
                "musa_to_cpu": self.compare(torch, values["musa"], values["cpu"]),
                "cpu_to_fp64": self.compare(torch, values["cpu"], values["fp64_cpu"]),
                "musa_to_fp64": self.compare(torch, values["musa"], values["fp64_cpu"]),
            }
        stages["blocked_probability_max_abs"] = {device: float(values["probabilities"].masked_select(blocked.expand_as(values["probabilities"])).abs().max())
                                                    for device, values in raw.items()}
        stages["qkv_cpu_capture"] = [tensor_layout(torch, x) for x in (q, k, v)]
        stages["qkv_transfer_stride_preserved"] = all(x.stride() == x.to("musa").stride() for x in (q, k, v))
        stages["fp64_scope"] = "Isolated kernel only; identical source-projected FP32 Q/K/V and mask. Not full FP64 model."
        return stages

    def norm_formula(self, x, module):
        x = x.double()
        return x * self.torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon) * module.weight.double()

    def norm_cross(self, cpu_norm, gpu_norm, cpu_entry, gpu_entry, valid):
        torch = self.torch
        hcpu, hgpu = cpu_entry["args"][0], gpu_entry["args"][0]
        out = {}
        references = {}
        stage_values = {}
        for label, x in (("cpu_input", hcpu), ("musa_input", hgpu)):
            references[label] = self.norm_formula(x, cpu_norm)
            for device, module in (("cpu", cpu_norm), ("musa", gpu_norm)):
                local = x.to(device)
                out[(device, label)] = module(local).detach().cpu()
                variance = local.pow(2).mean(-1, keepdim=True)
                reciprocal = torch.rsqrt(variance + module.variance_epsilon)
                stage_values[(device, label)] = {"variance": variance.cpu(), "rsqrt": reciprocal.cpu(),
                                                 "normalized": (local * reciprocal).cpu()}
        rows = {}
        for label in references:
            rows[label] = {
                "cpu_to_fp64": self.compare(torch, out[("cpu", label)][valid], references[label][valid]),
                "musa_to_fp64": self.compare(torch, out[("musa", label)][valid], references[label][valid]),
                "musa_to_cpu": self.compare(torch, out[("musa", label)][valid], out[("cpu", label)][valid]),
                "stages_musa_to_cpu": {name: self.compare(torch, stage_values[("musa", label)][name][valid], stage_values[("cpu", label)][name][valid])
                                       for name in ("variance", "rsqrt", "normalized")},
            }
        cpu_base, gpu_actual = out[("cpu", "cpu_input")], out[("musa", "musa_input")]
        kernel_delta = gpu_actual.double() - out[("cpu", "musa_input")].double()
        propagation_delta = out[("cpu", "musa_input")].double() - cpu_base.double()
        total_delta = gpu_actual.double() - cpu_base.double()
        return {"same_input_rows": rows,
                "input_musa_to_cpu": self.compare(torch, hgpu[valid], hcpu[valid]),
                "source_cpu_capture_replay_exact": torch.equal(cpu_base, cpu_entry["output"]),
                "source_musa_capture_replay_exact": torch.equal(gpu_actual, gpu_entry["output"]),
                "delta_decomposition": {"max_abs_reconstruction_error": float((total_delta - kernel_delta - propagation_delta).abs().max()),
                    "total_valid_l2": float(total_delta[valid].norm()), "kernel_valid_l2": float(kernel_delta[valid].norm()),
                    "propagated_input_valid_l2": float(propagation_delta[valid].norm()),
                    "formula": "N_musa(h_musa)-N_cpu(h_cpu) = [N_musa(h_musa)-N_cpu(h_musa)] + [N_cpu(h_musa)-N_cpu(h_cpu)]"},
                "fp64_scope": "Manual formula; .double() on source RMSNorm would still cast to FP32 and is not used."}
