# Kernel contract adapted from Transformers 4.51.3, Apache-2.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Qwen3/SigLIP source revision: 5f4ecf2d9f867a1255131d2461d75793c0cf1db2.
"""Instance-bound kernel replacement; the real Attention.forward stays intact.

Clone the pinned source function's bytecode with a private globals dictionary
and replace only its eager kernel lookup. No class/global patch, custom global
attention registry entry, FlashAttention shim, or altered availability check.
The probe must verify the complete installed source hashes before binding.
"""

from __future__ import annotations

import math
import types


class BackboneAttentionKernel:
    def __init__(self, family, source_kernel, repeat_kv=None, fallback=True, trace=None):
        if family not in ("qwen3", "siglip"):
            raise ValueError("Only pinned Qwen3/SigLIP attention is supported")
        self.family = family
        self.source_kernel = source_kernel
        self.repeat_kv = repeat_kv
        self.fallback = fallback
        self.trace = trace

    def __call__(self, module, query, key, value, attention_mask,
                 scaling, dropout=0.0, **kwargs):
        import torch
        if query.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("This component probe is scoped to FP32/BF16")
        if module.training or dropout != 0.0:
            raise ValueError("Frozen eval forward requires zero attention dropout")
        if kwargs.get("sliding_window") is not None:
            raise ValueError("Sliding-window/cache behavior is outside this probe")
        if not math.isclose(scaling, query.shape[-1] ** -0.5, rel_tol=1e-12):
            raise ValueError("Unexpected source attention scaling")
        if not all(bool(torch.isfinite(x).all()) for x in (query, key, value)):
            raise FloatingPointError("Non-finite source-projected Q/K/V")
        if self.family == "qwen3":
            if attention_mask is None:
                raise ValueError("Qwen3 source eager needs the model-generated causal mask")
            if self.repeat_kv is None:
                raise ValueError("Pinned Qwen3 repeat_kv is required")
            key_states = self.repeat_kv(key, module.num_key_value_groups)
            value_states = self.repeat_kv(value, module.num_key_value_groups)
        else:
            if module.is_causal or kwargs.get("is_causal", False):
                raise ValueError("This SigLIP vision scope is noncausal")
            key_states, value_states = key, value
        blocked = None
        used_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.dtype == torch.bool:
                raise ValueError("Expected real source 4D additive mask")
            used_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            if bool(torch.isnan(used_mask).any()) or bool((used_mask == float("inf")).any()):
                raise FloatingPointError("Invalid additive mask")
            blocked = ((used_mask == torch.finfo(used_mask.dtype).min)
                       | (used_mask == float("-inf")))
            if bool(blocked.all(dim=-1).any()):
                raise ValueError("Fully masked rows are explicitly outside this right-padding probe")
        if self.fallback:
            scores = torch.matmul(query.float(), key_states.float().transpose(-1, -2)) * scaling
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Non-finite unmasked FP32 scores")
            if used_mask is not None:
                scores = scores + used_mask.float()
            if bool(torch.isnan(scores).any()) or bool((scores == float("inf")).any()):
                raise FloatingPointError("Invalid masked scores")
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            output = torch.matmul(probabilities, value_states.float()).to(query.dtype)
            output = output.transpose(1, 2).contiguous()
            weights = probabilities.to(query.dtype)
        else:
            output, weights = self.source_kernel(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs)
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("Non-finite attention result")
        if self.trace is not None:
            blocked_max = None
            if blocked is not None and bool(blocked.any()):
                # Torch-MUSA masked_select does not support BF16 input. This
                # cast only reads returned weights for an exact-zero diagnostic;
                # it does not change the attention output, weights, or VJP.
                blocked_max = float(weights.float().masked_select(blocked.expand_as(weights)).abs().max())
            self.trace({
                "family": self.family, "kernel": "fp32_fallback" if self.fallback else "source_eager",
                "qkv_shapes": [list(x.shape) for x in (query, key, value)],
                "qkv_strides": [list(x.stride()) for x in (query, key, value)],
                "qkv_contiguous": [x.is_contiguous() for x in (query, key, value)],
                "qkv_dtype": str(query.dtype),
                "gqa_groups": getattr(module, "num_key_value_groups", 1),
                "scale": scaling, "dropout": dropout,
                "mask_shape": None if used_mask is None else list(used_mask.shape),
                "mask_stride": None if used_mask is None else list(used_mask.stride()),
                "mask_dtype": None if used_mask is None else str(used_mask.dtype),
                "causal_mask_already_composed_by_model": self.family == "qwen3",
                "blocked_probability_max_abs": blocked_max,
                "kernel_output_layout": "BSHD", "kernel_output_contiguous": output.is_contiguous(),
            })
        return output, weights


def bind_source_kernel(module, kernel):
    """Bind an identical source forward to one instance with one private lookup."""
    if module.config._attn_implementation != "eager":
        raise ValueError("The probe must explicitly configure source eager")
    original = type(module).forward
    if "eager_attention_forward" not in original.__globals__:
        raise ValueError("Unexpected source forward globals")
    private_globals = dict(original.__globals__)
    private_globals["eager_attention_forward"] = kernel
    clone = types.FunctionType(original.__code__, private_globals, original.__name__,
                               original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    clone.__annotations__ = original.__annotations__
    module.forward = types.MethodType(clone, module)
    return {
        "source_class": type(module).__module__ + "." + type(module).__name__,
        "same_source_code_object": clone.__code__ is original.__code__,
        "private_globals": clone.__globals__ is not original.__globals__,
        "instance_only": True,
    }
