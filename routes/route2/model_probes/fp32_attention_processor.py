# Adapted from Diffusers 0.30.2 AttnProcessor2_0, Apache-2.0.
# Copyright 2024 The HuggingFace Team. All rights reserved.
# https://github.com/huggingface/diffusers/blob/f63c12633f154c2a1d79c17f4238fb073133652c/src/diffusers/models/attention_processor.py
"""Instance-bound GR00T action MHA processor with FP32 attention accumulation.

The source processor's normalization, projection, output layout, dropout module,
residual and rescale operations are retained. Only its attention kernel changes.
No class/global monkey-patch, FlashAttention shim or dependency installation.
"""

from __future__ import annotations


class FP32AttentionProcessor:
    def __init__(self, trace=None):
        self.trace = trace

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        import torch
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            from diffusers.utils import deprecate
            deprecate("scale", "1.0.0", "The scale argument is deprecated and ignored, as in AttnProcessor2_0.")
        if (not attn.scale_qk or attn.inner_kv_dim != attn.inner_dim
                or attn.added_kv_proj_dim is not None or attn.pre_only
                or attn.context_pre_only is not None):
            raise ValueError("This GR00T processor requires ordinary MHA with default QK scale and no added KV")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        for norm in (attn.norm_q, attn.norm_k):
            if norm is not None and hasattr(norm, "normalized_shape") and tuple(norm.normalized_shape) != (head_dim,):
                raise ValueError("Only per-head Q/K normalization is supported")
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if query.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("This probe processor is scoped to FP32 and BF16")
        if self.trace is not None:
            self.trace({
                "qkv_shapes": [list(x.shape) for x in (query, key, value)],
                "qkv_strides": [list(x.stride()) for x in (query, key, value)],
                "qkv_contiguous": [x.is_contiguous() for x in (query, key, value)],
                "qkv_dtype": str(query.dtype), "accumulation_dtype": "torch.float32",
                "mask_shape": None if attention_mask is None else list(attention_mask.shape),
                "mask_stride": None if attention_mask is None else list(attention_mask.stride()),
                "mask_dtype": None if attention_mask is None else str(attention_mask.dtype),
                "scale": head_dim**-0.5, "kernel_dropout_p": 0.0, "is_causal": False,
            })
        # Keep the source BHSD transpose layout; promote only kernel operands.
        if not all(bool(torch.isfinite(x).all()) for x in (query, key, value)):
            raise FloatingPointError("Non-finite projected Q/K/V")
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * head_dim**-0.5
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Non-finite attention scores before masking")
        valid = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
                valid = attention_mask.any(dim=-1, keepdim=True)
            else:
                if bool(torch.isnan(attention_mask).any()) or bool((attention_mask == float("inf")).any()):
                    raise FloatingPointError("NaN or positive infinity in additive attention mask")
                scores = scores + attention_mask.float()
                valid = (attention_mask != float("-inf")).any(dim=-1, keepdim=True)
        if bool(torch.isnan(scores).any()) or bool((scores == float("inf")).any()):
            raise FloatingPointError("Invalid attention scores after masking")
        # Defined kernel-only empty-row policy. The module still applies its
        # output bias/residual. This probe's required masks have no empty rows.
        # Only a mask explicitly blocking every key can trigger this policy.
        if valid is not None:
            scores = torch.where(valid, scores, torch.zeros_like(scores))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        if valid is not None:
            probs = torch.where(valid, probs, torch.zeros_like(probs))
        hidden_states = torch.matmul(probs, value.float()).to(query.dtype)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
