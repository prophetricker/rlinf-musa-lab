"""Portable S4000 fallback for the GR00T action head.

The fallback is intentionally process-local: it binds the already validated
FP32 attention processor to action-head Attention instances and keeps the
Eagle backbone in its original dtype.  It does not modify Diffusers globals or
the upstream GR00T source tree.
"""

from __future__ import annotations

from typing import Any


def apply_action_head_fp32_fallback(model: Any) -> dict[str, Any]:
    """Bind the S4000-safe action-head path and return its audit metadata."""
    import torch
    import tree
    from diffusers.models.attention_processor import Attention

    from fp32_attention_processor import FP32AttentionProcessor

    action_attention = []
    for name, module in model.action_head.named_modules():
        if isinstance(module, Attention):
            module.set_processor(FP32AttentionProcessor())
            action_attention.append(name)

    if not action_attention:
        raise RuntimeError("action-head FP32 fallback found no Diffusers Attention modules")

    model.action_head.float()
    backbone_dtype = next(model.backbone.parameters()).dtype
    action_dtype = next(model.action_head.parameters()).dtype
    if action_dtype != torch.float32:
        raise RuntimeError(f"action-head fallback produced {action_dtype}, expected torch.float32")
    if backbone_dtype != torch.bfloat16:
        raise RuntimeError(f"fallback changed backbone dtype to {backbone_dtype}")

    def prepare_mixed(inputs):
        backbone_inputs, action_inputs = (
            model.backbone.prepare_input(inputs),
            model.action_head.prepare_input(inputs),
        )

        def move(value, dtype):
            if torch.is_floating_point(value):
                return value.to(model.device, dtype=dtype)
            return value.to(model.device)

        return (
            tree.map_structure(lambda value: move(value, backbone_dtype), backbone_inputs),
            tree.map_structure(lambda value: move(value, action_dtype), action_inputs),
        )

    model.prepare_input = prepare_mixed
    original_process_backbone = model.action_head.process_backbone_output

    def process_backbone(output):
        output["backbone_features"] = output["backbone_features"].float()
        return original_process_backbone(output)

    model.action_head.process_backbone_output = process_backbone
    return {
        "processor": "FP32AttentionProcessor",
        "attention_modules": len(action_attention),
        "attention_names": action_attention,
        "backbone_dtype": str(backbone_dtype),
        "action_head_dtype": str(action_dtype),
        "scope": "process-local action head only",
    }
