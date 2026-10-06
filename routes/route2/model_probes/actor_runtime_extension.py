"""Explicit RLinf model extension for the fixed S4000 GR00T N1.5 stack.

Load with RLINF_EXT_MODULE=actor_runtime_extension. Both official Actor and
Rollout keep their upstream factory; the builder uses the original N1.5
loader and applies the previously validated action-head fallback at load time.
Requires the fixed eager GR00T tree and lazy optional PyTorch3D import patch.
"""
from __future__ import annotations

import os


def build_gr00t(cfg, torch_dtype):
    import torch
    from transformers import AutoConfig, AutoModel
    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import (
        Eagle2_5_VLConfig,
    )
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
        Eagle2_5_VLForConditionalGeneration,
    )
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from gr00t_action_head_fallback import apply_action_head_fp32_fallback

    if cfg.embodiment_tag != "libero_franka" or torch_dtype != torch.bfloat16:
        raise ValueError("S4000 extension requires LIBERO-Franka and a BF16 backbone")
    if not cfg.get("musa_eager_attention", False):
        raise ValueError("S4000 extension requires explicit musa_eager_attention")
    AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig, exist_ok=True)
    AutoModel.register(
        Eagle2_5_VLConfig, Eagle2_5_VLForConditionalGeneration, exist_ok=True
    )
    model = get_model(cfg, torch_dtype)
    eagle = model.backbone.eagle_model
    implementations = {
        "top": eagle.config._attn_implementation,
        "vision": eagle.vision_model.config._attn_implementation,
        "language": eagle.language_model.config._attn_implementation,
    }
    if any(value != "eager" for value in implementations.values()):
        raise RuntimeError(f"Expected actual Eagle eager attention: {implementations}")
    metadata = apply_action_head_fp32_fallback(model)
    metadata["backbone_attention"] = implementations
    metadata["entrypoint"] = "RLinf model registry extension"
    model.s4000_fallback_metadata = metadata
    return model


def register():
    import torch_musa  # noqa: F401 - register the real installed backend
    from rlinf.models import register_model

    if os.environ.get("RLINF_MUSA_FSDP_LEGACY_NORM", "0") == "1":
        from musa_fsdp_norm import apply
        apply(mode="legacy")
    if os.environ.get("RLINF_MUSA_FSDP_OPTIM_DEVICE_HANDLE", "0") == "1":
        from musa_fsdp_optim_device import apply as apply_optim_device
        apply_optim_device()
    register_model("gr00t", build_gr00t, category="embodied", force=True)
