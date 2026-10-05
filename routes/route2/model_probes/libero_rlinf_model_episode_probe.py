#!/usr/bin/env python3
"""Run the official RLinf GR00T N1.5 wrapper through a short LIBERO episode.

This probe intentionally stops before PPO.  It verifies the production RLinf
model wrapper, its observation/action transforms, and the real LIBERO env
contract on the MUSA runtime.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import types
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(torch, value) -> bool:
    return bool(torch.isfinite(value).all())


def shape(value):
    return list(value.shape) if hasattr(value, "shape") else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--worker-start-method", choices=("spawn", "fork"), default="spawn")
    parser.add_argument("--rollout-mode", choices=("train", "eval"), default="train")
    parser.add_argument("--ppo-one-step", action="store_true")
    parser.add_argument("--action-head-fp32", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error("steps must be positive and output must be new")

    result = {
        "schema_version": 1,
        "scope": "Official RLinf GR00T N1.5 wrapper and real LIBERO-Spatial short episode",
        "probe_sha256": sha256(Path(__file__)),
        "steps_requested": args.steps,
        "status": "fail",
    }
    try:
        # LIBERO's subprocess wrapper is compatible with fork on this isolated
        # Linux runtime; stdin-launched probes cannot be re-imported by spawn.
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
        sys.path.insert(0, str(args.rlinf_source.resolve()))
        sys.path.insert(0, str(args.gr00t_source.resolve()))
        sys.path.insert(0, str(args.rlinf_source.parent / "model_probes"))

        # GR00T imports the optional rotation helper at module import time.
        # LIBERO-Franka's N1.5 transform path does not call it; keep the
        # optional dependency isolated rather than changing the environment.
        pytorch3d = types.ModuleType("pytorch3d")
        pytorch3d.transforms = types.ModuleType("pytorch3d.transforms")
        sys.modules.setdefault("pytorch3d", pytorch3d)
        sys.modules.setdefault("pytorch3d.transforms", pytorch3d.transforms)

        import torch
        import torch_musa  # noqa: F401
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoModel
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if not torch.musa.is_available():
            raise RuntimeError("MUSA is unavailable")

        # The bundled eager GR00T source does not register its custom Eagle
        # architecture with Transformers.  Register the actual source classes
        # for this process only; the model code and weights remain unmodified.
        from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import (
            Eagle2_5_VLConfig,
        )
        from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
            Eagle2_5_VLForConditionalGeneration,
        )

        registered = []
        for registry, key, value in (
            (CONFIG_MAPPING, "eagle_2_5_vl", Eagle2_5_VLConfig),
        ):
            try:
                registry.register(key, value)
                registered.append((registry, key, value))
            except ValueError:
                if registry[key] is not value:
                    raise
        try:
            AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig)
        except ValueError:
            if AutoConfig.for_model("eagle_2_5_vl") is not Eagle2_5_VLConfig:
                raise
        try:
            AutoModel.register(Eagle2_5_VLConfig, Eagle2_5_VLForConditionalGeneration)
        except ValueError:
            pass

        import rlinf.models.embodiment.gr00t.gr00t_n1d5.musa_patches as musa_patches

        # The rental image has no vendor flash_attn package.  The prepared
        # Eagle config is eager, so only the optional binding is disabled.
        original_bind = musa_patches.bind_vendor_flash_attn_in_transformers
        musa_patches.bind_vendor_flash_attn_in_transformers = lambda: None
        try:
            from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

            model_cfg = OmegaConf.create(
                {
                    "embodiment_tag": "libero_franka",
                    "model_path": str(args.model_path),
                    "denoising_steps": 4,
                    "num_action_chunks": 1,
                    "obs_converter_type": "libero",
                    "rl_head_config": OmegaConf.create(
                        {
                            "joint_logprob": False,
                            "noise_method": "flow_sde",
                            "ignore_last": False,
                            "safe_get_logprob": False,
                            "noise_anneal": False,
                            "noise_params": [0.7, 0.3, 400],
                            "noise_level": 0.5,
                            "add_value_head": True,
                            "chunk_critic_input": False,
                            "detach_critic_input": True,
                            "disable_dropout": True,
                            "use_vlm_value": False,
                            "value_vlm_mode": "mean_token",
                            "padding_value": 570,
                        }
                    ),
                }
            )
            started = time.monotonic()
            model = get_model(model_cfg, torch_dtype=torch.bfloat16)
            model = model.to("musa:0")
            # GR00T's override of eval() does not return self.
            model.eval()
            from diffusers.models.attention_processor import Attention
            from fp32_attention_processor import FP32AttentionProcessor

            action_attention = []
            for name, module in model.action_head.named_modules():
                if isinstance(module, Attention):
                    module.set_processor(FP32AttentionProcessor())
                    action_attention.append(name)
            if args.action_head_fp32:
                import tree

                model.action_head.float()
                backbone_dtype = next(model.backbone.parameters()).dtype
                action_dtype = next(model.action_head.parameters()).dtype
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
        finally:
            musa_patches.bind_vendor_flash_attn_in_transformers = original_bind

        result["model"] = {
            "class": type(model).__name__,
            "device": str(next(model.parameters()).device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_load_seconds": time.monotonic() - started,
            "weights": str(args.model_path),
            "action_attention_processor": "FP32AttentionProcessor",
            "action_attention_modules": len(action_attention),
            "action_head_dtype": str(next(model.action_head.parameters()).dtype),
        }

        cfg = OmegaConf.load(args.env_config)
        cfg.specific_reset_id = 0
        cfg.total_num_envs = 1
        cfg.group_size = 1
        cfg.auto_reset = False
        cfg.ignore_terminations = False
        cfg.is_eval = False
        cfg.max_episode_steps = max(args.steps, int(cfg.max_episode_steps))
        import rlinf.envs.sim.libero.venv as libero_venv

        # Keep the worker method explicit so stdin-launched compatibility runs
        # can use fork, while the file-entry probe can verify RLinf's default
        # spawn behavior.
        original_get_context = libero_venv.multiprocessing.get_context
        libero_venv.multiprocessing.get_context = lambda method=None: original_get_context(
            args.worker_start_method if method == "spawn" else method
        )
        from rlinf.envs.sim.libero.libero_env import LiberoEnv

        env = LiberoEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
        obs, reset_info = env.reset()
        result["reset"] = {
            "main_shape": shape(obs["main_images"]),
            "wrist_shape": shape(obs["wrist_images"]),
            "state_shape": shape(obs["states"]),
            "state_finite": finite(torch, obs["states"]),
            "task": list(obs["task_descriptions"]),
            "info_keys": sorted(reset_info),
        }

        rows = []
        update_details = None
        for step_idx in range(args.steps):
            started_step = time.monotonic()
            with torch.no_grad():
                action, details = model.predict_action_batch(obs, mode=args.rollout_mode)
            if update_details is None:
                update_details = details
            action = action.detach().cpu()
            if action.ndim == 3:
                action = action[:, 0]
            if list(action.shape) != [1, 7]:
                raise RuntimeError(f"Unexpected converted action shape: {list(action.shape)}")
            if not finite(torch, action):
                raise RuntimeError("Converted action contains non-finite values")
            next_obs, reward, terminated, truncated, info = env.step(action, auto_reset=False)
            rows.append(
                {
                    "step": step_idx,
                    "action_shape": list(action.shape),
                    "action_min": float(action.min()),
                    "action_max": float(action.max()),
                    "action_finite": finite(torch, action),
                    "reward": reward.tolist(),
                    "terminated": terminated.tolist(),
                    "truncated": truncated.tolist(),
                    "next_state_finite": finite(torch, next_obs["states"]),
                    "details_keys": sorted(details),
                    "info_keys": sorted(info),
                    "elapsed_seconds": time.monotonic() - started_step,
                }
            )
            obs = next_obs
            if bool(terminated.any() or truncated.any()):
                break

        if args.ppo_one_step:
            if update_details is None:
                raise RuntimeError("No rollout details available for PPO update")
            from rlinf.algorithms.losses import compute_ppo_actor_critic_loss

            forward_inputs = update_details["forward_inputs"]
            old_logprobs = update_details["prev_logprobs"].detach().float()
            old_values = update_details["prev_values"].detach().float().reshape(-1)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("Model has no trainable parameters")
            before_sample = trainable[0].detach().flatten()[:16].float().cpu().tolist()
            optimizer = torch.optim.AdamW(trainable, lr=1.0e-8, foreach=False)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                current = model(
                    forward_inputs=forward_inputs,
                    prev_logprobs=old_logprobs,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=True,
                )
                logprobs = current["logprobs"].float()
                values = current["values"].float().reshape(-1)
                advantages = torch.ones_like(values, dtype=torch.float32)
                returns = old_values + 1.0
                ppo_loss, ppo_metrics = compute_ppo_actor_critic_loss(
                    logprobs=logprobs,
                    old_logprobs=old_logprobs,
                    values=values,
                    returns=returns,
                    prev_values=old_values,
                    advantages=advantages,
                    clip_ratio_low=0.2,
                    clip_ratio_high=0.2,
                    clip_ratio_c=3.0,
                    value_clip=0.2,
                    huber_delta=10.0,
                )
                if not bool(torch.isfinite(ppo_loss)):
                    raise RuntimeError("PPO loss is non-finite")
                ppo_loss.backward()
            torch.musa.synchronize()
            grad_sq = torch.zeros((), device="musa:0", dtype=torch.float32)
            grad_max = 0.0
            grad_count = 0
            grad_finite = True
            for parameter in trainable:
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    grad_finite = grad_finite and bool(torch.isfinite(gradient).all())
                    grad_sq = grad_sq + gradient.float().square().sum()
                    grad_max = max(grad_max, float(gradient.detach().abs().max()))
                    grad_count += 1
            if not grad_finite:
                raise RuntimeError("PPO gradient is non-finite")
            optimizer.step()
            torch.musa.synchronize()
            after_sample = trainable[0].detach().flatten()[:16].float().cpu().tolist()
            result["ppo_one_step"] = {
                "status": "pass",
                "loss": float(ppo_loss.detach().cpu()),
                "logprobs_shape": list(logprobs.shape),
                "values_shape": list(values.shape),
                "old_logprobs_shape": list(old_logprobs.shape),
                "old_values_shape": list(old_values.shape),
                "trainable_parameter_count": sum(p.numel() for p in trainable),
                "gradient_tensor_count": grad_count,
                "gradient_finite": grad_finite,
                "gradient_l2": float(grad_sq.sqrt().cpu()),
                "gradient_max_abs": grad_max,
                "parameter_sample_changed": before_sample != after_sample,
                "ppo_metrics": {key: float(value.detach().cpu()) for key, value in ppo_metrics.items() if hasattr(value, "detach")},
                "target_note": "one synthetic one-sample PPO target: advantages=1 and returns=prev_values+1; this validates the RLinf loss/backward/update path, not learning quality",
            }

        result["steps"] = rows
        result["episode"] = {
            "steps_completed": len(rows),
            "return": float(sum(float(row["reward"][0]) for row in rows)),
            "terminated": bool(rows[-1]["terminated"][0]) if rows else False,
            "truncated": bool(rows[-1]["truncated"][0]) if rows else False,
            "rollout_mode": args.rollout_mode,
        }
        result["musa_memory"] = {
            "allocated": int(torch.musa.memory_allocated()),
            "reserved": int(torch.musa.memory_reserved()),
        }
        env.close()
        result["status"] = "pass"
    except Exception as error:
        result.update(error_class=type(error).__name__, error=str(error), traceback=traceback.format_exc(limit=12))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
