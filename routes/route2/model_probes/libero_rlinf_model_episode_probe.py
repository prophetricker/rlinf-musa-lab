#!/usr/bin/env python3
"""Run the official RLinf GR00T N1.5 wrapper through a short LIBERO episode."""
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
    parser.add_argument("--ppo-multi-step", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--action-head-fp32", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error("steps must be positive and output must be new")
    if args.ppo_one_step and args.ppo_multi_step:
        parser.error("choose at most one PPO update mode")
    if args.ppo_multi_step and not args.action_head_fp32:
        parser.error("--ppo-multi-step requires --action-head-fp32")

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
            from gr00t_action_head_fallback import apply_action_head_fp32_fallback

            if args.action_head_fp32:
                fallback_metadata = apply_action_head_fp32_fallback(model)
                result["fallback"] = fallback_metadata
        finally:
            musa_patches.bind_vendor_flash_attn_in_transformers = original_bind

        result["model"] = {
            "class": type(model).__name__,
            "device": str(next(model.parameters()).device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_load_seconds": time.monotonic() - started,
            "weights": str(args.model_path),
            "action_attention_processor": result.get("fallback", {}).get(
                "processor", "model default"
            ),
            "action_attention_modules": result.get("fallback", {}).get(
                "attention_modules", 0
            ),
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
        rollout_details = []
        for step_idx in range(args.steps):
            started_step = time.monotonic()
            with torch.no_grad():
                action, details = model.predict_action_batch(obs, mode=args.rollout_mode)
            if args.ppo_one_step or args.ppo_multi_step:
                rollout_details.append(details)
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

        if args.ppo_one_step or args.ppo_multi_step:
            update_details = rollout_details[0] if rollout_details else None
            if update_details is None:
                raise RuntimeError("No rollout details available for PPO update")
            from rlinf.algorithms.losses import compute_ppo_actor_critic_loss

            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("Model has no trainable parameters")
            before_sample = trainable[0].detach().flatten()[:16].float().cpu().tolist()
            optimizer = torch.optim.AdamW(trainable, lr=1.0e-8, foreach=False)
            optimizer.zero_grad(set_to_none=True)
            if args.ppo_one_step:
                old_logprobs = update_details["prev_logprobs"].detach().float()
                old_values = update_details["prev_values"].detach().float().reshape(-1)
                advantages = torch.ones_like(old_values, dtype=torch.float32)
                returns = old_values + 1.0
                loss_steps = [
                    (update_details, old_logprobs, old_values, advantages, returns)
                ]
                gae_metadata = {
                    "source": "synthetic",
                    "gamma": None,
                    "gae_lambda": None,
                }
            else:
                if len(rollout_details) != len(rows):
                    raise RuntimeError("multi-step rollout metadata length mismatch")
                with torch.no_grad():
                    _, bootstrap_details = model.predict_action_batch(
                        obs, mode=args.rollout_mode
                    )
                value_steps = [
                    details["prev_values"].detach().float().reshape(-1)
                    for details in rollout_details
                ]
                values_with_bootstrap = torch.stack(
                    [*value_steps, bootstrap_details["prev_values"].detach().float().reshape(-1)]
                )
                rewards = torch.tensor(
                    [float(row["reward"][0]) for row in rows],
                    device=values_with_bootstrap.device,
                    dtype=torch.float32,
                ).unsqueeze(-1)
                dones = torch.tensor(
                    [False, *[
                        bool(row["terminated"][0] or row["truncated"][0])
                        for row in rows
                    ]],
                    device=values_with_bootstrap.device,
                    dtype=torch.bool,
                ).unsqueeze(-1)
                from rlinf.algorithms.advantages import (
                    compute_gae_advantages_and_returns,
                )

                advantages, returns = compute_gae_advantages_and_returns(
                    rewards=rewards,
                    values=values_with_bootstrap,
                    dones=dones,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    normalize_advantages=False,
                )
                if not bool(torch.isfinite(advantages).all()):
                    raise RuntimeError("GAE advantages are non-finite")
                if not bool(torch.isfinite(returns).all()):
                    raise RuntimeError("GAE returns are non-finite")
                loss_steps = [
                    (
                        details,
                        details["prev_logprobs"].detach().float(),
                        details["prev_values"].detach().float().reshape(-1),
                        advantages[idx],
                        returns[idx],
                    )
                    for idx, details in enumerate(rollout_details)
                ]
                gae_metadata = {
                    "source": "environment rewards and done flags",
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "rewards": rewards.squeeze(-1).detach().cpu().tolist(),
                    "dones": dones.squeeze(-1).detach().cpu().tolist(),
                    "bootstrap_values": values_with_bootstrap[-1].detach().cpu().tolist(),
                    "values": values_with_bootstrap[:-1].detach().cpu().tolist(),
                    "advantages": advantages.squeeze(-1).detach().cpu().tolist(),
                    "returns": returns.squeeze(-1).detach().cpu().tolist(),
                }

            ppo_loss = None
            ppo_metrics = {}
            logprobs = None
            values = None
            with torch.enable_grad():
                for details, old_logprobs, old_values, advantages_t, returns_t in loss_steps:
                    current = model(
                        forward_inputs=details["forward_inputs"],
                        prev_logprobs=old_logprobs,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=True,
                    )
                    logprobs = current["logprobs"].float()
                    values = current["values"].float().reshape(-1)
                    step_loss, step_metrics = compute_ppo_actor_critic_loss(
                        logprobs=logprobs,
                        old_logprobs=old_logprobs,
                        values=values,
                        returns=returns_t,
                        prev_values=old_values,
                        advantages=advantages_t,
                        clip_ratio_low=0.2,
                        clip_ratio_high=0.2,
                        clip_ratio_c=3.0,
                        value_clip=0.2,
                        huber_delta=10.0,
                    )
                    if not bool(torch.isfinite(step_loss)):
                        raise RuntimeError("PPO loss is non-finite")
                    ppo_loss = step_loss if ppo_loss is None else ppo_loss + step_loss
                    ppo_metrics = step_metrics
                    (step_loss / len(loss_steps)).backward()
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
            if grad_count == 0 or grad_max == 0.0:
                raise RuntimeError("PPO backward produced no non-zero gradients")
            optimizer.step()
            torch.musa.synchronize()
            after_sample = trainable[0].detach().flatten()[:16].float().cpu().tolist()
            parameter_sample_changed = before_sample != after_sample
            if not parameter_sample_changed:
                raise RuntimeError("optimizer step did not change the parameter sample")
            result["ppo_update"] = {
                "status": "pass",
                "mode": "one_step_synthetic" if args.ppo_one_step else "multi_step_real_gae",
                "loss_sum": float(ppo_loss.detach().cpu()),
                "loss_mean": float((ppo_loss / len(loss_steps)).detach().cpu()),
                "logprobs_shape": list(logprobs.shape),
                "values_shape": list(values.shape),
                "old_logprobs_shape": list(old_logprobs.shape),
                "old_values_shape": list(old_values.shape),
                "trainable_parameter_count": sum(p.numel() for p in trainable),
                "gradient_tensor_count": grad_count,
                "gradient_finite": grad_finite,
                "gradient_l2": float(grad_sq.sqrt().cpu()),
                "gradient_max_abs": grad_max,
                "parameter_sample_changed": parameter_sample_changed,
                "ppo_metrics": {key: float(value.detach().cpu()) for key, value in ppo_metrics.items() if hasattr(value, "detach")},
                "target_note": (
                    "one synthetic one-sample target; not learning quality"
                    if args.ppo_one_step
                    else "real environment rewards/dones with RLinf GAE; not a success-rate result"
                ),
                "gae": gae_metadata,
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
