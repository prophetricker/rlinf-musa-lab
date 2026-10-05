#!/usr/bin/env python3
"""Compare small GR00T-shaped attention kernels without loading model weights.

Default execution is CPU-only. A coordinator must explicitly pass --device musa
to run on the accelerator. The script does not install or monkey-patch packages.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import platform
import time
import traceback


def repeat_kv(x, groups: int):
    """Repeat each KV head as Transformers 4.51.3 does, preserving head order."""
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


def attention(torch, q, k, v, mask, kind: str):
    """Accept BHSD inputs and bool True=allowed or additive attention masks."""
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    v = repeat_kv(v, q.shape[1] // v.shape[1])
    if kind == "sdpa":
        # Match the Transformers SDPA integration's contiguous Q/K/V boundary.
        return torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask, dropout_p=0.0, is_causal=False,
        )
    if kind == "sdpa_layout":
        # Diffusers AttnProcessor2_0 transposes BHSD without making it contiguous.
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
        )
    original_dtype = q.dtype
    if kind in ("eager_fp32", "reference"):
        accum = torch.float64 if q.dtype == torch.float64 else torch.float32
        q, k, v = q.to(accum), k.to(accum), v.to(accum)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.to(scores.dtype)
    # Qwen eager promotes softmax; Diffusers' default eager processor does not.
    softmax_dtype = torch.float32 if kind == "eager_qwen" else scores.dtype
    if kind == "reference":
        softmax_dtype = torch.float64
    probs = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(v.dtype)
    if kind in ("eager_fp32", "reference"):
        # Explicit fallback contract for rows with no allowed keys: zero output.
        valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
        probs = torch.where(valid, probs, torch.zeros_like(probs))
    return torch.matmul(probs, v).to(original_dtype)


def geometries(profile: str) -> list[dict]:
    """Small kernels by default; larger dimensions are declared assumptions."""
    if profile == "feature-570":
        b, aq, seq, ah, ad, vh, vd, lh, lkv, ld = 1, 49, 570, 8, 64, 16, 72, 16, 8, 128
    else:
        b, aq, seq, ah, ad, vh, vd, lh, lkv, ld = 2, 17, 31, 4, 64, 4, 72, 4, 2, 128
    return [
        dict(name="action_cross_unmasked", b=b, h=ah, kv=ah, q=aq, k=seq, d=ad, mask="none", backward=True, required=True),
        dict(name="action_self_unmasked", b=b, h=ah, kv=ah, q=aq, k=aq, d=ad, mask="none", backward=True, required=True),
        dict(name="vl_self_unmasked", b=b, h=ah, kv=ah, q=seq, k=seq, d=ad, mask="none", backward=True, required=True),
        dict(name="eagle_gqa_causal_rightpad", b=b, h=lh, kv=lkv, q=seq, k=seq, d=ld, mask="causal_right", backward=False, required=True),
        dict(name="vision_packed_unmasked", b=b, h=vh, kv=vh, q=17 if profile == "small" else 256, k=17 if profile == "small" else 256, d=vd, mask="none", backward=False, required=True, packed=True),
        dict(name="padding_bool_contract", b=b, h=ah, kv=ah, q=aq, k=seq, d=ad, mask="key_bool", backward=True, required=True),
        dict(name="padding_additive_contract", b=b, h=ah, kv=ah, q=aq, k=seq, d=ad, mask="key_additive", backward=True, required=True),
        dict(name="eagle_leftpad_empty_rows_edge", b=b, h=lh, kv=lkv, q=seq, k=seq, d=ld, mask="causal_left", backward=False, required=False),
    ]


def make_inputs(torch, case: dict, dtype, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    b, h, kv, nq, nk, d = (case[x] for x in ("b", "h", "kv", "q", "k", "d"))
    if case.get("packed"):
        packed = torch.randn(b, nq, 3, h, d, generator=generator).to(dtype) * 0.5
        q, k, v = [packed[:, :, i].transpose(1, 2) for i in range(3)]
    else:
        q = (torch.randn(b, nq, h, d, generator=generator) * 0.5).to(dtype).transpose(1, 2)
        k = (torch.randn(b, nk, kv, d, generator=generator) * 0.5).to(dtype).transpose(1, 2)
        v = (torch.randn(b, nk, kv, d, generator=generator) * 0.5).to(dtype).transpose(1, 2)
    mask = None
    if case["mask"] != "none":
        positions = torch.arange(nk)
        lengths = torch.tensor([max(1, nk - 3 - i) for i in range(b)])
        if case["mask"] == "causal_left":
            key_valid = positions[None, :] >= (nk - lengths)[:, None]
        else:
            key_valid = positions[None, :] < lengths[:, None]
        mask = key_valid[:, None, None, :]
        if case["mask"].startswith("causal"):
            causal = torch.arange(nq)[:, None] >= positions[None, :]
            mask = mask & causal[None, None, :, :]
        if case["mask"] == "key_additive":
            mask = torch.zeros(mask.shape, dtype=dtype).masked_fill(~mask, float("-inf"))
    upstream = torch.randn(b, h, nq, d, generator=generator).to(dtype) / math.sqrt(nq * d)
    return (q, k, v), mask, upstream


def synchronize(torch, device: str) -> None:
    if device.startswith("musa"):
        torch.musa.synchronize()


def run_kernel(torch, case, inputs, mask, upstream, kind: str, device: str, backward: bool):
    if case.get("packed"):
        # Transfer the packed base before selecting Q/K/V so the target really
        # exercises RADIO's non-dense B,S,3,H,D slices rather than compact copies.
        packed = torch.stack([x.transpose(1, 2) for x in inputs], dim=2).to(device)
        xs = [packed[:, :, i].transpose(1, 2).detach().requires_grad_(backward) for i in range(3)]
    else:
        xs = [x.transpose(1, 2).contiguous().to(device).transpose(1, 2).detach().requires_grad_(backward) for x in inputs]
    local_mask = None if mask is None else mask.to(device)
    local_upstream = upstream.to(device)
    synchronize(torch, device)
    start = time.perf_counter()
    with nullcontext() if backward else torch.no_grad():
        output = attention(torch, *xs, local_mask, kind)
        if backward:
            # A fixed vector-Jacobian product tests all output positions.
            (output * local_upstream).sum().backward()
    synchronize(torch, device)
    elapsed = time.perf_counter() - start
    finite = bool(torch.isfinite(output).all().item())
    grads = [x.grad.detach().cpu().double() for x in xs] if backward else []
    result = {
        "output": output.detach().cpu().double(), "grads": grads,
        "finite": finite and all(bool(torch.isfinite(x).all()) for x in grads),
        "seconds": elapsed, "output_dtype": str(output.dtype),
        "input_strides": [list(x.stride()) for x in xs],
        "inputs_contiguous": [x.is_contiguous() for x in xs],
    }
    del output, xs, local_upstream, local_mask
    if case.get("packed"):
        del packed
    return result


def error_metrics(torch, actual, expected, dtype_name: str) -> dict:
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "pass": False, "max_abs": None, "relative_l2": None}
    delta = actual - expected
    absolute = float(delta.abs().max())
    relative = float(delta.norm() / expected.norm().clamp_min(1e-12))
    atol, rtol, l2_limit = (2e-4, 2e-4, 2e-4) if dtype_name == "fp32" else (0.025, 0.04, 0.04)
    return {
        "finite": True, "max_abs": absolute, "relative_l2": relative,
        "atol": atol, "rtol": rtol, "relative_l2_limit": l2_limit,
        "pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)) and relative <= l2_limit,
    }


def numerics(torch, cases, args, emit) -> bool:
    success = True
    for case in cases:
        for dtype_name in args.dtypes.split(","):
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            inputs, mask, upstream = make_inputs(torch, case, dtype, args.seed)
            ref = run_kernel(torch, case, [x.double() for x in inputs], None if mask is None else mask if mask.dtype == torch.bool else mask.double(), upstream.double(), "reference", "cpu", case["backward"])
            peers = {}
            kinds = args.kinds.split(",")
            devices = ["cpu"] if args.device == "cpu" else ["cpu", args.device]
            for device in devices:
                for kind in kinds:
                    row = {"phase": "numerics", "case": case, "dtype": dtype_name, "kind": kind, "device": device, "reference": "CPU FP64 on the same quantized inputs; deterministic VJP"}
                    try:
                        actual = run_kernel(torch, case, inputs, mask, upstream, kind, device, case["backward"])
                        row.update({x: actual[x] for x in ("seconds", "finite", "output_dtype", "input_strides", "inputs_contiguous")})
                        row["forward_error"] = error_metrics(torch, actual["output"], ref["output"], dtype_name)
                        row["gradient_errors"] = [error_metrics(torch, a, b, dtype_name) for a, b in zip(actual["grads"], ref["grads"])]
                        row["pass"] = actual["finite"] and row["forward_error"]["pass"] and all(x["pass"] for x in row["gradient_errors"])
                        if device == "cpu":
                            peers[kind] = actual
                        elif kind in peers:
                            row["cpu_peer_forward_error"] = error_metrics(torch, actual["output"], peers[kind]["output"], dtype_name)
                            row["cpu_peer_gradient_errors"] = [error_metrics(torch, a, b, dtype_name) for a, b in zip(actual["grads"], peers[kind]["grads"])]
                            row["pass"] &= row["cpu_peer_forward_error"]["pass"] and all(x["pass"] for x in row["cpu_peer_gradient_errors"])
                    except Exception as exc:
                        row.update({"pass": False, "error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc(limit=3)})
                    emit(row)
                    if case["required"] and not row["pass"]:
                        success = False
            del ref, peers, inputs, mask, upstream
            gc.collect()
    return success


def primitives(torch, args, emit) -> bool:
    """Exercise BF16 operations needed around attention, including parameter VJP."""
    success = True
    for dtype_name in args.dtypes.split(","):
        dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
        gen = torch.Generator().manual_seed(args.seed)
        values = [(torch.randn(*shape, generator=gen) * 0.25).to(dtype) for shape in ((2, 7, 16), (16, 32), (32, 16), (16,))]
        def execute(device, reference=False):
            x, w1, w2, weight = [v.to(device=device, dtype=torch.float64 if reference else dtype).detach().requires_grad_() for v in values]
            # RMSNorm promotes reductions as Qwen3 does; LayerNorm + GELU/SiLU
            # cover DiT/connector, not a full pretrained GR00T transformer block.
            accum = x.double() if reference else x.float()
            rms = (accum * torch.rsqrt(accum.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
            norm = torch.nn.functional.layer_norm(rms, (16,), eps=1e-5)
            y = torch.nn.functional.gelu(norm @ w1, approximate="tanh") @ w2
            y = torch.nn.functional.silu(y)
            # Category-specific linear uses gathered weights and batched GEMM.
            selected = torch.stack((w2, w2), dim=0)
            z = torch.bmm(torch.nn.functional.gelu(norm @ w1), selected)
            # Actual sinusoidal/rotary elementwise layout (rotate_half), no fused RoPE.
            angles = torch.arange(7, device=device).float()[:, None] * torch.arange(8, device=device).float()[None, :] / 100
            cos = torch.cat((angles.cos(), angles.cos()), -1).to(x.dtype)
            sin = torch.cat((angles.sin(), angles.sin()), -1).to(x.dtype)
            a, b = x.chunk(2, dim=-1)
            rope = x * cos + torch.cat((-b, a), -1) * sin
            output = y + z + rope
            output.square().mean().backward()
            synchronize(torch, device)
            return output.detach().cpu().double(), [v.grad.detach().cpu().double() for v in (x, w1, w2, weight)]
        ref, refgrads = execute("cpu", reference=True)
        peer = None
        for device in (["cpu"] if args.device == "cpu" else ["cpu", args.device]):
            row = {"phase": "primitives", "device": device, "dtype": dtype_name, "operators": ["linear/matmul", "gathered category weight + bmm", "RMSNorm fp32 reduction", "LayerNorm", "GELU tanh", "SiLU", "rotary cos/sin/rotate_half"]}
            try:
                out, grads = execute(device)
                row["forward_error"] = error_metrics(torch, out, ref, dtype_name)
                row["gradient_errors"] = [error_metrics(torch, a, b, dtype_name) for a, b in zip(grads, refgrads)]
                row["pass"] = row["forward_error"]["pass"] and all(x["pass"] for x in row["gradient_errors"])
                if peer is None:
                    peer = (out, grads)
                else:
                    row["cpu_peer_forward_error"] = error_metrics(torch, out, peer[0], dtype_name)
                    row["cpu_peer_gradient_errors"] = [error_metrics(torch, a, b, dtype_name) for a, b in zip(grads, peer[1])]
                    row["pass"] &= row["cpu_peer_forward_error"]["pass"] and all(x["pass"] for x in row["cpu_peer_gradient_errors"])
            except Exception as exc:
                row.update({"pass": False, "error": type(exc).__name__ + ": " + str(exc)})
            emit(row)
            success &= row["pass"]
    return success


def memory(torch, cases, args, emit) -> bool:
    """Allocator peaks include inputs, output and requested backward, one layer."""
    success = True
    for case in cases:
        if not case["required"]:
            continue
        for dtype_name in args.dtypes.split(","):
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            inputs, mask, upstream = make_inputs(torch, case, dtype, args.seed)
            score_elements = case["b"] * case["h"] * case["q"] * case["k"]
            for kind in args.kinds.split(","):
                row = {"phase": "memory", "case": case, "dtype": dtype_name, "kind": kind, "device": args.device,
                       "score_elements": score_elements, "one_fp32_score_matrix_mib": score_elements * 4 / 2**20,
                       "one_dtype_score_matrix_mib": score_elements * inputs[0].element_size() / 2**20,
                       "scope": "one kernel including input allocation and optional VJP; not full model peak"}
                api = getattr(torch, "musa", None) if args.device.startswith("musa") else None
                gc.collect()
                if api is not None:
                    api.empty_cache()
                    api.synchronize()
                available = api is not None and all(hasattr(api, name) for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"))
                if available:
                    baseline = {"allocated": api.memory_allocated(), "reserved": api.memory_reserved()}
                    api.reset_peak_memory_stats()
                try:
                    result = run_kernel(torch, case, inputs, mask, upstream, kind, args.device, case["backward"])
                    row.update({"finite": result["finite"], "seconds": result["seconds"], "measurement": "musa_allocator" if available else "theoretical_only"})
                    if available:
                        row.update({"baseline_bytes": baseline,
                                    "peak_allocated_bytes": api.max_memory_allocated(), "peak_reserved_bytes": api.max_memory_reserved(),
                                    "incremental_peak_allocated_bytes": api.max_memory_allocated() - baseline["allocated"],
                                    "incremental_peak_reserved_bytes": api.max_memory_reserved() - baseline["reserved"]})
                    else:
                        row["limitation"] = "No supported allocator peak API; CPU matrix byte estimate is not measured RSS."
                except Exception as exc:
                    row.update({"error": type(exc).__name__ + ": " + str(exc)})
                success &= bool(row.get("finite", False)) and "error" not in row
                emit(row)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa", "musa:0"), default="cpu")
    parser.add_argument("--phase", choices=("numerics", "primitives", "memory", "all"), default="numerics")
    parser.add_argument("--profile", choices=("small", "feature-570"), default="small")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--kinds", default="sdpa,sdpa_layout,eager_native,eager_qwen,eager_fp32")
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="auto")
    parser.add_argument("--cases", default="all", help="Comma-separated geometry names, or all")
    parser.add_argument("--max-score-mib", type=float, default=128)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", required=True, help="JSON evidence path")
    args = parser.parse_args()
    if any(x not in {"fp32", "bf16"} for x in args.dtypes.split(",")):
        parser.error("--dtypes must contain fp32 and/or bf16")
    if any(x not in {"sdpa", "sdpa_layout", "eager_native", "eager_qwen", "eager_fp32"} for x in args.kinds.split(",")):
        parser.error("Unknown attention implementation")
    import torch
    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    cases = geometries(args.profile)
    if args.cases != "all":
        requested = set(args.cases.split(","))
        cases = [case for case in cases if case["name"] in requested]
        if not cases or requested != {case["name"] for case in cases}:
            parser.error("Unknown --cases name")
    for case in cases:
        if case["b"] * case["h"] * case["q"] * case["k"] * 4 / 2**20 > args.max_score_mib:
            parser.error("Case exceeds --max-score-mib; explicit budget change required")
    rows = []
    def emit(row):
        rows.append(row)
        compact = {k: row[k] for k in ("phase", "dtype", "kind", "device", "pass", "error") if k in row}
        if "case" in row:
            compact["case"] = row["case"]["name"]
        print(json.dumps(compact, ensure_ascii=False), flush=True)
    result = {"scope": "Synthetic operator numerical and memory gate, not a GR00T model/inference/training result",
              "torch_version": torch.__version__, "platform": platform.platform(), "arguments": vars(args), "rows": rows}
    if args.device.startswith("musa"):
        result["musa_device"] = torch.musa.get_device_name(0)
        sample = torch.empty(1, device=args.device)
        result["musa_tensor_device"] = str(sample.device)
        result["musa_tensor_is_cuda"] = sample.is_cuda
        result["musa_tensor_is_musa"] = getattr(sample, "is_musa", None)
        del sample
    success = True
    backend_context = nullcontext()
    if args.sdpa_backend == "math":
        backend_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    result["sdpa_backend_note"] = "Requested via shared ATen context; no profiler claim about the dispatched kernel"
    with backend_context:
        if args.phase in ("numerics", "all"):
            success &= numerics(torch, cases, args, emit)
        if args.phase in ("primitives", "all"):
            success &= primitives(torch, args, emit)
        if args.phase in ("memory", "all"):
            memory_success = memory(torch, cases, args, emit)
            result["memory_kernels_finite_without_error"] = memory_success
    result["required_numerics_pass"] = success if args.phase != "memory" else None
    if args.phase in ("memory", "all"):
        success &= memory_success
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
