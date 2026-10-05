#!/usr/bin/env python3
"""Diagnose SDPA key-mask broadcasting without changing the original probe.

CPU is the default. The coordinator explicitly selects --device musa and runs
auto/math requests serially. Original B,1,1,K failures remain in the output;
this probe neither changes numerical gates nor installs or patches packages.
Requires attention_probe.py alongside this file for unchanged input/oracle code.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import traceback

from attention_probe import attention, error_metrics, make_inputs, synchronize


VARIANTS = ("B11K", "B1QK_contiguous", "BHQK_expand_view", "BHQK_contiguous")


def mask_variant(base, case, variant):
    """Expand on the target device so the view retains zero-stride axes."""
    if variant == "B11K":
        return base
    if variant == "B1QK_contiguous":
        return base.expand(case["b"], 1, case["q"], case["k"]).contiguous()
    expanded = base.expand(case["b"], case["h"], case["q"], case["k"])
    return expanded if variant == "BHQK_expand_view" else expanded.contiguous()


def execute(torch, inputs, base_mask, upstream, case, variant, kind, device, metadata_sink=None):
    # All variants share contiguous BHSD Q/K/V; only the mask boundary changes.
    xs = [x.to(device).contiguous().detach().requires_grad_() for x in inputs]
    local_mask = mask_variant(base_mask.to(device), case, variant)
    local_upstream = upstream.to(device)
    metadata = {
        "mask_shape": list(local_mask.shape), "mask_stride": list(local_mask.stride()),
        "mask_contiguous": local_mask.is_contiguous(), "mask_dtype": str(local_mask.dtype),
        "qkv_shapes": [list(x.shape) for x in xs],
        "qkv_strides": [list(x.stride()) for x in xs],
    }
    if metadata_sink is not None:
        metadata_sink.update(metadata)
    output = attention(torch, *xs, local_mask, kind)
    (output * local_upstream).sum().backward()
    synchronize(torch, device)
    output = output.detach().cpu().double()
    grads = [x.grad.detach().cpu().double() for x in xs]
    return {"output": output, "grads": grads, "metadata": metadata}


def compare(torch, actual, expected, dtype_name):
    result = {
        "forward_error": error_metrics(torch, actual["output"], expected["output"], dtype_name),
        "gradient_errors": [error_metrics(torch, a, b, dtype_name)
                            for a, b in zip(actual["grads"], expected["grads"])],
    }
    result["pass"] = result["forward_error"]["pass"] and all(x["pass"] for x in result["gradient_errors"])
    return result


def blocked_gradient(torch, grads, allowed):
    blocked = (~allowed).expand_as(grads[1])
    metrics = []
    for name, grad in zip(("K", "V"), grads[1:]):
        values = grad[blocked]
        finite = bool(torch.isfinite(values).all())
        maximum = float(values.abs().max()) if finite else None
        # Zero is the mathematical mask contract, not an adjusted tolerance.
        metrics.append({"input": name, "finite": finite, "max_abs": maximum,
                        "pass": finite and bool((values == 0).all())})
    return {"gradients": metrics, "pass": all(x["pass"] for x in metrics)}


def perturb_blocked(torch, inputs, allowed, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    changed = [inputs[0]]
    for x in inputs[1:]:
        replacement = (torch.randn(x.shape, generator=generator) * 8 + 3).to(x.dtype)
        changed.append(torch.where(allowed, x, replacement))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "musa", "musa:0"), default="cpu")
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="auto")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dtypes = args.dtypes.split(",")
    if any(x not in {"fp32", "bf16"} for x in dtypes):
        parser.error("--dtypes must contain fp32 and/or bf16")
    import torch
    torch_musa_version = None
    if args.device.startswith("musa"):
        import torch_musa
        torch_musa_version = getattr(torch_musa, "__version__", None)
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    rows = []
    source_dir = Path(__file__).resolve().parent
    result = {
        "scope": "Synthetic key-mask representation diagnosis; not model validation or a kernel fix",
        "torch_version": torch.__version__, "torch_musa_version": torch_musa_version,
        "platform": platform.platform(), "arguments": vars(args),
        "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in ("attention_probe.py", "mask_contract_probe.py")},
        "variants": list(VARIANTS), "rows": rows,
        "reference": "CPU FP64 on the same quantized inputs; deterministic VJP",
        "gates": {"fp32": {"atol": 2e-4, "rtol": 2e-4, "relative_l2_limit": 2e-4},
                  "bf16": {"atol": 0.025, "rtol": 0.04, "relative_l2_limit": 0.04}},
        "backend_note": "Shared ATen flags requested; no profiler claim about dispatch",
        "mask_semantics": "bool True=allowed; additive 0=allowed/-inf=blocked; no empty rows",
    }
    context = nullcontext()
    if args.sdpa_backend == "math":
        context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    devices = ["cpu"] if args.device == "cpu" else ["cpu", args.device]
    with context:
        for mask_kind in ("key_bool", "key_additive"):
            case = dict(name="padding_" + mask_kind + "_contract", b=2, h=4, kv=4,
                        q=17, k=31, d=64, mask=mask_kind, backward=True, required=True)
            for dtype_name in dtypes:
                dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
                inputs, mask, upstream = make_inputs(torch, case, dtype, args.seed)
                allowed = mask if mask.dtype == torch.bool else torch.isfinite(mask)
                assert bool(allowed.any(-1).all()), "Empty rows are outside this probe"
                key_allowed = allowed.transpose(-1, -2)  # B,1,K,1 for K/V tensors.
                changed = perturb_blocked(torch, inputs, key_allowed, args.seed)
                ref_mask = mask if mask.dtype == torch.bool else mask.double()
                ref = execute(torch, [x.double() for x in inputs], ref_mask, upstream.double(),
                              case, "B11K", "reference", "cpu")
                ref_changed = execute(torch, [x.double() for x in changed], ref_mask, upstream.double(),
                                      case, "B11K", "reference", "cpu")
                reference_invariance = compare(torch, ref_changed, ref, dtype_name)
                if not reference_invariance["pass"]:
                    raise RuntimeError("CPU FP64 mask oracle changed under blocked-key perturbation")
                peers, bases = {}, {}
                for device in devices:
                    for kind in ("sdpa", "eager_fp32"):
                        for variant in VARIANTS:
                            row = {"case": case, "dtype": dtype_name, "device": device,
                                   "kind": kind, "variant": variant, "pass": False}
                            try:
                                actual = execute(torch, inputs, mask, upstream, case, variant, kind, device, row)
                                perturbed = execute(torch, changed, mask, upstream, case, variant, kind, device)
                                row.update(actual["metadata"])
                                row["numerics"] = compare(torch, actual, ref, dtype_name)
                                row["perturbed_numerics"] = compare(torch, perturbed, ref_changed, dtype_name)
                                row["blocked_key_invariance"] = compare(torch, perturbed, actual, dtype_name)
                                row["blocked_key_gradients"] = blocked_gradient(torch, actual["grads"], key_allowed)
                                key = (kind, variant)
                                if device == "cpu":
                                    peers[key] = actual
                                elif key in peers:
                                    row["cpu_peer"] = compare(torch, actual, peers[key], dtype_name)
                                base_key = (device, kind)
                                if variant == "B11K":
                                    bases[base_key] = actual
                                elif base_key in bases:
                                    row["original_representation_peer"] = compare(torch, actual, bases[base_key], dtype_name)
                                checked = ("numerics", "perturbed_numerics", "blocked_key_invariance",
                                           "blocked_key_gradients", "cpu_peer", "original_representation_peer")
                                row["pass"] = all(row[name]["pass"] for name in checked if name in row)
                            except Exception as exc:
                                row.update(error=type(exc).__name__ + ": " + str(exc),
                                           traceback=traceback.format_exc(limit=3))
                            rows.append(row)
                            print(json.dumps({name: row[name] for name in
                                  ("dtype", "device", "kind", "variant", "pass", "error") if name in row},
                                  ensure_ascii=False), flush=True)
    result["all_variants_pass"] = all(row["pass"] for row in rows)
    result["expanded_variants_pass"] = all(row["pass"] for row in rows if row["variant"] != "B11K")
    result["diagnostic_note"] = "Nonzero exit preserves any original or expanded numerical/contract failure. Read each row; expansion can run yet fail the unchanged FP32 gradient gate."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0 if result["all_variants_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
