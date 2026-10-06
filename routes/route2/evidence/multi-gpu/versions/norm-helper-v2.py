"""Explicit local L2 norm alternatives for the fixed S4000 FSDP1 runtime."""
from __future__ import annotations


def chunked_l2_norm(tensors, *, chunk_size=1 << 20):
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    partials = []
    for value in tensors:
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_size):
            partials.append(torch.linalg.vector_norm(flat[start:start + chunk_size], 2, dtype=torch.float32))
    if not partials:
        raise ValueError("norm requires at least one nonempty tensor")
    # The L2 norm of the per-chunk L2 norms equals the original global L2 norm.
    return torch.linalg.vector_norm(torch.stack(partials), 2, dtype=torch.float32)


def legacy_l2_norm(tensors):
    import torch

    grads = [value.detach().to(torch.float32) for value in tensors if value.numel()]
    if not grads:
        raise ValueError("norm requires at least one nonempty tensor")
    return torch.norm(torch.stack([torch.norm(grad, 2) for grad in grads]), 2)


def apply(*, mode="legacy"):
    import torch
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as strategy

    original = strategy.get_grad_norm_for_mixed_precision
    if mode not in {"legacy", "chunked"}:
        raise ValueError(f"unsupported norm mode {mode}")
    if getattr(original, "s4000_norm_mode", None) == mode:
        return

    def bounded_norm(params, norm_type, zero, device):
        params = list(params)
        grads = [param.grad for param in params if param.grad is not None and param.grad.numel()]
        if float(norm_type) != 2 or not grads or any(grad.device.type != "musa" for grad in grads):
            return original(params, norm_type, zero, device)
        operation = legacy_l2_norm if mode == "legacy" else chunked_l2_norm
        return operation(grads).to(device)

    bounded_norm.s4000_norm_mode = mode
    strategy.get_grad_norm_for_mixed_precision = bounded_norm
