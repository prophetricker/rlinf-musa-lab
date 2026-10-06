"""Bound the size of FP32 L2 reductions on experimental Torch-MUSA FSDP1."""
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


def apply():
    import torch
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as strategy

    original = strategy.get_grad_norm_for_mixed_precision
    if getattr(original, "s4000_chunked_norm", False):
        return

    def bounded_norm(params, norm_type, zero, device):
        params = list(params)
        grads = [param.grad for param in params if param.grad is not None and param.grad.numel()]
        if float(norm_type) != 2 or not grads or any(grad.device.type != "musa" for grad in grads):
            return original(params, norm_type, zero, device)
        return chunked_l2_norm(grads).to(device)

    bounded_norm.s4000_chunked_norm = True
    strategy.get_grad_norm_for_mixed_precision = bounded_norm
