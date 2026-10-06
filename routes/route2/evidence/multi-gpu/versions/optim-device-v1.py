"""Opt-in device-handle dispatch for Torch 2.2 FSDP optimizer gathering.

The installed Torch package is never edited. Only the known CUDA diagnostic
and synchronization calls in this function are rewritten in this process;
non-MUSA invocations retain the original function.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap


def rewrite_source(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("unexpected FSDP optimizer source structure")
    function = tree.body[0]
    if function.name != "_allgather_orig_param_states" or function.decorator_list:
        raise RuntimeError("unexpected FSDP optimizer function/decorators")
    counts = {"memory_summary": 0, "synchronize": 0}

    class DeviceCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            target = node.func
            if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "torch"
                and target.value.attr == "cuda"):
                if target.attr not in counts or node.args or node.keywords:
                    raise RuntimeError("unexpected CUDA call in FSDP optimizer gathering")
                counts[target.attr] += 1
                target.value = ast.Attribute(
                    value=ast.Name(id="fsdp_state", ctx=ast.Load()),
                    attr="_device_handle", ctx=ast.Load(),
                )
            return node

    DeviceCalls().visit(tree)
    if counts != {"memory_summary": 1, "synchronize": 3}:
        raise RuntimeError(f"unsupported FSDP optimizer source calls: {counts}")
    return ast.fix_missing_locations(tree), counts


def apply():
    import torch
    import torch.distributed.fsdp._optim_utils as optim_utils

    original = optim_utils._allgather_orig_param_states
    previous = getattr(original, "s4000_device_adapter", None)
    if previous is not None:
        return previous
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "2"]:
        raise RuntimeError("device-handle adapter is scoped to audited Torch 2.2")
    source = textwrap.dedent(inspect.getsource(original))
    tree, counts = rewrite_source(source)
    namespace = {}
    exec(compile(tree, "<s4000-fsdp-optim-device-adapter>", "exec"),
         original.__globals__, namespace)
    adapted = namespace[original.__name__]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        state = bound.arguments["fsdp_param_info"].state
        if state.compute_device.type != "musa":
            return original(*args, **kwargs)
        for method in counts:
            if not callable(getattr(state._device_handle, method, None)):
                raise RuntimeError(f"MUSA FSDP device handle lacks {method}")
        return adapted(*args, **kwargs)

    metadata = {
        "torch_version": torch.__version__,
        "function": original.__name__,
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "replacement_calls": counts,
        "scope": "process-local, MUSA-only device-handle dispatch",
    }
    dispatch.s4000_device_adapter = metadata
    optim_utils._allgather_orig_param_states = dispatch
    return metadata
