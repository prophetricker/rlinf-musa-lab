"""Run legacy-DTensor and vendor-detection regression checks without a GPU job."""

from __future__ import annotations

import argparse
import builtins
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


def main() -> None:
    """Check patched public behavior using fakes only for external vendor APIs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch

    original_import = builtins.__import__

    def legacy_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch.distributed.tensor" and "DTensor" in fromlist:
            raise ImportError("DTensor uses the legacy namespace in Torch 2.2")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=legacy_import):
        utils = importlib.import_module("rlinf.utils.utils")
    from torch.distributed._tensor import DTensor

    from rlinf.scheduler.hardware.accelerators.musa_gpu import MUSAGPUManager

    assert utils.DTensor is DTensor, "The legacy namespace must supply DTensor"
    tensor = torch.tensor([1.0, 2.0])
    assert utils.materialize_tensor(tensor) is tensor, "Dense tensors must pass through"
    mtml = ModuleType("pymtml")
    mtml.mtmlLibraryInit = mock.Mock(side_effect=RuntimeError("MTML not found"))
    platform = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _index: "MTT S4000",
    )
    with (
        mock.patch.dict(
            sys.modules, {"pymtml": mtml, "torch_musa": ModuleType("torch_musa")}
        ),
        mock.patch.object(torch, "musa", platform, create=True),
        mock.patch.dict(os.environ),
    ):
        os.environ.pop("RLINF_MUSA_TORCH_DETECTION_FALLBACK", None)
        assert MUSAGPUManager.get_num_devices() == 0, "Default MTML behavior changed"
        assert MUSAGPUManager.get_accelerator_model() == "UNKNOWN", (
            "Default MTML behavior changed"
        )
        os.environ["RLINF_MUSA_TORCH_DETECTION_FALLBACK"] = "1"
        assert MUSAGPUManager.get_num_devices() == 1, "Opt-in runtime count was ignored"
        assert MUSAGPUManager.get_accelerator_model() == "MTT S4000", (
            "Opt-in runtime model was ignored"
        )
        platform.is_available = lambda: False
        assert MUSAGPUManager.get_num_devices() == 0, (
            "Unavailable runtime reported a device"
        )
        assert MUSAGPUManager.get_accelerator_model() == "UNKNOWN", (
            "Unavailable runtime reported a model"
        )
    print(
        json.dumps(
            {
                "probe": "cpu_compat_regressions",
                "status": "pass",
                "legacy_dtensor": True,
                "musa_detection_cases": 3,
            }
        )
    )


if __name__ == "__main__":
    main()
