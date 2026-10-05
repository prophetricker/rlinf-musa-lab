"""Isolate the old muDNN single-observation layout failure from RLinf and PPO."""

import argparse
import json

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch_musa  # noqa: F401

    torch.manual_seed(7)
    cpu_layer = torch.nn.Linear(3, 64)
    layer = torch.nn.Linear(3, 64).to("musa")
    layer.load_state_dict(cpu_layer.state_dict())
    raw = np.array([0.7, 0.7, 0.8], dtype=np.float32)[None]
    rows = []
    for name, values in (
        ("zero_stride_view", raw),
        ("explicit_c_order_copy", np.array(raw, copy=True, order="C")),
    ):
        inputs = torch.tensor(values, device="musa")
        row = {
            "case": name,
            "numpy_strides": list(values.strides),
            "tensor_strides": list(inputs.stride()),
        }
        try:
            actual = layer(inputs).detach().cpu()
            expected = cpu_layer(
                torch.tensor(np.array(values, copy=True, order="C"))
            ).detach()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            row.update(
                status="pass", max_abs_error=float((actual - expected).abs().max())
            )
        except Exception as error:
            row.update(
                status="blocked", error_type=type(error).__name__, error=str(error)
            )
        rows.append(row)
    report = {
        "date": "2026-10-05",
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "rows": rows,
    }
    from pathlib import Path

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if rows[-1]["status"] != "pass":
        raise RuntimeError("Copy-layout candidate failed")


if __name__ == "__main__":
    main()
