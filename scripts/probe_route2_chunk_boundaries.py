#!/usr/bin/env python3
"""CPU-check upstream loss-mask semantics at real chunk boundaries, without Ray."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    import torch
    source = args.rlinf_source / "rlinf/utils/metric_utils.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "compute_loss_mask")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    rows = []
    for chunks in (1, 5):
        for ends in ((None, None), (0, 1), (3, 4), (5, 239), (239, None)):
            dones = torch.zeros((240 // chunks + 1, 2, chunks), dtype=torch.bool)
            expected_actions = torch.zeros((2, 240), dtype=torch.bool)
            for lane, end in enumerate(ends):
                count = 240 if end is None else end + 1
                expected_actions[lane, :count] = True
                if end is not None:
                    # Include repeated boundary flags after the first ending.
                    flat = torch.zeros(240, dtype=torch.bool)
                    flat[end:] = True
                    dones[1:, lane, :] = flat.reshape(240 // chunks, chunks)
            mask, sums = namespace["compute_loss_mask"](dones)
            expected = expected_actions.reshape(2, 240 // chunks, chunks).transpose(0, 1)
            checks = {"exact_action_prefix": torch.equal(mask, expected),
                      "valid_action_counts": torch.equal(sums[0, :, 0], expected_actions.sum(dim=1)),
                      "chunk_level_any_mask": torch.equal(mask.any(dim=-1), expected.any(dim=-1))}
            rows.append({"actions_per_chunk": chunks, "first_terminal_slots_zero_based": ends,
                         "valid_action_counts": expected_actions.sum(dim=1).tolist(),
                         "valid_chunk_counts": mask.any(dim=-1).sum(dim=0).tolist(), "checks": checks})
    result = {"status": "pass" if all(all(row["checks"].values()) for row in rows) else "fail",
              "scope": "AST-extracted actual upstream compute_loss_mask on CPU; no environment, Ray, policy, GPU, GAE or PPO execution",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "case_count": len(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "case_count": len(rows)}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
