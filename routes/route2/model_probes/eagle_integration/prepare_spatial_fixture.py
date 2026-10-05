#!/usr/bin/env python3
"""Emit a full Spatial geometry eager config/manifest; no source-tree edits.

Preserve full bundled dimensions and original 28 LLM layers. The real wrapper
select_layer12 prunes to the checkpoint's 12 layers. Real weights are loaded
into a CPU FP32 model later; this helper does not load/download any tensors.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--original-config", type=Path, default=root / "audit_sources/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json")
    parser.add_argument("--patch-manifest", type=Path, default=root / "prepared/source-preparation.json")
    parser.add_argument("--output-dir", type=Path, default=root / "prepared/spatial-full")
    args = parser.parse_args()
    raw = args.original_config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "e6957b163b6c62b685f69fd9d601e183509d109246290ad579d461692d925dba":
        raise ValueError("Original full Eagle config hash mismatch")
    cfg = json.loads(raw)
    cfg.pop("auto_map", None)
    cfg["_attn_implementation"] = "eager"
    cfg["torch_dtype"] = "float32"
    for component in ("text_config", "vision_config"):
        cfg[component]["_attn_implementation"] = "eager"
        cfg[component]["torch_dtype"] = "float32"
    payload = (json.dumps(cfg, indent=2) + "\n").encode()
    manifest = copy.deepcopy(json.loads(args.patch_manifest.read_text()))
    manifest["preparation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["profile"] = "Full Spatial checkpoint geometry, not tiny"
    manifest["changes"][-1] = "Full geometry config removes auto_map and sets explicit eager/FP32; real local model registration is temporary and restored"
    manifest.pop("tiny_config_sha256", None)
    manifest["full_config_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["wrapper_settings"] = {"tune_visual": False, "tune_llm": False, "select_layer": 12, "project_to_dim": None}
    for record in manifest["files"]:
        if record["path"].endswith("eagle2_hg_model/config.json"):
            record["prepared_sha256"] = manifest["full_config_sha256"]
            record.pop("tiny_config_replacement", None)
            record["spatial_full_config_fixture"] = True
    manifest["scope"] = "Full Qwen3 H16/KV8/D128 hidden2048 (28 constructed,12 retained), full SigLIP H16/D72 hidden1152/27layers/image224/patch14; connector1152->2048 and Identity. CPU initial FP32; actual checkpoint backbone585BF16 tensors converted toFP32 at strict load."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spatial-eager-config.json").write_bytes(payload)
    (args.output_dir / "source-preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"full_config_sha256": manifest["full_config_sha256"], "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
