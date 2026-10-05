#!/usr/bin/env python3
"""Construct actual full GR00T on meta and compare all 899 checkpoint keys/shapes.

Reads only fixed public index/headers/config; no weight download/load or GPU.
Requires genuine dm-tree for the fixed GR00T class (policy/data modules unused).
Temporary distribution argument validation is disabled only for meta constructor
and restored, since meta tensors cannot evaluate Beta support predicates.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

from eagle_composite_probe import verify_sources, register_real_classes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=root / "prepared/spatial-full/source-preparation.json")
    parser.add_argument("--checkpoint-config", type=Path, default=root.parent / "backbone_sources/checkpoint-config.json")
    parser.add_argument("--metadata-dir", type=Path, default=root / "checkpoint_metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evidence already exists")
    prep, sources = verify_sources(args)
    if sha(args.checkpoint_config) != "6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526":
        raise ValueError("Official Spatial config hash mismatch")
    if sha(args.metadata_dir / "model.safetensors.index.json") != "bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746":
        raise ValueError("Official index hash mismatch")
    from download_spatial_weights import SHARDS
    headers, header_records = {}, []
    for name, (_, _, digest) in SHARDS.items():
        path = args.metadata_dir / (name + ".header.json")
        if sha(path) != digest:
            raise ValueError("Public expected header hash mismatch")
        values = json.loads(path.read_text())
        headers.update({key: value for key, value in values.items() if key != "__metadata__"})
        header_records.append({"path": str(path.resolve()), "sha256": digest})
    import torch
    import transformers
    if transformers.__version__ != "4.51.3":
        raise ValueError("Requires fixed Transformers4.51.3")
    from transformers import AutoConfig, AutoModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "gr00t" in sys.modules:
        raise RuntimeError("Requires a fresh process")
    sys.path.insert(0, str(args.source_tree.resolve()))
    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
    from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
    extra_hashes = {
        "gr00t/model/gr00t_n1.py": "1b4a9653f3818f7417f2e99aa558be0abf19dc1cd05e3d026a987dde294f8138",
        "gr00t/model/action_head/flow_matching_action_head.py": "8a8e6cf7ec63e2a335559990c4ab62bbb81e487d82ea4a969f452a93e0dbdd69",
        "gr00t/model/action_head/cross_attention_dit.py": "1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62",
    }
    for rel, expected in extra_hashes.items():
        if sha(args.source_tree / rel) != expected:
            raise ValueError(f"Actual full model source mismatch: {rel}")
        sources[rel] = {"path": str((args.source_tree / rel).resolve()), "sha256": expected}
    config_dict = json.loads(args.checkpoint_config.read_text())
    # RLinf's explicit builder overrides. They do not alter checkpoint shapes.
    config_dict["backbone_cfg"].update(tune_visual=False, tune_llm=False)
    config = GR00T_N1_5_Config(**config_dict)
    registration = {}
    old_validation = torch.distributions.Distribution._validate_args
    old_default_dtype = torch.get_default_dtype()
    try:
        torch.distributions.Distribution.set_default_validate_args(False)
        torch.set_default_dtype(torch.float32)
        with register_real_classes(AutoConfig, AutoModel, CONFIG_MAPPING,
                                   Eagle2_5_VLConfig, Eagle2_5_VLForConditionalGeneration, registration):
            with torch.device("meta"):
                model = GR00T_N1_5(config, local_model_path="metadata-only-no-weight-load")
    finally:
        torch.distributions.Distribution.set_default_validate_args(old_validation)
        torch.set_default_dtype(old_default_dtype)
    actual = model.state_dict()
    missing = sorted(set(actual) - set(headers))
    unexpected = sorted(set(headers) - set(actual))
    shape_mismatches = {key: {"actual": list(actual[key].shape), "checkpoint": headers[key]["shape"]}
                        for key in set(actual) & set(headers) if list(actual[key].shape) != headers[key]["shape"]}
    tensor_layout = {key: {"shape": list(value.shape), "constructor_dtype": str(value.dtype),
                           "checkpoint_dtype": headers[key]["dtype"] if key in headers else None,
                           "device": str(value.device)} for key, value in actual.items()}
    all_meta = all(value.device.type == "meta" for value in actual.values())
    result = {"pass": not missing and not unexpected and not shape_mismatches and all_meta,
              "probe_sha256": sha(__file__), "sources": sources, "source_preparation": prep,
              "checkpoint_config_sha256": sha(args.checkpoint_config), "headers": header_records,
              "actual_model_class": type(model).__module__ + "." + type(model).__name__,
              "actual_model_source": inspect.getfile(type(model)), "tensor_count": len(actual),
              "checkpoint_tensor_count": len(headers), "missing_in_checkpoint": missing,
              "unexpected_checkpoint_keys": unexpected, "shape_mismatches": shape_mismatches,
              "all_state_tensors_meta": all_meta, "tensor_layout": tensor_layout,
              "temporary_model_registration": registration,
              "distribution_validation_restored": torch.distributions.Distribution._validate_args == old_validation,
              "default_dtype_restored": torch.get_default_dtype() == old_default_dtype,
              "scope": "Exact full GR00T_N1_5 key/shape audit against public headers on meta. No weight payload, GPU, forward, action inference or learning."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "tensor_count": len(actual), "missing": len(missing),
                      "unexpected": len(unexpected), "shape_mismatches": len(shape_mismatches)}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
