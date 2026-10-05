#!/usr/bin/env python3
"""Read fixed GR00T sources and emit a minimal patch + tiny config.

This tool never edits the supplied source tree. The coordinator applies the
patch and copies the generated config into a separate fixed GR00T checkout.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path


COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
EAGLE_DIR = "gr00t/model/backbone/eagle2_hg_model"
SOURCE_HASHES = {
    "gr00t/__init__.py": "dd7d5f5e544e1d2923bf4f1aa548504255938f58a6e74a119356573e6ce9e36e",
    "gr00t/model/__init__.py": "279c47a1bcae3a98ac661206e04d0b1e9a15c73ce8262536431ba433589c3854",
    "gr00t/model/backbone/__init__.py": "00a176f47c8d95fd7e6cdbab55f293a6b4bd59f1efe97c47f35d061d99af73ef",
    "gr00t/model/backbone/eagle_backbone.py": "553d642fe1b5f7fca0b4d09a719a8df76ee21cca864d6292336c7656c5bc0b50",
    EAGLE_DIR + "/configuration_eagle2_5_vl.py": "d1578a2d5b7654da9a5e4cb18acd5d9d92b04a227ddab9365c2e3265d6cf2f93",
    EAGLE_DIR + "/modeling_eagle2_5_vl.py": "96554c1ca204cc6fb0bff636356b1f878d2ee47678b4a82e6378b59e98bb01d0",
    EAGLE_DIR + "/config.json": "e6957b163b6c62b685f69fd9d601e183509d109246290ad579d461692d925dba",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one fixed source match: {old!r}")
    return text.replace(old, new, 1)


def patched_sources(original):
    result = dict(original)
    name = EAGLE_DIR + "/configuration_eagle2_5_vl.py"
    result[name] = replace_once(result[name], "from .radio_model import RADIOConfig\n", "")
    result[name] = replace_once(result[name],
        '        elif vision_config["model_type"] == "radio":\n',
        '        elif vision_config["model_type"] == "radio":\n'
        '            from .radio_model import RADIOConfig\n\n')
    name = EAGLE_DIR + "/modeling_eagle2_5_vl.py"
    result[name] = replace_once(result[name], "from .radio_model import RADIOModel\n", "")
    result[name] = replace_once(result[name], "from peft import LoraConfig, get_peft_model\n", "")
    result[name] = replace_once(result[name],
        '                config.vision_config._attn_implementation = "flash_attention_2"\n',
        '                config.vision_config._attn_implementation = config._attn_implementation\n')
    result[name] = replace_once(result[name],
        '            elif config.vision_config.model_type == "radio":\n',
        '            elif config.vision_config.model_type == "radio":\n'
        '                from .radio_model import RADIOModel\n\n')
    for method in ("wrap_backbone_lora", "wrap_llm_lora"):
        result[name] = replace_once(result[name],
            f"    def {method}(self, r=128, lora_alpha=256, lora_dropout=0.05):\n",
            f"    def {method}(self, r=128, lora_alpha=256, lora_dropout=0.05):\n"
            "        from peft import LoraConfig, get_peft_model\n\n")
    name = "gr00t/model/__init__.py"
    result[name] = replace_once(result[name],
        "from .gr00t_n1 import GR00T_N1_5\nfrom .policy import Gr00tPolicy\n",
        '__all__ = ["GR00T_N1_5", "Gr00tPolicy"]\n\n\n'
        'def __getattr__(name):\n'
        '    if name == "GR00T_N1_5":\n'
        '        from .gr00t_n1 import GR00T_N1_5\n\n'
        '        return GR00T_N1_5\n'
        '    if name == "Gr00tPolicy":\n'
        '        from .policy import Gr00tPolicy\n\n'
        '        return Gr00tPolicy\n'
        '    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n')
    return result


def tiny_config(original):
    cfg = copy.deepcopy(original)
    # Resolve public local model registration instead of HF dynamic code's
    # static recursive scan of unused RADIO imports. Real classes are retained.
    cfg.pop("auto_map", None)
    cfg.update(_attn_implementation="eager", force_image_size=56,
               image_token_index=255, select_layer=-1, use_backbone_lora=0,
               use_llm_lora=0, mlp_checkpoint=False, torch_dtype="float32")
    cfg["text_config"].update(
        _attn_implementation="eager", hidden_size=512, intermediate_size=1536,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        num_hidden_layers=3, max_window_layers=3, vocab_size=256,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        torch_dtype="float32", use_cache=False, attention_dropout=0.0,
        attention_bias=False, use_sliding_window=False, sliding_window=None,
        rms_norm_eps=1e-6, rope_theta=1000000.0, rope_scaling=None)
    cfg["vision_config"].update(
        _attn_implementation="eager", hidden_size=288, intermediate_size=1076,
        num_attention_heads=4, num_hidden_layers=2, image_size=56, patch_size=14,
        num_channels=3, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
        attention_dropout=0.0, vision_use_head=True, torch_dtype="float32")
    assert cfg["mlp_connector_layers"] == 1 and cfg["use_pixel_shuffle"] is False
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path,
                        default=Path(__file__).parent / "audit_sources/Isaac-GR00T")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "prepared")
    args = parser.parse_args()
    original = {}
    files = []
    for name, expected in SOURCE_HASHES.items():
        raw = (args.source_tree / name).read_bytes()
        if digest(raw) != expected:
            raise RuntimeError(f"Fixed original source hash mismatch: {name}")
        original[name] = raw.decode()
    patched = patched_sources(original)
    diff = ""
    for name, text in original.items():
        if name.endswith(".py"):
            ast.parse(patched[name])
        if patched[name] != text:
            diff += "".join(difflib.unified_diff(text.splitlines(keepends=True),
                patched[name].splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name))
        files.append({"path": name, "original_sha256": SOURCE_HASHES[name],
                      "prepared_sha256": digest(patched[name].encode()),
                      "source_modified": patched[name] != text})
    cfg = tiny_config(json.loads(original[EAGLE_DIR + "/config.json"]))
    config_bytes = (json.dumps(cfg, indent=2) + "\n").encode()
    config_record = next(f for f in files if f["path"] == EAGLE_DIR + "/config.json")
    config_record["prepared_sha256"] = digest(config_bytes)
    config_record["source_modified"] = True
    config_record["tiny_config_replacement"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eagle-eager-lazy.patch").write_text(diff)
    (args.output_dir / "tiny-eagle-config.json").write_bytes(config_bytes)
    manifest = {"gr00t_commit": COMMIT, "source_tree_read_only": str(args.source_tree.resolve()),
                "preparation_script_sha256": digest(Path(__file__).read_bytes()),
                "files": files, "patch_sha256": digest(diff.encode()),
                "tiny_config_sha256": digest(config_bytes),
                "wrapper_settings": {"tune_visual": False, "tune_llm": False,
                                     "select_layer": 2, "project_to_dim": None},
                "scope": "Actual EagleBackbone constructor, real local AutoConfig/AutoModel model registration; tiny config only. Source forward/extract_feature/mlp1/substitution unchanged. No LoRA/RADIO/weights.",
                "changes": ["RADIOConfig/RADIOModel branch-only imports", "PEFT imported only by actual LoRA methods",
                            "SigLIP constructor respects top-level explicit eager", "Public gr00t.model exports lazy via PEP562",
                            "Tiny random config removes auto_map; real local model registration is temporary and restored"]}
    (args.output_dir / "source-preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "patch_sha256": manifest["patch_sha256"],
                      "tiny_config_sha256": manifest["tiny_config_sha256"]}))


if __name__ == "__main__":
    main()
