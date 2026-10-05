"""Read package paths/versions for the isolated Eagle integration experiment."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

result = {
    "scope": "CPU-only package inventory; no model import, weights or GPU tensors",
    "executable": sys.executable,
    "python": platform.python_version(),
    "packages": {},
    "module_sources": {},
}
for name in (
    "torch", "torch_musa", "transformers", "diffusers", "peft", "accelerate",
    "huggingface_hub", "safetensors", "tokenizers", "regex", "einops", "timm", "gr00t", "dm-tree",
):
    try:
        result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][name] = None
    spec = importlib.util.find_spec("tree" if name == "dm-tree" else name)
    if spec is not None and spec.origin is not None:
        source = Path(spec.origin)
        result["module_sources"][name] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
print(json.dumps(result))
