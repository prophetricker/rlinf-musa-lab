#!/usr/bin/env bash
# Separate Transformers from the validated Worker and action Attention environments.
set -euo pipefail
research_root=/root/autodl-tmp/s4000-research
backbone_env="$research_root/envs/route2-backbone"
base_python="$research_root/envs/route2/bin/python"
if [[ ! -x "$backbone_env/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$backbone_env"
fi
"$backbone_env/bin/python" -m pip install --no-deps \
  'transformers==4.51.3' 'huggingface_hub==0.30.2' \
  'tokenizers==0.21.4' 'regex==2024.11.6' 'safetensors==0.8.0'
"$backbone_env/bin/python" - <<'PY'
import importlib.metadata
import json
import torch
import torch_musa
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

print(json.dumps({
    "scope": "isolated Transformers dependency import, no model weights or GPU tensors",
    "versions": {name: importlib.metadata.version(name) for name in (
        "transformers", "huggingface_hub", "tokenizers", "regex", "torch", "torch_musa"
    )},
    "torch_runtime": torch.__version__,
    "torch_path": torch.__file__,
    "torch_musa_path": torch_musa.__file__,
    "real_qwen3_attention": Qwen3Attention.__module__,
    "real_siglip_vision": SiglipVisionModel.__module__,
}))
PY
