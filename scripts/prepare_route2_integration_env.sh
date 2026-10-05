#!/usr/bin/env bash
# Keep prior action/backbone environments intact while combining pinned packages.
set -euo pipefail
research_root=/root/autodl-tmp/s4000-research
integration_env="$research_root/envs/route2-integration"
base_python="$research_root/envs/route2/bin/python"
if [[ ! -x "$integration_env/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$integration_env"
fi
"$integration_env/bin/python" -m pip install --no-deps \
  'transformers==4.51.3' 'huggingface_hub==0.30.2' \
  'tokenizers==0.21.4' 'regex==2024.11.6' 'safetensors==0.8.0' \
  'diffusers==0.30.2' 'dm-tree==0.1.8'
"$integration_env/bin/python" - <<'PY'
import importlib.metadata
import json
import torch
import torch_musa
from diffusers.models.attention_processor import Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

assert '/root/miniconda3/lib/python3.10/site-packages/torch/' in torch.__file__
print(json.dumps({
    "scope": "isolated dependencies only; no model construction or GPU tensors",
    "versions": {name: importlib.metadata.version(name) for name in (
        'torch', 'torch_musa', 'transformers', 'huggingface_hub',
        'tokenizers', 'regex', 'safetensors', 'diffusers', 'dm-tree'
    )},
    "torch_runtime": torch.__version__,
    "torch_musa_runtime": torch_musa.__version__,
    "torch_path": torch.__file__,
    "torch_musa_path": torch_musa.__file__,
    "model_classes": [Attention.__module__, Qwen3ForCausalLM.__module__, SiglipVisionModel.__module__],
}))
PY
