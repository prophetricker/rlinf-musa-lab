#!/usr/bin/env bash
# Isolate action-model dependencies from the validated Worker/FSDP environment.
set -euo pipefail
research_root=/root/autodl-tmp/s4000-research
model_env="$research_root/envs/route2-models"
base_python="$research_root/envs/route2/bin/python"
if [[ ! -x "$model_env/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$model_env"
  "$model_env/bin/python" - "$model_env" <<'PY'
import pathlib, sys
env = pathlib.Path(sys.argv[1])
site = next((env / 'lib').glob('python*/site-packages'))
(site / 'research-base.pth').write_text('/root/autodl-tmp/rlinf-s4000/lib/python3.10/site-packages\n')
PY
fi
"$model_env/bin/python" -m pip install --no-deps 'diffusers==0.30.2'
"$model_env/bin/python" - <<'PY'
import hashlib, importlib.metadata, inspect, json
from pathlib import Path
import torch, torch_musa
import diffusers.models.attention_processor as attention
from diffusers import ConfigMixin, ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
print(json.dumps({
    'scope': 'isolated dependency import only; no MUSA tensor or training',
    'versions': {n: importlib.metadata.version(n) for n in ['diffusers','torch','torch_musa','huggingface_hub','accelerate']},
    'torch_runtime': torch.__version__,
    'attention_processor_sha256': hashlib.sha256(Path(inspect.getfile(attention)).read_bytes()).hexdigest(),
    'attention_processor_path': inspect.getfile(attention),
}))
PY
