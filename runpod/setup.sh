#!/usr/bin/env bash
# Run this once on a fresh RunPod pod, from anywhere:
#   git clone https://github.com/SampleBias/Coding_Decision_Agent.git
#   cd Coding_Decision_Agent && bash runpod/setup.sh
#
# Uses the pod's existing CUDA torch if import torch works and torch.cuda.is_available().
# Otherwise installs the default PyTorch wheel (needs a CUDA driver on the pod).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA torch already available: $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
else
  echo "installing PyTorch (CUDA 12.4 wheel)"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install -e ".[train]"
# vllm is large; install it only for the teacher stage. Uncomment if you want it now:
# python -m pip install -e ".[train,teacher]"

if [[ -n "${HF_TOKEN:-}" ]]; then
  python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
print("Hugging Face login ok")
PY
else
  echo "HF_TOKEN is not set. Export a write token before the publish stage."
fi

mkdir -p "${WORKDIR:-/workspace/cda_artifacts}"
echo "setup done. next: bash runpod/run_pipeline.sh data"
