#!/usr/bin/env bash
set -euo pipefail

# Create a dedicated Python 3.10 env and install the Ministral-3 stack
# Uses transformers dev build (required for Mistral-3) and no vLLM.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv310_mistral3"

echo "==> Installing Python 3.10 via uv (if needed)"
uv python install 3.10

echo "==> Creating venv at ${ENV_DIR}"
uv venv --python 3.10 "${ENV_DIR}"

echo "==> Installing Torch 2.9.0+cu128"
uv pip install --python "${ENV_DIR}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0+cu128 torchvision==0.24.0+cu128

echo "==> Installing base deps (repo + notebook stack)"
uv pip install --python "${ENV_DIR}/bin/python" -r "${ROOT_DIR}/requirements_unsloth_qwen3vl_full.txt"

echo "==> Upgrading transformers to dev build (required for Mistral-3)"
uv pip install --python "${ENV_DIR}/bin/python" --upgrade \
  git+https://github.com/huggingface/transformers.git

echo "==> Done. Activate with:"
echo "    source ${ENV_DIR}/bin/activate"
echo "Then run:"
echo "    FIM_FAST_INFERENCE=0 python train_gspo_fim_mistral3.py"
