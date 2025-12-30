#!/usr/bin/env bash
set -euo pipefail

venv_dir=".venv-fastinf"
echo "[env] venv_dir=${venv_dir}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH" >&2
  exit 1
fi

uv venv "${venv_dir}"

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

# Ensure pip exists inside the venv (uv venv may omit it).
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi

# Install Unsloth via pip (brings its dependencies).
python -m pip install unsloth

# vLLM needs its dependencies; install with deps via uv.
uv pip install vllm --torch-backend=auto

# Install pinned transformers from git without resolving deps (last).
python -m pip install --no-deps \
  git+https://github.com/huggingface/transformers.git@a7f29523361b2cc12e51c1f5133d95f122f6f45c

echo
echo "Done."
echo "Activate: source ${venv_dir}/bin/activate"
echo "Run: FIM_FAST_INFERENCE=1 bash scripts/run_train_mistral3.sh"
