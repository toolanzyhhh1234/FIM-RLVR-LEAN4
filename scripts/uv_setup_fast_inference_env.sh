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

# Install using pip to avoid uv's resolver/compatibility checks.
python -m pip install --no-deps -r ministral-training-requirements.txt

# vLLM often conflicts with other pins; install it without resolving deps.
uv pip install --no-deps vllm --torch-backend=auto

echo
echo "Done."
echo "Activate: source ${venv_dir}/bin/activate"
echo "Run: FIM_FAST_INFERENCE=1 bash scripts/run_train_mistral3.sh"
