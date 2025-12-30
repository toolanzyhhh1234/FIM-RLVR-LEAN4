#!/usr/bin/env bash
set -euo pipefail

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
branch_sanitized="$(echo "${branch}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
venv_dir=".venv-fastinf-${branch_sanitized}"

echo "[env] branch=${branch}"
echo "[env] venv_dir=${venv_dir}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH" >&2
  exit 1
fi

uv venv "${venv_dir}"

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

uv pip install -r ministral-training-requirements.txt

# vLLM often conflicts with other pins; install it without resolving deps.
uv pip install --no-deps vllm --torch-backend=auto

echo
echo "Done."
echo "Activate: source ${venv_dir}/bin/activate"
echo "Run: FIM_FAST_INFERENCE=1 bash scripts/run_train_mistral3.sh"

