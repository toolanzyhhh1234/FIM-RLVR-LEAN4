#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir="training_logs/run_${timestamp}"
mkdir -p "${run_dir}"

export FIM_LOG_DIR="${run_dir}"

python train_gspo_fim_qwen3-vl-32b.py 2>&1 | tee "${run_dir}/train.log"
