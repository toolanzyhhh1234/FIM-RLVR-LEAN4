#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir="training_logs/run_${timestamp}"
mkdir -p "${run_dir}"

export FIM_LOG_DIR="${run_dir}"
# Unsloth vLLM standby is incompatible with expandable segments (CUDA/HIP alloc conf).
_alloc_conf="${PYTORCH_ALLOC_CONF-}"
_alloc_conf="${_alloc_conf/,expandable_segments:True/}"
_alloc_conf="${_alloc_conf/expandable_segments:True,/}"
_alloc_conf="${_alloc_conf/expandable_segments:True/}"
export PYTORCH_ALLOC_CONF="${_alloc_conf}"
unset _alloc_conf

_cuda_alloc_conf="${PYTORCH_CUDA_ALLOC_CONF-}"
_cuda_alloc_conf="${_cuda_alloc_conf/,expandable_segments:True/}"
_cuda_alloc_conf="${_cuda_alloc_conf/expandable_segments:True,/}"
_cuda_alloc_conf="${_cuda_alloc_conf/expandable_segments:True/}"
export PYTORCH_CUDA_ALLOC_CONF="${_cuda_alloc_conf}"
unset _cuda_alloc_conf

_hip_alloc_conf="${PYTORCH_HIP_ALLOC_CONF-}"
_hip_alloc_conf="${_hip_alloc_conf/,expandable_segments:True/}"
_hip_alloc_conf="${_hip_alloc_conf/expandable_segments:True,/}"
_hip_alloc_conf="${_hip_alloc_conf/expandable_segments:True/}"
export PYTORCH_HIP_ALLOC_CONF="${_hip_alloc_conf}"
unset _hip_alloc_conf

python train_gspo_fim_qwen3-vl-8b.py 2>&1 | tee "${run_dir}/train.log"
