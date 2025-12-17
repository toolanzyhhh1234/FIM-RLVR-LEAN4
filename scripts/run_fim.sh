#!/usr/bin/env bash
set -euo pipefail

# Activate env with torch 2.6 + cu124 + flash-attn wheel
conda activate unsloth-py310

LOGDIR="training_logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d-%H%M%S")
LOGFILE="$LOGDIR/run-$TS.log"

# Allow user overrides without editing this file
: "${FIM_PARQUET_PATH:=data/data/train-00000-of-00001.parquet}"
: "${FIM_MAX_STEPS:=}"              # empty = use script default
: "${FIM_NUM_GENERATIONS:=}"        # empty = use script default
: "${FIM_MAX_COMPLETION_LENGTH:=}"  # empty = use script default

export PYTHONUNBUFFERED=1

TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE=0 \
UNSLOTH_DISABLE_DYNAMO=1 \
FIM_PARQUET_PATH="$FIM_PARQUET_PATH" \
FIM_MAX_STEPS="$FIM_MAX_STEPS" \
FIM_NUM_GENERATIONS="$FIM_NUM_GENERATIONS" \
FIM_MAX_COMPLETION_LENGTH="$FIM_MAX_COMPLETION_LENGTH" \
PYTORCH_CUDA_ALLOC_CONF= \
python -u train_gspo_fim_20b.py 2>&1 | tee "$LOGFILE"
