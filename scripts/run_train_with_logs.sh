#!/usr/bin/env bash
set -euo pipefail

# Run training while streaming stdout to terminal and logging to disk,
# plus record GPU VRAM trajectory in parallel.
#
# Usage:
#   ./scripts/run_train_with_logs.sh [log_dir] [interval_seconds] -- <train_command...>
# Example:
#   ./scripts/run_train_with_logs.sh /tmp/train_logs 1 -- bash run_qwen3moe-30b_megatron_lora.sh

LOG_DIR=${1:-/tmp/train_logs}
INTERVAL=${2:-1}

shift 2 || true
if [ "${1:-}" != "--" ]; then
  echo "ERROR: expected -- before the training command."
  exit 1
fi
shift 1

if [ "$#" -eq 0 ]; then
  echo "ERROR: no training command provided."
  exit 1
fi

mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="${LOG_DIR}/train_${TS}.log"

GPU_LOG_DIR="${LOG_DIR}/gpu_${TS}"
mkdir -p "${GPU_LOG_DIR}"

./scripts/gpu_vram_logger.sh "${GPU_LOG_DIR}" "${INTERVAL}" &
GPU_LOGGER_PID=$!

cleanup() {
  kill "${GPU_LOGGER_PID}" 2>/dev/null || true
  echo "Logs saved to:"
  echo "  ${TRAIN_LOG}"
  echo "  ${GPU_LOG_DIR}/gpu_mem_${TS}.csv"
  echo "  ${GPU_LOG_DIR}/gpu_procs_${TS}.csv"
}
trap cleanup EXIT

set +e
("$@") 2>&1 | tee "${TRAIN_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

exit "${EXIT_CODE}"
