#!/usr/bin/env bash
set -euo pipefail

# Simple GPU memory logger for pre-crash diagnosis.
# Usage:
#   ./scripts/gpu_vram_logger.sh /path/to/logdir [interval_seconds]
# Example:
#   ./scripts/gpu_vram_logger.sh /tmp/gpu_logs 1

LOG_DIR=${1:-/tmp/gpu_logs}
INTERVAL=${2:-1}

mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

GPU_LOG="${LOG_DIR}/gpu_mem_${TS}.csv"
PROC_LOG="${LOG_DIR}/gpu_procs_${TS}.csv"

# Header
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv > "${GPU_LOG}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "${PROC_LOG}"

# Stream logs
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l "${INTERVAL}" >> "${GPU_LOG}" &
GPU_PID=$!

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l "${INTERVAL}" >> "${PROC_LOG}" &
PROC_PID=$!

trap 'kill ${GPU_PID} ${PROC_PID} 2>/dev/null || true; echo "Logs saved to ${GPU_LOG} and ${PROC_LOG}"' EXIT

# Keep running until interrupted
while true; do
  sleep 3600
done
