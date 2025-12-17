#!/usr/bin/env bash
set -euo pipefail

INTERVAL=${GPU_MON_INTERVAL:-5}
LOGDIR=training_logs
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d-%H%M%S")
LOGFILE="$LOGDIR/gpu-usage-$TS.csv"

echo "Logging GPU stats every ${INTERVAL}s to $LOGFILE (Ctrl+C to stop)"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,pstate,clocks.sm \
  --format=csv \
  -l $INTERVAL | tee "$LOGFILE"
