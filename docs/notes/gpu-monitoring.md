# GPU Monitoring Cheatsheet

Use these during training to confirm the A100 is busy and sized correctly.

- Live dashboard: `watch -n1 nvidia-smi` (1s refresh; util, mem, power).
- Smoothed telemetry: `nvidia-smi dmon -s pucmem -d 2` (power/util/clock/mem every 2s; Ctrl+C to stop).
- Continuous log to file: `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 5 >> training_logs/gpu-usage.csv` (append a line every 5s for later plotting).
- One-shot check: `nvidia-smi` before/after a step to see memory reservation.

Tip: run training in one tmux pane and `watch -n1 nvidia-smi` in another for realtime visibility.
