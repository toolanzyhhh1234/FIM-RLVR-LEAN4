#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p training_logs

# Generate timestamp for the log file (e.g., train_2024-12-16_15-30-00.log)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="training_logs/train_${TIMESTAMP}.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting training run at $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Run the python script
# 2>&1 redirects stderr (errors) to stdout so they are also captured in the log
# | tee -a "$LOG_FILE" pipes the output to both the terminal screen and the log file
python train_grpo_fim.py 2>&1 | tee -a "$LOG_FILE"

# Capture the exit status of the python command (the first command in the pipe)
EXIT_STATUS=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
if [ $EXIT_STATUS -ne 0 ]; then
    echo "❌ Training CRASHED or Failed with exit code $EXIT_STATUS" | tee -a "$LOG_FILE"
else
    echo "✅ Training finished successfully" | tee -a "$LOG_FILE"
fi
echo "==================================================" | tee -a "$LOG_FILE"
