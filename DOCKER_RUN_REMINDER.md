Docker Run Reminder (Training)

When running in a VM or on a host where you start the container yourself, use a larger shared
memory segment for Ray/vLLM to avoid performance warnings and slowdowns.

Recommended command:

docker run --rm -it --gpus all --shm-size=10.24gb \
  -v /root/FIM-RLVR-LEAN4:/workspace \
  -v /root/FIM-RLVR-LEAN4/verl_logs:/app/verl_logs \
  -v /root/FIM-RLVR-LEAN4/data:/app/data \
  -v /root/FIM-RLVR-LEAN4/checkpoints:/app/checkpoints \
  -w /app \
  fim-rlvr-lean4:verl-vllm bash -lc "/workspace/run_test_gspo_3b_math_patched.sh"

Guidance:
- Set --shm-size to at least 30% of host RAM (e.g., 20gb for a 64GB host).
- If your cloud provider starts you inside the container directly (no VM / no docker run),
  ignore this note because you cannot change /dev/shm from inside the container.
