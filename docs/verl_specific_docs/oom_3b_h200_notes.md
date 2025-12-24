# 3B GSPO OOM (H200) and Working Overrides

## Summary
Running `run_test_gspo_3b_math_patched.sh` on a single H200 hit CUDA OOM during
`compute_log_prob` even though the model is small. The issue was caused by large
sequence lengths and batch sizing in the default script. The fix was to pass
smaller overrides at runtime (no code changes required) and *not* to use
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` because it conflicts with
vLLM's memory pool.

## Symptoms
- OOM during log-prob calculation (`verl/workers/fsdp_workers.py` -> `compute_log_prob`)
- Error mentions a huge allocation (e.g., ~100 GiB) in `softmax`
- GPU had free memory but the kernel still failed due to oversized tensors
- Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` caused vLLM to crash:
  "Expandable segments are not compatible with memory pool."

## Root cause
Default script values were large for a single-GPU run:
- `max_prompt_length=2048`
- `max_response_length=4096`
- `train_batch_size=64`
- `rollout.n=4`
- `log_prob_max_token_len_per_gpu=18432`

These compound in `compute_log_prob`, which is memory heavy, leading to OOM.

## Working command (runtime overrides)
Run from `/workspace/verl` inside the container:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
bash run_test_gspo_3b_math_patched.sh \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.train_batch_size=8 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

## Notes
- These overrides keep VRAM usage moderate (~70-80 GB on H200 in our run).
- You can scale up gradually once the run is stable:
  - Increase `max_prompt_length`/`max_response_length`
  - Increase `train_batch_size`
  - Increase `rollout.n`
  - Increase `max_num_batched_tokens` and log-prob token limits
- Do not use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with vLLM v1.
