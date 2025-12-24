# Session Summary (2025-12-24)

## Goal
Stand up a working VERL training environment (Megatron + vLLM) on a Blackwell GPU and debug official GSPO examples. Provide patched scripts and a Dockerfile path for reproducible installs.

## Key Environment Notes
- GPU: NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0)
- PyTorch: 2.9.x (cu128) in the live env
- FlashAttention2 not installed; SDPA used as fallback
- Megatron core requires `numpy<2.0`

## Critical Fixes Applied
1) **Missing python deps**
- Installed `packaging`, `pyzmq`.
- Attempted `vllm` install; it pulled many deps and downgraded `torch` to 2.9.0, then numpy was pinned back to 1.26.4. `opencv-python-headless` pinned to 4.11.0.86 to keep numpy<2 compatibility.

2) **Megatron + Blackwell arch**
- Set `TORCH_CUDA_ARCH_LIST="12.0+PTX"` to avoid JIT arch list crash during Megatron extension builds.

3) **Hydra config errors**
- Empty `ARNOLD_WORKER_GPU`/`ARNOLD_WORKER_NUM` caused TypeError in config validation; defaults were added to patched scripts.

4) **Dataset paths**
- HF dataset roots do not work with `datasets.load_dataset("parquet", data_files=...)`.
- Patched dataset paths to explicit parquet files:
  - `hf://datasets/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet`
  - `hf://datasets/BytedTsinghua-SIA/AIME-2024/data/aime-2024.parquet`

5) **Megatron mbridge**
- Missing `mbridge` caused crash; it’s required when `use_mbridge=True`. Script toggled as needed.

6) **Apex not installed**
- `gradient_accumulation_fusion=True` requires Apex CUDA extensions. Disabled in patched script to avoid crash.

7) **FlashAttention dependency**
- FSDP default `attn_implementation=flash_attention_2` requires FA2.
- For 3B test script, set `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` and disabled `use_remove_padding` to avoid FA2 usage.

8) **vLLM dependency**
- `vllm` was missing in the conda env; attempted install added many deps (and torch downgrade). Potentially disruptive.

## Files Added / Modified
- `run_qwen30b_gspo_patched_official_example.sh`
  - Uses HF model repo and parquet dataset files.
  - Single-GPU settings (TP=1, EP=1, infer_tp=1).
  - Default env vars: `ARNOLD_WORKER_GPU`, `ARNOLD_WORKER_NUM`, `TORCH_CUDA_ARCH_LIST`.
  - Logs to `verl_logs/run_qwen30b_gspo_<timestamp>.log`.
  - Disabled `gradient_accumulation_fusion`.

- `run_test_gspo_3b_math_patched.sh`
  - Single-GPU sanity test for 3B model.
  - GSM8K preprocessing if missing.
  - SDPA attention override and `use_remove_padding=false` to avoid FA2.
  - Logs to `verl_logs/run_test_gspo_3b_math_<timestamp>.log`.

- `Dockerfile.verl-enhanced-vllm`
  - Based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
  - Installs PyTorch cu124, FA2, vLLM.
  - Clones `verl` and installs requirements.
  - Installs `megatron-core`, `mbridge`, pins `numpy<2.0` + `opencv-python-headless<4.12`.
  - Added Miniconda and `uv` for package management.

- `docs/verl_specific_docs/README.md`
- `docs/verl_specific_docs/cloud_environment_setup.md`
- `AGENTS.md` updated with **user-led installs** policy for heavy packages.
- `verl_logs/` contains run logs from multiple attempts.

## Current Errors / Status
- 30B Megatron run hits GPU OOM on single GPU (expected).
- 3B FSDP test now runs further but hit missing `vllm` then FA2 issues; SDPA/disable remove-padding is the workaround.

## H200 Run (Success)
We got the 3B GSPO run to complete on a single H200 by reducing sequence length
and batch sizes at runtime (no code changes). Command (from `/workspace/verl`
inside the container):

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

Note: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not compatible with
vLLM v1 memory pool, so it must be unset.

## Recommended Next Steps
1) Run `run_test_gspo_3b_math_patched.sh` again after installing `vllm` in the active env.
2) If SDPA path works, keep it for environment validation.
3) For production, build FA2 from source with throttled flags:
   - `MAX_JOBS=4 NVCC_APPEND_FLAGS="--threads 2" TORCH_CUDA_ARCH_LIST="12.0" FLASH_ATTN_FAST_BUILD=1 pip install flash-attn --no-build-isolation`
4) Decide whether to build a custom Docker image (cu124 base) to leverage FA2 wheels.

## Codex session logs
- Most recent logs copied to `docs/codex_session_dump/2025/12/23` (gitignored).
