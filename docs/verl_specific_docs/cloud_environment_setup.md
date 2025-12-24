# verl-specific cloud environment setup notes

This file summarizes what we had to install on top of the `verlai/verl:vllm011.latest`
base image to run the Megatron-based GSPO example in this repo.

## What the base image provides
- CUDA toolkit and runtime libraries.
- vLLM stack and related CUDA tooling.

## Python deps we installed or needed
- `packaging` (required by `verl` import path).
- `verl` runtime deps from `verl/requirements.txt` (datasets, ray, transformers, etc.).
- `megatron-core` (required by the Megatron backend).
- `mbridge` (required when `actor_rollout_ref.actor.megatron.use_mbridge=True`).
- `torch` and CUDA wheels may be pulled if not already present in the env.

## Optional but recommended for Megatron fusions
- `apex` built from source with CUDA extensions
  (`APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .`).
  Needed for `gradient_accumulation_fusion=True` and other fused kernels.

## Notes for this repo
- The patched script `run_qwen30b_gspo_patched_official_example.sh` sets:
  - HF repo IDs for model paths.
  - HF parquet file paths for datasets.
  - single-GPU Megatron settings (TP=1, EP=1, infer_tp=1).
  - default env vars: `ARNOLD_WORKER_GPU`, `ARNOLD_WORKER_NUM`,
    `TORCH_CUDA_ARCH_LIST`.
  - log teeing to `verl_logs/run_qwen30b_gspo_<timestamp>.log`.
*** End Patch}Lark would error due to trailing text? We'll produce proper patch. Let's craft correct. Use proper patch format. beware instructions: tool expects patch only. We'll craft.### We'll try.】
