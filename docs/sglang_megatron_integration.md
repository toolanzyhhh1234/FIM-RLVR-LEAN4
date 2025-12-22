# SGLang Integration with Megatron-LM and LoRA

This document outlines the steps and configurations required to use **SGLang** as the rollout engine within the FIM-RLVR framework, specifically when using the **Megatron-LM** backend with **LoRA**.

## Overview

SGLang provides a high-throughput inference engine that can be used as a drop-in replacement for vLLM in the `verl` framework. When integrated with Megatron-LM (used for training MoE models), it operates in a "Hybrid" mode where weights are synchronized via the **Megatron-Bridge**.

## Configuration

To switch from vLLM to SGLang, the following changes are required in your training script (e.g., `run_qwen3moe_30b_fim_gspo.sh`).

### 1. Enable SGLang Rollout

Update the `rollout_name` variable to `sglang`:

```bash
rollout_name="sglang"
```

This maps to the configuration parameter:
`actor_rollout_ref.rollout.name=sglang`

### 2. Required Environment Variables

To ensure stability and prevent memory management conflicts between the Megatron training process and the SGLang inference server, export the following environment variables at the top of your execution script:

```bash
# Disable expandable segments to avoid memory allocation issues (specific to Megatron+SGLang)
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1

# Prevent Ray from restricting GPU visibility. 
# SGLang needs to see all GPUs to manage its own internal distributed process group.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Explicitly set CUDA visibility if using Tensor Parallelism (TP) >= 8
# This ensures SGLang can initialize across the entire node.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 3. Memory Management

For large models (like Qwen3-30B MoE), it is critical to release GPU memory during the training phase so that SGLang does not hold onto resources needed by the actor/critic.

Ensure the following configuration is set (usually true by default in our examples):

```bash
actor_rollout_ref.rollout.free_cache_engine=True
```

## How It Works

### Weight Synchronization
*   **Mechanism**: Weight synchronization happens via `verl`'s integration with the **Megatron-Bridge**.
*   **Merging**: When the actor (Megatron) updates weights, the LoRA adapters are **merged** into the base model weights.
*   **Transfer**: These merged weights are exported and sent to the SGLang server (via HTTP/REST).
*   **Inference**: SGLang treats the received model as a standard BF16 model. It does **not** need to handle LoRA adapters natively in this setup, as the "adapter" logic is handled upstream by the merge step.

### Tensor Parallelism
*   **Distribution**: SGLang will launch its own distributed process group.
*   **TP Size**: This is controlled by `actor_rollout_ref.rollout.tensor_model_parallel_size`. For 30B models on H100s, this is typically set to 8 (matching the node size).

## Troubleshooting

*   **OOM Errors**: If you encounter Out-Of-Memory errors during the switch between rollout and training, verify that `free_cache_engine=True` is set and that `MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1` is exported.
*   **Hang/Timeout**: If SGLang hangs during initialization, check `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`. Ray's default behavior is to restrict GPU visibility per worker, which breaks SGLang's internal assumption that it can orchestrate its own world size.
