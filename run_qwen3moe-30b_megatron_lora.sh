#!/usr/bin/env bash
set -xeuo pipefail

# Need to install Megatron-Bridge
# NOTE: Make sure you use Megatron-Bridge later than 0.2.0 
# (Recommend https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/a489bed3a2410ed9b000ec13a3c90176fec7d99c or later)
# for proper MoE LoRA support.

# For Megatron communication/computation overlapping
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

# Parallelism settings - adjust based on available GPUs
# For single GPU, all must be 1
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}


rollout_name="vllm"
project_name='fim_rlvr_lean4'
exp_name='qwen3_30b_megatron_lora_fim'
adv_estimator=grpo

# FIM-RLVR-LEAN4 data paths - preprocessed NuminaMath-LEAN dataset
# Run: python scripts/preprocess_numinamath_lean.py to generate these files
fim_train_path=${FIM_TRAIN_PATH:-$HOME/data/numinamath_lean/train.parquet}
fim_test_path=${FIM_TEST_PATH:-$HOME/data/numinamath_lean/val.parquet}

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${fim_train_path}
    data.val_files=${fim_test_path}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.train_batch_size=8
    data.max_prompt_length=1024
    data.max_response_length=512
    data.truncation='left'
    data.filter_overlong_prompts=True
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Instruct-2507
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.lora.rank=32
    actor_rollout_ref.model.lora.alpha=64
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    # # Optional: Use canonical LoRA
    # actor_rollout_ref.model.lora.type="canonical_lora"
    # actor_rollout_ref.model.lora.target_modules='["linear_q","linear_k","linear_v","linear_proj","linear_fc1_up","linear_fc1_gate","linear_fc2"]'

    # # Optional: Add dropout to LoRA layers
    # actor_rollout_ref.model.lora.dropout=0.05
    # actor_rollout_ref.model.lora.dropout_position=pre
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    # Use FSDP strategy for single GPU (not Megatron which requires multiple GPUs)
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # KL loss configuration
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    # GSPO configuration (sequence-level importance sampling)
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    # CPU offloading for memory efficiency on single GPU
    +actor_rollout_ref.actor.fsdp.param_offload=True
    +actor_rollout_ref.actor.fsdp.optimizer_offload=True
    +actor_rollout_ref.actor.fsdp.grad_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # Rollout quantization: FP8 for vLLM rollout server
    +actor_rollout_ref.rollout.quantization=fp8
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=2
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    # Use FSDP for ref model as well
    +actor_rollout_ref.ref.fsdp.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
)

REWARD=(
    reward_model.reward_manager=lean_verifier
    +reward_model.reward_kwargs.lean_env_path=${LEAN_ENV_PATH:-/workspace/verl/verification_env}
    +reward_model.reward_kwargs.verification_timeout=30
    +reward_model.reward_kwargs.parallel_workers=20
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=10
    trainer.test_freq=10
    trainer.total_epochs=10
)

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
