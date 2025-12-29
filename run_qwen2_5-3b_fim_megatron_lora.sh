#!/usr/bin/env bash
set -xeuo pipefail

# 3B pipeline sanity script (single GPU) using Megatron config.
# Keep Megatron so the same knobs apply when scaling to multi-GPU for 30B-A3B.

export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

# Megatron parallelism (single GPU = 1)
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

# Megatron LoRA requires Megatron-Bridge.
USE_MBRIDGE=${USE_MBRIDGE:-True}
VANILLA_MBRIDGE=${VANILLA_MBRIDGE:-False}

# Offload toggles for Megatron engine
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-False}
GRAD_OFFLOAD=${GRAD_OFFLOAD:-False}

# Rollout quantization (set to fp8 to save VRAM if supported)
ROLLOUT_QUANT=${ROLLOUT_QUANT:-null}

rollout_name="vllm"
project_name='fim_rlvr_lean4'
exp_name='qwen2_5_3b_fim_megatron_lora'
adv_estimator=grpo

# FIM-RLVR-LEAN4 data paths (preprocessed parquet with prompt column)
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
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.lora.rank=32
    actor_rollout_ref.model.lora.alpha=64
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28

    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIM_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE}
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${VANILLA_MBRIDGE}
)

ROLLOUT=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.quantization=${ROLLOUT_QUANT}
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.max_model_len=4096
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${PARAM_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_mbridge=${USE_MBRIDGE}
    actor_rollout_ref.ref.megatron.vanilla_mbridge=${VANILLA_MBRIDGE}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
)

REWARD=(
    reward_model.reward_manager=lean_verifier
    +reward_model.reward_kwargs.lean_env_path=${LEAN_ENV_PATH:-/workspace/verl/verification_env}
    +reward_model.reward_kwargs.verification_timeout=30
    +reward_model.reward_kwargs.parallel_workers=8
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
    trainer.total_epochs=3
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
