#!/usr/bin/env bash
set -euo pipefail
set -x

# NOTE: For Ray/vLLM performance inside Docker, run with larger shared memory.
# Example: docker run --shm-size=10.24gb ...  (recommended >=30% of host RAM)

mkdir -p verl_logs
log_ts=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "verl_logs/run_test_gspo_3b_math_${log_ts}.log") 2>&1

export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export NNODES=${NNODES:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"12.0+PTX"}
export VLLM_USE_V1=1
export WANDB_MODE=${WANDB_MODE:-disabled}

# ------------------------------------- Setup params ---------------------------------------
project_name='RL-GSPO'

adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
offload=false
rollout_engine=vllm
rollout_mode=async
return_raw_chat="True"
gpu_memory_utilization=0.6
reward_manager=dapo
shuffle_dataset=true

# smaller defaults for single GPU
train_batch_size=64
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=2
n_resp_per_prompt=4

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
# dapo reward manager params
enable_overlong_buffer=false
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

test_freq=10
save_freq=10
total_epochs=1
total_training_steps=100
val_before_train=false

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003
clip_ratio_high=0.0004

# Paths and namings
SFT_MODEL=$(basename $MODEL_PATH)
exp_name="${loss_mode}-epslow-${clip_ratio_low}-epshigh-${clip_ratio_high}-${SFT_MODEL}-RL"
CKPTS_DIR="$PWD/checkpoints/${loss_mode}/${exp_name}"

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

gen_tp=1
entropy_checkpointing=true

# ------------------------------------- train/val data preparation ---------------------------------------
DATA_DIR="$PWD/data/gsm8k"
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    python verl/examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi

gsm8k_train_path="$DATA_DIR/train.parquet"
gsm8k_test_path="$DATA_DIR/test.parquet"

# set the paths
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    +actor_rollout_ref.ref.override_config.attn_implementation=sdpa \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.return_raw_chat=${return_raw_chat} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    "$@"
