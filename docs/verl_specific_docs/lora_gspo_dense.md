# LoRA + GSPO on Dense Models in Verl

This note documents how Verl supports **LoRA** with the **GSPO** algorithm on dense (non‑MoE) models, and provides a high-level flowchart of the training loop.

## Evidence in repo
- GSPO loss is implemented as `compute_policy_loss_gspo` in `verl.trainer.ppo.core_algos` (registered under `"gspo"`). The loss computes sequence-level importance ratios and clipping suitable for GSPO.@verl/verl/trainer/ppo/core_algos.py#1015-1089
- LoRA is validated and accepted in configs; Verl checks `lora_rank` when using vLLM rollouts to ensure supported ranks.@verl/verl/utils/config.py#201-203
- Advanced docs describe LoRA enablement for FSDP and Megatron backends, including dense-model examples (e.g., `run_qwen2-7b_math_megatron_lora.sh`) and rank/target-module settings.@verl/docs/advance/ppo_lora.rst#1-202

**Conclusion:** Verl supports GSPO training while using LoRA on dense models via its PPO/GRPO stack and configuration checks. Users can pair the GSPO loss with LoRA-enabled actors on dense architectures using the documented configs and examples.

## Mermaid flowchart — LoRA + GSPO (dense)
```mermaid
flowchart TD
    A[Start: choose dense base model] --> B[Load base weights in training backend (FSDP or Megatron)]
    B --> C[Attach LoRA adapters (rank, alpha, target_modules)]
    C --> D[Launch rollout workers (e.g., vLLM) with LoRA-enabled actor]
    D --> E[Generate responses for prompts]
    E --> F[Compute rewards (task-specific; may include verifier or RM)]
    F --> G[Run GSPO loss (sequence IS + clipping) on actor grads]
    G --> H[Update LoRA adapter weights]
    H --> I[Sync updated actor to rollout workers (merge or adapter transfer)]
    I --> J{More steps?}
    J -->|Yes| D
    J -->|No| K[Save adapters & checkpoints]
    K --> L[End]
```

## Practical notes
- Keep `lora_rank <= 512` when using vLLM rollout workers (enforced in config validation).@verl/verl/utils/config.py#201-203
- For Megatron dense runs, follow the LoRA YAML section in `advance/ppo_lora.rst` and the dense example script `run_qwen2-7b_math_megatron_lora.sh` as a template.
- Pair GSPO by selecting the `"gspo"` loss in configs or using recipes that register it; the loss integrates with the standard PPO/GRPO trainer stack.
