# GRPO: `num_generations` and "relative advantage" (TRL)

This repo uses TRL's `GRPOTrainer` in several training entrypoints.

## What `num_generations` controls
`num_generations` is the number of completions sampled **per prompt** at each sampling step.
In this repo it's exposed as `FIM_NUM_GENERATIONS` and passed into TRL as `num_generations`.

## How TRL computes advantages (why `num_generations=1` is special)
In TRL, rewards are normalized *within each prompt's group* of size `num_generations`.
Concretely (simplified), TRL computes:

- `mean_grouped_rewards = mean(rewards over the group for a given prompt)`
- `advantages = rewards - mean_grouped_rewards`
- optionally scales by a reward std (group-wise or batch-wise), but the mean-subtraction is still group-wise

Implication:
- If `num_generations=1`, then `advantages = reward - mean([reward]) = 0` for every sample.
  This typically makes the policy-gradient term ~0, so learning can stall (you may see
  `reward_std` and/or `advantages` collapse to 0 in logs).

So, `FIM_NUM_GENERATIONS=1` is useful for:
- smoke tests (checking that generation, verification, logging, and the training loop run)
- debugging OOMs or correctness issues with the pipeline

But it is usually **not** suitable for:
- actual GRPO training where you want a per-prompt "relative" advantage signal

## How many samples contribute to one optimizer step
There are two "counts" that matter:

1) **Group size for relative advantage** (per prompt): `num_generations`.
2) **Total sampled completions used by an optimizer update** (across devices + grad accumulation):
   `per_device_train_batch_size * num_generations * gradient_accumulation_steps * world_size`.

Note that increasing `gradient_accumulation_steps` increases the number of total samples in an update,
but it does **not** change the fact that TRL's advantage baseline is computed per prompt-group of size
`num_generations`.

## Practical guidance for OOM tuning (without shortening context length)
If you must keep long context/rollouts but hit OOM, the usual approach is:
- keep `num_generations >= 2` for a non-degenerate advantage signal
- reduce peak memory via batching controls (e.g., vLLM `max_num_batched_tokens`) and allocator tweaks
  (e.g., `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)

