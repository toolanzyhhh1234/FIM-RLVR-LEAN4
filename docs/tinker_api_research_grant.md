# Tinker API Research Grant Proposal

## Overview

This document outlines the rationale for using [Tinker API](https://tinker-docs.thinkingmachines.ai/) (by Thinking Machines Lab) for the Lean4 FIM + RLVR project, and serves as preparation for the **$5,000 Research Grant** application.

---

## Why Tinker API?

### Problem: GPU Resource Constraints

Our current training infrastructure faces significant cost and resource challenges:

| Model Size | Min GPU Requirement | Estimated Monthly Cost (Cloud) |
|------------|---------------------|-------------------------------|
| 8B dense   | 1× H200 (80GB)      | ~$2,500-4,000/mo              |
| 30B MoE    | 2-4× H200           | ~$8,000-15,000/mo             |
| 120B MoE   | 8+ H200 cluster     | ~$30,000+/mo                  |

**Key insight**: `gpt-oss-120b` (ultra-sparse MoE) achieved **60% accuracy** on our Lean4 FIM task out-of-box (5-sample test), while smaller models (<14B) struggle to even follow the prompting format correctly.

### Solution: Tinker's Token-Based Pricing

Tinker abstracts away infrastructure management with usage-based pricing:

- **Fine-tune 30B model**: ~$0.62 for ~3M tokens ([source](https://recapio.com/digest/fine-tune-a-30b-model-for-0-62-prompt-distillation-with-tinker-by-llm-implementation))
- **Supports large MoE models**: Qwen3-235B-A22B, Llama 70B via LoRA
- **No infrastructure overhead**: Distributed training handled automatically

### Tinker Features Relevant to Our Project

1. **Low-level API primitives**: `forward_backward()`, `optim_step()`, `sample()` — compatible with custom GRPO training loops
2. **LoRA fine-tuning**: Efficient for our iterative RLVR experiments
3. **Large model support**: Can train 120B+ MoE models that show promise on our task
4. **Tinker Cookbook**: Open-source implementations of post-training methods including RL algorithms

---

## Technical Integration Plan

### Supported Models (Relevant to Our Task)

| Model | Type | Notes |
|-------|------|-------|
| `openai/gpt-oss-120b` | MoE | Best performer on our task (60% accuracy) |
| `Qwen/Qwen3-235B-A22B` | MoE | Large capacity, cost-effective |
| `meta-llama/Llama-3.1-70B` | Dense | Strong baseline |
| `deepseek-ai/DeepSeek-V3.1` | MoE | Reasoning-focused |

MoE models are more cost-effective since pricing is proportional to **active parameters**, not total parameters.

### Tinker API Primitives

```python
import tinker

# Core primitives for our RL loop:
# 1. forward_backward() - compute gradients with custom loss
# 2. optim_step() - update LoRA weights
# 3. sample() - generate candidate completions

# Supported loss functions:
# - "importance_sampling" 
# - "ppo" 
# - "cispo" (Clipped Importance Sampling Policy Optimization) ← RECOMMENDED FOR MoE
# - "dro" (Direct Reward Optimization)
```

### Algorithm Choice: CISPO over GRPO for MoE Stability

**Problem with GRPO + MoE**: GRPO's token-level clipping discards "critical but low-probability" tokens (e.g., reasoning pivots like "Wait" or "Aha"), causing expert routing collapse in MoE architectures.

**Solution: CISPO** (Clipped Importance Sampling Policy Optimization)
- Introduced in [MiniMax-M1 Technical Report (arXiv:2506.13585)](https://arxiv.org/abs/2506.13585)
- Clips **importance sampling weights** rather than policy updates
- Proven stable for large hybrid MoE models (MiniMax-M1)
- **Tinker natively supports CISPO** via `loss_fn="cispo"`

**Fallback plan**:
1. Primary: Use Tinker's built-in CISPO
2. Secondary: Check tinker-cookbook for GSPO implementation
3. Tertiary: Implement custom GSPO if needed

### Custom Environment for Lean4 Verification

Tinker's `Env` interface maps directly to our RLVR setup:

```python
from tinker_cookbook.rl.types import Env, StepResult, Observation, StopCondition

class Lean4FIMEnv(Env):
    """Environment for Lean4 proof infilling with verification rewards."""
    
    def __init__(self, prefix: str, suffix: str, ground_truth_middle: str):
        self.prefix = prefix
        self.suffix = suffix
        self.ground_truth = ground_truth_middle
        
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Return FIM prompt: prefix + <HOLE> + suffix
        prompt = self._format_fim_prompt()
        return Observation(prompt), StopCondition(max_tokens=512)
    
    async def step(self, action: list[int]) -> StepResult:
        # Decode model output
        middle_completion = self._decode(action)
        
        # Reconstruct full proof
        full_proof = self.prefix + middle_completion + self.suffix
        
        # Verify with Lean4 (our existing lean_verifier.py)
        verified = await self._verify_with_lean(full_proof)
        
        # Binary reward: 1 if verified, 0 otherwise
        reward = 1.0 if verified else 0.0
        
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=None,
            next_stop_condition=None
        )
```

### Training Loop Integration

Our existing GRPO curriculum logic can integrate with Tinker's RL training system:

```python
# Tinker supports three execution modes:
# 1. Synchronous on-policy (strict on-policy RL)
# 2. Streaming minibatch (overlapped sampling + training)
# 3. Asynchronous off-policy (max throughput)

# Our curriculum sampling integrates via EnvGroupBuilder:
class CurriculumEnvGroupBuilder:
    def __init__(self, theorem_id: str, current_ratio: float):
        self.theorem_id = theorem_id
        self.ratio = current_ratio
    
    def make_envs(self) -> list[Lean4FIMEnv]:
        # Sample G environments for group sampling
        return [self._make_env_at_ratio(self.ratio) for _ in range(G)]
    
    def compute_group_rewards(self, trajectories) -> list[float]:
        # Return raw rewards; CISPO handles clipping internally
        return [t.total_reward for t in trajectories]
```

---

## Project Summary (For Grant Application)

### Title
**Reinforcement Learning from Verifiable Rewards for Lean4 Proof Infilling**

### Research Objective
Train a language model to infill missing proof segments in Lean4 tactic proofs using RLVR (Reinforcement Learning with Verification Rewards), where the Lean4 compiler serves as a ground-truth verifier.

### Technical Approach

1. **Task**: Fill-in-the-Middle (FIM) for Lean4 proofs
   - Input: `prefix + <HOLE> + suffix` of a tactic proof
   - Output: The missing `middle` segment
   - Reward: Binary (1 if Lean verifies reconstructed proof, 0 otherwise)

2. **Training Algorithm**: CISPO (Clipped Importance Sampling Policy Optimization)
   - Sample G candidate completions per hole
   - Verify each with Lean4 compiler
   - Update policy with importance-sampling clipped objective
   - **Why CISPO over GRPO**: GRPO is unstable with MoE models due to token-level clipping discarding critical low-probability tokens, causing expert routing collapse. CISPO clips importance weights instead, proven stable for MoE (MiniMax-M1).
   - Reference: [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)

3. **Curriculum Learning**: Mastery-based hole-size progression
   - Start with small holes (10% of proof)
   - Promote to larger holes upon mastery
   - Prevents catastrophic forgetting via mixed sampling

### Why Large Models Matter

| Model | FIM Format Compliance | Proof Accuracy (5-sample) |
|-------|----------------------|---------------------------|
| <14B models | Poor (format errors) | <10% |
| gpt-oss-120b | Excellent | ~60% |

Smaller models fail to follow the structured FIM format required for Lean4 proofs. The 120B ultra-sparse MoE shows the task is learnable with sufficient model capacity.

### Evaluation
- **Benchmarks**: MiniF2F, ProofNet, LeanDojo Benchmark
- **Metrics**: Pass@N for verified proof reconstructions
- **Ablations**: Curriculum vs fixed difficulty, CISPO vs other RL algorithms

### Key References
- **CISPO / MiniMax-M1**: [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) - MoE-stable RL algorithm

---

## Budget Justification (~$5,000)

| Item | Estimated Cost | Notes |
|------|----------------|-------|
| Initial experiments (30B model) | $500 | Hyperparameter search, ~800M tokens |
| Main RLVR training (120B MoE) | $2,500 | ~4B tokens across curriculum stages |
| Ablation studies | $1,000 | Curriculum vs baseline comparisons |
| Evaluation runs | $500 | MiniF2F/ProofNet benchmarking |
| Buffer for iteration | $500 | Unexpected reruns |
| **Total** | **$5,000** | |

---

## Expected Outcomes

1. **Trained model checkpoint** capable of Lean4 proof infilling
2. **Open-source training code** adapted for Tinker API (will extend existing `fim_rlvr_lean4/` codebase)
3. **Research findings** on curriculum learning for formal verification tasks
4. **Benchmark results** on MiniF2F/ProofNet with Pass@N metrics

---

## Team & Resources

- **Existing infrastructure**: 
  - Lean4 verification pipeline (`verification_env/`)
  - FIM data pipeline (`data_pipeline/`)
  - GRPO training scripts (Unsloth + TRL based)
  
- **Codebase**: This repository contains working implementations for local training; Tinker integration would replace the GPU-bound components while preserving the Lean verification and curriculum logic.

---

## Links

- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/
- **Tinker Cookbook** (GitHub): https://github.com/thinking-machines-lab/tinker-cookbook
- **Research Grant Application**: https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/

### Relevant Cookbook Examples

- `tinker_cookbook/recipes/rl_loop.py` - Basic RL training loop
- `tinker_cookbook/recipes/math_rl.py` - Mathematical reasoning with verifiable rewards
- `tinker_cookbook/rl/types.py` - `Env` and `EnvGroupBuilder` interfaces
- `tinker_cookbook/rl/envs/math_env.py` - Example of programmatic verification

---

## Next Steps

1. [ ] Submit research grant application (~$5,000)
2. [ ] Clone tinker-cookbook and study `math_env.py` as reference
3. [ ] Implement `Lean4FIMEnv` extending Tinker's `Env` interface
4. [ ] Adapt `CurriculumEnvGroupBuilder` for mastery-based progression
5. [ ] Validate 120B model (`gpt-oss-120b`) performance on larger sample set

---

## Appendix: Architecture Mapping

| Our Current Component | Tinker Equivalent |
|-----------------------|-------------------|
| `train_grpo_fim_local.py` | `tinker_cookbook/recipes/rl_loop.py` |
| `fim_rlvr_lean4/lean_verifier.py` | Custom `Env.step()` reward logic |
| `fim_rlvr_lean4/curriculum.py` | `EnvGroupBuilder` + `RLDataset` |
| Unsloth LoRA | Tinker's built-in LoRA (`create_lora_training_client`) |
| TRL GRPO trainer | Tinker's `cispo` loss (MoE-stable) |

---

*Document created: 2026-01-06*
*Last updated: 2026-01-06*
