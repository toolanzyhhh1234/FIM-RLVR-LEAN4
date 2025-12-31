# 80/20 Guide: DL Frameworks for Algorithm Researchers

Quick-reference guide for grasping deep learning infrastructure without becoming an infrastructure engineer.

## The 3 Knobs That Matter Most

In `train_gspo_fim_qwen3-vl-8b.py`, these are the critical memory/speed tradeoffs:

```python
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # RL memory savings (30%+)
LOAD_IN_4BIT = True                        # quantization
FAST_INFERENCE = False                     # vLLM backend toggle
```

### Mental Model

For RL training, you have 3 memory pools competing:

1. **Model weights** (4-bit vs 16-bit)
2. **Generation cache** (vLLM vs native)
3. **Gradients/optimizer states**

`VLLM_STANDBY` swaps the vLLM KV cache to CPU when doing gradient updates — that's why it saves 30%+ for RL specifically.

## Framework Selection Heuristic

| If you need... | Use... | Why |
|---|---|---|
| Just inference | vLLM directly | Fastest, simplest |
| Fine-tuning only | Unsloth/PEFT | Memory efficient |
| RL (GRPO/PPO/DPO) | Unsloth + TRL | Handles the generation<->training dance |

---

## Diagnostic Chain: "Latent Space Features" for Codebases

The key observability chain for RL training:

```
prompt -> raw_completion -> extracted_code -> verifier_input -> reward
```

### What to Log

Current logging in `lean_validity_reward_factory()`:

```python
raw_logs.append({
    "theorem_id": ...,
    "task_type": ...,
    "raw_completion": generated_text,  # What model actually outputs
    "extracted_code": ...,             # After tag extraction
    "verifier_input": ...,             # What Lean sees
})
```

### Extensions to Consider

1. **Add reward context** — Log `curriculum.get_mask_ratio(th_id)` alongside success/failure to see if harder masks correlate with failures.

2. **Log the thinking block** — The `[THINK]...[/THINK]` content is often the most diagnostic signal for why the model succeeded or failed.

3. **Hash-based deduplication** — Add a `completion_hash` to detect mode collapse (same outputs for different prompts).

### Quick Debug One-Liner

Add after the verification loop (around line 604):

```python
if idx == 0:  # Log first sample each batch
    print(f"[DBG] th={th_id} mask={curriculum.get_mask_ratio(th_id):.2f} "
          f"tag_ok={tag_ok_list[idx]} lean_ok={success}")
```

Gives instant feedback on the prompt->completion->verification chain.

---

## 80/20 Reading Strategy for Any DL Codebase

1. **Find the main training loop** — `trainer.train()` is your anchor
2. **Trace the reward function signature** — `completions, **kwargs` tells you what data flows through
3. **Check what `kwargs` actually contains** — Discover hidden data like `completion_ids` by logging `kwargs.keys()`
4. **Ignore everything else until you hit a bug**

---

## Config Comparison: 8B Scripts

| Config | `train_gspo_fim_qwen3-vl-8b.py` | `train_gspo_fim_qwen3-vl-8b_4bit_no_vllm.py` |
|--------|----------------------------------|---------------------------------------------|
| `LOAD_IN_4BIT` | `False` (default) | `True` (default) |
| `FAST_INFERENCE` | `True` (default) | `False` (default) |
| Use case | 16-bit LoRA + vLLM | 4-bit QLoRA, no vLLM |
| VRAM requirement | Higher (~48GB+) | Lower (~32GB) |

The `_4bit_no_vllm` variant is for when vLLM + 4-bit quantization conflicts (common in containers without proper CUDA runtime).

---

## Key Files to Understand

| File | Purpose | Read Priority |
|------|---------|---------------|
| `train_gspo_fim_*.py` | Training entry point | 1st |
| `fim_rlvr_lean4/lean_verifier.py` | Reward computation | 2nd |
| `fim_rlvr_lean4/curriculum.py` | Difficulty scheduling | 3rd |
| `fim_rlvr_lean4/masking.py` | FIM mask generation | 4th |

Everything else is supporting infrastructure.
