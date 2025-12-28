## Unsloth GRPO ref_hidden_states error (note)

Observed error:
```
TypeError: grpo_accumulated_loss() missing 1 required positional argument: 'ref_hidden_states'
```

Environments where it shows up:
- Qwen3-VL env (`.venv310`): transformers==4.57.0, trl==0.22.2, unsloth==2025.9.5
- Mistral-3 env (`.venv310_mistral3`): transformers==5.0.0.dev0, trl==0.22.2, unsloth==2025.9.5

Behavior:
- Deleting `unsloth_compiled_cache` regenerates `UnslothGRPOTrainer.py` with the same missing arg.
- Manual cache patch (`ref_hidden_states=None`) is overwritten on regeneration.
- Vanilla TRL GRPO (non-DR loss) still hits the same `ref_hidden_states` issue.

Working hypothesis:
- Likely an upstream Unsloth/TRL compatibility bug.
- Not stable to patch generated cache.

New suspicion to keep in mind:
- Using `unsloth==2025.9.5` may be the problematic version (especially for Mistral-3).
- Note: Qwen3-VL example notebook also uses that Unsloth version, so version alone may not explain Qwen3-VL; still worth tracking.

Reference (main branch):
- This matches the previous issue recorded in `docs/debug_logs/unsloth_ministral3_error.md`.
- We did not encounter this error in the new run.
