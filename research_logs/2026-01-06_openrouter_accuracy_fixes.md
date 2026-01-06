# OpenRouter Accuracy Pipeline Fixes

**Date**: 2026-01-06

## Summary

Fixed multiple issues in the OpenRouter synthetic data generation and accuracy testing pipeline that were causing 0% Lean verification pass rate. After fixes, achieved 60% pass rate (3/5 samples).

## Issues Identified and Fixed

### 1. Missing `middle_truth` in Dryrun Data

**Problem**: The `test_openrouter_accuracy.py` script was reading `middle_truth` from the dryrun JSONL file, but the field was never populated, resulting in empty ground truth comparisons.

**Fix**: Added logic to compute `middle_truth` from `formal_ground_truth` by extracting the portion between `fim_prefix` and `fim_suffix`:

```python
# In test_openrouter_accuracy.py dryrun processing
formal = record_info.get("formal_ground_truth", "")
if formal and prefix and formal.startswith(prefix):
    remainder = formal[len(prefix):]
    if suffix and remainder.endswith(suffix):
        middle_truth = remainder[:-len(suffix)]
    # ... fallback logic
```

### 2. Incorrect `formal_ground_truth` Storage

**Problem**: In `generate_openrouter_synthetic.py`, the `formal_ground_truth` field was being set to `full_generated` (model's reconstruction) instead of the original proof from the dataset.

**Fix**: Changed to store the original `full_code`:

```python
# Before (wrong)
"formal_ground_truth": full_generated,

# After (correct)
"formal_ground_truth": full_code,
```

### 3. Model Over-Generating (Main Issue)

**Problem**: The model was generating code that overlapped with the suffix already present in the prompt. For example, when only a declaration header was masked, the model would output the complete proof body, duplicating code that was already in the suffix.

**Root Cause**: The system prompt only showed FIM examples without suffix context. The model didn't understand that `[MISSING_BLOCK]` marks a gap with code following it.

**Fix**: Updated `_build_system_prompt()` in both files to:
1. Add explicit rule: "Do NOT repeat code that appears after [MISSING_BLOCK]"
2. Add example showing FIM with suffix context

```python
"4) For fill-in-the-middle: [MISSING_BLOCK] marks a gap. Output ONLY the missing code. "
"Do NOT repeat code that appears after [MISSING_BLOCK] in the prompt.\n"
...
"Example (fill-in-the-middle with suffix):\n\n"
"[USER]\n"
"theorem example (n : ℕ) : n + 0 = n := by\n"
"  [MISSING_BLOCK]\n"
"  rfl\n\n"
"[ASSISTANT]\n"
"<FIM_CODE>\n"
"  simp only [Nat.add_zero]\n"
"</FIM_CODE>\n\n"
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Lean Pass Rate | 0/5 (0%) | 3/5 (60%) |
| Avg Similarity | 0.32 | 0.55 |
| Exact Match | 0/5 | 0/5 |

## Remaining Issues

- Sample 4 had empty `middle_truth` (edge case in masking)
- Model still diverges on some complex proofs

## FIM Token Format Experiment

Tested standard FIM tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) as an alternative prompt format.

### Initial Issue: Empty Responses

First run with FIM tokens returned empty `content` fields. Investigation revealed:
- Model returns `reasoning` trace showing it understands the task
- But `content` was empty with `finish_reason: "length"`
- The model's internal reasoning consumed all tokens (default 1024) before generating output

### Fix: Increased Token Limit

Increased `max_tokens` from 1024 to 4096 to accommodate reasoning overhead.

### Results Comparison

| Format | Lean Pass | Avg Similarity | Notes |
|--------|-----------|----------------|-------|
| Custom `[MISSING_BLOCK]` | 3/5 (60%) | 0.55 | Clear instructions, suffix-aware example |
| FIM tokens `<\|fim_*\|>` | 0/5 (0%) | 0.12 | Inconsistent, some empty outputs |

### Conclusion

The custom prompt format with `[MISSING_BLOCK]` significantly outperforms standard FIM tokens for `gpt-oss-120b:free`. This model wasn't trained on FIM special tokens and struggles to interpret them consistently. The explicit instructions and examples in the custom format provide better guidance.

**Recommendation**: Use the custom `[MISSING_BLOCK]` format for OpenRouter models unless the specific model is known to support FIM tokens natively.

## Files Changed

- `data_pipeline/test_openrouter_accuracy.py`: Fixed middle_truth extraction, updated system prompt
- `data_pipeline/generate_openrouter_synthetic.py`: Fixed formal_ground_truth storage, updated system prompt
- `data_pipeline/test_openrouter_accuracy_fim_tokens.py`: New script for FIM token format testing
