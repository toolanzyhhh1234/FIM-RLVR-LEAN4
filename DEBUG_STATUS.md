# BPE Tokenizer Decoding Debug Status

## RESOLVED

The byte-level BPE decoding issue has been fixed.

## Issue Summary
Model completions contained byte-level BPE artifacts (`Ġ` for space, `Ċ` for newline) that corrupted Lean code verification.

## Root Cause
The `LlamaTokenizer` (used by Mistral 3 via PixtralProcessor) lacks a `byte_decoder` attribute. Its `decode()` and `batch_decode()` methods return raw BPE byte representations instead of converting them back to actual UTF-8 characters.

From `tokenizer_debug.log`:
```
Has byte_decoder: False
batch_decode result: 'TheĠtaskĠisĠtoĠprove...'  # Ġ should be space
batch_decode has artifacts: True
```

## Solution Implemented
Added manual GPT-2 style BPE byte decoding in `train_gspo_fim_mistral3.py`:

1. **`_build_byte_decoder()`** - Creates inverse mapping from GPT-2's `bytes_to_unicode()`:
   - `Ġ` (U+0120) → byte 0x20 (space)
   - `Ċ` (U+010A) → byte 0x0A (newline)
   - All 256 byte mappings

2. **`_fix_bpe_artifacts(text)`** - Converts BPE bytes back to UTF-8:
   - Maps each char through `_byte_decoder`
   - Collects bytes and decodes as UTF-8
   - Handles multi-byte UTF-8 chars (e.g., `ℕ`)

3. **Applied after `batch_decode`**: `texts = [_fix_bpe_artifacts(t) for t in texts]`

## Verification
From `tokenizer_debug.log` after fix:
```
After _fix_bpe_artifacts[0][:200]: 'The task is to prove that addition of natural numbers is commutative...'
After fix has artifacts: False
```

## Attempted Fixes That Did NOT Work
1. **TRL 0.22.2 downgrade** - Same issue
2. **Using `completion_ids` with `batch_decode`** - Still had artifacts (tokenizer bug)

## Files Modified
- `train_gspo_fim_mistral3.py` - Added `_build_byte_decoder()`, `_fix_bpe_artifacts()`, and applied fix in `_decode_completions()`

## Related Files
- `training_logs/tokenizer_debug.log` - Shows before/after comparison
- `training_logs/raw_completions.log` - Clean completions with proper spaces/newlines
