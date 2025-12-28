# BPE Tokenizer Decoding Debug Status

## Issue Summary
Model completions contain byte-level BPE artifacts (`Ġ` for space, `Ċ` for newline) that corrupt Lean code verification.

## What We Know

### From `reward_debug.log`:
- TRL passes `kwargs.keys(): ['prompts', 'completion_ids', 'trainer_state']`
- `completion_ids` is available as a list
- Raw `completions` strings contain BPE artifacts

### Attempted Fixes:
1. **TRL 0.22.2 downgrade** - Did NOT work, same issue
2. **Decode from `completion_ids` using `batch_decode`** - Did NOT work, still has artifacts

## Current Hypothesis
The tokenizer's `batch_decode()` method itself is returning BPE-encoded strings instead of proper UTF-8.

Possible causes:
1. Tekken tokenizer (used by Mistral 3) handles decoding differently
2. There's another wrapper layer we're not unwrapping
3. The tokenizer lacks proper `byte_decoder` mapping

## Debug Logging Added
`training_logs/tokenizer_debug.log` will capture:
- Tokenizer type hierarchy (PixtralProcessor -> LlamaTokenizer -> ?)
- Whether `byte_decoder` attribute exists
- Individual token ID -> string mappings
- `convert_ids_to_tokens` output
- `convert_tokens_to_string` output
- `batch_decode` output comparison

## Next Steps After Debug Log
1. If `convert_tokens_to_string` works: use it instead of `batch_decode`
2. If tokenizer lacks `byte_decoder`: manually apply BPE byte mapping fix
3. If deeper wrapper exists: unwrap further

## Files Modified
- `train_gspo_fim_mistral3.py` - Added `_decode_completions` with deep debug logging

## Related Files
- `training_logs/reward_debug.log` - Initial kwargs inspection
- `training_logs/tokenizer_debug.log` - Deep tokenizer analysis (generated on next run)
- `fix_detokenization_corruption_plan.md` - GPT-5.2's original fix plan
