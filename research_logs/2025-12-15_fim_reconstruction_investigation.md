# FIM Reconstruction Investigation - 2025-12-15

## Problem Identified
FIM reconstruction failing due to improper line-level splitting that breaks Lean syntax.

## Update (2025-12-15): Additional Root Cause Found
Even when the split point is “reasonable”, reconstruction was often invalid due to *string concatenation* issues:

- **Glued-line bug**: `header + prefix_body` (and `prefix + middle + suffix`) were frequently concatenated without a separating newline, producing invalid Lean like `:= by  have ...` on one line.
- **Fixed generator still ends holes unsafely**: `fim_generator_fixed.py` selected a “safe” start split point but computed the hole end via `end_line = start_line + k`, which can still cut mid-block even if the start was safe.

### Code Changes Made
- `data_pipeline/fim_generator.py`: preserve boundary newlines between prefix/middle/suffix and ensure `header` is followed by a newline.
- `data_pipeline/fim_generator_streaming.py`: same newline-preserving fix.
- `data_pipeline/fim_generator_fixed.py`: choose both start/end from safe split points and preserve boundary newlines.
- `data_pipeline/test_fim_reconstruction.py`: add a quick “boundary glue” regression check.

## Investigation Results

### Original FIM Generator (fim_generator.py)
- **Success Rate**: 20% (1/5 samples)
- **Issue**: Line-based splitting without syntax awareness
- **Problem**: Breaks tactic blocks, creates incomplete statements

### Fixed FIM Generator (fim_generator_fixed.py)  
- **Approach**: Added safe split point detection
- **Rules**: Avoid splitting after `by`, `:=`, `have`, `let`
- **Success Rate**: 0% (0/20 samples)
- **Issue**: Too restrictive, still breaks syntax

### Simple FIM Generator (fim_simple_fix.py)
- **Approach**: Clean line boundaries, skip first/last lines
- **Success Rate**: 20% (2/10 samples) 
- **Generated**: 1000 samples successfully
- **Conclusion**: Line-level approach works for subset of cases

## Key Findings

1. **Line-level FIM is viable** - 20-30% success rate achievable
2. **Perfect syntax preservation is hard** - Lean has complex indentation rules
3. **Filtering approach is practical** - Train only on successful reconstructions
4. **Completion model value** - Even partial success creates useful model for open-source

## Technical Analysis

### Working Cases (20%)
- Simple tactic sequences
- Clean line boundaries  
- Proper indentation preserved

### Failing Cases (80%)
- Mid-tactic splits
- Indentation mismatches
- Incomplete statements
- Complex proof structures

## Recommendations

### Option 1: Filter & Train (Pragmatic)
- Use 20% successful samples for SFT training
- Creates working completion model
- Can iterate and improve later
- Faster path to results

### Option 2: Improve FIM Generator (Perfectionist)  
- Implement AST-based splitting
- Higher success rate but more complex
- Delays training phase
- May over-engineer the solution

### Future Direction (Statement-Only Solving)
If the long-term goal is a model that can solve Lean problems from a theorem statement (traditional theorem proving), we likely want a dataset format closer to:

- condition on *imports + theorem statement* (or even mask/standardize imports)
- predict the entire proof (Option A: proof-only), rather than doing generic file-level FIM

In that setting, Option A remains relevant, but we should also revisit how we treat/import-mask the header so the model is trained in the “given a Lean statement, produce a proof” regime.

## Data Generated
- `fim_train_fixed.jsonl`: 3000 samples (0% success rate)
- `fim_simple.jsonl`: 1000 samples (20% success rate)
- `mvp_train.jsonl`: Original data (20% success rate)

## Next Steps Decision Point
Choose between pragmatic filtering approach vs. perfect data generation.

## Update (2025-12-15 v2): Exact Byte-by-Byte Solution
We have successfully implemented an "exact line splitting" strategy (`splitlines(keepends=True)`) that solves the FIM reconstruction issues entirely for proof-body extraction.

### Results
- **Success Rate**: 100% (50/50 samples verified)
- **Method**: 
  - Locate `:= by` to separate header/proof.
  - Split proof body using `splitlines(keepends=True)` to preserve all whitespace and newlines.
  - Randomly select a contiguous block of lines for the middle "hole".
  - Reassemble as `header + prefix_body + middle + suffix_body`.
- **Outcome**: 
  - `exact_match_rate`: 1.0
  - `glued_boundary`: 0
  - No newlines lost, no whitespace limits needed.
  

## Future Data Strategy (Reminder)
We have decided to mix different data formats for the upcoming SFT phase:
1.  **Instruction-FIM**: 
    - **Prompt**: Context (Prefix + Suffix) + `[MISSING_BLOCK: N lines]`
    - **Gold Label**: The exact `middle` block extracted from the original valid proof. (Model predicts *only* the missing lines).
2.  **Standard SFT**: Full generation (Prompt: Theorem Statement, Answer: Full Proof).


**Note**: The "Standard SFT" data can also be easily generated using our new **exact byte-by-byte reconstruction** pipeline. We simply treat the entire proof body as the "middle" block, ensuring perfect syntax preservation for full-proof samples just as we do for partial ones.

## Thought-Augmented Data Strategy (Rationalization)
To preserve the reasoning capabilities of our base model (e.g., `gpt-oss-120b`), we **must** include `<think>` traces in the training data. Training on raw code without thought traces risks "lobotomizing" the reasoning model.

**Plan:**
1.  **Generate Raw Pairs**: Use `fim_generator.py` to create `(Prompt, Gold_Code)` samples.
2.  **Generate Thoughts (Rationalization)**:
    -   **API**: Use OpenRouter's free endpoint `openai/gpt-oss-120b:free`.
    -   **Why**: This is effectively the same model as our target base model. Using it allows us to mimic "on-policy" data generation (the model teaching itself) without the cost/complexity of running local inference for the entire dataset.
    -   **Input**: The FIM prompt context + the *known correct* Gold Code.
    -   **Task**: "Explain the reasoning behind this solution in a `<think>` block."
3.  **Final Training Target Assembly**:
    -   **Action**: Extract *only* the `<think>...</think>` block from the API response.
    -   **Assembly**: Concatenate `<think>...extracted thoughts...</think>` + `\n` + `Gold_Code`.
    -   **Robustness Note**: We do *not* rely on the API to output the code correctly. The API model might be chatty or make small syntax errors. By discarding its code output and re-attaching our guaranteed-correct Gold Code, we ensure the training data is 100% syntax-valid while capturing the high-level reasoning.

## Update (2025-12-15 v3): FIM-RLVR Pipeline Verification (Stage 0 Complete)

We have successfully verified the data-to-training pipeline end-to-end on a local setup.

### Achievements:
1.  **FIM Reconstruction**: Used `data_pipeline/fim_exact_line.py` to generate 1000 proof-only FIM samples with exact line preservation.
2.  **Environment Setup**: Installed `Unsloth` + `TRL` + `PEFT` on local machine.
3.  **Training Script**: Created `train_grpo_fim.py` which:
    - Loads `unsloth/Qwen2.5-0.5B-Instruct` (as a lightweight proxy for `gpt-oss-20b`).
    - Preprocesses `<PFX>/<SFX>` raw formatting into standard Chat Templates.
    - Implements **RLVR** (Reinforcement Learning with Verification Rewards) via `LeanVerifier`.
4.  **Verification**: 
    - The loop ran successfully for >1 steps.
    - `rewards/format_reward` was high (pipeline works).
    - `rewards/lean_validity_reward` was computed (verifier rejected invalid code from untrained model as expected).

### Next Steps:
We are moving to Cloud GPU for the actual training (Stage 1).
- **Model**: `unsloth/gpt-oss-20b-bn` (or similar base model).
- **Data**: Scale to ~10k samples (Small to Medium holes).
- **Infrastructure**: The scripts `train_grpo_fim.py` and `data_pipeline/generate_dataset.py` are production-ready.
