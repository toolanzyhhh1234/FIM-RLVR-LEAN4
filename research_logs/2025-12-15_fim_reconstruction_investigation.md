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
