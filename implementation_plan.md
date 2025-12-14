# Implementation Plan - FIM-RLVR-LEAN4

## Phase 1: Data Pipeline & AST Extraction
- [ ] **Setup LeanDojo**: Install and verify LeanDojo on the local machine.
- [ ] **Trace Mathlib**: Run LeanDojo's trace tool on a subset of Mathlib4 to generate raw extracted data.
- [ ] **Data Parser**: Write a Python script to parse LeanDojo's `traced_tactics` (JSON).
    - Extract: `theorem_name`, `file_path`, `full_source`, `tactic_ranges`.
    - Validation: Ensure tactic ranges correspond to clean text blocks in the source.
- [ ] **FIM Generator**: Implement the **Structure-Aware Masking** logic.
    - Input: parsed theorem data.
    - Logic: Select random `tactic_range`. Create `prefix`, `suffix`, `middle`.
    - Retrieval Hook: (Optional for v0) Add placeholder for `<CTX>` logic.
- [ ] **Dataset Export**: Save processed examples to `data/train.jsonl` and `data/val.jsonl`.

## Phase 2: Model Training (SFT)
- [ ] **Tokenizer Setup**: Ensure tokenizer handles Lean 4 syntax and special tokens (`<PFX>`, `<SFX>`, `<MID>`).
- [ ] **Training Loop**: Set up a standard SFT loop (HuggingFace Trainer or custom).
- [ ] **Baseline Training**: Train a small model (e.g., DeepSeek-Coder-1.3B or similar small model for testing) on the generated data.

## Phase 3: RLVR Infrastructure
- [ ] **Environment Setup**: Create a Python class `LeanVerifier` that wraps LeanDojo interaction.
    - Input: `(file_path, theorem_name, proof_candidate)`.
    - Logic: Inject candidate into file -> Run `lean` build -> Parse stderr/stdout.
    - Fast-Fail: Implement regex-based syntax check before calling Lean.
- [ ] **RL Loop (GRPO)**:
    - Implement Group Relative Policy Optimization loop.
    - Reward Function: `1` if verifier returns success, `0` otherwise.
- [ ] **Curriculum Logic**:
    - Track pass rates per theorem.
    - Implement the "promotion" logic (increase hole size after success).

## Phase 4: Evaluation
- [ ] **Benchmark Setup**: Prepare the MiniF2F or ProofNet harness.
- [ ] **Eval Script**: Write a script to run the model on the benchmark and compute Pass@1 / Pass@16.
