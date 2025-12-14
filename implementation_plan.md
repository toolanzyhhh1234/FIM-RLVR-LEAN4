# Implementation Plan - FIM-RLVR-LEAN4

## Phase 1: The "Inner Loop" (Verifier & Environment) - **HIGH PRIORITY**
*Goal: Ensure we can reliably execute and verify a tactic string before generating massive datasets.*
- [ ] **Setup & Pinning**:
    - Install `elan` (Lean version manager).
    - Pin `LeanDojo` to a specific stable version.
    - Pin `Mathlib4` to a specific commit hash (compatible with that LeanDojo version).
- [ ] **LeanVerifier Class**: Implement the Python wrapper for LeanDojo.
    - Input: `(file_path, theorem_name, proof_candidate_string)`.
    - Logic: Use `LeanDojo.Dojo` to modify the file/proof state and check for success.
    - **Speed Test**: Measure average seconds per verification. Optimizing this early is critical.
- [ ] **Environment Isolation Test**: Create a test script that picks a random theorem, corrupts it, attempts to verify (should fail), then provides the correct proof (should pass).

## Phase 2: Data Pipeline (HF Source)
- [ ] **Data Sourcing**:
    - Download `AI-MO/NuminaMath-LEAN` and `charliemeyer2000/leandojo_benchmark_lean4_17_0`.
    - **Metadata Audit**: Check which Mathlib4 commit each dataset aligns with.
    - **Decision**: Select the dataset that is easiest to reproduce locally (likely the one with clearer dependency metadata).
- [ ] **Tactic Parser**:
    - Adapt parser to read from Parquet/JSONL instead of raw LeanDojo trace files.
    - Focus on extracting top-level tactic blocks.
- [ ] **FIM Generator**:
    - Implement `prefix + <HOLE> + suffix` generation.
    - **Deduplication**: Filter out one-liner trivial proofs (e.g., `by rfl`, `by simp`).
- [ ] **Dataset Export**: Save `mvp_train.jsonl` and `mvp_val.jsonl`.

## Phase 3: Model Training (SFT - MVP)
- [ ] **Tokenizer**: Setup for Lean4 + Special Tokens.
- [ ] **Baseline SFT**:
    - Train a small model (e.g., DeepSeek-Coder-1.3B) on the `mvp_train.jsonl`.
    - Eval: Check syntax validity of generated outputs (before full verification).

## Phase 4: RLVR Loop (GRPO)
- [ ] **Orchestrator**: Combine `LeanVerifier` + `SFT Model`.
- [ ] **GrpoTrainer**:
    - Implement the group sampling loop (Group Size G=4 initially).
    - **Async Verification**: (Stretch) Parallelize verifier calls if single-thread is too slow.
- [ ] **Logging & Checkpointing**:
    - Log: `Verifier Pass Rate` (most important), `KL Divergence`, `Mean Reward`.
    - Save checkpoints every X steps (RL is unstable).
- [ ] **Curriculum**:
    - Simple schedule: Start with fixed 20% hole size. Only enable dynamic sizing after stability is proven.

## Phase 4: Evaluation
- [ ] **Benchmark Setup**: Prepare the MiniF2F or ProofNet harness.
- [ ] **Eval Script**: Write a script to run the model on the benchmark and compute Pass@1 / Pass@16.
