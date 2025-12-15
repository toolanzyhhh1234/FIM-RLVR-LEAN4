# Implementation Plan - FIM-RLVR-LEAN4

## Phase 1: The "Inner Loop" (Verifier & Environment) - **DONE**
*Goal: Ensure we can reliably execute and verify a tactic string before generating massive datasets.*
- [x] **Setup & Pinning**: Verifier environment setup with `verification_env/lakefile.lean`.
- [x] **LeanVerifier Class**: Implemented in `fim_rlvr_lean4/lean_verifier.py`.
- [x] **Environment Isolation Test**: Verified with `train_grpo_fim.py` loop.

## Phase 2: Data Pipeline (HF Source) - **DONE**
- [x] **Data Sourcing**: Using `AI-MO/NuminaMath-LEAN`.
- [x] **FIM Generator**: 
    - Implemented `data_pipeline/fim_exact_line.py` (Exact Line Splitting).
    - Implemented `data_pipeline/generate_dataset.py` (Batch Generation).
- [x] **Dataset Export**: Generated `data/fim_fresh.jsonl`.


## Phase 3: RLVR Loop (GRPO)
- [x] **Orchestrator**: Combine `LeanVerifier` + `SFT Model`.
- [x] **GrpoTrainer**:
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
