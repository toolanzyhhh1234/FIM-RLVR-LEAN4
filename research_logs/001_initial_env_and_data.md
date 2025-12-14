# Research Log 001: Data Pipeline & Environment Initialization
**Date:** 2025-12-15
**Author:** Antigravity (AI Assistant)

## 1. Summary of Activities
We initialized the FIM-RLVR-LEAN4 project, focusing on the critical "Inner Loop": setting up a reproducible Lean 4 verification environment and validating our data source (`AI-MO/NuminaMath-LEAN`).

## 2. Key Findings

### A. The "Version Drift" Problem
*   **Observation:** Lean 4 projects are highly sensitive to library versions. A proof written for `Mathlib` from July 2024 will often fail to compile in December 2024 due to lemma renaming and syntax changes.
*   **Evidence:** Initial scripts attempting to load `NuminaMath-LEAN` samples failed with "build failed" errors because the environment was not matched to the data.
*   **Mitigation:** We pinned the verification environment to **Lean v4.15.0** and **Mathlib v4.15.0** (Release Tag) to approximate the generation environment of the dataset.

### B. Data Verification Pass Rates
We implemented a `LeanVerifier` and ran a smoke test on samples from `NuminaMath-LEAN`:
*   **Test Size:** Small random subsets (20 items).
*   **Initial Pass Rate:** ~50%
*   **Failure Modes:**
    1.  **Missing Imports:** Some samples rely on obscure Mathlib modules not included in standard `import Mathlib`.
    2.  **Syntax Drift:** Deprecated tactic usage.
    3.  **Parsing/Splitting Artifacts:** Our regex-based FIM generator sometimes slices inside comments or strings, creating invalid code.

### C. FIM Dataset Generation
*   **Source:** `AI-MO/NuminaMath-LEAN` (filtered for compilability).
*   **Method:** "Structure-Aware" heuristic using regex (`:= by` splitting).
*   **Output:** Generates `prompt` (Prefix + Suffix) and `completion` (Middle) pairs.
*   **Validation:** Reconstructed code from the generated JSONL passes the verifier at roughly the same rate as the raw source (~50%).

## 3. Decisions & Pivot
*   **Hybrid Sourcing Strategy:**
    *   Use **NuminaMath-LEAN** for SFT (Phase 2) to get volume and variety.
    *   Use **LeanDojo Benchmark** (or a strictly curated Numina subset) for RLVR (Phase 3) to ensure valid reward signals.
*   **Environment:** Permanently pinned to `v4.15.0` to prevent future drift.

## 4. Next Steps
1.  Initialize the RL Training Loop (GRPO).
2.  Refine the FIM splitter to use AST data if possible to improve the 50% sanity pass rate.
