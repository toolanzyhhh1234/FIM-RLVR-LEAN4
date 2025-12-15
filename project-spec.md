## Project spec (Lean4 FIM + RLVR)

### Objective
Build a Lean4-capable model that can **infill a missing proof segment** given `prefix + <HOLE> + suffix`, then improve it using reinforcement learning from verifiable rewards (Lean verification).[1][2]
LeanDojo will be used as the primary infrastructure for data extraction (proof states/tactics/premises) and programmatic interaction with Lean4.[3][1]

### Scope decisions (recommended defaults)
- Target language: Lean4 `by`-style tactic proofs.  
- Hole definition: line-based spans (no AST parsing); masking is based on exact-line splits so recomposition is byte-for-byte.[6]  
- Verification: compile/check the reconstructed proof with Lean; success is binary (verified / not verified).[2]
- Optional “process signal”: parse Lean feedback to find the earliest failing step and use it for partial credit, following “Lean as a symbolic process oracle” framing.[4][5]

***

## Data and task design

### Raw source corpora
- Primary: mathlib4-style Lean repos; extract theorem statements and full proofs via LeanDojo’s extractor.[1][3]
- Secondary (optional): benchmark theorem sets used by modern Lean4 provers (for evaluation and/or additional training).[5][2]

### FIM transformation (offline)
For each theorem proof script:
- Parse into a sequence of tactic lines (exact-line splits).
- Sample a hole location and a hole ratio \(r\) (e.g., 0.10 means 10% of tactic lines removed).
- Produce a single training instance:
  - `prompt = <PFX> prefix <CTX> retrieved_context <K=n> <SFX> suffix <MID>` (Context is optional but recommended)
  - `target = middle` (the exact removed block)
- Use exact line-based masking (no AST parsing) to keep recomposition byte-aligned.
- **Reconstruction integrity:** The pipeline enforces byte-for-byte equality `prefix + middle + suffix == original` via `data_pipeline/byte_reconstruction_check.py` (and the same concatenation is used in the training loop, with `middle` coming from the model at runtime). Older Lean-only smoke tests (e.g., `data_pipeline/test_fim_reconstruction.py`) are deprecated and should not be treated as the integrity gate.

### Suggested JSONL schema
- `theorem_id`: stable identifier (file + theorem name).
- `theorem_stmt`: the statement line(s) (or a canonical form).
- `prefix`, `suffix`, `middle`: strings.
- `k_lines`: integer number of missing tactic lines.
- `mask_ratio`: float.
- `hole_span`: (start_idx, end_idx) in the tactic list.
- `meta`: repo commit, file path, imports, etc. (LeanDojo can help capture dependencies).[3][1]

***

## Training algorithm

### Phase 1: RLVR (Reinforcement Learning with Verification Rewards) - **Current Focus**
Skipping traditional SFT based on recent findings that clever prompt engineering (Chat Templates) with `prefix + [MISSING] + suffix` is sufficient to elicit good FIM behavior from base models (like `gpt-oss-20b`).

Core loop for each training episode:
1. Sample a theorem \(T\) and a hole ratio \(r\), generate a hole `(prefix, suffix)` as in the FIM transformation.
2. Sample a *group* of \(G\) candidate `middle` completions from the policy (group sampling is compatible with GRPO-style updates).[2][5]
3. For each candidate:
   - **Fast-Fail Check:** Check syntax/elaboration. If invalid, reward = 0 immediately (saves compute).
   - **Full Verification:** Reconstruct full proof and run Lean verification.
4. Reward:
   - Outcome reward: \(R=1\) if Lean verifies, else \(0\).[2]
   - Optional process reward: partial credit based on locally valid tactics.
5. Update with a group-relative objective (e.g., GRPO-style).


**Key point on hole-size reward:** do not scale terminal reward as \(1+\text{mask\_frac}\) by default; instead push difficulty through curriculum and sampling, and reserve reward shaping for the process-reward variant.[4][5]

***

## Curriculum on hole size (mastery-based)

### State tracked per theorem
Maintain per theorem \(T\):
- Current ratio level `r[T]` in a discrete ladder (e.g., 0.10, 0.20, …, 0.50 up to fully blank).
- A rolling window of recent verifier outcomes at each ratio (e.g., last \(W=8\) attempts).  
- A “mastered” flag per ratio if successes in window \(\ge m\) (e.g., \(m=5\)).

### Promotion rule (your idea, made robust)
- Train on `r[T]` until the model succeeds \(m\) times on *different* hole placements at that ratio within the last \(W\) trials.  
- Then promote: `r[T] = min(r[T] + Δr, r_max)`.

### Sampling policy (prevents forgetting)
Even after promotion, don’t show *only* larger holes:
- 70%: sample at `r[T]` (current level).
- 20%: sample at a smaller ratio (“review”).
- 10%: sample at `r[T] + Δr` (“challenge/probe”).
This keeps positive rewards flowing and reduces regressions.

### Demotion / remediation (optional but useful)
If performance at the new ratio collapses (e.g., 0 successes in last \(W\)), either:
- temporarily increase review sampling, or
- demote by one level for a fixed number of steps.

***

## Experimentation plan (what to run and what to report)

### Metrics (Standard Benchmarks)
- **Primary Eval:** Use established Lean4 benchmarks (e.g., **MiniF2F**, **ProofNet**, or **LeanDojo Benchmark**) to ensure comparability and avoid custom-split leakage issues.
- **Pass@N**: Probability at least one of \(N\) samples yields Lean-verified reconstruction on these benchmarks.
- Breakdown by hole ratio: pass@N for 10%, 20%, … (shows curriculum impact).
- Sample efficiency: verifier calls per additional % pass@1 improvement.

### Baselines / ablations
1. RLVR outcome-only (binary reward).
2. RLVR with process rewards (earliest failure / locally-valid tactics).[5][4]
3. Curriculum vs no curriculum (fixed hole ratio distribution).
4. Structure-aware masking (tactic-boundary holes) vs naive token-span masking (expected to be worse / noisier).[6]

### Practical engineering checks
- Ensure each reconstructed proof is checked in a clean Lean environment (consistent imports, same toolchain) to avoid false negatives.[1]
- Log Lean error locations for failed attempts; this is needed if process rewards are implemented.[4][5]
- Add per-theorem sampling caps so a few “easy” theorems don’t dominate training updates.[5]
- Track duplicate exposure: optionally dedup identical `(theorem, hole_start, hole_len)` when generating data, or downweight repeats in the sampler. Low priority now, but keep in mind if we observe overfitting on a handful of holes.

If the coding agent needs one crisp “MVP target,” implement: (SFT on exact-line FIM) → (outcome-only GRPO RLVR) → (mastery-based hole-size curriculum), and leave process rewards as the stretch goal.[2][5]

[1](https://github.com/lean-dojo/LeanDojo)
[2](https://openreview.net/forum?id=I4YAIwrsXa)
[3](https://papers.neurips.cc/paper_files/paper/2023/file/4441469427094f8873d0fecb0c4e1cee-Paper-Datasets_and_Benchmarks.pdf)
[4](https://openreview.net/forum?id=sKvVHgmRP2)
[5](https://openreview.net/pdf?id=sKvVHgmRP2)
[6](https://arxiv.org/abs/2506.00204)
[7](https://leandojo.readthedocs.io)
[8](https://leandojo.org/leandojo.html)
[9](https://lean-lang.org/learn/)
[10](https://leanprover-community.github.io/archive/stream/219941-Machine-Learning-for-Theorem-Proving/topic/Releasing.20LeanDojo.html)
[11](https://www.emergentmind.com/topics/deepseek-prover-v1-5)
[12](https://iclr.cc/virtual/2025/poster/30193)
[13](https://leandojo.org)
[14](https://openreview.net/forum?id=FAe9Gts2Qd)
[15](https://arxiv.org/abs/2408.08152)
[16](https://leandojo.readthedocs.io/en/latest/user-guide.html)
[17](https://www.lesswrong.com/posts/y3NgXbtCK8qQNzhYq/distillation-of-deepseek-prover-v1-5)
[18](https://arxiv.org/html/2312.14188v1)
[19](https://openreview.net/pdf?id=P00k4DFaXF)
[20](https://github.com/tpn/pdfs/blob/master/DeepSeek-Prover-V1.5%20-%20Harnessing%20Proof%20Assistant%20Feedback%20for%20Reinforcement%20Learning%20and%20Monte-Carlo%20Tree%20Search%20-%202024%20(2408.08152v1).pdf)
[21](https://openreview.net/forum?id=g7OX2sOJtn&noteId=EJxdCMebal)
