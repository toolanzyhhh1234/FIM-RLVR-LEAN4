## Dataset Options Analysis

### Option 1: AI-MO/NuminaMath-LEAN
- **Pros**: Massive (100k+), likely high quality, specifically designed for competition math.
- **Cons**: Need to determine exactly which Mathlib commit it depends on. If it doesn't ship with `imports` metadata, it's hard to compile.

### Option 2: charliemeyer2000/leandojo_benchmark_lean4_17_0
- **Pros**: Explicitly formatted for LeanDojo. Likely contains the `file_path`, `commit_hash`, and `trace_data` we need.
- **Cons**: Might be older (Lean 4.17.0 is implied by the name?).

### Recommendation
Switch Phase 1 (Data) to **"Hybrid Sourcing"**:
1. Default to **NuminaMath-LEAN** for the bulk training data (SFT).
2. Use **LeanDojo Benchmark** (or a small Mathlib slice) for the **RLVR pipeline** initially, because we know it compiles.

**Why?**
RLVR requires a *working compiler environment*. Recompiling random snippets from NuminaMath without their exact dependencies is very hard (the "context/import hell" problem).
For SFT (Phase 2), we can be looser (just learn the syntax).
For RLVR (Phase 3), we need strict reproducibility.
