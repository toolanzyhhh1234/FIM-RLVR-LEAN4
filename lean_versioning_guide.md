# Lean 4 Version Drift: The "Silent Killer" of RLVR

## What is Version Drift?
Unlike Python (where code from 2020 usually runs today) or C++ (backward compatible for decades), **Lean 4 is a moving target**. It is a research language under active development.

### The Mechanism of Failure
1.  **Language Changes**: The syntax of tactics itself changes. (e.g., the way `rw` handles specific cases, or how `simp` config options are passed).
2.  **Mathlib Churn**: `Mathlib4` is a monolith. Lemmas are constantly renamed, moved, or deleted.
    *   *Yesterday:* `theorem foo` existed in `Mathlib.Algebra.Group`.
    *   *Today:* It was renamed to `foo_of_bar` and moved to `Mathlib.Algebra.Ring`.
3.  **The Result**: A proof string that was **100% correct** on Monday might be a **Compilation Error** on Tuesday.

## Why this matters for us (NuminaMath-LEAN)
We looked at a sample from NuminaMath:
```lean
import Mathlib
theorem algebra_4013 ...
```
**It is missing the critical lockfile.**
It says `import Mathlib`, but *which* Mathlib?
- If this dataset was generated using Mathlib from **July 2024**, and we try to verify it using Mathlib from **December 2024**, 30-50% of the proofs might effectively "rot" (fail to compile).

## The Consequence for RLVR
If we don't pin the version exactly:
1.  The model generates a correct proof.
2.  Our Verifier (running a different Mathlib) says "Error: unknown identifier `mul_div_mul_left`".
3.  **We give the model a Negative Reward (0).**
4.  **We lobotomize our AI**: We train it to unlearn correct math because our environment is broken.

## The Solution
1.  **Find the "Golden Commit"**: We must check the dataset documentation to find the exact `lean-toolchain` and `lake-manifest.json` used during generation.
2.  **Pin Global Version**: We will configure our `LeanVerifier` to use *only* that specific version.
3.  **Fallback**: Use `LeanDojo Benchmark`, which explicitly ships with `file_path` and `commit_hash` for every single datapoint, guaranteeing reproducibility.
