# 2026-01-13 — Testing strategy: deterministic tests for LLM-facing pipelines

This note captures a key lesson from debugging Tinker FIM verification failures:

- LLM outputs are nondeterministic.
- But many real failures come from **deterministic pipeline logic** (prompt formatting, boundary normalization, tag extraction, stitching), which *can* and *should* be tested reliably.

---

## What to test (deterministically)

Instead of trying to test “the model solves the Lean theorem,” test the **pipeline invariants**:

1) **Indentation preservation**
   - Extraction must not remove leading spaces that are syntactically meaningful in Lean tactic blocks.
   - Regression target: avoid `.strip()` on extracted Lean snippets.

2) **Harmony final-channel preference**
   - When the model outputs both analysis and final channels, extraction should prefer the final channel.
   - Regression target: avoid extracting tags that appear only in analysis.

3) **Boundary normalization consistency (Option A)**
   - If prompts use `{prefix}[MISSING_BLOCK]\n{suffix}`, then verification stitching must match that boundary.
   - Regression target: avoid suffix “glueing” that turns valid `constructor\n  · ...` into invalid `constructor  · ...`.

4) **Markdown-fence stripping without destroying indentation**
   - If the model wraps code in ``` fences, stripping fences must not left-strip indentation.

5) **Stitching invariant: never concatenate suffix onto a token**
   - Reconstructed full code must not accidentally merge the suffix onto the last token of the extracted snippet when a separator is required.

---

## How to test despite nondeterministic LLM outputs

The trick is to supply **synthetic model responses** as fixtures.

Even though real LLM responses vary, the pipeline bugs are deterministic: if we reproduce the problematic response shape (extra newlines, missing end-tags, tags in analysis vs final, fenced code, etc.), then a unit test can reliably verify correct behavior.

Two practical test styles:

### A) Invariant tests (high ROI)

Assert properties that must always hold regardless of model content.

Examples:

- `extract_code_from_response()` preserves leading indentation.
- Extraction prefers Harmony final channel when present.
- `normalize_boundaries()` never adds a newline after a prefix that ends with indentation spaces.

### B) Regression fixtures (pattern-based)

Convert known failure patterns into minimal reproducible strings.

Example pattern observed in logs:

- Lean error: `expected '{' or indented tactic sequence`
- Often caused by extracted code being shifted to column 0 (indentation stripped)

So the test fixture can simply be:

- a prefix ending with `:= by\n  ` (indentation context)
- a response whose `<FIM_CODE>` contains `"  exact heq"`
- an assertion that the extracted snippet still starts with two spaces after extraction + fence stripping

---

## Why this matters

Without these tests, small refactors (e.g., accidentally reintroducing `.strip()`) can silently tank verification success rates.

With these tests, we can ensure:

- pipeline regressions are caught immediately
- measured “success rate” changes reflect model behavior (capability) rather than avoidable formatting/stitching bugs
