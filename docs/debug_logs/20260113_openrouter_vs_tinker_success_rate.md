# 2026-01-13 — Why OpenRouter FIM accuracy looks higher than Tinker (success-rate contrast + root cause)

This note captures an observed discrepancy:

- `data_pipeline/test_openrouter_accuracy.py` achieves **materially higher Lean pass rate** on the same style of FIM tasks.
- The Tinker training loop (Tinker API) shows a **much lower verification success rate**, dominated by Lean **parsing/indentation** errors.

The goal is to provide enough context for another engineer (or AI) to immediately reproduce and investigate.

---

## Observed success rates

### OpenRouter script (dryrun)

Command:

```bash
python data_pipeline/test_openrouter_accuracy.py
```

Observed output (user report):

```text
Wrote 5 accuracy samples to data/openrouter_accuracy.jsonl
Summary: exact=0/5 lean_pass=3 lean_fail=2 avg_similarity=0.5528 source=dryrun
```

So, on this run:

- **Lean pass rate:** `3/5 = 60%`
- **Exact match:** `0/5`

Notes:

- This was a **dryrun**, using `data/openrouter_synthetic_dryrun.jsonl` as input (the script switches to dryrun if that file exists).
- The results are persisted to: `data/openrouter_accuracy.jsonl`.

### Tinker training loop (latest run window)

Source: `logs/tinker_fim/debug_samples.jsonl`.

In the most recent run window (identified around the latest two `verification_success=true` events), we observed:

- **Verification successes:** `2`
- **Total entries in window:** `~24`

Approximate success rates:

- **Per-sample success:** `2/24 ≈ 8.3%`
- (Earlier run summaries also used group-size assumptions like `2/20 = 10%`; exact denominator depends on grouping and windowing, but the success rate is clearly far below the OpenRouter dryrun result.)

Failure mode breakdown in that window:

- **Formatting/syntax (Lean parse failures):** `17`
  - `unexpected identifier; expected command`: `6`
  - `unexpected token ... expected command`: `6`
  - `expected '{' or indented tactic sequence`: `5`
- **Pipeline: missing/empty tag:** `2` (`<skipped: missing/empty tag>`)
- **Model: unsolved goals:** `2`

Interpretation: the dominant issue in Tinker is **not theorem proving**, it is **producing (and preserving) syntactically valid Lean inside the hole**, especially indentation.

---

## Key code-path differences (most likely cause)

### OpenRouter accuracy script preserves indentation

In `data_pipeline/test_openrouter_accuracy.py`, tag extraction is:

```py
def _extract_tagged_code(text: str, tag: str) -> Optional[str]:
    ...
    return match.group(1).strip("\n")
```

This is important because it **does NOT strip leading spaces**, and leading spaces are often required when the hole is inside a Lean `by` block / bullet block.

### Tinker pipeline used to strip indentation during extraction (bug)

In the Tinker path, extraction happens via `tinker_integration/prompt_formatter.py`:

- `extract_code_from_response()`
- `strip_markdown_fences()`

Previously:

- `_extract_from()` used `.strip()` in the “missing end-tag” branch.
- `strip_markdown_fences()` returned `... .strip()`.

Both can remove **leading indentation** on the first line of the extracted snippet, which is syntactically meaningful for Lean tactic blocks.

This aligns with the observed dominant Tinker failures:

- `expected '{' or indented tactic sequence`

We also recorded 5 concrete examples (with prefix/extracted/suffix and ground truth) here:

- `docs/debug_logs/20260113_tinker_indentation_failures_latest_run.md`

---

## Fix applied

File: `tinker_integration/prompt_formatter.py`

Changes:

- Preserve indentation by using `.strip("\n")` instead of `.strip()` in extraction and fence-stripping.

Rationale:

- Many holes are inside indented tactic blocks; left-trimming can shift the snippet to column 0, causing immediate parse failures even if the tactic content is correct.

---

## Why the discrepancy matters

The OpenRouter script’s “Lean pass” metric is more indicative of **model capability**.

The Tinker pipeline’s low pass rate was dominated by **formatting failures**, which can be caused by:

- model output (not matching indentation), AND/OR
- pipeline extraction/normalization (accidentally removing indentation)

If indentation is being stripped, the Tinker success rate underestimates model capability.

---

## Reproduction checklist

1) Run OpenRouter dryrun and record summary:

```bash
python data_pipeline/test_openrouter_accuracy.py
```

2) Run a short Tinker trial (e.g., 5 steps) and then analyze:

- `logs/tinker_fim/debug_samples.jsonl`
- look for `verification_success=true`
- compare failure categories (parse vs unsolved goals)

3) If parse failures dominate:

- inspect whether extracted code kept its leading spaces
- compare against ground truth middle

---

## Expected next outcome

After preserving indentation in Tinker extraction:

- we expect a significant reduction in parse/indentation errors
- success/failure should shift toward “unsolved goals” and other semantic errors, which better reflects model capability
