# 2026-01-13 — Python `strip()` vs `strip("\n")` and why it broke Lean indentation

This note explains a subtle but important Python behavior that caused the Tinker FIM pipeline to accidentally destroy Lean indentation during tag extraction.

---

## What `str.strip()` does

In Python, `s.strip()` removes **whitespace characters** from **both ends** of the string (left and right), repeatedly, until the next character is not whitespace.

“Whitespace” includes spaces, tabs, and newlines.

Key point: `strip()` does **not** remove whitespace from the *middle* of the string — but it **will** remove indentation spaces if those spaces are at the very start of the extracted snippet.

Example:

```py
s = "    intro x\n    simp\n"
print(repr(s.strip()))
# 'intro x\n    simp'
```

The indentation before `intro x` was removed because it was at the **left edge** of the string.

---

## What `str.strip("\n")` does

`s.strip("\n")` removes only newline characters (`\n`) from **both ends** of the string.

It does not remove spaces/tabs, and it does not touch interior newlines.

Example:

```py
s = "\n\n    intro x\n    simp\n\n"
print(repr(s.strip("\n")))
# '    intro x\n    simp'
```

---

## Why this matters for Lean FIM

Many holes occur inside an indented Lean context (e.g. a `by` block or a bullet `·` block). In those contexts, **leading spaces are syntactically meaningful**.

Extra leading/trailing newlines around the extracted snippet are usually harmless, but removing leading spaces can shift the snippet to column 0 and trigger parse/layout errors such as:

```text
error: expected '{' or indented tactic sequence
```

---

## Practical rule used in this repo

- Avoid `.strip()` on extracted Lean snippets.
- If trimming is desired, use `.strip("\n")` (or `rstrip("\n")`) so indentation is preserved.
