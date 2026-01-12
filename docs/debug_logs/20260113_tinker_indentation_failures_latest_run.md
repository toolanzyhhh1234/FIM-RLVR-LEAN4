# 2026-01-13 — Tinker FIM: “expected '{' or indented tactic sequence” failure cases

This note records 5 representative failures from the latest Tinker run window where Lean failed with:

```
error: expected '{' or indented tactic sequence
```

For each case, we include:
- the **prefix tail** (context immediately before the hole)
- the model **extracted code** (what we inserted)
- the **ground truth middle** (the dataset’s actual middle)
- the **suffix head** (context immediately after the hole)

In all cases below, the dominant issue is **structural/indentation mismatch**: the model emits code at the wrong indentation level or repeats a header (`have ... := by`) when the hole expects only the proof body.

---

## Example 1

- `theorem_id`: `theorem_6755`
- `step`: `0`
- `mask_ratio`: `0.1`
- `timestamp`: `1768223378.7571552`
- `ground_truth_similarity`: `0.1923`
- verifier (first error line):
  - `Verify_eb14f322-c5a8-404b-8135-2bc75c849e69.lean:30:2`

### Prefix tail

```lean
  constructor
  · -- solve equation
    intro heq
    -- factorization
    replace heq: (1 + cos x) * ( (cos x)^2 - 2) = 0 := by
      rw [← heq, this]
      ring

    -- (cos x)^2 - 2 ≠ 0, so 1 + cos x = 0, cos x = -1
    replace: (cos x)^2 - 2 ≠ 0 := by
      apply ne_of_lt
      linarith only [cos_sq_le_one x]
    simp only [mul_eq_zero, this, or_false] at heq
    replace heq: cos x = -1 := by
  linarith only [heq]

    -- so x = (2k+1) pi
    rw [cos_eq_neg_one_iff] at heq
```

### Extracted code

```lean
exact heq
```

### Ground truth middle

```lean
    rcases heq with ⟨k, rfl⟩
    use k
    ring
```

### Suffix head

```lean
  · -- verify
    rintro ⟨k,hk⟩
    rw [this]
    replace: cos x = -1 := by
      rw [cos_eq_neg_one_iff]
      use k
      rw [hk]
      ring
    rw [this]
    ring
```

### Notes

The hole is inside an indented tactic/bullet block, but the extracted code is at column 0.

---

## Example 2

- `theorem_id`: `theorem_6755`
- `step`: `0`
- `mask_ratio`: `0.1`
- `timestamp`: `1768223401.8605983`
- `ground_truth_similarity`: `0.6129`
- verifier (first error line):
  - `Verify_001a6e97-2738-4fdc-b58e-50c75c40f94b.lean:30:2`

### Prefix tail

```lean
  constructor
  · -- solve equation
    intro heq
    -- factorization
    replace heq: (1 + cos x) * ( (cos x)^2 - 2) = 0 := by
      rw [← heq, this]
      ring

    -- (cos x)^2 - 2 ≠ 0, so 1 + cos x = 0, cos x = -1
    replace: (cos x)^2 - 2 ≠ 0 := by
      apply ne_of_lt
      linarith only [cos_sq_le_one x]
    simp only [mul_eq_zero, this, or_false] at heq
    replace heq: cos x = -1 := by
  linarith only [heq]
```

### Extracted code

```lean
have heq : ∃ k : ℤ, x = (2 * k + 1) * π := (cos_eq_neg_one_iff.mp heq)
```

### Ground truth middle

```lean

    -- so x = (2k+1) pi
    rw [cos_eq_neg_one_iff] at heq

```

### Suffix head

```lean
    rcases heq with ⟨k, rfl⟩
    use k
    ring
```

### Notes

Even if the content were reasonable, it’s emitted without the indentation expected by the surrounding `by` block.

---

## Example 3

- `theorem_id`: `theorem_375`
- `step`: `4`
- `mask_ratio`: `0.1`
- `timestamp`: `1768224064.902866`
- `ground_truth_similarity`: `0.4698`
- verifier (first error line):
  - `Verify_b1ba2a22-da40-4eb9-af4e-5bf7c075a0ab.lean:9:2`

### Prefix tail

```lean
  have hy' : 1/y = 5 := by
  linarith
  have hy : y = 1/5 := by
```

### Extracted code

```lean
have hy : y = 1/5 := by
    rw [show (1 : ℝ) / 5 = (5 : ℝ)⁻¹ by norm_num, hy'.symm]
    rw [one_div, inv_inv]
```

### Ground truth middle

```lean
    rw [one_div, hy'.symm, one_div, inv_inv]
```

### Suffix head

```lean
  -- Our desired sum is ...
  rw [hx, hy]
  linarith
```

### Notes

The hole is the *body* of `have hy : y = 1/5 := by`, but the model repeats the entire `have hy : ... := by` header.

---

## Example 4

- `theorem_id`: `theorem_375`
- `step`: `4`
- `mask_ratio`: `0.1`
- `timestamp`: `1768224134.0042803`
- `ground_truth_similarity`: `0.2923`
- verifier (first error line):
  - `Verify_9524ea87-ad46-412b-af72-d33aaa97c9d2.lean:9:2`

### Prefix tail

```lean
  have hx': 1/x = -2 := by
  linarith
  have hx : x = -1/2 := by
```

### Extracted code

```lean
have := congrArg (fun t : ℝ => t⁻¹) hx'
    simpa [one_div, inv_inv] using this
```

### Ground truth middle

```lean
    rw [show -(1 : ℝ)/2 = (-2)⁻¹ by norm_num, hx'.symm]
```

### Suffix head

```lean
    rw [one_div, inv_inv]
  --Similarly, we find ...
  have hy' : 1/y = 5 := by
```

### Notes

The extracted code is plausible proof content, but it’s not aligned/indented as part of the `by` block.

---

## Example 5

- `theorem_id`: `theorem_375`
- `step`: `4`
- `mask_ratio`: `0.1`
- `timestamp`: `1768224151.732967`
- `ground_truth_similarity`: `0.08`
- verifier (first error line):
  - `Verify_7de85e74-3680-449c-9a4a-592b61003f47.lean:9:2`

### Prefix tail

```lean
  have hx : x = -1/2 := by
    rw [show -(1 : ℝ)/2 = (-2)⁻¹ by norm_num, hx'.symm]
```

### Extracted code

```lean
skip
```

### Ground truth middle

```lean
    rw [one_div, inv_inv]
```

### Suffix head

```lean
  --Similarly, we find ...
  have hy' : 1/y = 5 := by
  linarith
```

### Notes

Even a “do nothing” tactic like `skip` must be correctly indented to belong to the surrounding `by` block.
