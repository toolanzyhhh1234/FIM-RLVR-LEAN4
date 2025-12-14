import Mathlib

open Real

/- Let $x$ be a real number, and $\lfloor x \rfloor$ denote the greatest integer that is less than or equal to $x$. Then, the smallest positive period of the function $f(x) = x - \lfloor x \rfloor$ is $\_\_\_\_\_$.-/
theorem algebra_9177 {f : ℝ → ℝ} (hf : f = fun (x : ℝ) => x - ⌊x⌋) :
    IsLeast {T | 0 < T ∧ f = fun x => f (x + T)} 1 := by
  constructor
  -- Verify that 1 is a period of $f(x)$.
  . constructor
    . norm_num
    simp [hf]
  -- Prove that 1 is the least period by contradiction.  rintro T ⟨htpos, htperiod⟩
  by_contra h
  push_neg at h  apply_fun (· 0) at htperiod
  simp [hf] at htperiod
  have : Int.fract T = T := by
    rw [Int.fract_eq_self]
    constructor <;> linarith
  linarith
