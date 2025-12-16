import Mathlib



/-- Let $\(a, b, c\)$ be distinct integers. Show that there does not exist a polynomial $\(P\)$ with integer coefficients such that $\(P(a)=b\)$, $\(P(b)=c\)$, and $\(P(c)=a\)$.-/
theorem algebra_4034
    (a b c : ℤ) (hnab: a ≠ b)(hnbc: b ≠ c)(hnac: a ≠ c)
    (P : Polynomial ℤ)
    (hPab : P.eval a = b) 
    (hPbc : P.eval b = c)
    (hPac : P.eval c = a) :
    False := by
  -- we get `a - b ∣ P a - P b`, by a lemma, and likewise
  have hab := P.sub_dvd_eval_sub a b
  have hbc := P.sub_dvd_eval_sub b c
  have hca := P.sub_dvd_eval_sub c a
  -- replace
  rw [hPab, hPbc, hPac] at *
  clear hPab hPbc hPac
  obtain ⟨m0, hm0⟩ := hab
  obtain ⟨m1, hm1⟩ := hbc
  obtain ⟨m2, hm2⟩ := hca

  have mul_all : (b - c) * (c - a) * (a - b) = ((a - b) * m0) * ((b - c) * m1) * ((c - a) * m2) := by 
    nth_rw 1 [hm0]
    nth_rw 1 [hm1]
    nth_rw 1 [hm2]
    ring

  have h7 : (b - c) * (c - a) * (a - b) = ((b - c) * (c - a) * (a - b)) * (m0 * m1 * m2) := by 
    nth_rw 1 [mul_all]
    ring
  
  have : (b - c) * (c - a) * (a - b) ≠ 0 := by 
    apply mul_ne_zero
    apply mul_ne_zero
    exact sub_ne_zero_of_ne hnbc
    exact sub_ne_zero_of_ne (id (Ne.symm hnac))
    exact sub_ne_zero_of_ne hnab

  simp [this] at h7

  simp [Int.mul_eq_one_iff_eq_one_or_neg_one, Int.mul_eq_neg_one_iff_eq_one_or_neg_one] at h7

  have hm0' : m0 = -1 ∨ m0 = 1 := by tauto
  have hm1' : m1 = -1 ∨ m1 = 1 := by tauto
  have hm2' : m2 = -1 ∨ m2 = 1 := by tauto

  have : a = b := by 
    cases' hm0' with hm0' hm0' <;> cases' hm1' with hm1' hm1' <;> cases' hm2' with hm2' hm2' 
      <;> subst hm0' hm1' hm2' <;> linarith

  exact hnab this
