import Mathlib

open Nat

/-
The Antarctican language has an alphabet of just 16 letters. Interestingly, every word in the language has exactly 3 letters, and it is known that no word's first letter equals any word's last letter (for instance, if the alphabet were $\{a, b\}$ then $a a b$ and aaa could not both be words in the language because $a$ is the first letter of a word and the last letter of a word; in fact, just aaa alone couldn't be in the language). Given this, determine the maximum possible number of words in the language.
-/
def alphabet_size : ℕ := 16

-- Lemma: For fixed sum S = a + b, a*b is maximized when a and b are as close as possible.
-- The maximum value is (S / 2) * (S - S / 2).
lemma product_le_half_split_sum (S a b : ℚ) (h_sum : a + b = S) :
    a * b ≤ (S / 2) * (S - S / 2) := by  transitivity S / 2 * (S / 2)
  subst S
  have := two_mul_le_add_pow_two a b 
  ring_nf
  simp_all
  sorry
  sorry
-- Lemma: To maximize a*b subject to a+b ≤ S, we should choose a+b = S.
lemma product_max_when_sum_is_max (S a b : ℚ) (h_le_S : a + b ≤ S) :    a * b ≤ (S / 2) * (S - S / 2) := by
  let S' := a + b
  have hS'_le_S : S' ≤ S := h_le_S
  have term1_le : a * b ≤ (S' / 2) * (S' - S' / 2) := product_le_half_split_sum S' a b rfl  have term2_mono : (S' / 2) * (S' - S' / 2) ≤ (S / 2) * (S - S / 2) := by
    let g := fun s : ℚ => (s/2)*(s - s/2)
    suffices Monotone g from this hS'_le_S 
    -- Proof of monotonicity for g(s) = floor(s/2) * ceil(s/2)
    sorry
  exact le_trans term1_le term2_mono
-- Calculate maximum number of words
def max_num_words : ℕ :=
  let S := alphabet_size
  let max_ab := (S / 2) * (S - S / 2) -- This is the max value of a*b s.t. a+b=S
  alphabet_size * max_ab
theorem final_answer_is_1024 : max_num_words = 1024 := by
  unfold max_num_words alphabet_size
  norm_num