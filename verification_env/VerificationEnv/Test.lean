import Mathlib
theorem logic_and_puzzles_609734 (currentTime correctTime : ℕ) (h1 : currentTime = 9 * 60 + 57) (h2 : correctTime = 10 * 60 + 10) : IsLeast { n | ∃ a b, currentTime + 9 * a - 20 * b = correctTime ∧ a + b = n } 24 := by  constructor
  · -- supply witnesses    simp only [Set.mem_setOf_eq]    apply Exists.intro 17
    apply Exists.intro 7
    omega
  · intros n hn
    -- obtain hypotheses
    obtain ⟨a,b,ha1,ha2⟩ := hn
    -- solve
    omega