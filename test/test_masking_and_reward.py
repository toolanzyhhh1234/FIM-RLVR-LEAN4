import pytest
from fim_rlvr_lean4.masking import apply_dynamic_mask
from train_gspo_fim_20b import lean_validity_reward_factory
from fim_rlvr_lean4.curriculum import CurriculumManager


def test_apply_dynamic_mask_fallback_non_empty_prefix_suffix():
    code = """
import Mathlib
theorem algebra_4013 {a b c : ℝ} (h : a * b * c = 1) (haux : 1 + a + a * b ≠ 0) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 1 :=
  sorry
"""
    prefix, suffix, middle = apply_dynamic_mask(code, ratio=0.5)
    # At least one of prefix/suffix should be non-empty so verifier has code to check
    assert prefix or suffix, "Masking produced empty prefix and suffix"
    # Middle must not be empty (we masked something)
    assert middle, "Masked middle should not be empty"


def test_reward_builds_full_code_when_prefix_suffix_empty():
    class MockVerifier:
        def __init__(self):
            self.seen = []

        def verify(self, code):
            self.seen.append(code)
            return True, "ok"

    curriculum = CurriculumManager()
    verifier = MockVerifier()
    reward_fn = lean_validity_reward_factory(verifier, curriculum)

    completions = ["lemma foo : True := by trivial"]
    fim_prefix = [""]
    fim_suffix = [""]
    theorem_id = ["test"]

    scores = reward_fn(completions, fim_prefix, fim_suffix, theorem_id)

    assert verifier.seen, "Verifier was not called when prefix/suffix empty"
    assert scores == [2.0], "Reward should be success=2.0 when verifier returns True"
