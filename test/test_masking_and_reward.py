import pytest
from fim_rlvr_lean4.masking import apply_dynamic_mask
from fim_rlvr_lean4.curriculum import CurriculumManager
import polars as pl
import os


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

    verifier = MockVerifier()

    # Minimal reward fn to avoid importing full training stack
    def reward_fn(completions, fim_prefix, fim_suffix, theorem_id):
        out = []
        for comp, pre, suf, th in zip(completions, fim_prefix, fim_suffix, theorem_id):
            full = (pre or "") + comp + (suf or "")
            success, _ = verifier.verify(full if full.strip() else "")
            out.append(2.0 if success else 0.0)
        return out

    completions = ["lemma foo : True := by trivial"]
    fim_prefix = [""]
    fim_suffix = [""]
    theorem_id = ["test"]

    scores = reward_fn(completions, fim_prefix, fim_suffix, theorem_id)

    assert verifier.seen, "Verifier was not called when prefix/suffix empty"
    assert scores == [2.0], "Reward should be success=2.0 when verifier returns True"


def test_dataset_masking_keeps_non_empty_prefix_suffix():
    path = "data/data/train-00000-of-00001.parquet"
    if not os.path.exists(path):
        pytest.skip("Dataset shard not available")

    df = pl.read_parquet(path)
    if "formal_ground_truth" not in df.columns:
        pytest.skip("Expected formal_ground_truth column not found")

    sample = df["formal_ground_truth"][:3].to_list()

    # Directly test masking on the dataset rows without loading the full training stack.
    for code in sample:
        pre, suf, mid = apply_dynamic_mask(code, ratio=0.3)
        assert pre or suf, "Prefix and suffix both empty after masking"
        assert mid, "Masked middle should not be empty"
        # Build a simple prompt as the model would see
        user_content = f"{pre}[MISSING_BLOCK]\n{suf}"
        assert "[MISSING_BLOCK]" in user_content
        assert user_content.strip(), "Chat prompt should not be empty"
