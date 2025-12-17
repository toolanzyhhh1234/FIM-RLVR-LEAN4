import unittest
from fim_rlvr_lean4.masking import apply_dynamic_mask, reconstruct_full_code


class TestCurriculumMasking(unittest.TestCase):
    def setUp(self):
        # A simple fake lean file content
        self.lean_code_simple = """
theorem basic (n : Nat) : n + 0 = n := by
  rw [Nat.add_zero]
  rfl
"""
        # A code with slightly more lines
        self.lean_code_multi = """
theorem arithmetic (a b : Nat) : a + b = b + a := by
  ind_simp
  ac_rfl
  simp
  trace
"""
        # No ':= by' marker
        self.lean_code_no_marker = """
def plain_def (n : Nat) := n + 1
"""

    def test_partial_masking_ratio_0_5(self):
        """Test partial masking (e.g. 50%)"""
        # Proof body lines:
        #   ind_simp
        #   ac_rfl
        #   simp
        #   trace
        # Total 4 lines (excluding ':=' line? The function splits after ':= by')

        # In `apply_dynamic_mask`, it looks for `:= by`.
        # `theorem arithmetic... := by` is line 1 (index 0 or 1 depending on split).
        # splitlines(keepends=True) on self.lean_code_multi:
        # [0] "" (because of leading newline in example string?)
        # [1] "theorem ... := by\n"
        # [2] "  ind_simp\n"
        # [3] "  ac_rfl\n"
        # [4] "  simp\n"
        # [5] "  trace\n"

        # If the input string starts with newline, splitlines might give an empty first line.
        full_code = self.lean_code_multi.strip()
        # Now:
        # [0] "theorem ... := by"
        # [1] "  ind_simp" ...

        prefix, suffix, middle = apply_dynamic_mask(full_code, ratio=0.5)

        # 4 proof lines. ratio 0.5 -> mask 2 lines.
        middle_lines = middle.splitlines()
        # Should mask roughly 2 lines.
        # Note: apply_dynamic_mask uses max(1, int(num * ratio))
        self.assertGreaterEqual(len(middle_lines), 1)
        self.assertLessEqual(len(middle_lines), 3)  # Depending on rounding

        # Check that prefix + middle + suffix == full_code
        self.assertEqual(prefix + middle + suffix, full_code, "Reconstruction failed")

        # Check that prefix ends with something before the masked part
        # and suffix starts after.

    def test_full_masking_ratio_1_0(self):
        """Test full masking (ratio >= 1.0) - should mask the entire proof body."""
        full_code = self.lean_code_simple.strip()
        # [0] theorem ... := by
        # [1]   rw ...
        # [2]   rfl

        prefix, suffix, middle = apply_dynamic_mask(full_code, ratio=1.0)

        # EXPECTATION:
        # Prefix should be the theorem statement up to `:= by` (and maybe the newline after?)
        # Implementation says: `prefix = "".join(lines[:proof_start_idx])`
        # `proof_start_idx` is index AFTER the line containing `:= by`.
        # So prefix contains the `:= by` line.

        self.assertIn(":= by", prefix)

        # Suffix should be empty (as per requirement "question only")
        self.assertEqual(suffix, "", "Suffix should be empty for ratio=1.0")

        # Middle should contain the proof lines
        self.assertIn("rw [Nat.add_zero]", middle)
        self.assertIn("rfl", middle)

        # Reconstruction check
        self.assertEqual(prefix + middle + suffix, full_code)

    def test_no_marker_fallback(self):
        """Test fallback when ':= by' is missing"""
        full_code = self.lean_code_no_marker.strip()
        prefix, suffix, middle = apply_dynamic_mask(full_code, ratio=0.5)

        # If no marker, it defaults to whole file as prefix? Or logic?
        # My implementation: "if proof_start_idx >= total_lines: proof_start_idx = 0"
        # Then "proof_lines = lines[0:]" -> everything is proof.
        # Then it masks 50% of the definition.

        self.assertEqual(prefix + middle + suffix, full_code)
        self.assertTrue(
            len(middle) > 0, "Should have masked something even without marker"
        )


if __name__ == "__main__":
    unittest.main()
