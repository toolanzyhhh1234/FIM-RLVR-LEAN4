from fim_rlvr_lean4.lean_verifier import LeanVerifier
import os

# Fix PATH for Lean 4
elan_bin = os.path.expanduser("~/.elan/bin")
if elan_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{elan_bin}:{os.environ['PATH']}"
    print(f"Added {elan_bin} to PATH")


def debug():
    print(f"Current PATH: {os.environ['PATH']}")
    # Check if lake is reachable
    import shutil

    lake_path = shutil.which("lake")
    print(f"Lake path: {lake_path}")

    verifier = LeanVerifier("./verification_env")

    # A simple valid code (the one from original file)
    valid_code = r"""
import Mathlib
theorem algebra_4013 {a b c : ℝ} (h : a * b * c = 1) (haux : 1 + a + a * b ≠ 0) :
    a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 1 := by
  have : a * b * c ≠ 0 := by rw [h]; norm_num
  have ha : a ≠ 0 := left_ne_zero_of_mul <| left_ne_zero_of_mul this
  have hb : b ≠ 0 := right_ne_zero_of_mul <| left_ne_zero_of_mul this
  conv => lhs; lhs; rhs; rw [← mul_div_mul_left _ _ ha]
  conv => lhs; rhs; rw [← mul_div_mul_left _ _ (mul_ne_zero ha hb)]
  rw [show a * (b * c + b + 1) = a*b*c + a*b + a by ring]
  rw [show a*b*(c * a + c + 1) = a*b*c*a + a*b*c + a*b by ring]
  rw [h, one_mul]
  ring_nf
  rw [← add_mul]
  nth_rw 2 [← one_mul (1 + a + a * b)⁻¹]
  rw [← add_mul, show a * b + a + 1 = 1 + a + a * b by ring]
  exact mul_inv_cancel₀ haux
"""

    print("Testing verification...")
    success, output = verifier.verify(valid_code)
    print(f"Success: {success}")
    if not success:
        print(f"Output:\n{output}")


if __name__ == "__main__":
    debug()
