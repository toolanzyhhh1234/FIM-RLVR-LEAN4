import subprocess
import os
from typing import Tuple

class LeanVerifier:
    def __init__(self, project_dir: str):
        """
        Initializes the LeanVerifier.
        
        Args:
            project_dir: Path to the Lean 4 project directory (should contain lakefile.lean).
        """
        self.project_dir = os.path.abspath(project_dir)
        self.test_file_path = os.path.join(self.project_dir, "VerificationEnv", "Test.lean")

    def verify(self, full_code: str) -> Tuple[bool, str]:
        """
        Verifies a Lean 4 proof by compiling it.

        Args:
            full_code: The complete Lean source code (imports + theorem + proof).

        Returns:
            (success, output): 
                success: True if compilation passed, False otherwise.
                output: The stderr output from the compiler (contains error messages).
        """
        # 1. Write code to the test file
        try:
            with open(self.test_file_path, "w") as f:
                f.write(full_code)
        except IOError as e:
            return False, f"Failed to write to verification file: {e}"

        # 2. Run lake build
        # We target the specific module 'VerificationEnv.Test'
        try:
            result = subprocess.run(
                ["lake", "build", "VerificationEnv.Test"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=False # We handle return code manually
            )
        except FileNotFoundError:
            return False, "Error: 'lake' command not found. Is Lean 4 installed?"

        # 3. Check result
        success = (result.returncode == 0)
        output = result.stderr + "\n" + result.stdout
        
        return success, output

if __name__ == "__main__":
    # Smoke test
    verifier = LeanVerifier("./verification_env")
    
    # Valid code (from Numina)
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

    print("Testing Valid Code...")
    success, out = verifier.verify(valid_code)
    print(f"Success: {success}")
    if not success: print(out)

    # Invalid code
    invalid_code = r"""
import Mathlib
theorem failure_test : 1 = 0 := by
  simp
"""
    print("\nTesting Invalid Code...")
    success, out = verifier.verify(invalid_code)
    print(f"Success: {success}")
    if not success: print(f"Output (truncated): {out[:200]}...")
