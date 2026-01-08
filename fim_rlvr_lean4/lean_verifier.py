import subprocess
import os
import re
from typing import Tuple


class LeanVerifier:
    def __init__(self, project_dir: str, *, no_sorries: bool = False):
        """
        Initializes the LeanVerifier.

        Args:
            project_dir: Path to the Lean 4 project directory (should contain lakefile.lean).
            no_sorries: If True, fail verification when the code contains `sorry` or `admit`.
                       This is checked at the Python level since Lean 4.15+ doesn't have
                       a --no-sorries flag.
        """
        self.project_dir = os.path.abspath(project_dir)
        self.no_sorries = bool(no_sorries)
        import uuid

        self.uuid_module = uuid  # Keep reference

    def _contains_sorry_or_admit(self, code: str) -> bool:
        """
        Check if code contains sorry or admit keywords.
        
        Uses word boundary matching to avoid false positives like
        variable names containing 'sorry'.
        """
        return bool(re.search(r'\bsorry\b', code) or re.search(r'\badmit\b', code))

    def verify(self, full_code: str) -> Tuple[bool, str]:
        """
        Verifies a Lean 4 proof by compiling it.
        Supports parallel execution by using unique temporary files.

        Args:
            full_code: The complete Lean source code (imports + theorem + proof).

        Returns:
            (success, output):
                success: True if compilation passed, False otherwise.
                output: The stderr output from the compiler (contains error messages).
        """
        # Check for sorry/admit if no_sorries is enabled
        if self.no_sorries and self._contains_sorry_or_admit(full_code):
            return False, "Verification failed: code contains 'sorry' or 'admit' (no_sorries=True)"
        
        # Generate unique ID for this verification task
        unique_id = str(self.uuid_module.uuid4())
        filename = f"Verify_{unique_id}.lean"
        # We put it in VerificationEnv folder to be safe with any relative assumptions, though not strictly required
        file_path = os.path.join(self.project_dir, "VerificationEnv", filename)

        # 1. Write code to the unique test file
        try:
            with open(file_path, "w") as f:
                f.write(full_code)
        except IOError as e:
            return False, f"Failed to write to verification file: {e}"

        # 2. Run 'lake env lean <file>'
        # This parses and checks the file using the project's environment (imports etc.)
        # but does not require the file to be listed in lakefile.lean.
        # Note: --no-sorries flag was removed as it's not supported in Lean 4.15+
        # We check for sorry/admit at the Python level instead.
        try:
            cmd = ["lake", "env", "lean", file_path]
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            # Clean up
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return False, "Error: 'lake' command not found. Is Lean 4 installed?"

        # 3. Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

        # 4. Check result
        success = result.returncode == 0
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
    if not success:
        print(out)

    # Invalid code
    invalid_code = r"""
import Mathlib
theorem failure_test : 1 = 0 := by
  simp
"""
    print("\nTesting Invalid Code...")
    success, out = verifier.verify(invalid_code)
    print(f"Success: {success}")
    if not success:
        print(f"Output (truncated): {out[:200]}...")
