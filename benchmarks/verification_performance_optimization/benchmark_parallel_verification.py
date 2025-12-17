import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from fim_rlvr_lean4.lean_verifier import LeanVerifier

# Valid code (from Numina) - takes a few seconds to verify usually due to imports
VALID_CODE = r"""
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


def run_sequential(verifier, n):
    print(f"Running {n} sequential verifications...")
    start = time.time()
    for i in range(n):
        print(f"  Seq [{i + 1}/{n}] starting...")
        success, _ = verifier.verify(VALID_CODE)
        print(f"  Seq [{i + 1}/{n}] done. Success: {success}")
    end = time.time()
    return end - start


def run_parallel(verifier, n, workers):
    print(f"Running {n} parallel verifications with {workers} workers...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verifier.verify, VALID_CODE) for _ in range(n)]
        for i, f in enumerate(futures):
            success, _ = f.result()
            print(f"  Par [{i + 1}/{n}] done. Success: {success}")
    end = time.time()
    return end - start


def main():
    verifier = LeanVerifier("./verification_env")

    # Warmup
    print("Warming up (1 run)...")
    verifier.verify(VALID_CODE)

    N = 2

    # Sequential
    time_seq = run_sequential(verifier, N)
    print(f"Sequential Time: {time_seq:.2f}s")

    # Parallel
    time_par = run_parallel(verifier, N, workers=2)
    print(f"Parallel Time: {time_par:.2f}s")

    speedup = time_seq / time_par
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.2:
        print("Parallelism IS effective.")
    else:
        print("Parallelism IS NOT effective (or overhead is too high).")


if __name__ == "__main__":
    main()
