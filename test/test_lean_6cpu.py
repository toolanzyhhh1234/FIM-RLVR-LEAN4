import time
import sys
import os
import resource
import argparse
from concurrent.futures import ThreadPoolExecutor

# Add the current directory to path so we can import fim_rlvr_lean4
sys.path.append(os.getcwd())

from fim_rlvr_lean4.lean_verifier import LeanVerifier

# Fix PATH for Lean 4
elan_bin = os.path.expanduser("~/.elan/bin")
if elan_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{elan_bin}:{os.environ['PATH']}"

# Increase open file limit just in case
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
except:
    pass

# Valid code (from Numina)
# NOTE: One verification takes roughly 10-20 seconds on a standard core.
# We must be careful not to schedule too many sequential proofs or the benchmark will take forever.
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


def run_benchmark(verifier, workers, num_to_run):
    """
    Runs verification on a subset of samples using a specific number of workers.
    """
    print(f"Benchmarking with {workers} workers (Target: {num_to_run} proofs)...")

    start_time = time.time()
    completed = 0
    successes = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(verifier.verify, VALID_CODE) for _ in range(num_to_run)
        ]
        for f in futures:
            try:
                is_success, _ = f.result()
                if is_success:
                    successes += 1
                completed += 1
            except Exception as e:
                print(f"Error in verification: {e}")

    end_time = time.time()
    duration = end_time - start_time
    throughput = completed / duration if duration > 0 else 0

    print(
        f"  Done in {duration:.2f}s. Throughput: {throughput:.2f} proofs/sec. Success Rate: {successes}/{completed}"
    )
    return throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark Lean 4 verification")
    parser.add_argument(
        "--cores",
        type=int,
        default=6,
        help="Number of CPU cores available (default: 6)",
    )
    args = parser.parse_args()

    verifier = LeanVerifier("./verification_env")

    # NOTE: Verification is slow (~10-20s per proof).
    # We want quick feedback, so we reduce the counts.

    # Warmup
    print("Warming up (verifying 1 proofs sequential)...")
    try:
        verifier.verify(VALID_CODE)
    except:
        pass

    # worker_counts to test: [1, cores/2, cores, cores * 1.5]
    # We want to see if oversubscribing helps (IO wait) or hurts (CPU contention).
    worker_counts = sorted(
        list(set([1, max(2, args.cores // 2), args.cores, int(args.cores * 1.5)]))
    )

    results = {}

    print(f"\nStarting Benchmark (Cores={args.cores})...")
    print("-" * 75)
    print(
        f"{'Workers':<10} | {'Count':<6} | {'Throughput (proofs/s)':<25} | {'Speedup':<10}"
    )
    print("-" * 75)

    baseline_throughput = 0

    for w in worker_counts:
        # Heuristic:
        # For 1 worker: run 2 proofs (enough to get an average without waiting 40s)
        # For N workers: run roughly 2 * N proofs so each worker gets ~2 tasks.
        # Max cap 20 to avoid waiting too long during testing.
        num_to_run = min(24, max(2, w * 2))

        tp = run_benchmark(verifier, w, num_to_run)
        results[w] = tp

        if w == 1:
            baseline_throughput = tp
            speedup = 1.0
        else:
            speedup = tp / baseline_throughput if baseline_throughput > 0 else 0

        print(f"{w:<10} | {num_to_run:<6} | {tp:<25.2f} | {speedup:<10.2f}")

    print("-" * 75)
    best_workers = max(results, key=results.get)
    print(f"\nPeak throughput observed at {best_workers} workers.")


if __name__ == "__main__":
    main()
