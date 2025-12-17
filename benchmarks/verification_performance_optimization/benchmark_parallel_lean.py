import time
import sys
import os
import resource
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from fim_rlvr_lean4.lean_verifier import LeanVerifier
import numpy as np

# Fix PATH for Lean 4
elan_bin = os.path.expanduser("~/.elan/bin")
if elan_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{elan_bin}:{os.environ['PATH']}"


# Increase open file limit just in case
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))


def load_verification_samples(n=100):
    """
    Loads samples from AI-MO/NuminaMath-LEAN.
    Returns a list of code strings to verify.
    """
    print(f"Loading {n} samples from AI-MO/NuminaMath-LEAN...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)

    samples = []
    count = 0
    for item in ds:
        # Prefer formal_ground_truth as it usually contains the full code
        code = item.get("formal_ground_truth")
        if not code:
            # Fallback: combine statement and proof if ground truth missing
            stmt = item.get("formal_statement", "")
            proof = item.get("formal_proof", "")
            if stmt and proof:
                code = stmt + "\n" + proof

        if code and "import Mathlib" in code:
            samples.append(code)
            count += 1
            if count >= n:
                break

    print(f"Loaded {len(samples)} samples.")
    return samples


def run_benchmark(verifier, samples, workers, num_to_run=48):
    """
    Runs verification on a subset of samples using a specific number of workers.
    """
    # Simply cycle through samples if we need more than we have
    test_set = (samples * (num_to_run // len(samples) + 1))[:num_to_run]

    print(f"Benchmarking with {workers} workers (Target: {num_to_run} proofs)...")

    start_time = time.time()

    completed = 0
    successes = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verifier.verify, code) for code in test_set]
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
    throughput = completed / duration

    print(
        f"  Done in {duration:.2f}s. Throughput: {throughput:.2f} proofs/sec. Success Rate: {successes}/{completed}"
    )
    return throughput


def main():
    verifier = LeanVerifier("./verification_env")

    # 1. Load Data
    # Load enough samples to have variety
    samples = load_verification_samples(n=50)

    if not samples:
        print("Error: No samples loaded.")
        return

    # 2. Warmup
    print("Warming up (verifying 2 proofs sequential)...")
    try:
        verifier.verify(samples[0])
        verifier.verify(samples[1])
    except:
        pass

    # 3. Benchmark

    worker_counts = [1, 4, 8, 12, 16, 20, 24, 32]
    results = {}

    print("\nStarting Benchmark...")
    print("-" * 60)
    print(
        f"{'Workers':<10} | {'Count':<6} | {'Throughput (proofs/s)':<25} | {'Speedup':<10}"
    )
    print("-" * 60)

    baseline_throughput = 0

    for w in worker_counts:
        # User feedback: 1 proof is ~20s.
        # For small workers, keep count low to avoid long wait.
        # For large workers, increase count to ensure saturation.
        # Rule: Run at least 3 waves of proofs per worker count.
        num_to_run = max(w * 3, 5)

        tp = run_benchmark(verifier, samples, w, num_to_run=num_to_run)
        results[w] = tp

        if w == 1:
            baseline_throughput = tp
            speedup = 1.0
        else:
            speedup = tp / baseline_throughput

        print(f"{w:<10} | {num_to_run:<6} | {tp:<25.2f} | {speedup:<10.2f}")

    print("-" * 60)

    # Recommendation
    best_workers = max(results, key=results.get)
    print(f"\nPeak throughput observed at {best_workers} workers.")

    # Simple heuristic: 1 thread per proof is usually fine if IO bound (writing files, spawning process).
    # Since LeanVerifier spawns `lake env lean`, the CPU heavy lifting is in the subprocess.
    # The python part is just waiting. So we are limited by CPU cores available for the subprocesses.
    # If we have 24 physical cores (or vCPUs), we expect saturation around 24.


if __name__ == "__main__":
    main()
