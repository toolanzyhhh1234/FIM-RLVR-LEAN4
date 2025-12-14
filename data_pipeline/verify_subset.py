import pandas as pd
import sys
import os
import random

# Add project root to path
sys.path.append(os.getcwd())

from fim_rlvr_lean4.lean_verifier import LeanVerifier


def main():
    parquet_path = "data/NuminaMath-LEAN/data/train-00000-of-00001.parquet"
    print(f"Loading dataset from {parquet_path}...")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return

    print(f"Total rows: {len(df)}")

    # We expect columns 'formal_statement' and 'formal_proof' or similar.
    # Numina usually combines them or puts them in 'formal_statement' (imports+stm) and 'formal_proof' (proof body).
    # Let's inspect the columns.
    print(f"Columns: {df.columns.tolist()}")

    # Filter for non-empty formal_ground_truth
    df = df[df["formal_ground_truth"].notna() & (df["formal_ground_truth"] != "")]

    # Sample 20 theorems
    sample_size = 20
    subset = df.sample(sample_size, random_state=42)

    verifier = LeanVerifier("./verification_env")

    passed = 0
    attempted = 0

    print(f"\nStarting Verification Check on {sample_size} random samples...")
    print("=" * 60)

    for idx, row in subset.iterrows():
        # Prefer formal_ground_truth which usually contains the full compile-ready file
        full_code = row.get("formal_ground_truth", "")

        # Fallback (unlikely to work if proof is missing)
        if not full_code:
            stmt = row.get("formal_statement", "")
            proof = row.get("formal_proof", "")
            full_code = (
                stmt
                + ("" if ":= by" in stmt else " := by\n")
                + (proof if proof else " sorry")
            )

        # Basic cleanup: Ensure imports.
        # Numina usually has "import Mathlib" at the top.
        if "import Mathlib" not in full_code:
            full_code = "import Mathlib\n" + full_code

        attempted += 1
        success, out = verifier.verify(full_code)

        status = "PASS" if success else "FAIL"
        if success:
            passed += 1

        print(f"[{attempted}/{sample_size}] ID: {idx} | Status: {status}")
        if not success:
            # Print first line of error for debug
            err_line = out.split("\n")[0]
            if len(err_line) > 100:
                err_line = err_line[:100] + "..."
            print(f"   Error: {err_line}")

    print("=" * 60)
    print(
        f"Verification Summary: {passed}/{attempted} passed ({passed / attempted * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
