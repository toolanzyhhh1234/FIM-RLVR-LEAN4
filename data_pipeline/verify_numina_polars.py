import polars as pl
import random
import sys
import os
from datasets import load_dataset

sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier


def main():
    print("Loading NuminaMath dataset...")
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train')
    df = pl.from_pandas(ds.to_pandas())
    
    # Filter complete proofs
    complete_df = df.filter(
        (pl.col('ground_truth_type') == 'complete') &
        (pl.col('formal_ground_truth').is_not_null()) &
        (pl.col('formal_ground_truth').str.len_chars() > 100)
    )
    
    print(f"Found {len(complete_df)} complete proofs")
    
    # Sample 20 for quick verification
    sample_df = complete_df.sample(n=min(20, len(complete_df)), seed=42)
    
    verifier = LeanVerifier("./verification_env")
    passed = 0
    
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        proof = row['formal_ground_truth']
        success, output = verifier.verify(proof)
        
        if success:
            passed += 1
            print(f"✓ {i+1}")
        else:
            error_msg = output.split('error:')[1][:80] if 'error:' in output else 'Unknown error'
            print(f"✗ {i+1}: {error_msg}...")
    
    print(f"\nNuminaMath Verification: {passed}/20 ({passed/20*100:.1f}%)")
    
    if passed >= 16:  # 80%+
        print("✅ Data quality looks good - proceed with FIM generation")
    else:
        print("⚠️  Low pass rate - check Lean environment")


if __name__ == "__main__":
    main()
