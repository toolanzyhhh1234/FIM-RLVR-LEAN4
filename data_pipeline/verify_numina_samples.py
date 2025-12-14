import json
import random
import sys
import os
from datasets import load_dataset

sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier


def main():
    print("Loading NuminaMath dataset (streaming)...")
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train', streaming=True)
    
    # Take first 50 complete proofs
    complete_proofs = []
    for example in ds:
        if (example['ground_truth_type'] == 'complete' and 
            example['formal_ground_truth'] and 
            len(example['formal_ground_truth'].strip()) > 100):
            complete_proofs.append(example['formal_ground_truth'])
            if len(complete_proofs) >= 50:
                break
    
    print(f"Testing {len(complete_proofs)} complete proofs from NuminaMath...")
    
    verifier = LeanVerifier("./verification_env")
    passed = 0
    
    for i, proof in enumerate(complete_proofs):
        success, output = verifier.verify(proof)
        if success:
            passed += 1
            print(f"✓ {i+1}")
        else:
            print(f"✗ {i+1}: {output.split('error:')[1][:100] if 'error:' in output else 'Unknown error'}...")
    
    print(f"\nNuminaMath Verification Rate: {passed}/{len(complete_proofs)} ({passed/len(complete_proofs)*100:.1f}%)")
    
    if passed < len(complete_proofs) * 0.8:
        print("⚠️  Low pass rate - check Lean environment compatibility")
    else:
        print("✅ NuminaMath data is compatible with your Lean environment")


if __name__ == "__main__":
    main()
