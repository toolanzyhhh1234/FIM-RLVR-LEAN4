import sys
import os
from datasets import load_dataset

sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier


def main():
    print("Loading NuminaMath dataset (first 100 examples)...")
    
    # Load small subset efficiently 
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train', streaming=True)
    
    verifier = LeanVerifier("./verification_env")
    
    tested = 0
    passed = 0
    
    for example in ds:
        if tested >= 10:  # Test just 10 real examples
            break
            
        # Only test complete proofs
        if (example['ground_truth_type'] != 'complete' or 
            not example['formal_ground_truth'] or
            len(example['formal_ground_truth'].strip()) < 50):
            continue
            
        proof = example['formal_ground_truth']
        tested += 1
        
        print(f"\n=== TESTING REAL EXAMPLE {tested} ===")
        print(f"Problem: {example['problem'][:100]}...")
        print(f"Proof length: {len(proof)} chars")
        
        success, output = verifier.verify(proof)
        
        if success:
            passed += 1
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            error_lines = [line for line in output.split('\n') if 'error:' in line]
            if error_lines:
                print(f"Error: {error_lines[0][:150]}...")
    
    print(f"\n{'='*50}")
    print(f"REAL NUMINA DATA TEST: {passed}/{tested} ({passed/tested*100:.1f}%)")
    
    if passed >= tested * 0.8:
        print("✅ NuminaMath dataset is compatible - proceed with FIM")
    else:
        print("⚠️  Dataset compatibility issues - need to investigate")
        print("Common issues: Lean version mismatch, missing imports, syntax changes")


if __name__ == "__main__":
    main()
