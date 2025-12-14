import json
import sys
import os

sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier


def parse_prompt(prompt, completion):
    try:
        pfx_start = prompt.find('<PFX>') + 5
        sfx_start = prompt.find('<SFX>')
        prefix = prompt[pfx_start:sfx_start]
        
        sfx_content_start = sfx_start + 5
        mid_start = prompt.find('<MID>')
        suffix = prompt[sfx_content_start:mid_start]
        
        return prefix + completion + suffix
    except:
        return ""


def main():
    print("Testing FIM reconstruction on existing data...")
    
    verifier = LeanVerifier("./verification_env")
    
    with open("data/mvp_train.jsonl", "r") as f:
        lines = f.readlines()[:5]  # Test first 5
    
    passed = 0
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        reconstructed = parse_prompt(data["prompt"], data["completion"])
        
        if "import Mathlib" not in reconstructed:
            reconstructed = "import Mathlib\n" + reconstructed
        
        print(f"\n=== FIM SAMPLE {i+1} ===")
        success, output = verifier.verify(reconstructed)
        
        if success:
            passed += 1
            print("✅ FIM reconstruction PASSED")
        else:
            print("❌ FIM reconstruction FAILED")
            # Show the problematic part
            error_lines = [line for line in output.split('\n') if 'error:' in line]
            if error_lines:
                print(f"Error: {error_lines[0][:100]}...")
            
            # Show reconstruction around the error
            print("Reconstructed code (first 400 chars):")
            print(reconstructed[:400])
    
    print(f"\n{'='*50}")
    print(f"FIM RECONSTRUCTION TEST: {passed}/{len(lines)} ({passed/len(lines)*100:.1f}%)")
    
    if passed >= len(lines) * 0.8:
        print("✅ FIM pipeline working - ready for training")
    else:
        print("⚠️  FIM reconstruction issues - need debugging")


if __name__ == "__main__":
    main()
