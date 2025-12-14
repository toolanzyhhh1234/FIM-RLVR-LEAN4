import json
import random
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
    verifier = LeanVerifier("./verification_env")
    
    with open("data/fim_train_fixed.jsonl", "r") as f:
        lines = f.readlines()
    
    # Test random sample
    sample_lines = random.sample(lines, min(20, len(lines)))
    
    print(f"Testing {len(sample_lines)} fixed FIM samples...")
    
    passed = 0
    for i, line in enumerate(sample_lines):
        data = json.loads(line)
        full_code = parse_prompt(data["prompt"], data["completion"])
        
        if "import Mathlib" not in full_code:
            full_code = "import Mathlib\n" + full_code
        
        success, _ = verifier.verify(full_code)
        if success:
            passed += 1
            print(f"✓ {i+1}")
        else:
            print(f"✗ {i+1}")
    
    print(f"\nFixed FIM Success Rate: {passed}/{len(sample_lines)} ({passed/len(sample_lines)*100:.1f}%)")
    
    if passed >= len(sample_lines) * 0.8:
        print("✅ FIM pipeline fixed - ready for SFT training!")
    else:
        print("⚠️  Still need more fixes")


if __name__ == "__main__":
    main()
