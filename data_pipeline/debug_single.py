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
    verifier = LeanVerifier("./verification_env")
    
    with open("data/mvp_train.jsonl", "r") as f:
        lines = f.readlines()
    
    # Test first failing case
    for i, line in enumerate(lines[:5]):
        data = json.loads(line)
        full_code = parse_prompt(data["prompt"], data["completion"])
        
        if "import Mathlib" not in full_code:
            full_code = "import Mathlib\n" + full_code
        
        print(f"\n=== SAMPLE {i+1} ===")
        print("RECONSTRUCTED CODE:")
        print(full_code[:800])
        print("\n" + "="*50)
        
        success, output = verifier.verify(full_code)
        print(f"SUCCESS: {success}")
        
        if not success:
            print("ERROR OUTPUT:")
            print(output[:600])
            break  # Stop at first failure to debug
        else:
            print("✅ This one passed")


if __name__ == "__main__":
    main()
