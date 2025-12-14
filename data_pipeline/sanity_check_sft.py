import json
import random
import sys
import os

sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier


def parse_prompt(prompt, completion):
    # Prompt: <PFX>{prefix}<SFX>{suffix}<MID>
    # Reconstruct: prefix + completion + suffix

    # Simple parsing assuming strict tags
    try:
        pfx_part = prompt.split("<PFX>")[1].split("<SFX>")[0]
        sfx_part = prompt.split("<SFX>")[1].split("<MID>")[0]
        return pfx_part + completion + sfx_part
    except:
        return ""


def main():
    verifier = LeanVerifier("./verification_env")

    with open("data/mvp_train.jsonl", "r") as f:
        lines = f.readlines()

    sample = random.sample(lines, 20)

    print(f"Sanity checking {len(sample)} SFT samples...")

    passed = 0
    for line in sample:
        data = json.loads(line)
        full_code = parse_prompt(data["prompt"], data["completion"])

        # Numina clean up (ensure import mathlib if lost)
        if "import Mathlib" not in full_code:
            full_code = "import Mathlib\n" + full_code

        success, _ = verifier.verify(full_code)
        if success:
            passed += 1
        else:
            print("Failed reconstruction!")

    print(
        f"Sanity Pass Rate: {passed}/{len(sample)} ({passed / len(sample) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
