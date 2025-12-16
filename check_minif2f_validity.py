import re
import os
import subprocess
import concurrent.futures
import random
import threading
from tqdm import tqdm

# Configuration
TEST_FILE = (
    "/home/admin1/CodeProjects/FIM-RLVR-LEAN4/benchmarks/miniF2F/formal/test.lean"
)
VERIFICATION_DIR = "/home/admin1/CodeProjects/FIM-RLVR-LEAN4/verification_env"
SAMPLE_SIZE = 5


def parse_theorems(file_path):
    """
    Parses the Lean file and extracts individual theorems.
    """
    with open(file_path, "r") as f:
        content = f.read()

    chunks = re.split(r"\n(?=theorem )", content)
    header = chunks[0]
    theorems = chunks[1:]
    return header, theorems


def verify_theorem(header, theorem_code):
    """
    Wraps the theorem in a unique file and runs lake build to check validity.
    """
    # Create a unique filename for this thread/check
    valid_id = threading.get_ident()
    temp_file_name = f"TestTheorem_{valid_id}_{random.randint(0, 10000)}.lean"
    temp_lean_file = os.path.join(VERIFICATION_DIR, temp_file_name)

    full_content = f"{header}\n\n{theorem_code}\n"

    with open(temp_lean_file, "w") as f:
        f.write(full_content)

    try:
        # Run lake env lean on the specific file
        result = subprocess.run(
            ["lake", "env", "lean", temp_file_name],
            cwd=VERIFICATION_DIR,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0, result.stderr
    finally:
        # Cleanup
        if os.path.exists(temp_lean_file):
            os.remove(temp_lean_file)


def main():
    print(f"Parsing {TEST_FILE}...")
    header, theorems = parse_theorems(TEST_FILE)
    print(f"Found {len(theorems)} theorems.")

    # Random sample
    sampled_theorems = random.sample(theorems, min(len(theorems), SAMPLE_SIZE))
    print(f"Randomly sampled {len(sampled_theorems)} theorems to check.")

    valid_count = 0
    invalid_count = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2
    ) as executor:  # Lower workers for CPU check
        future_to_thm = {
            executor.submit(verify_theorem, header, thm): thm
            for thm in sampled_theorems
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_thm), total=len(sampled_theorems)
        ):
            thm = future_to_thm[future]
            is_valid, error_msg = future.result()

            thm_name = thm.split()[1] if len(thm.split()) > 1 else "Unknown"
            if is_valid:
                valid_count += 1
                print(f"[PASS] {thm_name}")
            else:
                invalid_count += 1
                print(f"[FAIL] {thm_name}")
                # print(f"Error: {error_msg[:200]}...")

    print(f"\nSummary:")
    print(f"Valid Statements: {valid_count}")
    print(f"Invalid Statements: {invalid_count}")
    print(f"Validity Rate: {valid_count / len(sampled_theorems) * 100:.2f}%")


if __name__ == "__main__":
    main()
