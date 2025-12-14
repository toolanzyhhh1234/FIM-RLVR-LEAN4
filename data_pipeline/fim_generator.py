import json
from datasets import load_dataset
from data_pipeline.fim_exact_line import generate_proof_only_exact_fim


def count_lines(s: str) -> int:
    return len(s.splitlines())


def main() -> None:
    output_path = "data/instruction_tuning_proof_exact.jsonl"
    target_samples = 5000

    print("Generating instruction-tuning proof data (exact line extraction)...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)

    samples_written = 0

    # System prompt to encourage validity over strict line matching
    system_instruction = (
        "You are a Lean 4 expert. Your task is to complete the missing part of the proof. "
        "The missing block is indicated by [MISSING_BLOCK: <approx_lines> lines]. "
        "The line count is a hint; you may use any number of lines as long as the proof remains valid and correct. "
        "Output ONLY the code that belongs in the missing block. Do not repeat the surrounding code."
    )

    with open(output_path, "w") as f:
        for example in ds:
            if samples_written >= target_samples:
                break

            if (
                example.get("ground_truth_type") != "complete"
                or not example.get("formal_ground_truth")
                or len(example["formal_ground_truth"].strip()) < 100
            ):
                continue

            code = example["formal_ground_truth"]

            prefix, middle, suffix = generate_proof_only_exact_fim(code, ratio=0.15)
            if not middle:
                continue

            missing_lines = count_lines(middle)

            # Construct the user user prompt with the placeholder
            # We ensure exact reconstruction is possible by just concatenating prefix + middle + suffix
            # But for the prompt, we insert the tag.
            user_content = f"{prefix}[MISSING_BLOCK: {missing_lines} lines]\n{suffix}"

            # Chat format for SFT
            sample = {
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": middle},
                ],
                # Metadata useful for debugging/splitting
                "metadata": {"missing_lines": missing_lines, "theorem_name": "unknown"},
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples_written += 1

            if samples_written % 500 == 0:
                print(f"Generated {samples_written} samples...")

    print(f"Saved {samples_written} samples to {output_path}")


if __name__ == "__main__":
    main()
