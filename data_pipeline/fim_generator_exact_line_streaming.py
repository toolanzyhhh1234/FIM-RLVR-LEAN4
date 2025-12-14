import json
from datasets import load_dataset

from data_pipeline.fim_exact_line import generate_proof_only_exact_fim


def main() -> None:
    output_path = "data/fim_proof_only_exact.jsonl"
    target_samples = 5000

    print("Generating exact line-by-line proof-only FIM data (no trimming)...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)

    samples_written = 0

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

            sample = {
                "prompt": f"<PFX>{prefix}<SFX>{suffix}<MID>",
                "completion": middle,
                "theorem_name": "unknown",
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples_written += 1

            if samples_written % 500 == 0:
                print(f"Generated {samples_written} samples...")

    print(f"Saved {samples_written} samples to {output_path}")


if __name__ == "__main__":
    main()
