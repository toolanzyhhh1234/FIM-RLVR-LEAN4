import json
import os
import sys
import gc
from datasets import load_dataset
from tqdm import tqdm

# Ensure we can import from local modules
sys.path.append(os.getcwd())
from data_pipeline.fim_exact_line import generate_proof_only_exact_fim


def count_lines(s: str) -> int:
    return len(s.splitlines())


def main():
    output_path = "data/fim_fresh.jsonl"
    target_samples = 1000

    print(f"Generating {target_samples} samples of FIM data...")
    
    samples_generated = 0
    
    try:
        ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)
        
        with open(output_path, "w") as f:
            for example in tqdm(ds):
                if samples_generated >= target_samples:
                    break

                # Basic validation
                if (
                    example.get("ground_truth_type") != "complete"
                    or not example.get("formal_ground_truth")
                    or len(example["formal_ground_truth"].strip()) < 100
                ):
                    continue

                code = example["formal_ground_truth"]

                # Generate FIM split
                prefix, middle, suffix = generate_proof_only_exact_fim(code, ratio=0.15)

                if not middle or not prefix:
                    continue

                # Create FIM prompt format
                prompt_text = f"<PFX>{prefix}<SFX>{suffix}<MID>"
                completion_text = middle

                sample = {
                    "prompt": prompt_text,
                    "completion": completion_text,
                    "metadata": {
                        "source": "NuminaMath-LEAN",
                        "missing_lines": count_lines(middle),
                        "theorem_name": example.get("name", "unknown"),
                    },
                }

                f.write(json.dumps(sample) + "\n")
                samples_generated += 1

        print(f"Successfully generated {samples_generated} samples to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Force cleanup
        gc.collect()
        os._exit(0)  # Force exit to avoid threading cleanup issues


if __name__ == "__main__":
    main()
