import json
import random
from typing import List, Tuple
from datasets import load_dataset


def get_tactic_blocks(proof_str: str) -> List[str]:
    """Split proof into lines, keeping non-empty ones."""
    lines = proof_str.split("\n")
    return [line for line in lines if line.strip()]


def join_lines(lines: List[str], trailing_newline: bool) -> str:
    if not lines:
        return ""
    s = "\n".join(lines)
    if trailing_newline:
        return s + "\n"
    return s


def generate_fim_sample(full_proof: str, ratio: float = 0.2) -> Tuple[str, str, str]:
    """Generate Prefix, Middle, Suffix from proof."""
    blocks = get_tactic_blocks(full_proof)
    if not blocks:
        return "", "", ""

    n_lines = len(blocks)
    k = max(1, int(n_lines * ratio))
    max_start = max(0, n_lines - k)
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + k

    prefix = join_lines(blocks[:start_idx], trailing_newline=bool(blocks[:start_idx]))
    middle = join_lines(blocks[start_idx:end_idx], trailing_newline=bool(blocks[start_idx:end_idx]))
    suffix = join_lines(blocks[end_idx:], trailing_newline=False)

    return prefix, middle, suffix


def main():
    output_path = "data/mvp_train.jsonl"
    target_samples = 5000  # Smaller for MVP
    
    print("Streaming NuminaMath dataset...")
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train', streaming=True)
    
    samples_written = 0
    
    with open(output_path, "w") as f:
        for example in ds:
            if samples_written >= target_samples:
                break
                
            # Filter for complete proofs
            if (example['ground_truth_type'] != 'complete' or 
                not example['formal_ground_truth'] or
                len(example['formal_ground_truth'].strip()) < 100):
                continue
            
            code = example['formal_ground_truth']
            
            # Extract proof body after ':= by'
            if ':= by' not in code:
                continue
                
            parts = code.split(':= by', 1)
            header = parts[0] + ':= by'
            proof_body = parts[1]
            
            # Generate FIM sample
            prefix_body, middle, suffix_body = generate_fim_sample(proof_body, ratio=0.15)
            
            if not middle.strip():
                continue
                
            full_prefix = header + "\n" + prefix_body
            full_suffix = suffix_body
            
            sample = {
                "prompt": f"<PFX>{full_prefix}<SFX>{full_suffix}<MID>",
                "completion": middle,
                "theorem_name": "unknown"
            }
            
            f.write(json.dumps(sample) + "\n")
            samples_written += 1
            
            if samples_written % 500 == 0:
                print(f"Generated {samples_written} samples...")
    
    print(f"Saved {samples_written} FIM samples to {output_path}")


if __name__ == "__main__":
    main()
