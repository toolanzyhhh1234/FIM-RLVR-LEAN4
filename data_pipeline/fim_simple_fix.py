import json
import random
from datasets import load_dataset


def generate_simple_fim(full_proof: str, ratio: float = 0.2) -> tuple:
    """Simple line-based FIM that ensures clean boundaries."""
    lines = full_proof.split("\n")
    if len(lines) < 5:
        return "", "", ""
    
    # Skip first/last lines to avoid header/footer issues
    n_lines = len(lines) - 2
    k = max(1, int(n_lines * ratio))
    
    # Random start position (skip first line)
    start_idx = random.randint(1, max(1, len(lines) - k - 1))
    end_idx = start_idx + k
    
    prefix = "\n".join(lines[:start_idx])
    middle = "\n".join(lines[start_idx:end_idx])  
    suffix = "\n".join(lines[end_idx:])
    
    return prefix, middle, suffix


def main():
    print("Generating simple line-based FIM data...")
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train', streaming=True)
    
    samples = []
    processed = 0
    
    for example in ds:
        if len(samples) >= 1000:  # Smaller test set
            break
            
        if (example['ground_truth_type'] != 'complete' or 
            not example['formal_ground_truth'] or
            len(example['formal_ground_truth'].strip()) < 200):
            continue
        
        code = example['formal_ground_truth']
        
        # Generate FIM sample on entire proof
        prefix, middle, suffix = generate_simple_fim(code, ratio=0.15)
        
        if not middle.strip() or len(middle.split('\n')) < 1:
            continue
            
        sample = {
            "prompt": f"<PFX>{prefix}<SFX>{suffix}<MID>",
            "completion": middle,
            "theorem_name": "unknown"
        }
        
        samples.append(sample)
        processed += 1
        
        if processed % 200 == 0:
            print(f"Processed {processed} examples, generated {len(samples)} samples")
    
    # Save
    with open("data/fim_simple.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Generated {len(samples)} simple FIM samples")


if __name__ == "__main__":
    main()
