import json
import random
from typing import List, Tuple
from datasets import load_dataset


def get_proof_lines(proof_str: str) -> List[str]:
    """Split proof into lines, preserving empty lines for structure."""
    return proof_str.split("\n")


def is_safe_split_point(lines: List[str], idx: int) -> bool:
    """Check if it's safe to split at this line index."""
    if idx == 0 or idx >= len(lines):
        return False
    
    current_line = lines[idx].strip()
    prev_line = lines[idx-1].strip() if idx > 0 else ""
    
    # Don't split if previous line ends with incomplete syntax
    if prev_line.endswith(('by', ':=', ':', 'have', 'let')):
        return False
    
    # Don't split on empty lines or comments
    if not current_line or current_line.startswith('--'):
        return False
        
    return True


def generate_fim_sample(full_proof: str, ratio: float = 0.2) -> Tuple[str, str, str]:
    """Generate FIM sample with safe line-level splits."""
    lines = get_proof_lines(full_proof)
    if len(lines) < 3:
        return "", "", ""
    
    # Find safe split points
    safe_points = [i for i in range(1, len(lines)) if is_safe_split_point(lines, i)]
    if len(safe_points) < 2:
        return "", "", ""
    
    # Calculate hole size
    k = max(1, int(len(lines) * ratio))
    k = min(k, len(safe_points) - 1)
    
    # Pick random start point
    max_start_idx = len(safe_points) - k
    start_point_idx = random.randint(0, max_start_idx)
    
    start_line = safe_points[start_point_idx]
    end_line = min(start_line + k, len(lines))
    
    prefix = "\n".join(lines[:start_line])
    middle = "\n".join(lines[start_line:end_line])
    suffix = "\n".join(lines[end_line:])
    
    return prefix, middle, suffix


def main():
    output_path = "data/fim_train_fixed.jsonl"
    target_samples = 3000
    
    print("Generating FIM data with safe line splits...")
    ds = load_dataset('AI-MO/NuminaMath-LEAN', split='train', streaming=True)
    
    samples_written = 0
    
    with open(output_path, "w") as f:
        for example in ds:
            if samples_written >= target_samples:
                break
                
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
            
            if not middle.strip() or len(middle.split('\n')) < 2:
                continue
                
            full_prefix = header + prefix_body
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
    
    print(f"Generated {samples_written} FIM samples with safe splits")


if __name__ == "__main__":
    main()
