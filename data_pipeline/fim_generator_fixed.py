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

    # Heuristic: avoid splitting where indentation level changes abruptly.
    # This reduces odds of cutting between a tactic header and its sub-block.
    def indent(s: str) -> int:
        return len(s) - len(s.lstrip(' '))

    prev_raw = lines[idx - 1]
    cur_raw = lines[idx]
    if indent(cur_raw) > indent(prev_raw) and prev_line.endswith((':', '=>', 'do', 'by')):
        return False
        
    return True


def join_lines(lines: List[str], trailing_newline: bool) -> str:
    if not lines:
        return ""
    s = "\n".join(lines)
    if trailing_newline:
        return s + "\n"
    return s


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
    
    # Choose both start/end from safe split points so we don't end the hole mid-block.
    start_idx = random.randint(0, len(safe_points) - 2)
    max_end_idx = min(len(safe_points) - 1, start_idx + max(1, k))
    end_idx = random.randint(start_idx + 1, max_end_idx)
    
    start_line = safe_points[start_idx]
    end_line = safe_points[end_idx]

    prefix = join_lines(lines[:start_line], trailing_newline=bool(lines[:start_line]))
    middle = join_lines(lines[start_line:end_line], trailing_newline=bool(lines[start_line:end_line]))
    suffix = join_lines(lines[end_line:], trailing_newline=False)
    
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
    
    print(f"Generated {samples_written} FIM samples with safe splits")


if __name__ == "__main__":
    main()
