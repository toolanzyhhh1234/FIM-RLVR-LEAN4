import pandas as pd
import json
import re
import random
from typing import List, Tuple


def get_tactic_blocks(proof_str: str) -> List[str]:
    """
    Heuristic splitter to approximate tactic blocks from a text proof.
    Splits on newlines but tries to keep 'by ...' or structured blocks together?

    MVP: Split by newline, but group indented lines with parent.
    Actually, Lean code is whitespace sensitive.
    Simple MVP: Split by line.
    """
    lines = proof_str.split("\n")
    # Filter empty lines
    return [l for l in lines if l.strip()]


def generate_fim_sample(full_proof: str, ratio: float = 0.2) -> Tuple[str, str, str]:
    """
    Generates Prefix, Middle, Suffix.
    middle is a contiguous block of lines.
    """
    blocks = get_tactic_blocks(full_proof)
    if not blocks:
        return "", "", ""

    n_lines = len(blocks)
    k = max(1, int(n_lines * ratio))

    # Pick start index
    # Don't pick 0 if possible (keep some prefix)
    max_start = max(0, n_lines - k)
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + k

    prefix = "\n".join(blocks[:start_idx])
    middle = "\n".join(blocks[start_idx:end_idx])
    suffix = "\n".join(blocks[end_idx:])

    return prefix, middle, suffix


def main():
    input_path = "data/NuminaMath-LEAN/data/train-00000-of-00001.parquet"
    output_path = "data/mvp_train.jsonl"

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)

    # Filter valid
    df = df[df["formal_ground_truth"].notna() & (df["formal_ground_truth"] != "")]

    print(f"Processing {len(df)} theorems...")

    samples = []

    for _, row in df.iterrows():
        # Source code
        code = row["formal_ground_truth"]

        # We need to extract the PROOF part for masking.
        # But for FIM, we mask the file.
        # Masking imports or theorem stmt is bad.
        # We should find ':= by' and only mask AFTER that.

        splitter = ":= by"
        if splitter not in code:
            continue

        parts = code.split(splitter, 1)
        header = parts[0] + splitter
        proof_body = parts[1]

        # Generate FIM on proof_body
        prefix_body, middle, suffix_body = generate_fim_sample(proof_body, ratio=0.15)

        if not middle.strip():
            continue

        full_prefix = header + prefix_body
        full_suffix = suffix_body

        # Format for SFT
        # Prompt: <PFX>{prefix}<SFX>{suffix}<MID>
        # Completion: {middle}

        samples.append(
            {
                "prompt": f"<PFX>{full_prefix}<SFX>{full_suffix}<MID>",
                "completion": middle,
                "theorem_name": "unknown",  # Todo: extract from header regex
            }
        )

    # Shuffle and save
    random.shuffle(samples)

    # Save a subset for MVP (e.g. 10k)
    subset = samples[:10000]

    print(f"Saving {len(subset)} samples to {output_path}...")
    with open(output_path, "w") as f:
        for s in subset:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
