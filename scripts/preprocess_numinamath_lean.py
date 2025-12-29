#!/usr/bin/env python3
"""
Pre-process NuminaMath-LEAN dataset for verl training.
Converts raw problem text into chat-formatted prompts.
"""

import pandas as pd
from datasets import load_dataset
import os

def main():
    print("Loading NuminaMath-LEAN dataset...")
    dataset = load_dataset("AI-MO/NuminaMath-LEAN", split="train")

    print(f"Loaded {len(dataset)} examples")

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    # Create chat-formatted prompts
    # The model should generate a Lean 4 proof given the problem and formal statement
    def create_prompt(row):
        problem = row.get('problem', '')
        formal_statement = row.get('formal_statement', '')

        # Create a chat message format
        prompt = [
            {
                "role": "user",
                "content": f"""Please provide a Lean 4 proof for the following mathematical problem.

**Problem:**
{problem}

**Formal Statement:**
```lean4
{formal_statement}
```

Please write a complete Lean 4 proof that compiles successfully."""
            }
        ]
        return prompt

    print("Converting to chat format...")
    df['prompt'] = df.apply(create_prompt, axis=1)

    # Keep relevant columns and add data_source for reward function selection
    df['data_source'] = 'lean_verifier'

    # Keep the formal_statement and formal_proof for verification
    # The reward manager will need access to these
    columns_to_keep = [
        'prompt', 'data_source',
        'problem', 'formal_statement', 'formal_proof', 'formal_ground_truth',
        'uuid', 'answer', 'source', 'problem_type'
    ]

    # Only keep columns that exist
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    df = df[columns_to_keep]

    # Create output directory
    output_dir = os.path.expanduser("~/data/numinamath_lean")
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/val (90/10)
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Save as parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    print(f"Saving {len(train_df)} train examples to {train_path}")
    train_df.to_parquet(train_path, index=False)

    print(f"Saving {len(val_df)} val examples to {val_path}")
    val_df.to_parquet(val_path, index=False)

    print("Done!")
    print(f"\nTo use in training, set:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={val_path}")

if __name__ == "__main__":
    main()
