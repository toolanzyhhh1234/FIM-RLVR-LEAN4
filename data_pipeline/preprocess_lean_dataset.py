#!/usr/bin/env python3
"""
Preprocess NuminaMath-LEAN dataset for FIM training.

Filters out invalid samples and saves a clean parquet file for faster loading.

Usage:
    python data_pipeline/preprocess_lean_dataset.py

Output:
    data/NuminaMath-LEAN-filtered.parquet
"""

import os
import re
import polars as pl
from pathlib import Path


def is_valid_lean_sample(text: str, exclude_sorry: bool = True) -> bool:
    """
    Check if a Lean code sample is valid for training.
    
    Filters out:
    - Samples shorter than 50 characters
    - Samples containing 'sorry' or 'admit' (if exclude_sorry=True)
    - Samples without 'theorem', 'lemma', or 'def' keywords
    """
    if not text or len(text.strip()) < 50:
        return False
    
    if exclude_sorry:
        if re.search(r'\bsorry\b', text) or re.search(r'\badmit\b', text):
            return False
    
    return ("theorem" in text) or ("lemma" in text) or ("def" in text)


def preprocess_dataset(
    input_path: str,
    output_path: str,
    exclude_sorry: bool = True,
) -> dict:
    """
    Preprocess the dataset and save filtered version.
    
    Args:
        input_path: Path to input parquet file.
        output_path: Path to output filtered parquet file.
        exclude_sorry: Whether to exclude samples with sorry/admit.
    
    Returns:
        Statistics dict with original_count, filtered_count, etc.
    """
    print(f"Loading dataset from {input_path}...")
    df = pl.read_parquet(input_path)
    original_count = len(df)
    print(f"Original dataset: {original_count} samples")
    
    # Find the text column
    text_col = None
    for col_name in ["formal_ground_truth", "full_code", "code", "prompt"]:
        if col_name in df.columns:
            text_col = col_name
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {df.columns}")
    
    print(f"Using text column: {text_col}")
    
    # Filter rows
    print("Filtering samples...")
    valid_mask = []
    for row in df.iter_rows(named=True):
        text = row.get(text_col, "")
        valid_mask.append(is_valid_lean_sample(text, exclude_sorry=exclude_sorry))
    
    df_filtered = df.filter(pl.Series(valid_mask))
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    print(f"Filtered dataset: {filtered_count} samples ({removed_count} removed)")
    
    # Save filtered dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    df_filtered.write_parquet(output_path)
    
    # Verify saved file
    df_verify = pl.read_parquet(output_path)
    print(f"Verified: {len(df_verify)} samples in output file")
    
    return {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": removed_count,
        "exclude_sorry": exclude_sorry,
        "input_path": input_path,
        "output_path": output_path,
    }


def main():
    # Default paths
    input_path = "data/NuminaMath-LEAN/data/train-00000-of-00001.parquet"
    output_path = "data/NuminaMath-LEAN-filtered.parquet"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please download the dataset first.")
        return 1
    
    # Run preprocessing
    stats = preprocess_dataset(input_path, output_path, exclude_sorry=True)
    
    print("\n=== Summary ===")
    print(f"Original: {stats['original_count']} samples")
    print(f"Filtered: {stats['filtered_count']} samples")
    print(f"Removed:  {stats['removed_count']} samples")
    print(f"Output:   {stats['output_path']}")
    print("\nTo use the filtered dataset, update your config:")
    print(f"  dataset_path: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
