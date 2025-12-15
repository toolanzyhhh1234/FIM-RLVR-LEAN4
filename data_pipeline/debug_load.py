import json
from datasets import load_dataset

try:
    print("Loading dataset...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)
    print("Dataset loaded. Fetching one item...")
    item = next(iter(ds))
    print("Item fetched successfully.")
    print(f"Item keys: {item.keys()}")
except Exception as e:
    print(f"Error: {e}")
