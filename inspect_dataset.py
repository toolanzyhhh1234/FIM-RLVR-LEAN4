from datasets import load_dataset
import pandas as pd


def inspect():
    print("Loading dataset AI-MO/NuminaMath-LEAN...")
    # Try loading a small streaming sample first to be fast
    try:
        ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)
        print("Dataset loaded (streaming).")

        print("\n--- First 3 Examples ---")
        msg_count = 0
        for sample in ds:
            print(f"\n[Example {msg_count}] Keys: {list(sample.keys())}")
            # Print a bit of content to identify the proof field
            for k, v in sample.items():
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"  {k}: {val_str}")

            msg_count += 1
            if msg_count >= 3:
                break

    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    inspect()
