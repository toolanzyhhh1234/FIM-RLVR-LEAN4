import argparse
import json
import os
import time
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

# Configuration
INPUT_FILE = "data/instruction_tuning_proof_exact.jsonl"
OUTPUT_FILE = "data/fim_sft_thought_augmented.jsonl"
MODEL_NAME = "deepseek-reasoner"


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please export your API key in .env: DEEPSEEK_API_KEY=...")
        exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def generate_thought_trace(
    client: OpenAI, fim_prompt: str, gold_completion: str, system_prompt: str
) -> tuple[Optional[str], Optional[str], float]:
    """
    Calls DeepSeek Reasoner.
    Returns (thoughts, raw_content, latency_seconds).
    """

    rationalization_user_prompt = (
        f"{fim_prompt}\n\n"
        f"The correct code for the [MISSING_BLOCK] is:\n```lean\n{gold_completion}\n```\n"
        "Please provide the step-by-step reasoning that leads to this solution."
    )

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rationalization_user_prompt},
            ],
            stream=False,
        )
        latency = time.time() - start_time

        # Accessing the raw dictionary from the PyPydantic model
        msg = response.choices[0].message

        # Check standard fields first
        thoughts = getattr(msg, "reasoning_content", None)

        # Helper to get the full raw content (thoughts or regular content) just in case
        raw_content = thoughts if thoughts else msg.content

        return thoughts if thoughts else msg.content, raw_content, latency

    except Exception as e:
        print(f"DeepSeek API Exception: {e}")
        return None, None, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic thoughts using DeepSeek Reasoner."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of new samples to generate in this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep time (seconds) between API calls.",
    )
    args = parser.parse_args()

    client = get_client()

    print(f"Reading input from {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} total input samples.")

    # Check if output exists to resume
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            processed_count = len(f.readlines())
            print(f"Resuming from {processed_count} samples...")

    # Calculate how many to process in this run
    start_idx = processed_count
    end_idx = min(len(lines), start_idx + args.limit)
    print(
        f"Processing batch of {end_idx - start_idx} samples (from {start_idx} to {end_idx})..."
    )

    # Open for appending
    with open(OUTPUT_FILE, "a") as f_out:
        for i in tqdm(range(start_idx, end_idx)):
            line = lines[i]
            data = json.loads(line)
            messages = data["messages"]

            # Extract components from the stored format
            system_prompt = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_prompt = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            gold_completion = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )

            # 1. Generate Thought
            thoughts, raw_response, latency = generate_thought_trace(
                client, user_prompt, gold_completion, system_prompt
            )

            if not thoughts:
                print(f"Skipping sample {i} due to API failure")
                continue

            # 2. Assemble Final Target: <think>...</think>\nGoldCode
            # DeepSeek Reasoner output is just the thought content string
            final_content = f"<think>\n{thoughts}\n</think>\n{gold_completion}"

            # 3. Save
            meta = data.get("metadata", {})
            meta["latency"] = latency
            # meta["thoughts_source"] = "deepseek-reasoner"

            new_sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": final_content},
                ],
                "metadata": meta,
                "raw_api_response": raw_response,
            }

            f_out.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            f_out.flush()  # Ensure we save progress

            # Rate limiting / Be nice to free tier
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
