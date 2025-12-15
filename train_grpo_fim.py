# train_grpo_fim.py
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch
import os
import sys

# Ensure we can import from local modules
sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier

# Configuration
MAX_SEQ_LENGTH = 1024
LORA_RANK = 4
MODEL_NAME = "unsloth/gpt-oss-20b"
# MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # Tiny model for local debug
OUTPUT_DIR = "outputs_fim_grpo"
DATA_FILE = "data/fim_fresh.jsonl"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        offload_embedding=True,
    )

    # ADD LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load Dataset
    print(f"Loading dataset from {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # Initialize Verifier
    verifier = LeanVerifier("./verification_env")

    # --- Preprocessing: Convert <PFX> format to Chat Instruction ---
    def process_fim_sample(sample):
        # Parse the raw PFX format we stored
        raw_prompt = sample["prompt"]
        try:
            pfx_idx = raw_prompt.find("<PFX>")
            sfx_idx = raw_prompt.find("<SFX>")
            mid_idx = raw_prompt.find("<MID>")

            if pfx_idx != -1 and sfx_idx != -1 and mid_idx != -1:
                prefix = raw_prompt[pfx_idx + 5 : sfx_idx]
                suffix = raw_prompt[sfx_idx + 5 : mid_idx]
            else:
                # Fallback if malformed
                prefix = ""
                suffix = ""
        except:
            prefix = ""
            suffix = ""

        # Construct a User-facing instruction
        # We use a clear marker for the model to see where the hole is.
        user_content = f"{prefix}[MISSING_BLOCK]\n{suffix}"

        messages = [
            {
                "role": "system",
                "content": "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. Output ONLY the missing code.",
            },
            {"role": "user", "content": user_content},
        ]

        # Apply chat template to get the actual text prompt for the model
        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": text_prompt,  # What the model sees
            "fim_prefix": prefix,  # For Reward Function (Ground Truth context)
            "fim_suffix": suffix,  # For Reward Function
        }

    print("Preprocessing dataset...")
    dataset = dataset.map(process_fim_sample)

    # Reward Functions
    def format_reward(completions, **kwargs):
        """Reward if the completion is non-empty and looks like Lean code."""
        scores = []
        for completion in completions:
            # completion is a string (the generated text)
            c = completion.strip()
            score = 0.0
            if len(c) > 0:
                score += 0.5
            if ":=" in c or "by" in c or "rw" in c or "simp" in c:
                score += 0.5
            scores.append(score)
        return scores

    def lean_validity_reward(completions, fim_prefix, fim_suffix, **kwargs):
        """
        Verify the completed code using Lean compiler.
        Uses the 'fim_prefix' and 'fim_suffix' columns we added during preprocessing.
        """
        scores = []
        # fim_prefix and fim_suffix are lists corresponding to the batch

        for generated_text, prefix, suffix in zip(completions, fim_prefix, fim_suffix):
            # generated_text is the string completion

            # Reconstruct the full file content
            # We simply glue them back together.
            full_code = prefix + generated_text + suffix

            if not prefix or not suffix:
                scores.append(0.0)
                continue

            # Verify
            success, _ = verifier.verify(full_code)
            scores.append(2.0 if success else -1.0)

        return scores

    # Training Arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=MAX_SEQ_LENGTH,  # Adjusted for full chat prompt
        max_completion_length=512,  # Enough for a tactic block
        max_steps=50,
        logging_steps=1,
        report_to="none",
        num_generations=4,
        temperature=0.8,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, lean_validity_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save model
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DIR, "final_model"), tokenizer, save_method="lora"
    )


if __name__ == "__main__":
    main()
