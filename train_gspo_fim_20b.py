# train_grpo_fim.py
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch
import os
import sys
import random
import re
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import from local modules
sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier
from fim_rlvr_lean4.curriculum import CurriculumManager
from fim_rlvr_lean4.masking import reconstruct_full_code, apply_dynamic_mask

# Configuration
MAX_SEQ_LENGTH = 1024
LORA_RANK = 4
MODEL_NAME = "unsloth/gpt-oss-20b"
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

    # Initialize Curriculum
    curriculum = CurriculumManager()

    # Dynamic Transform
    def dynamic_transform(batch):
        prompts = []
        fim_prefixes = []
        fim_suffixes = []
        theorem_ids = []

        # Batch is a dict of lists
        for i in range(len(batch["prompt"])):
            raw_p = batch["prompt"][i]
            mid_truth = batch["completion"][i]  # The original middle from dataset

            # Reconstruct full code
            p_pre, p_suf = reconstruct_full_code(raw_p)
            if p_pre is None:
                # Fallback: Just use as is (no dynamic masking possible easily without full code)
                # But we need full code for dynamic.
                # If we assume 'prompt' + 'completion' ~ full code relative to the static split
                full_code = (
                    raw_p.replace("<MID>", "") + mid_truth
                )  # Crude approximation if PFX/SFX wrappers exist
                # Actually reconstruct_full_code returns the text strings.
                full_code = ""  # Fail safe
            else:
                full_code = p_pre + mid_truth + p_suf

            # Get Theorem ID (using metadata name)
            # Check safely
            th_name = str(i)  # Default to index if missing
            if (
                "metadata" in batch
                and batch["metadata"][i]
                and "theorem_name" in batch["metadata"][i]
            ):
                th_name = batch["metadata"][i]["theorem_name"]
            theorem_ids.append(th_name)

            # Get current curriculum ratio
            ratio = curriculum.get_mask_ratio(th_name)

            # Apply dynamic masking
            if full_code:
                new_pre, new_suf, new_mid = apply_dynamic_mask(full_code, ratio)
            else:
                # Fallback if reconstruction failed
                new_pre, new_suf, new_mid = "", "", ""

            # Construct Chat Prompt
            user_content = f"{new_pre}[MISSING_BLOCK]\n{new_suf}"

            messages = [
                {
                    "role": "system",
                    "content": "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. Output ONLY the missing code.",
                },
                {"role": "user", "content": user_content},
            ]

            text_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompts.append(text_prompt)
            fim_prefixes.append(new_pre)
            fim_suffixes.append(new_suf)

        return {
            "prompt": prompts,
            "fim_prefix": fim_prefixes,
            "fim_suffix": fim_suffixes,
            "theorem_id": theorem_ids,
        }

    # Set transform instead of map
    print("Setting up dynamic curriculum transform...")
    dataset.set_transform(dynamic_transform)

    # Reward Functions

    def lean_validity_reward(completions, fim_prefix, fim_suffix, theorem_id, **kwargs):
        """
        Verify the completed code using Lean compiler.
        Updates curriculum based on outcome.
        """
        # Prepare inputs for parallel execution
        verification_inputs = []
        for generated_text, prefix, suffix in zip(completions, fim_prefix, fim_suffix):
            if not prefix and not suffix:  # Handling empty/failure cases
                verification_inputs.append(None)
            else:
                full_code = prefix + generated_text + suffix
                verification_inputs.append(full_code)

        # Helper for the executor
        def verify_single(code):
            if code is None:
                return False
            success, _ = verifier.verify(code)
            return success

        with ThreadPoolExecutor(max_workers=len(completions)) as executor:
            results = list(executor.map(verify_single, verification_inputs))

        # Convert to scores and Update Curriculum
        scores = []
        for success, th_id in zip(results, theorem_id):
            # Update curriculum state for this theorem
            curriculum.update_outcome(th_id, success)

            scores.append(2.0 if success else 0.0)

        return scores

    # Training Arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=512,
        max_steps=50,
        logging_steps=1,
        report_to="none",
        num_generations=4,
        temperature=0.8,
        dataloader_num_workers=0,  # IMPORTANT: Ensure main process for shared curriculum state
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[lean_validity_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training with Curriculum...")
    trainer.train()
    print("Training finished.")

    # Save model
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DIR, "final_model"), tokenizer, save_method="lora"
    )


if __name__ == "__main__":
    main()
