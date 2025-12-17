# train_grpo_fim_120b.py
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch
import os
import sys
from concurrent.futures import ThreadPoolExecutor


# Ensure we can import from local modules
sys.path.append(os.getcwd())
# Assuming the verification environment is set up similarly in the cloud
# If 'fim_rlvr_lean4' module structure exists there.
from fim_rlvr_lean4.lean_verifier import LeanVerifier

# Configuration
MAX_SEQ_LENGTH = 2048  # Increased for larger model context
LORA_RANK = 16  # Increased rank for larger model capacity
MODEL_NAME = "unsloth/gpt-oss-120b"
OUTPUT_DIR = "outputs_fim_grpo_120b"
DATA_FILE = "data/fim_fresh.jsonl"


def main():
    print(f"Loading model: {MODEL_NAME}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            fast_inference=True,  # Enables vLLM backend
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            offload_embedding=False,  # do not move embedding to cpu otherwise it will slow down the process
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback explanation or exit
        return

    # ADD LoRA adapters
    # Unsloth patches gpt-oss modules to standard names (q_proj, k_proj...) automatically
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
    if not os.path.exists(DATA_FILE):
        print(f"Warning: {DATA_FILE} not found. Please ensure data is present.")

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # Initialize Verifier
    # Ensure verification_env exists
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
                # Fallback
                prefix = ""
                suffix = ""
        except:
            prefix = ""
            suffix = ""

        # Construct a User-facing instruction
        user_content = f"{prefix}[MISSING_BLOCK]\n{suffix}"

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

        return {
            "prompt": text_prompt,
            "fim_prefix": prefix,
            "fim_suffix": suffix,
        }

    print("Preprocessing dataset...")
    dataset = dataset.map(process_fim_sample)

    # Reward Functions

    def lean_validity_reward(completions, fim_prefix, fim_suffix, **kwargs):
        """
        Verify the completed code using Lean compiler.
        """
        verification_inputs = []
        for generated_text, prefix, suffix in zip(completions, fim_prefix, fim_suffix):
            if not prefix or not suffix:
                verification_inputs.append(None)
            else:
                full_code = prefix + generated_text + suffix
                verification_inputs.append(full_code)

        def verify_single(code):
            if code is None:
                return False
            success, _ = verifier.verify(code)
            return success

        # Parallel verification
        with ThreadPoolExecutor(max_workers=len(completions)) as executor:
            results = list(executor.map(verify_single, verification_inputs))

        scores = []
        for success, inp in zip(results, verification_inputs):
            if inp is None:
                scores.append(0.0)
            else:
                # Reward 2.0 for valid code
                scores.append(2.0 if success else 0.0)

        return scores

    # Training Arguments
    # For 120B model, batch size MUST be small on single node unless H100s
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,  # Slightly lower LR for larger model
        per_device_train_batch_size=1,  # Keep strict 1
        gradient_accumulation_steps=4,  # Increase accumulation for stability
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=512,
        max_steps=100,  # Adjust as needed for cloud run
        logging_steps=1,
        report_to="none",
        num_generations=4,  # 4 generations per prompt
        temperature=0.8,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[lean_validity_reward],
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
