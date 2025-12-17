# train_grpo_fim.py
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from datasets import Dataset
import polars as pl
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
CURRICULUM_STATE_PATH = os.path.join(OUTPUT_DIR, "curriculum_state.json")
DATA_PARQUET = os.environ.get(
    "FIM_PARQUET_PATH",
    "hf://datasets/AI-MO/NuminaMath-LEAN/data/train-00000-of-00001.parquet",
)
MAX_STEPS = int(os.environ.get("FIM_MAX_STEPS", "50"))
NUM_GENERATIONS = int(os.environ.get("FIM_NUM_GENERATIONS", "4"))
# Completion cap: bounds reasoning + code tokens; stay well below model's 128k ctx.
# Default 65536 keeps long traces while preventing runaway generation.
MAX_COMPLETION_LENGTH = int(os.environ.get("FIM_MAX_COMPLETION_LENGTH", "65536"))
# Verifier workers: cap Lean checks; default = cores-1 to leave headroom for trainer/logging.
DEFAULT_VERIFIERS = max(1, (os.cpu_count() or 4) - 1)
MAX_VERIFIERS = int(os.environ.get("FIM_MAX_VERIFIERS", str(DEFAULT_VERIFIERS)))


def load_training_dataset(parquet_path: str) -> Dataset:
    """
    Load training data from a Parquet shard with Polars and return a HF Dataset.
    Expects columns for prompt/completion; attempts light remapping when possible.
    """
    df = pl.read_parquet(parquet_path)
    cols = set(df.columns)

    prompt_col = None
    completion_col = None

    for candidate in ["prompt", "input", "problem", "question"]:
        if candidate in cols:
            prompt_col = candidate
            break

    for candidate in ["completion", "formal_ground_truth", "answer", "output"]:
        if candidate in cols:
            completion_col = candidate
            break

    if not prompt_col or not completion_col:
        raise ValueError(
            f"Dataset missing prompt/completion columns in parquet {parquet_path}. "
            f"Found columns: {sorted(cols)}"
        )

    select_cols = [prompt_col, completion_col]
    metadata_col = "metadata" if "metadata" in cols else None
    if metadata_col:
        select_cols.append(metadata_col)

    df = df.select(select_cols)
    df = df.rename({prompt_col: "prompt", completion_col: "completion"})
    return Dataset.from_polars(df)


def filter_valid_rows(dataset: Dataset, reconstruct_fn=reconstruct_full_code) -> Dataset:
    """Drop samples whose prompt cannot be reconstructed into prefix/suffix."""

    def _is_valid(example):
        p_pre, p_suf = reconstruct_fn(example["prompt"])
        return p_pre is not None and p_suf is not None

    before = len(dataset)
    filtered = dataset.filter(_is_valid)
    after = len(filtered)
    print(f"Filtered dataset for valid FIM structure: {before} -> {after}")
    if after == 0:
        raise ValueError("All samples were filtered out; check dataset format and reconstruct_full_code().")
    return filtered


def build_dynamic_transform(
    tokenizer,
    curriculum,
    reconstruct_fn=reconstruct_full_code,
    mask_fn=apply_dynamic_mask,
):
    """
    Returns a Hugging Face transform function that applies dynamic FIM masking
    based on curriculum ratios.
    """

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
            p_pre, p_suf = reconstruct_fn(raw_p)
            full_code = None if p_pre is None else p_pre + mid_truth + p_suf

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
            if full_code is None:
                # Should not happen because we filter beforehand; keep lengths aligned.
                new_pre, new_suf, new_mid = "", "", ""
            else:
                new_pre, new_suf, new_mid = mask_fn(full_code, ratio)

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

    return dynamic_transform


def lean_validity_reward_factory(verifier, curriculum):
    """
    Creates a reward function closure that verifies Lean code and updates the curriculum.
    """

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

        max_workers = max(1, min(len(completions), MAX_VERIFIERS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(verify_single, verification_inputs))

        # Convert to scores and Update Curriculum
        scores = []
        for success, th_id in zip(results, theorem_id):
            # Update curriculum state for this theorem
            curriculum.update_outcome(th_id, success)

            scores.append(2.0 if success else 0.0)

        return scores

    return lean_validity_reward


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Apply Unsloth's TRL RL patches (enables GSPO and other RL fixes)
    PatchFastRL()

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        offload_embedding=False,  # Do not load embedding in cpu(it will slow the process down)
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
    print(f"Loading dataset from {DATA_PARQUET}")
    dataset = load_training_dataset(DATA_PARQUET)

    # Drop samples where reconstruct_full_code fails so we don't feed empty rewards
    dataset = filter_valid_rows(dataset)

    # Initialize Verifier
    verifier = LeanVerifier("./verification_env")

    # Initialize Curriculum
    if os.path.exists(CURRICULUM_STATE_PATH):
        print(f"Loading curriculum state from {CURRICULUM_STATE_PATH}")
        curriculum = CurriculumManager.load(CURRICULUM_STATE_PATH)
    else:
        curriculum = CurriculumManager()

    # Set transform instead of map
    print("Setting up dynamic curriculum transform...")
    dataset.set_transform(build_dynamic_transform(tokenizer, curriculum))

    # Reward Functions

    # Training Arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # TRL upstream does not recognize "gspo"; use "grpo" with seq-level IS to mirror GSPO.
        loss_type="grpo",
        importance_sampling_level="sequence",
        max_steps=MAX_STEPS,
        logging_steps=1,
        report_to="none",
        num_generations=NUM_GENERATIONS,  # We can tune this depending our VRAM consumption, potentially speed up here
        temperature=0.8,
        dataloader_num_workers=0,  # IMPORTANT: Ensure main process for shared curriculum state
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[lean_validity_reward_factory(verifier, curriculum)],
        args=training_args,
        train_dataset=dataset,
    )

    class CurriculumSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = kwargs.get("checkpoint_folder") or args.output_dir
            path = os.path.join(ckpt_dir, "curriculum_state.json")
            print(f"Saving curriculum state to {path}")
            curriculum.save(path)
            # Adapter-only checkpoint to keep saves light; merge happens only at final save.
            print(f"Saving LoRA adapters to {ckpt_dir}")
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            return control

    trainer.add_callback(CurriculumSaveCallback())

    print("Starting training with Curriculum...")
    trainer.train()
    print("Training finished.")

    print(f"Saving curriculum state to {CURRICULUM_STATE_PATH}")
    curriculum.save(CURRICULUM_STATE_PATH)

    # Save model
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DIR, "final_model"), tokenizer, save_method="lora"
    )


if __name__ == "__main__":
    main()
