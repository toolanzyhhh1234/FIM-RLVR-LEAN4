#!/usr/bin/env python3
"""
FIM-RLVR-LEAN4: Fill-in-the-Middle + Reinforcement Learning with Verification Rewards for Lean 4
Adapted from vision GRPO example for LEAN4 formal verification training.
"""

from unsloth import FastVisionModel  # No PatchFastRL - use standard TRL trainer
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
from packaging.version import Version
from transformers import __version__ as transformers_version

# Ensure we can import from local modules
sys.path.append(os.getcwd())
from fim_rlvr_lean4.lean_verifier import LeanVerifier
from fim_rlvr_lean4.curriculum import CurriculumManager
from fim_rlvr_lean4.masking import apply_dynamic_mask

# Configuration
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
MODEL_NAME = os.environ.get(
    "FIM_MODEL_NAME",
    "unsloth/Ministral-3-3B-Instruct-2512",
)
OUTPUT_DIR = "outputs_fim_grpo_mistral3"
CURRICULUM_STATE_PATH = os.path.join(OUTPUT_DIR, "curriculum_state.json")
DATA_PARQUET = os.environ.get(
    "FIM_PARQUET_PATH",
    "data/fim_fresh.jsonl",  # Default to local JSONL if available
)

def _int_env(name, default):
    val = os.environ.get(name, "")
    if val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _bool_env(name, default: bool) -> bool:
    val = os.environ.get(name, "")
    if val.strip() == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

MAX_STEPS = _int_env("FIM_MAX_STEPS", 100)
NUM_GENERATIONS = _int_env("FIM_NUM_GENERATIONS", 2)
LOAD_IN_4BIT = _bool_env("FIM_LOAD_IN_4BIT", False)  # False for 16-bit LoRA (matches example)
FAST_INFERENCE = _bool_env("FIM_FAST_INFERENCE", False)
MAX_COMPLETION_LENGTH = _int_env("FIM_MAX_COMPLETION_LENGTH", 1024)
DEFAULT_VERIFIERS = max(1, (os.cpu_count() or 4) - 1)
MAX_VERIFIERS = int(os.environ.get("FIM_MAX_VERIFIERS", str(DEFAULT_VERIFIERS)))
LOG_VERIFICATION = bool(int(os.environ.get("FIM_LOG_VERIFICATION", "1")))
LOG_VERIFICATION_LIMIT = int(os.environ.get("FIM_LOG_VERIFICATION_LIMIT", "3"))
LOG_DIR = "training_logs"
LOG_PROMPTS = bool(int(os.environ.get("FIM_LOG_PROMPTS", "1")))
LOG_PROMPTS_LIMIT = int(os.environ.get("FIM_LOG_PROMPTS_LIMIT", "3"))
TRUST_REMOTE_CODE = _bool_env("FIM_TRUST_REMOTE_CODE", True)


def _ensure_transformers_compat(model_name: str) -> None:
    """Fail fast with a clear message if transformers is too old for the model."""
    lowered = model_name.lower()
    if "mistral-3" in lowered or "ministral-3" in lowered or "mistral3" in lowered:
        if Version(transformers_version) < Version("5.0.0.dev0"):
            raise RuntimeError(
                "This model requires transformers>=5.0.0.dev0. "
                "Update with `uv pip install --upgrade git+https://github.com/huggingface/transformers.git` "
                "or `pip install --upgrade git+https://github.com/huggingface/transformers.git`."
            )


def load_training_dataset(data_path: str) -> Dataset:
    """
    Load training data from JSONL or Parquet with Lean code.
    """
    if data_path.endswith('.jsonl'):
        # Load JSONL format
        import json
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: {data_path} not found, creating minimal sample dataset")
            # Create minimal sample for testing
            data = [
                {
                    "formal_ground_truth": """theorem add_comm (a b : ℕ) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_add, ih]""",
                    "uuid": "sample_1"
                },
                {
                    "formal_ground_truth": """theorem zero_add (n : ℕ) : 0 + n = n := by
  rfl""",
                    "uuid": "sample_2"
                }
            ]
        
        df = pl.DataFrame(data)
    else:
        # Load Parquet format
        df = pl.read_parquet(data_path)
    
    cols = set(df.columns)
    if "formal_ground_truth" not in cols:
        raise ValueError(f"Expected 'formal_ground_truth' column. Found: {sorted(cols)}")

    # Prepare dataset
    select_cols = ["formal_ground_truth"]
    if "uuid" in cols:
        select_cols.append("uuid")
    
    df = df.select(select_cols)
    df = df.rename({"formal_ground_truth": "prompt"})
    df = df.with_columns(pl.col("prompt").alias("completion"))
    
    return Dataset.from_polars(df)


def filter_valid_rows(dataset: Dataset) -> Dataset:
    """Filter out invalid Lean code samples."""
    def _is_valid(example):
        txt = example["prompt"]
        if not txt or len(txt.strip()) < 50:
            return False
        return ("theorem" in txt) or ("lemma" in txt) or ("def" in txt)

    before = len(dataset)
    filtered = dataset.filter(_is_valid)
    after = len(filtered)
    print(f"Filtered dataset: {before} -> {after} valid Lean samples")
    
    if after == 0:
        raise ValueError("All samples filtered out - check dataset format")
    return filtered


def build_dynamic_transform(tokenizer, curriculum):
    """Transform function for dynamic FIM masking based on curriculum."""
    
    def dynamic_transform(batch):
        prompts = []
        fim_prefixes = []
        fim_suffixes = []
        theorem_ids = []
        logged = 0

        for i in range(len(batch["prompt"])):
            full_code = batch["prompt"][i]
            
            # Get theorem ID
            th_name = str(i)
            if "uuid" in batch:
                th_name = str(batch["uuid"][i])
            theorem_ids.append(th_name)

            # Get curriculum mask ratio
            ratio = curriculum.get_mask_ratio(th_name)
            
            # Apply dynamic masking
            new_pre, new_suf, new_mid = apply_dynamic_mask(full_code, ratio)

            # Create chat prompt
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

            # Optional logging
            if LOG_PROMPTS and logged < LOG_PROMPTS_LIMIT:
                os.makedirs(LOG_DIR, exist_ok=True)
                with open(os.path.join(LOG_DIR, "prompt_samples.log"), "a", encoding="utf-8") as f:
                    preview = text_prompt[:800] + "..." if len(text_prompt) > 800 else text_prompt
                    f.write(f"[prompt] th={th_name} ratio={ratio:.2f}\n{preview}\n---\n")
                print(f"[prompt-log] th={th_name} ratio={ratio:.2f}")
                logged += 1

        return {
            "prompt": prompts,
            "fim_prefix": fim_prefixes,
            "fim_suffix": fim_suffixes,
            "theorem_id": theorem_ids,
        }

    return dynamic_transform


def lean_validity_reward_factory(verifier, curriculum):
    """Creates reward function for Lean verification."""
    
    def lean_validity_reward(completions, fim_prefix, fim_suffix, theorem_id, **kwargs):
        """Verify completed Lean code and update curriculum."""
        
        # Prepare verification inputs
        verification_inputs = []
        for generated_text, prefix, suffix in zip(completions, fim_prefix, fim_suffix):
            full_code = (prefix or "") + generated_text + (suffix or "")
            verification_inputs.append(full_code if full_code.strip() else None)

        # Parallel verification
        def verify_single(code):
            if code is None:
                return False
            success, _ = verifier.verify(code)
            return success

        max_workers = max(1, min(len(completions), MAX_VERIFIERS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(verify_single, verification_inputs))

        # Convert to scores and update curriculum
        scores = []
        logs = []
        
        for idx, (success, th_id) in enumerate(zip(results, theorem_id)):
            curriculum.update_outcome(th_id, success)
            scores.append(2.0 if success else 0.0)

            if LOG_VERIFICATION and len(logs) < LOG_VERIFICATION_LIMIT:
                preview = verification_inputs[idx] or "<empty>"
                if len(preview) > 400:
                    preview = preview[:400] + "..."
                logs.append({
                    "theorem_id": th_id,
                    "success": success,
                    "gen_len": len(completions[idx]),
                    "code_preview": preview,
                })

        # Log verification results
        if LOG_VERIFICATION and logs:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "verifier_samples.log"), "a", encoding="utf-8") as f:
                for entry in logs:
                    f.write(f"[verify] success={entry['success']} th={entry['theorem_id']} "
                           f"gen_len={entry['gen_len']}\n{entry['code_preview']}\n---\n")

        return scores

    return lean_validity_reward


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    _ensure_transformers_compat(MODEL_NAME)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,  # False for 16-bit LoRA
    )

    # Add LoRA adapters (matching Unsloth example setup)
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,  # *2 speeds up training (per example)
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load dataset
    print(f"Loading dataset from {DATA_PARQUET}")
    dataset = load_training_dataset(DATA_PARQUET)
    dataset = filter_valid_rows(dataset)

    # Initialize verifier and curriculum
    verifier = LeanVerifier("./verification_env")
    
    if os.path.exists(CURRICULUM_STATE_PATH):
        print(f"Loading curriculum from {CURRICULUM_STATE_PATH}")
        curriculum = CurriculumManager.load(CURRICULUM_STATE_PATH)
    else:
        curriculum = CurriculumManager()

    # Set dynamic transform
    print("Setting up dynamic curriculum transform...")
    dataset.set_transform(build_dynamic_transform(tokenizer, curriculum))

    # Training configuration (matching Unsloth example setup)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=50,
        report_to="none",
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[lean_validity_reward_factory(verifier, curriculum)],
        args=training_args,
        train_dataset=dataset,
    )

    # Add curriculum save callback
    class CurriculumSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = kwargs.get("checkpoint_folder") or args.output_dir
            path = os.path.join(ckpt_dir, "curriculum_state.json")
            print(f"Saving curriculum state to {path}")
            curriculum.save(path)
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            return control

    trainer.add_callback(CurriculumSaveCallback())

    print("Starting LEAN4 FIM training with curriculum...")
    trainer.train()
    print("Training finished.")

    # Final saves
    curriculum.save(CURRICULUM_STATE_PATH)
    final_dir = os.path.join(OUTPUT_DIR, "final_adapters")
    print(f"Saving final LoRA adapters to {final_dir}")
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
