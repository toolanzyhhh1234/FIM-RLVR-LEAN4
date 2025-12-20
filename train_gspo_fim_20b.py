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
LORA_RANK = 16
MODEL_NAME = "unsloth/gpt-oss-20b"
OUTPUT_DIR = "outputs_fim_grpo"
CURRICULUM_STATE_PATH = os.path.join(OUTPUT_DIR, "curriculum_state.json")
DATA_PARQUET = os.environ.get(
    "FIM_PARQUET_PATH",
    "hf://datasets/AI-MO/NuminaMath-LEAN/data/train-00000-of-00001.parquet",
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

MAX_STEPS = _int_env("FIM_MAX_STEPS", 50)
NUM_GENERATIONS = _int_env("FIM_NUM_GENERATIONS", 4)
# Allow turning off 4bit if vLLM + bitsandbytes has incompatibilities.
LOAD_IN_4BIT = _bool_env("FIM_LOAD_IN_4BIT", True)
# Completion cap: bounds reasoning + code tokens; stay well below model's 128k ctx.
# Default 32768 gives a 32k reasoning budget; raise if needed but watch VRAM/time.
MAX_COMPLETION_LENGTH = _int_env("FIM_MAX_COMPLETION_LENGTH", 32768)
# Verifier workers: cap Lean checks; default = cores-1 to leave headroom for trainer/logging.
DEFAULT_VERIFIERS = max(1, (os.cpu_count() or 4) - 1)
MAX_VERIFIERS = int(os.environ.get("FIM_MAX_VERIFIERS", str(DEFAULT_VERIFIERS)))
# Optional debug logging of verifier inputs/outputs
LOG_VERIFICATION = bool(int(os.environ.get("FIM_LOG_VERIFICATION", "0")))
LOG_VERIFICATION_LIMIT = int(os.environ.get("FIM_LOG_VERIFICATION_LIMIT", "5"))  # per step
LOG_DIR = "training_logs"
# Optional prompt logging (what is fed to the model). Keep limits to avoid huge logs.
LOG_PROMPTS = bool(int(os.environ.get("FIM_LOG_PROMPTS", "0")))
LOG_PROMPTS_LIMIT = int(os.environ.get("FIM_LOG_PROMPTS_LIMIT", "5"))


def load_training_dataset(parquet_path: str) -> Dataset:
    """
    Load training data from a Parquet shard with Polars and return a HF Dataset.
    This dataset has Lean code in `formal_ground_truth` (full theorem + proof).
    We use it as the source to mask for FIM; `completion` is unused by masking.
    """
    df = pl.read_parquet(parquet_path)
    cols = set(df.columns)

    if "formal_ground_truth" not in cols:
        raise ValueError(
            f"Expected 'formal_ground_truth' column in {parquet_path}. "
            f"Found columns: {sorted(cols)}"
        )

    prompt_col = "formal_ground_truth"
    select_cols = [prompt_col]
    if "uuid" in cols:
        select_cols.append("uuid")

    # Keep a completion column for API compatibility; duplicate the same text.
    df = df.select(select_cols)
    df = df.rename({prompt_col: "prompt"})
    df = df.with_columns(pl.col("prompt").alias("completion"))
    # Optionally keep uuid as metadata if present
    if "uuid" in cols:
        df = df.with_columns(pl.col("uuid"))
    return Dataset.from_polars(df)


def filter_valid_rows(dataset: Dataset) -> Dataset:
    """
    Drop clearly invalid/empty rows so masking doesn't produce empty prompts.
    Heuristics: non-empty, minimum length, and contains a theorem/lemma keyword.
    """

    def _is_valid(example):
        txt = example["prompt"]
        if not txt:
            return False
        stripped = txt.strip()
        if len(stripped) < 50:
            return False
        return ("theorem" in stripped) or ("lemma" in stripped)

    before = len(dataset)
    filtered = dataset.filter(_is_valid)
    after = len(filtered)
    print(f"Filtered dataset for non-empty Lean code: {before} -> {after}")
    if after == 0:
        raise ValueError("All samples were filtered out; dataset may be empty or malformed.")
    return filtered


def build_dynamic_transform(
    tokenizer,
    curriculum,
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
        logged = 0

        # Batch is a dict of lists
        for i in range(len(batch["prompt"])):
            full_code = batch["prompt"][i]  # full Lean code (formal_ground_truth)

            # Get Theorem ID (using metadata name)
            # Check safely
            th_name = str(i)  # Default to index if nothing else
            if "uuid" in batch:
                th_name = str(batch["uuid"][i])

            theorem_ids.append(th_name)

            # Get current curriculum ratio
            ratio = curriculum.get_mask_ratio(th_name)

            # Apply dynamic masking on full code
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

            if LOG_PROMPTS and logged < LOG_PROMPTS_LIMIT:
                os.makedirs(LOG_DIR, exist_ok=True)
                with open(
                    os.path.join(LOG_DIR, "prompt_samples.log"), "a", encoding="utf-8"
                ) as f:
                    preview = text_prompt if len(text_prompt) < 1200 else text_prompt[:1200] + "... [truncated]"
                    f.write(
                        f"[prompt] th={theorem_ids[-1]} pre_len={len(new_pre)} suf_len={len(new_suf)} mid_len={len(new_mid)}\n"
                        f"{preview}\n---\n"
                    )
                print(f"[prompt-log] th={theorem_ids[-1]} pre_len={len(new_pre)} suf_len={len(new_suf)}")
                logged += 1

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
            # Always build a candidate code string; even if prefix/suffix empty, still verify
            full_code = (prefix or "") + generated_text + (suffix or "")
            verification_inputs.append(full_code if full_code.strip() else None)

        # Optional logging buffer
        logs = []

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
        for idx, (success, th_id) in enumerate(zip(results, theorem_id)):
            # Update curriculum state for this theorem
            curriculum.update_outcome(th_id, success)

            scores.append(2.0 if success else 0.0)

            if LOG_VERIFICATION and len(logs) < LOG_VERIFICATION_LIMIT:
                preview = verification_inputs[idx]
                if preview is None:
                    preview = "<empty>"
                # truncate to keep log light
                if len(preview) > 800:
                    preview = preview[:800] + "... [truncated]"
                logs.append(
                    {
                        "theorem_id": th_id,
                        "success": success,
                        "prefix_len": len(fim_prefix[idx]),
                        "suffix_len": len(fim_suffix[idx]),
                        "gen_len": len(completions[idx]),
                        "code_preview": preview,
                    }
                )

        if LOG_VERIFICATION and logs:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "verifier_samples.log"), "a", encoding="utf-8") as f:
                for entry in logs:
                    f.write(
                        f"[step?] success={entry['success']} th={entry['theorem_id']} "
                        f"gen_len={entry['gen_len']} p={entry['prefix_len']} s={entry['suffix_len']}\n"
                        f"{entry['code_preview']}\n---\n"
                    )

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
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference = True,  # Enable vLLM fast inference
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
        lora_alpha=32,
        lora_dropout=0.0,
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
        loss_type="dr_grpo",
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

    # Final save: adapters only (lightweight). If a fully merged fp16 model is ever needed,
    # call save_pretrained_merged to a separate path.
    final_dir = os.path.join(OUTPUT_DIR, "final_adapters")
    print(f"Saving final LoRA adapters to {final_dir}")
    model.save_pretrained(final_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
