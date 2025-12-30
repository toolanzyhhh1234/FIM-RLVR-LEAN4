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
    "unsloth/Ministral-3-14B-Reasoning-2512",
)
OUTPUT_DIR = "outputs_fim_grpo_mistral3"
CURRICULUM_STATE_PATH = os.path.join(OUTPUT_DIR, "curriculum_state.json")
DATA_DIR = os.environ.get("FIM_DATA_DIR", "/data")
DATA_PARQUET = os.environ.get("FIM_PARQUET_PATH", "")
HF_DATASET_REPO = os.environ.get("FIM_HF_DATASET", "AI-MO/NuminaMath-LEAN")
HF_DATASET_FILE = os.environ.get(
    "FIM_HF_DATASET_FILE",
    "data/train-00000-of-00001.parquet",
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
LOG_RAW = bool(int(os.environ.get("FIM_LOG_RAW", "1")))
LOG_RAW_LIMIT = int(os.environ.get("FIM_LOG_RAW_LIMIT", "3"))
LOG_DIR = os.environ.get("FIM_LOG_DIR", "training_logs")
LOG_PROMPTS = bool(int(os.environ.get("FIM_LOG_PROMPTS", "1")))
LOG_PROMPTS_LIMIT = int(os.environ.get("FIM_LOG_PROMPTS_LIMIT", "3"))
LOG_PROMPTS_MAX_CHARS = int(os.environ.get("FIM_LOG_PROMPTS_MAX_CHARS", "0"))

_PROMPT_LOG_SEEN: set[str] = set()
TRUST_REMOTE_CODE = _bool_env("FIM_TRUST_REMOTE_CODE", True)

FIM_CODE_TAG = "FIM_CODE"
FULL_CODE_TAG = "FULL_CODE"


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


def _find_local_parquet(data_dir: str) -> str | None:
    if not data_dir or not os.path.isdir(data_dir):
        return None
    candidates = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".parquet")
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def _download_hf_parquet(data_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    os.makedirs(data_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=HF_DATASET_FILE,
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
    )


def resolve_parquet_path() -> str:
    if DATA_PARQUET:
        if os.path.exists(DATA_PARQUET):
            return DATA_PARQUET
        print(f"Warning: FIM_PARQUET_PATH not found at {DATA_PARQUET}; falling back.")

    local_path = _find_local_parquet(DATA_DIR)
    if local_path:
        return local_path

    print(f"No parquet found in {DATA_DIR}; downloading from Hugging Face.")
    return _download_hf_parquet(DATA_DIR)


def load_training_dataset(parquet_path: str) -> Dataset:
    """
    Load training data from Parquet with Lean code.
    """
    df = pl.read_parquet(parquet_path)
    
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
        task_types = []
        logged = 0

        for i in range(len(batch["prompt"])):
            full_code = batch["prompt"][i]
            full_solution_required = "[FULL-SOLUTION-REQUIRED]" in full_code

            # Get theorem ID
            th_name = str(i)
            if "uuid" in batch:
                th_name = str(batch["uuid"][i])
            theorem_ids.append(th_name)

            # Get curriculum mask ratio
            ratio = curriculum.get_mask_ratio(th_name)
            
            # Apply dynamic masking (unless full solution is required)
            cleaned_code = full_code.replace("[FULL-SOLUTION-REQUIRED]", "").strip()
            if full_solution_required:
                new_pre, new_suf = "", ""
                user_content = cleaned_code
                task_type = "full"
            else:
                new_pre, new_suf, _ = apply_dynamic_mask(cleaned_code, ratio)
                user_content = f"{new_pre}[MISSING_BLOCK]\n{new_suf}"
                task_type = "fim"
            task_types.append(task_type)

            system_prompt = (
                "You are a Lean 4 expert. Solve the task strictly following this format:\n"
                "1) First write your reasoning inside [THINK]...[/THINK].\n"
                f"2) Then output ONLY the code inside <{FIM_CODE_TAG}>...</{FIM_CODE_TAG}> "
                f"for fill-in-the-middle tasks, or <{FULL_CODE_TAG}>...</{FULL_CODE_TAG}> "
                "for full solutions.\n"
                "3) Do NOT include markdown fences or extra text outside the tags.\n"
                "4) The tagged code must be valid Lean 4.\n"
                "If the user includes [FULL-SOLUTION-REQUIRED], output a full solution in <FULL_CODE>.\n\n"
                "[USER]\n"
                "theorem simple_add (n : ℕ) : 0 + n = n := by\n"
                "  [MISSING_BLOCK]\n\n"
                "[ASSISTANT]\n"
                "[THINK]\n"
                "The definition of addition recurses on the second argument, so 0+n requires induction or a lemma. \n"
                "`simp` uses Nat.zero_add to solve this.\n"
                "[/THINK]\n"
                f"<{FIM_CODE_TAG}>\n"
                "  simp\n"
                f"</{FIM_CODE_TAG}>\n\n"
                "Example (full):\n\n"
                "[USER]\n"
                "theorem add_zero_triv (n : ℕ) : n + 0 = n :=\n\n"
                "[ASSISTANT]\n"
                "[THINK]\n"
                "Addition is defined by recursion on the second argument. \n"
                "Therefore, `n + 0 = n` is true by definition (reflexivity).\n"
                "[/THINK]\n"
                f"<{FULL_CODE_TAG}>\n"
                "by\n"
                "  rfl\n"
                f"</{FULL_CODE_TAG}>"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
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
                log_key = f"{th_name}|{ratio:.4f}|{hash(text_prompt)}"
                if log_key not in _PROMPT_LOG_SEEN:
                    _PROMPT_LOG_SEEN.add(log_key)
                    with open(os.path.join(LOG_DIR, "prompt_samples.log"), "a", encoding="utf-8") as f:
                        if LOG_PROMPTS_MAX_CHARS > 0:
                            preview = (
                                text_prompt[:LOG_PROMPTS_MAX_CHARS] + "..."
                                if len(text_prompt) > LOG_PROMPTS_MAX_CHARS
                                else text_prompt
                            )
                        else:
                            preview = text_prompt
                        f.write(f"[prompt] th={th_name} ratio={ratio:.2f}\n{preview}\n---\n")
                    print(f"[prompt-log] th={th_name} ratio={ratio:.2f}")
                    logged += 1

        return {
            "prompt": prompts,
            "fim_prefix": fim_prefixes,
            "fim_suffix": fim_suffixes,
            "theorem_id": theorem_ids,
            "task_type": task_types,
        }

    return dynamic_transform


def lean_validity_reward_factory(verifier, curriculum, tokenizer):
    """Creates reward function for Lean verification.

    Args:
        verifier: LeanVerifier instance
        curriculum: CurriculumManager instance
        tokenizer: The tokenizer/processor used by the model (needed to decode completion_ids)
    """

    # Get the actual tokenizer if wrapped in a processor (e.g., PixtralProcessor)
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer

    # Build GPT-2 style byte decoder for fixing BPE artifacts
    # This maps Unicode chars like Ġ (U+0120) back to bytes like space (0x20)
    def _build_byte_decoder():
        """Build the inverse of GPT-2's bytes_to_unicode mapping."""
        # GPT-2 byte encoder: maps bytes 0-255 to Unicode chars
        # Printable ASCII stays as-is, others get shifted to U+0100+ range
        bs = list(range(ord("!"), ord("~") + 1))  # 33-126
        bs += list(range(ord("¡"), ord("¬") + 1))  # 161-172
        bs += list(range(ord("®"), ord("ÿ") + 1))  # 174-255
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        byte_encoder = dict(zip(bs, cs))
        return {chr(v): k for k, v in byte_encoder.items()}

    _byte_decoder = _build_byte_decoder()

    def _fix_bpe_artifacts(text: str) -> str:
        """Convert GPT-2 BPE byte representations back to actual UTF-8 text."""
        if not text:
            return text
        # Convert each character through byte_decoder, collect bytes
        byte_list = []
        for char in text:
            if char in _byte_decoder:
                byte_list.append(_byte_decoder[char])
            else:
                # Regular character - encode to UTF-8 bytes
                byte_list.extend(char.encode('utf-8'))
        # Decode collected bytes as UTF-8
        try:
            return bytes(byte_list).decode('utf-8', errors='replace')
        except Exception:
            return text  # Fallback to original if decode fails

    def _decode_completions(completions, **kwargs):
        """Decode completions from token IDs to avoid BPE artifacts.

        TRL passes completion_ids in kwargs which we can decode properly,
        avoiding the Ġ/Ċ byte-level BPE artifacts that appear in the
        pre-decoded completions string.
        """
        completion_ids = kwargs.get("completion_ids")

        if completion_ids is not None:
            # Handle torch tensors
            if hasattr(completion_ids, "tolist"):
                completion_ids = completion_ids.tolist()

            # Decode using the tokenizer
            texts = actual_tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # === DEEP DEBUG: Log tokenizer details once ===
            if not hasattr(_decode_completions, "_deep_debug_done"):
                _decode_completions._deep_debug_done = True
                os.makedirs(LOG_DIR, exist_ok=True)
                with open(os.path.join(LOG_DIR, "tokenizer_debug.log"), "w", encoding="utf-8") as f:
                    f.write("=== TOKENIZER DEEP DEBUG ===\n\n")
                    f.write(f"tokenizer type: {type(tokenizer)}\n")
                    f.write(f"actual_tokenizer type: {type(actual_tokenizer)}\n")
                    f.write(f"actual_tokenizer class name: {actual_tokenizer.__class__.__name__}\n")
                    f.write(f"actual_tokenizer MRO: {[c.__name__ for c in actual_tokenizer.__class__.__mro__]}\n\n")

                    # Check for vocab/byte decoder
                    if hasattr(actual_tokenizer, 'byte_decoder'):
                        f.write(f"Has byte_decoder: True\n")
                        f.write(f"byte_decoder sample: {dict(list(actual_tokenizer.byte_decoder.items())[:10])}\n\n")
                    else:
                        f.write(f"Has byte_decoder: False\n\n")

                    # Sample completion_ids
                    if completion_ids and len(completion_ids) > 0:
                        sample_ids = completion_ids[0][:20] if len(completion_ids[0]) > 20 else completion_ids[0]
                        f.write(f"Sample completion_ids[0][:20]: {sample_ids}\n\n")

                        # Decode each token individually
                        f.write("Individual token decodes:\n")
                        for tid in sample_ids[:10]:
                            try:
                                single = actual_tokenizer.decode([tid], skip_special_tokens=False)
                                f.write(f"  ID {tid} -> {repr(single)}\n")
                            except Exception as e:
                                f.write(f"  ID {tid} -> ERROR: {e}\n")
                        f.write("\n")

                        # Try convert_ids_to_tokens
                        if hasattr(actual_tokenizer, 'convert_ids_to_tokens'):
                            tokens = actual_tokenizer.convert_ids_to_tokens(sample_ids[:10])
                            f.write(f"convert_ids_to_tokens: {tokens}\n\n")

                        # Try convert_tokens_to_string
                        if hasattr(actual_tokenizer, 'convert_tokens_to_string') and hasattr(actual_tokenizer, 'convert_ids_to_tokens'):
                            tokens = actual_tokenizer.convert_ids_to_tokens(sample_ids[:10])
                            try:
                                string = actual_tokenizer.convert_tokens_to_string(tokens)
                                f.write(f"convert_tokens_to_string result: {repr(string)}\n")
                                has_artifacts = ("Ġ" in string) or ("Ċ" in string)
                                f.write(f"convert_tokens_to_string has artifacts: {has_artifacts}\n\n")
                            except Exception as e:
                                f.write(f"convert_tokens_to_string error: {e}\n\n")

                    # batch_decode result
                    if texts:
                        f.write(f"batch_decode result[0][:200]: {repr(texts[0][:200])}\n")
                        has_artifacts = ("Ġ" in texts[0]) or ("Ċ" in texts[0])
                        f.write(f"batch_decode has artifacts: {has_artifacts}\n\n")

                    # Check for _tekken or special attributes
                    for attr in ['_tekken', 'sp_model', 'backend_tokenizer', 'vocab']:
                        f.write(f"Has {attr}: {hasattr(actual_tokenizer, attr)}\n")

                    # Test BPE fix
                    if texts:
                        fixed_sample = _fix_bpe_artifacts(texts[0][:200])
                        f.write(f"\nAfter _fix_bpe_artifacts[0][:200]: {repr(fixed_sample)}\n")
                        has_artifacts_after = ("Ġ" in fixed_sample) or ("Ċ" in fixed_sample)
                        f.write(f"After fix has artifacts: {has_artifacts_after}\n")

                    f.write("\n=== END DEEP DEBUG ===\n")
                print("[DEBUG] Tokenizer deep debug written to training_logs/tokenizer_debug.log")
            # === END DEEP DEBUG ===

            # Apply BPE byte decoding fix to all texts
            texts = [_fix_bpe_artifacts(t) for t in texts]

            # Verify fix worked
            if texts and (("Ġ" in texts[0]) or ("Ċ" in texts[0])):
                print("[WARN] BPE artifacts still present after _fix_bpe_artifacts!")

            return texts

        # Fallback: chat-format completions [[{"content": "..."}], ...]
        if completions and isinstance(completions[0], list) and completions[0] and isinstance(completions[0][0], dict):
            return [c[0].get("content", "") for c in completions]

        # Fallback: use raw completions (may have artifacts)
        if completions and isinstance(completions[0], str) and (("Ġ" in completions[0]) or ("Ċ" in completions[0])):
            print("[WARN] Using raw completions with BPE artifacts - completion_ids not available!")

        return completions

    def _extract_tagged_code(text: str, tag: str) -> str | None:
        if not text:
            return None
        start = text.find(f"<{tag}>")
        if start == -1:
            return None
        start += len(f"<{tag}>")
        end = text.find(f"</{tag}>", start)
        if end == -1:
            return None
        return text[start:end].strip()

    def _strip_markdown_fences(text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        cleaned = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned).strip()

    def lean_validity_reward(completions, fim_prefix, fim_suffix, theorem_id, task_type=None, **kwargs):
        """Verify completed Lean code and update curriculum."""

        # Decode completions from token IDs to avoid BPE artifacts (Ġ/Ċ)
        decoded_completions = _decode_completions(completions, **kwargs)

        # Prepare verification inputs
        verification_inputs = []
        raw_logs = []
        for idx, (generated_text, prefix, suffix) in enumerate(zip(decoded_completions, fim_prefix, fim_suffix)):
            task = None
            if task_type is not None and idx < len(task_type):
                task = task_type[idx]

            if LOG_RAW and len(raw_logs) < LOG_RAW_LIMIT:
                raw_logs.append({
                    "theorem_id": theorem_id[idx] if idx < len(theorem_id) else "unknown",
                    "task_type": task or "unknown",
                    "raw_completion": generated_text,
                })

            if task == "fim":
                extracted = _extract_tagged_code(generated_text, FIM_CODE_TAG)
            elif task == "full":
                extracted = _extract_tagged_code(generated_text, FULL_CODE_TAG)
            else:
                extracted = None

            if extracted is None:
                extracted = generated_text

            extracted = _strip_markdown_fences(extracted)
            full_code = (prefix or "") + extracted + (suffix or "")
            verification_inputs.append(full_code if full_code.strip() else None)

            if LOG_RAW and len(raw_logs) <= LOG_RAW_LIMIT:
                raw_logs[-1]["extracted_code"] = extracted
                raw_logs[-1]["verifier_input"] = full_code if full_code.strip() else "<empty>"

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

        if LOG_RAW and raw_logs:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "raw_completions.log"), "a", encoding="utf-8") as f:
                for entry in raw_logs:
                    f.write(
                        "[raw]\n"
                        f"th={entry['theorem_id']} task={entry['task_type']}\n"
                        f"{entry['raw_completion']}\n"
                        "---\n"
                        "[extracted]\n"
                        f"{entry.get('extracted_code','')}\n"
                        "---\n"
                        "[verifier_input]\n"
                        f"{entry.get('verifier_input','')}\n"
                        "===\n"
                    )

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
    data_path = resolve_parquet_path()
    print(f"Loading dataset from {data_path}")
    dataset = load_training_dataset(data_path)
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
        reward_funcs=[lean_validity_reward_factory(verifier, curriculum, tokenizer)],
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
