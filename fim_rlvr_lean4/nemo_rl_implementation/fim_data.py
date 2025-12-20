from __future__ import annotations

from typing import Any, Optional

import os

import polars as pl
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec

from fim_rlvr_lean4.curriculum import CurriculumManager
from fim_rlvr_lean4.masking import apply_dynamic_mask


def load_fim_dataset(parquet_path: str) -> Dataset:
    """Load training data from a Parquet shard with Polars and return a HF Dataset."""
    df = pl.read_parquet(parquet_path)
    cols = set(df.columns)

    if "formal_ground_truth" not in cols:
        raise ValueError(
            f"Expected 'formal_ground_truth' column in {parquet_path}. "
            f"Found columns: {sorted(cols)}"
        )

    select_cols = ["formal_ground_truth"]
    if "uuid" in cols:
        select_cols.append("uuid")

    df = df.select(select_cols).rename({"formal_ground_truth": "prompt"})
    if "uuid" in cols:
        df = df.with_columns(pl.col("uuid"))
    return Dataset.from_polars(df)


def filter_valid_rows(dataset: Dataset) -> Dataset:
    """Drop clearly invalid/empty rows so masking doesn't produce empty prompts."""

    def _is_valid(example: dict[str, Any]) -> bool:
        txt = example.get("prompt")
        if not txt:
            return False
        stripped = str(txt).strip()
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


class CurriculumStateReader:
    def __init__(self, state_path: Optional[str]) -> None:
        self.state_path = state_path
        self._curriculum = CurriculumManager()
        self._last_mtime: float | None = None
        if state_path and os.path.exists(state_path):
            self._curriculum = CurriculumManager.load(state_path)
            self._last_mtime = os.path.getmtime(state_path)

    def get_ratio(self, theorem_id: str) -> float:
        if self.state_path and os.path.exists(self.state_path):
            mtime = os.path.getmtime(self.state_path)
            if self._last_mtime is None or mtime > self._last_mtime:
                self._curriculum = CurriculumManager.load(self.state_path)
                self._last_mtime = mtime
        return self._curriculum.get_mask_ratio(theorem_id)


def fim_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    idx: int,
    *,
    curriculum_reader: CurriculumStateReader,
    mask_fn=apply_dynamic_mask,
) -> DatumSpec:
    """Convert a Lean datum into a Nemo-RL DatumSpec with dynamic FIM masking."""
    full_code = datum_dict.get("prompt", "")
    if not full_code:
        raise ValueError("Missing 'prompt' content for FIM datum.")

    theorem_id = str(datum_dict.get("uuid", idx))
    ratio = curriculum_reader.get_ratio(theorem_id)
    fim_prefix, fim_suffix, _ = mask_fn(full_code, ratio)

    user_content = f"{fim_prefix}[MISSING_BLOCK]\n{fim_suffix}"
    system_prompt = (
        "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. "
        "Output ONLY the missing code."
    )

    message_log: LLMMessageLogType = []

    sys_message = {"role": "system", "content": system_prompt}
    sys_text = tokenizer.apply_chat_template(
        [sys_message],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    sys_message["token_ids"] = tokenizer(
        sys_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    sys_message["content"] = sys_text
    message_log.append(sys_message)

    user_message = {"role": "user", "content": user_content}
    user_text = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        user_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = user_text
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length > max_seq_length:
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : max(1, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    extra_env_info = {
        "theorem_id": theorem_id,
        "fim_prefix": fim_prefix,
        "fim_suffix": fim_suffix,
        "ground_truth": full_code,
    }

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name or "fim_lean4",
    }
