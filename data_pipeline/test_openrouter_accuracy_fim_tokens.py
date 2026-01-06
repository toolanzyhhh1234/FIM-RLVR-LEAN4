"""
Test OpenRouter accuracy using standard FIM tokens format.
Uses <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|> tokens.
"""
import json
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import polars as pl
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from fim_rlvr_lean4.lean_verifier import LeanVerifier
from fim_rlvr_lean4.masking import apply_dynamic_mask

# Standard FIM tokens
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _default_parquet_path() -> str:
    return os.environ.get(
        "FIM_PARQUET_PATH",
        "data/NuminaMath-LEAN/data/train-00000-of-00001.parquet",
    )


def _iter_rows(df: pl.DataFrame) -> Iterable[Dict[str, Any]]:
    for row in df.iter_rows(named=True):
        yield row


def _is_valid_lean_sample(txt: str, exclude_sorry: bool) -> bool:
    if not txt:
        return False
    stripped = txt.strip()
    if len(stripped) < 50:
        return False
    if exclude_sorry:
        if re.search(r"\bsorry\b", stripped) or re.search(r"\badmit\b", stripped):
            return False
    return any(tok in stripped for tok in ["theorem", "lemma", "def"])


def _build_fim_system_prompt() -> str:
    """System prompt for standard FIM token format."""
    return (
        "You are a Lean 4 expert performing fill-in-the-middle (FIM) code completion.\n\n"
        "The user will provide code in FIM format with these tokens:\n"
        f"- {FIM_PREFIX} marks the start of prefix code\n"
        f"- {FIM_SUFFIX} marks the start of suffix code\n"
        f"- {FIM_MIDDLE} signals where you should output the missing middle code\n\n"
        "Your task: Output ONLY the code that fills the gap between prefix and suffix.\n"
        "Do NOT include the FIM tokens in your response.\n"
        "Do NOT repeat any code from the prefix or suffix.\n"
        "Output valid Lean 4 code only.\n\n"
        "Example:\n"
        f"User: {FIM_PREFIX}theorem test : 1 + 1 = 2 := by\n"
        f"  {FIM_SUFFIX}\n"
        f"  rfl{FIM_MIDDLE}\n\n"
        "Assistant: simp"
    )


def _build_fim_prompt(prefix: str, suffix: str) -> str:
    """Build user prompt in standard FIM format."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def _request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    max_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", "5"))
    backoff = float(os.environ.get("OPENROUTER_BACKOFF", "2"))
    timeout = float(os.environ.get("OPENROUTER_TIMEOUT", "60"))

    for _attempt in range(1, max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            retry_after = 0
            try:
                retry_after = int(resp.json().get("error", {}).get("retry_after", 0))
            except Exception:
                retry_after = 0
            time.sleep(max(backoff, retry_after))
            continue

        if 500 <= resp.status_code < 600:
            time.sleep(backoff)
            continue

        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    raise RuntimeError("OpenRouter API retries exhausted")


def _similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _prepare_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    app_url = os.environ.get("OPENROUTER_APP_URL")
    if app_url:
        headers["HTTP-Referer"] = app_url
    app_title = os.environ.get("OPENROUTER_APP_TITLE")
    if app_title:
        headers["X-Title"] = app_title
    return headers


def _clean_response(content: str) -> str:
    """Clean model response - remove any accidentally included FIM tokens or markdown."""
    result = content.strip()
    # Remove FIM tokens if model included them
    for token in [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]:
        result = result.replace(token, "")
    # Remove markdown code fences
    if result.startswith("```"):
        lines = result.split("\n")
        if len(lines) > 2:
            result = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return result.strip()


def main() -> None:
    _load_dotenv(str(REPO_ROOT / ".env"))
    _load_dotenv()

    seed_raw = os.environ.get("OPENROUTER_MASK_SEED") or os.environ.get("FIM_MASK_SEED")
    if seed_raw:
        random.seed(int(seed_raw))

    input_path = _default_parquet_path()
    output_path = os.environ.get(
        "OPENROUTER_FIM_OUTPUT_PATH", "data/openrouter_accuracy_fim_tokens.jsonl"
    )
    mask_ratio = float(os.environ.get("FIM_MASK_RATIO", "0.15"))
    exclude_sorry = _bool_env("FIM_EXCLUDE_SORRY", True)
    target_samples = int(os.environ.get("OPENROUTER_ACCURACY_SAMPLES", "5"))

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
    url = base_url.rstrip("/") + "/v1/chat/completions"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input parquet not found: {input_path}. Set FIM_PARQUET_PATH to override."
        )

    df = pl.read_parquet(input_path)
    if "formal_ground_truth" in df.columns:
        source_col = "formal_ground_truth"
    elif "prompt" in df.columns:
        source_col = "prompt"
    else:
        raise ValueError("Expected 'formal_ground_truth' or 'prompt' column")

    system_prompt = _build_fim_system_prompt()
    headers = _prepare_headers(api_key)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    verifier = LeanVerifier(
        "./verification_env", no_sorries=_bool_env("FIM_NO_SORRIES", False)
    )

    generated = 0
    progress = tqdm(total=target_samples, desc="FIM tokens accuracy", unit="sample")
    exact_count = 0
    lean_pass_count = 0
    lean_fail_count = 0
    similarity_sum = 0.0

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in _iter_rows(df):
            if generated >= target_samples:
                break

            full_code = row.get(source_col) or ""
            if not _is_valid_lean_sample(full_code, exclude_sorry):
                continue

            prefix, suffix, middle_truth = apply_dynamic_mask(full_code, ratio=mask_ratio)
            
            # Skip if empty middle (edge case)
            if not middle_truth.strip():
                continue

            user_content = _build_fim_prompt(prefix, suffix)

            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                # Higher token limit needed for reasoning models that use internal reasoning tokens
                "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096")),
                "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7")),
            }

            response = _request_with_retries(url, headers, payload)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or ""

            generated_middle = _clean_response(content)
            
            exact_match = generated_middle.strip() == middle_truth.strip()
            similarity = _similarity_ratio(generated_middle, middle_truth)
            similarity_sum += similarity
            if exact_match:
                exact_count += 1

            # Reconstruct full code
            full_generated = (
                f"{prefix}{generated_middle}\n{suffix}"
                if suffix
                else f"{prefix}{generated_middle}"
            )

            lean_pass = None
            lean_output = ""
            if not exact_match:
                lean_pass, lean_output = verifier.verify(full_generated)
                if lean_pass:
                    lean_pass_count += 1
                else:
                    lean_fail_count += 1

            record = {
                "model": OPENROUTER_MODEL,
                "prompt_format": "fim_tokens",
                "mask_ratio": mask_ratio,
                "exact_match": exact_match,
                "similarity": similarity,
                "lean_pass": lean_pass,
                "fim_prefix": prefix,
                "fim_suffix": suffix,
                "middle_truth": middle_truth,
                "generated_middle": generated_middle,
                "raw_response": content,
            }

            if not exact_match:
                record["lean_output"] = lean_output

            handle.write(json.dumps(record) + "\n")
            generated += 1
            progress.update(1)

    progress.close()
    avg_similarity = similarity_sum / max(1, generated)
    print(f"Wrote {generated} FIM token accuracy samples to {output_path}")
    print(
        "Summary: "
        f"exact={exact_count}/{generated} "
        f"lean_pass={lean_pass_count} lean_fail={lean_fail_count} "
        f"avg_similarity={avg_similarity:.4f}"
    )


if __name__ == "__main__":
    main()
