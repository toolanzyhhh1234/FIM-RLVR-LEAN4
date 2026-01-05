import os
import re
import time
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from fim_rlvr_lean4.masking import apply_dynamic_mask

FIM_CODE_TAG = "FIM_CODE"
FULL_CODE_TAG = "FULL_CODE"

# NOTE: Prefer LongCat-Flash-Thinking for now. If we want non-reasoning,
# switch to LongCat-Flash-Chat and keep the rest unchanged.
LONGCAT_MODEL = "LongCat-Flash-Thinking"


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


def _build_system_prompt() -> str:
    return (
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


def _extract_tagged_code(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip("\n")


def _request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    max_retries = int(os.environ.get("LONGCAT_MAX_RETRIES", "5"))
    backoff = float(os.environ.get("LONGCAT_BACKOFF", "2"))
    timeout = float(os.environ.get("LONGCAT_TIMEOUT", "60"))

    for attempt in range(1, max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        if resp.status_code == 429:
            retry_after = 0
            try:
                retry_after = int(resp.json().get("error", {}).get("retry_after", 0))
            except Exception:
                retry_after = 0
            sleep_for = max(backoff, retry_after)
            time.sleep(sleep_for)
            continue

        if 500 <= resp.status_code < 600:
            time.sleep(backoff)
            continue

        raise RuntimeError(f"LongCat API error {resp.status_code}: {resp.text}")

    raise RuntimeError("LongCat API retries exhausted")


def _write_parquet_row(writer_state: Dict[str, Any], row: Dict[str, Any], out_path: str) -> None:
    if writer_state.get("writer") is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover - best effort fallback
            writer_state["buffer"].append(row)
            writer_state["fallback"] = True
            writer_state["fallback_exc"] = exc
            return

        table = pa.Table.from_pylist([row])
        writer_state["schema"] = table.schema
        writer_state["writer"] = pq.ParquetWriter(out_path, table.schema)
        writer_state["writer"].write_table(table)
        return

    if writer_state.get("fallback"):
        writer_state["buffer"].append(row)
        return

    import pyarrow as pa

    table = pa.Table.from_pylist([row], schema=writer_state["schema"])
    writer_state["writer"].write_table(table)


def _finalize_writer(writer_state: Dict[str, Any], out_path: str) -> None:
    if writer_state.get("writer") is not None:
        writer_state["writer"].close()
        return

    if writer_state.get("fallback"):
        df = pl.DataFrame(writer_state["buffer"])
        df.write_parquet(out_path)


def main() -> None:
    _load_dotenv()

    input_path = _default_parquet_path()
    output_path = os.environ.get("LONGCAT_OUTPUT_PATH", "data/longcat_synthetic.parquet")
    target_samples = int(os.environ.get("LONGCAT_TARGET_SAMPLES", "1000"))
    mask_ratio = float(os.environ.get("FIM_MASK_RATIO", "0.15"))
    exclude_sorry = _bool_env("FIM_EXCLUDE_SORRY", True)

    api_key = os.environ.get("LONGCAT_API_KEY")
    if not api_key:
        raise RuntimeError("LONGCAT_API_KEY is required")

    base_url = os.environ.get("LONGCAT_BASE_URL", "https://api.longcat.chat/openai")
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
        raise ValueError("Expected 'formal_ground_truth' or 'prompt' column in input parquet")

    system_prompt = _build_system_prompt()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    writer_state = {"writer": None, "schema": None, "buffer": [], "fallback": False}
    generated = 0

    for row in _iter_rows(df):
        if generated >= target_samples:
            break

        full_code = row.get(source_col) or ""
        if not _is_valid_lean_sample(full_code, exclude_sorry):
            continue

        prefix, suffix, _ = apply_dynamic_mask(full_code, ratio=mask_ratio)
        user_content = f"{prefix}[MISSING_BLOCK]\n{suffix}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": LONGCAT_MODEL,
            "messages": messages,
            "max_tokens": int(os.environ.get("LONGCAT_MAX_TOKENS", "1024")),
            "temperature": float(os.environ.get("LONGCAT_TEMPERATURE", "0.2")),
        }

        content = _request_with_retries(url, headers, payload)
        generated_middle = _extract_tagged_code(content, FIM_CODE_TAG) or ""
        if not generated_middle.strip():
            continue

        full_generated = f"{prefix}{generated_middle}\n{suffix}" if suffix else f"{prefix}{generated_middle}"

        record = {
            "formal_ground_truth": full_generated,
            "fim_prefix": prefix,
            "fim_suffix": suffix,
            "generated_middle": generated_middle,
            "mask_ratio": mask_ratio,
            "source": "LongCat",
            "base_source": "NuminaMath-LEAN",
            "model": LONGCAT_MODEL,
        }

        _write_parquet_row(writer_state, record, output_path)
        generated += 1

    _finalize_writer(writer_state, output_path)
    print(f"Wrote {generated} synthetic samples to {output_path}")


if __name__ == "__main__":
    main()
