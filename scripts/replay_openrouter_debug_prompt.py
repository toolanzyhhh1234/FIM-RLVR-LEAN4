import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from fim_rlvr_lean4.lean_verifier import LeanVerifier

SYSTEM_RE = re.compile(r"<\|start\|>system<\|message\|>(.*?)<\|end\|>", re.S)
DEVELOPER_RE = re.compile(r"<\|start\|>developer<\|message\|>(.*?)<\|end\|>", re.S)
USER_RE = re.compile(r"<\|start\|>user<\|message\|>(.*?)<\|end\|>", re.S)
FIM_CODE_TAG = "FIM_CODE"
FULL_CODE_TAG = "FULL_CODE"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _request_with_retries(
    url: str, headers: Dict[str, str], payload: Dict[str, Any]
) -> Dict[str, Any]:
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


def _extract_roles(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    system = SYSTEM_RE.search(prompt)
    developer = DEVELOPER_RE.search(prompt)
    user = USER_RE.search(prompt)
    return (
        system.group(1) if system else None,
        developer.group(1) if developer else None,
        user.group(1) if user else None,
    )


def _read_line(path: Path, line_no: int) -> str:
    if line_no < 1:
        raise ValueError("line number must be 1-based")
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if idx == line_no:
                return line
    raise ValueError(f"line {line_no} not found in {path}")


def _extract_tagged_code(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip("\n")


def _resolve_code_outputs(content: str) -> Tuple[Optional[str], Optional[str]]:
    fim_code = _extract_tagged_code(content, FIM_CODE_TAG)
    if fim_code is not None:
        return fim_code, None
    full_code = _extract_tagged_code(content, FULL_CODE_TAG)
    return None, full_code


def _extract_response_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _split_fim_prompt(user_msg: str) -> Tuple[str, str]:
    marker = "[MISSING_BLOCK]"
    if marker not in user_msg:
        return user_msg, ""
    prefix, suffix = user_msg.split(marker, 1)
    return prefix, suffix


def _similarity_ratio(a: str, b: str) -> Optional[float]:
    if not a or not b:
        return None
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a Tinker debug prompt through OpenRouter and extract messages."
    )
    parser.add_argument(
        "--input",
        default="logs/tinker_fim/debug_samples.jsonl",
        help="Path to debug_samples.jsonl",
    )
    parser.add_argument("--line", type=int, default=170, help="1-based line number")
    parser.add_argument(
        "--output",
        default="data/openrouter_replay_line170.json",
        help="Where to write the OpenRouter response JSON",
    )
    parser.add_argument(
        "--extract-output",
        default="data/openrouter_replay_line170_extracted.json",
        help="Where to write extracted system/developer/user messages",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="Actually call OpenRouter (otherwise just extract)",
    )
    parser.add_argument(
        "--merge-system",
        action="store_true",
        help="Merge system + developer into a single system message",
    )
    parser.add_argument(
        "--result-output",
        default="data/openrouter_replay_line170_result.json",
        help="Where to write the verification + similarity result JSON",
    )

    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")
    _load_dotenv(Path(".env"))

    line = _read_line(Path(args.input), args.line)
    record = json.loads(line)
    prompt = record.get("prompt", "")
    ground_truth_middle = record.get("ground_truth_middle") or ""

    system_msg, developer_msg, user_msg = _extract_roles(prompt)

    extracted = {
        "system": system_msg,
        "developer": developer_msg,
        "user": user_msg,
    }

    Path(args.extract_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.extract_output).write_text(
        json.dumps(extracted, indent=2), encoding="utf-8"
    )

    if not args.call:
        print("Extracted messages written to:", args.extract_output)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.8"))
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "32000"))
    # note this max_token is different from max_completion token in tinker yaml file,because this likely takes acccount of prompt length

    messages = []
    # NOTE: Do not forward the system prompt to OpenRouter; the provider applies its own.
    if developer_msg:
        if args.merge_system and messages:
            messages[0]["content"] += "\n\n" + developer_msg
        else:
            messages.append({"role": "system", "content": developer_msg})
    if user_msg:
        messages.append({"role": "user", "content": user_msg})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = _prepare_headers(api_key)
    response = _request_with_retries(url, headers, payload)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(response, indent=2), encoding="utf-8")
    print("OpenRouter response written to:", args.output)

    response_content = _extract_response_content(response)
    fim_code, full_code = _resolve_code_outputs(response_content)
    extracted_code = fim_code if fim_code is not None else full_code
    tag_ok = extracted_code is not None and extracted_code.strip() != ""

    prefix, suffix = _split_fim_prompt(user_msg or "")
    full_code_sent = None
    verification_success = False
    verification_output = "<skipped: missing/empty tag>"
    if tag_ok:
        full_code_sent = f"{prefix}{extracted_code}{suffix}"
        verifier = LeanVerifier(
            "./verification_env",
            no_sorries=os.environ.get("FIM_NO_SORRIES", "0") in {"1", "true", "yes"},
        )
        verification_success, verification_output = verifier.verify(full_code_sent)

    extracted_norm = (extracted_code or "").strip()
    truth_norm = (ground_truth_middle or "").strip()
    ground_truth_similarity = (
        _similarity_ratio(extracted_norm, truth_norm)
        if (tag_ok and extracted_norm and truth_norm)
        else None
    )
    ground_truth_exact_match = (
        (tag_ok and extracted_norm == truth_norm) if truth_norm else None
    )

    result = {
        "line": args.line,
        "model": model,
        "tag_extraction_ok": tag_ok,
        "extracted_code": extracted_code,
        "ground_truth_middle": ground_truth_middle,
        "ground_truth_similarity": ground_truth_similarity,
        "ground_truth_exact_match": ground_truth_exact_match,
        "full_code_sent_to_lean": full_code_sent,
        "verification_success": verification_success if tag_ok else False,
        "verification_output": verification_output,
    }

    Path(args.result_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Verification + similarity written to:", args.result_output)


if __name__ == "__main__":
    main()
