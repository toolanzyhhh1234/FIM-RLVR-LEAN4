#!/usr/bin/env python3
"""Export full ground truth and per-sample masked Lean files from Tinker debug JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

USER_RE = re.compile(r"<\|start\|>user<\|message\|>(.*?)<\|end\|>", re.S)
MISSING_BLOCK = "[MISSING_BLOCK]"
FULL_SOLUTION_REQUIRED = "[FULL-SOLUTION-REQUIRED]"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def _extract_user_payload(prompt: str) -> str:
    match = USER_RE.search(prompt or "")
    if match:
        return match.group(1)
    return prompt or ""


def _split_masked_prompt(user_payload: str) -> Optional[Tuple[str, str]]:
    if MISSING_BLOCK not in user_payload:
        return None
    return user_payload.split(MISSING_BLOCK, 1)


def _build_output_names(sample_idx: int, record: Dict[str, object]) -> str:
    theorem_id = _safe_slug(str(record.get("theorem_id") or "no_theorem_id"))
    step = str(record.get("step") if record.get("step") is not None else "na")
    base = f"sample_{sample_idx:04d}_step_{step}_{theorem_id}"
    return f"{base}_masked_prompt.lean"


def _reconstruct_full_ground_truth(user_payload: str, ground_truth_middle: str) -> str:
    split_payload = _split_masked_prompt(user_payload)
    if split_payload is not None:
        prefix, suffix = split_payload
        # training_loop logs prompts with normalize_boundaries(), which prepends a newline
        # to suffix in FIM mode. Remove that transport newline to recover original full_code.
        if suffix.startswith("\n"):
            suffix = suffix[1:]
        return prefix + ground_truth_middle + suffix
    if FULL_SOLUTION_REQUIRED in user_payload:
        # full-solution prompts are built as: "{prefix}\\n[FULL-SOLUTION-REQUIRED]".
        # Remove the marker line and append the full ground-truth middle.
        token_with_newline = "\n" + FULL_SOLUTION_REQUIRED
        if token_with_newline in user_payload:
            prefix, _ = user_payload.rsplit(token_with_newline, 1)
            return prefix + ground_truth_middle
        prefix, _ = user_payload.split(FULL_SOLUTION_REQUIRED, 1)
        return prefix + ground_truth_middle
    # Fallback for records without known marker: best available full text is user payload.
    return user_payload


def export_samples(input_path: Path, output_dir: Path, print_to_stdout: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    theorem_dir = output_dir / "full_ground_truth_by_group"
    masked_dir = output_dir / "masked_by_sample"
    theorem_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    fim_count = 0
    full_count = 0
    group_to_candidates: Dict[str, List[str]] = {}
    group_to_info: Dict[str, Tuple[str, str]] = {}

    with input_path.open("r", encoding="utf-8") as handle:
        for total, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            ground_truth_middle = str(record.get("ground_truth_middle") or "")
            prompt = str(record.get("prompt") or "")
            user_payload = _extract_user_payload(prompt)
            split_payload = _split_masked_prompt(user_payload)

            theorem_id = str(record.get("theorem_id") or "no_theorem_id")
            step = str(record.get("step") if record.get("step") is not None else "na")
            group_key = f"step_{step}__{theorem_id}"
            masked_name = _build_output_names(total, record)
            masked_path = masked_dir / masked_name
            full_ground_truth = _reconstruct_full_ground_truth(user_payload, ground_truth_middle)

            group_to_candidates.setdefault(group_key, []).append(full_ground_truth)
            group_to_info[group_key] = (step, theorem_id)

            if split_payload is None:
                full_count += 1
                note = (
                    "-- NOTE: no [MISSING_BLOCK] marker in user prompt.\n"
                    "-- This appears to be a full-code task.\n\n"
                )
                masked_path.write_text(note + user_payload, encoding="utf-8")
            else:
                fim_count += 1
                prefix, suffix = split_payload
                masked_path.write_text(prefix + MISSING_BLOCK + suffix, encoding="utf-8")

    total_inconsistent_samples = 0
    for group_key, candidates in group_to_candidates.items():
        canonical = max(candidates, key=len)
        group_slug = _safe_slug(group_key)
        theorem_path = theorem_dir / f"{group_slug}.lean"
        theorem_path.write_text(canonical, encoding="utf-8")
        total_inconsistent_samples += sum(1 for c in candidates if c != canonical)

    if print_to_stdout:
        # Print each group's canonical theorem once.
        def _sort_key(item: Tuple[str, Tuple[str, str]]) -> Tuple[int, str]:
            step_text, theorem_text = item[1]
            try:
                step_num = int(step_text)
            except ValueError:
                step_num = 10**9
            return (step_num, theorem_text)

        for group_key, (step, theorem_id) in sorted(group_to_info.items(), key=_sort_key):
            canonical = max(group_to_candidates[group_key], key=len)
            print(f"=== Group step={step} | theorem_id={theorem_id} ===")
            print(canonical)
            print()

    print(f"Processed {total} samples from {input_path}")
    print(f"FIM samples: {fim_count}")
    print(f"Full-code samples: {full_count}")
    print(f"Unique step+theorem groups exported: {len(group_to_candidates)}")
    print(f"Samples that differ from their group's canonical full ground truth: {total_inconsistent_samples}")
    print(f"Full ground truth files: {theorem_dir}")
    print(f"Masked files (per sample): {masked_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export full ground truth Lean files (one per step+theorem group) and "
            "masked Lean files (one per sample) from Tinker debug_samples JSONL."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/tinker_fim/debug_samples_last_run_20260113_003650.jsonl"),
        help="Path to debug_samples JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification_env/VerificationEnv/tinker_debug_exports/"
            "debug_samples_last_run_20260113_003650"
        ),
        help="Directory for exported Lean files",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Do not print human-readable ground truth to stdout",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    export_samples(
        input_path=args.input,
        output_dir=args.output_dir,
        print_to_stdout=not args.no_stdout,
    )


if __name__ == "__main__":
    main()
