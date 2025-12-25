#!/usr/bin/env python3
"""Condense a training log for LLM-sized contexts.

Outputs head, tail, sampled step lines, checkpoint events, and warnings, while
respecting a character budget. Conversion: ~2 chars/token (empirical).

Presets:
  --preset small   → ~10k tokens (20k chars)
  --preset medium  → ~20k tokens (40k chars)
"""
from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple


STEP_RE = re.compile(r"\bstep:(\d+)\b")
CHECKPOINT_RE = re.compile(r"Saved (model|optim|extra_state).*global_step_(\d+)")
WARNING_RE = re.compile(r"warn|Warn|WARN|WARNING")

# Presets: (head_lines, tail_lines, step_interval, max_step_samples, max_ckpts, max_warns, max_chars)
PRESETS: Dict[str, Tuple[int, int, int, int, int, int, int]] = {
    "small": (30, 60, 20, 12, 6, 6, 20_000),   # ~10k tokens
    "medium": (50, 100, 10, 20, 10, 10, 40_000),  # ~20k tokens
}


def take_tail(path: Path, n: int) -> List[str]:
    tail_buf: deque[str] = deque(maxlen=n)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tail_buf.append(line.rstrip("\n"))
    return list(tail_buf)


def collect_samples(path: Path, step_interval: int, max_samples: int) -> List[str]:
    samples: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step % step_interval == 0 or step in (0, 1, 2, 3):
                samples.append(line.rstrip("\n"))
                if len(samples) >= max_samples:
                    break
    return samples


def collect_checkpoints(path: Path, max_items: int) -> List[str]:
    ckpts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if CHECKPOINT_RE.search(line):
                ckpts.append(line.rstrip("\n"))
                if len(ckpts) >= max_items:
                    break
    return ckpts


def collect_warnings(path: Path, max_items: int) -> List[str]:
    warnings: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if WARNING_RE.search(line):
                warnings.append(line.rstrip("\n"))
                if len(warnings) >= max_items:
                    break
    return warnings


def load_head(path: Path, n: int) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = []
        for idx, line in enumerate(fh):
            if idx >= n:
                break
            head.append(line.rstrip("\n"))
    return head


Mode = Literal["front", "tail"]


def slice_for_budget(lines: Sequence[str], budget: int, mode: Mode) -> Tuple[List[str], bool]:
    """Return lines that fit in budget (chars), flag if truncated.

    mode="front" keeps earliest lines; mode="tail" keeps latest lines.
    """
    if budget <= 0:
        return [], bool(lines)

    total = 0
    chosen: List[str] = []
    iterable = lines if mode == "front" else reversed(lines)
    for ln in iterable:
        ln_len = len(ln) + 1  # +1 for newline
        if total + ln_len > budget:
            break
        chosen.append(ln)
        total += ln_len
    truncated = len(chosen) < len(lines)
    if mode == "tail":
        chosen = list(reversed(chosen))
    return chosen, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a long VERL log")
    parser.add_argument("log_path", type=Path, help="Path to the log file")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Use a preset config: 'small' (~10k tokens) or 'medium' (~20k tokens)",
    )
    parser.add_argument("--head-lines", type=int, default=None, help="Number of lines from the top")
    parser.add_argument("--tail-lines", type=int, default=None, help="Number of lines from the bottom")
    parser.add_argument("--step-interval", type=int, default=None, help="Sample every Nth step line")
    parser.add_argument("--max-step-samples", type=int, default=None, help="Cap on sampled step lines")
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Max checkpoint lines to show")
    parser.add_argument("--max-warnings", type=int, default=None, help="Max warning lines to show")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Character budget for all output (~2 chars/token).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. Defaults to <log>.summary.md next to the log file.",
    )
    args = parser.parse_args()

    # Apply preset defaults, then allow overrides
    preset = PRESETS.get(args.preset or "medium")
    head_lines = args.head_lines if args.head_lines is not None else preset[0]
    tail_lines = args.tail_lines if args.tail_lines is not None else preset[1]
    step_interval = args.step_interval if args.step_interval is not None else preset[2]
    max_step_samples = args.max_step_samples if args.max_step_samples is not None else preset[3]
    max_checkpoints = args.max_checkpoints if args.max_checkpoints is not None else preset[4]
    max_warnings = args.max_warnings if args.max_warnings is not None else preset[5]
    max_chars = args.max_chars if args.max_chars is not None else preset[6]

    path = args.log_path
    if not path.exists():
        raise SystemExit(f"Log not found: {path}")

    head = load_head(path, head_lines)
    tail = take_tail(path, tail_lines)
    steps = collect_samples(path, step_interval, max_step_samples)
    ckpts = collect_checkpoints(path, max_checkpoints)
    warns = collect_warnings(path, max_warnings)

    sections: List[Tuple[str, List[str], Mode]] = [
        (f"HEAD (first {len(head)} lines)", head, "front"),
        (f"STEP SAMPLES (every {step_interval}, max {max_step_samples})", steps, "front"),
        (f"CHECKPOINTS (max {max_checkpoints})", ckpts, "front"),
        (f"WARNINGS (max {max_warnings})", warns, "front"),
        (f"TAIL (last {len(tail)} lines)", tail, "tail"),
    ]

    remaining = max_chars
    output_lines: List[str] = []

    def emit(title: str, lines: Iterable[str], mode: Mode) -> None:
        nonlocal remaining
        header = f"===== {title} ====="
        header_len = len(header) + 1
        if remaining <= 0:
            return
        output_lines.append(header)
        remaining -= header_len

        lines_list = list(lines)
        chosen, truncated = slice_for_budget(lines_list, remaining, mode)
        for ln in chosen:
            if remaining <= 0:
                break
            output_lines.append(ln)
            remaining -= len(ln) + 1
        if truncated and remaining > 0:
            marker = f"... truncated, kept {len(chosen)}/{len(lines_list)} lines ..."
            if len(marker) + 1 <= remaining:
                output_lines.append(marker)
                remaining -= len(marker) + 1
        if remaining > 0:
            output_lines.append("")
            remaining -= 1

    for title, lines_list, mode in sections:
        emit(title, lines_list, mode)

    # Emit to stdout and optionally to markdown file.
    for ln in output_lines:
        print(ln)

    out_path = args.output or (path.with_suffix(path.suffix + ".summary.md"))
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(output_lines) + "\n")
    except OSError as exc:
        print(f"[warn] failed to write summary to {out_path}: {exc}")


if __name__ == "__main__":
    main()
