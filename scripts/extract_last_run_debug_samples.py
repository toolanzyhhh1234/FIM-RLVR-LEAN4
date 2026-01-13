#!/usr/bin/env python3
"""Extract the last run (from the first step==0 of the last run) from a JSONL log.

Usage:
  python scripts/extract_last_run_debug_samples.py \
    --input logs/tinker_fim/debug_samples.jsonl
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import re


STEP_PATTERN = re.compile(rb'"step"\s*:\s*([0-9]+)')


def _find_last_run_start_offset(path: Path) -> Optional[int]:
    """Return byte offset of the first step==0 line of the last run."""
    last_run_offset = None
    prev_step: Optional[int] = None
    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            match = STEP_PATTERN.search(line)
            if not match:
                continue
            step = int(match.group(1))
            if step == 0 and (prev_step is None or prev_step != 0):
                last_run_offset = line_start
            prev_step = step
    return last_run_offset


def _build_output_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir if output_dir is not None else input_path.parent
    filename = f"debug_samples_last_run_{timestamp}.jsonl"
    return out_dir / filename


def extract_last_run(input_path: Path, output_path: Path) -> int:
    """Write the last run to output_path. Returns number of lines written."""
    last_offset = _find_last_run_start_offset(input_path)
    if last_offset is None:
        raise ValueError("No line containing step==0 found.")

    lines_written = 0
    with input_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(last_offset)
        for line in src:
            dst.write(line)
            lines_written += 1

    return lines_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the last run (from the first step==0 of the last run to EOF) from a JSONL log."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/tinker_fim/debug_samples.jsonl"),
        help="Path to debug_samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output JSONL (default: input file directory)",
    )
    args = parser.parse_args()

    output_path = _build_output_path(args.input, args.output_dir)
    lines_written = extract_last_run(args.input, output_path)
    print(f"Wrote {lines_written} lines to {output_path}")


if __name__ == "__main__":
    main()
