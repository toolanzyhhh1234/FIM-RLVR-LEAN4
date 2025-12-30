#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _dedup_blocks(text: str, delimiter: str) -> tuple[str, int, int]:
    parts = [p.strip("\n") for p in text.split(delimiter)]
    blocks = [p for p in parts if p.strip()]

    seen: set[str] = set()
    unique_blocks: list[str] = []
    for block in blocks:
        if block in seen:
            continue
        seen.add(block)
        unique_blocks.append(block)

    if not unique_blocks:
        return "", 0, 0

    out = delimiter.join(unique_blocks) + delimiter
    if not out.endswith("\n"):
        out += "\n"
    return out, len(blocks), len(unique_blocks)


def _process_file(path: Path, delimiter: str, in_place: bool) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    deduped, before, after = _dedup_blocks(raw, delimiter)
    if before == 0:
        print(f"[skip] {path} (no blocks found)")
        return

    out_path = path if in_place else path.with_suffix(path.suffix + ".dedup")
    out_path.write_text(deduped, encoding="utf-8")
    print(f"[dedup] {path.name}: {before} -> {after} ({out_path.name})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deduplicate per-run training logs by repeated blocks."
    )
    ap.add_argument("run_dir", type=Path, help="Path like training_logs/run_YYYYmmdd_HHMMSS")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files instead of writing *.dedup files.",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    files = [
        ("prompt_samples.log", "\n---\n"),
        ("verifier_samples.log", "\n---\n"),
        ("raw_completions.log", "\n===\n"),
    ]

    for name, delim in files:
        path = run_dir / name
        if not path.exists():
            continue
        _process_file(path, delim, in_place=args.in_place)


if __name__ == "__main__":
    main()

