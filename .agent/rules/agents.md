---
trigger: always_on
---

# Repository Guidelines

## Project Structure & Module Organization
- Core code: `fim_rlvr_lean4/` (Lean verifier, curriculum logic, masking utilities).
- Training entrypoints: `train_gspo_fim_20b.py`, `train_gspo_fim_120b.py`, `train_gspo_fim_local.py`.
- Data pipeline and analyses: `data_pipeline/`, `dataset_analysis.md`, `dataset.md`.
- Tests: `test/` for unit tests; additional smoke scripts in `data_pipeline/`.
- Benchmarks: `benchmarks/verification_performance_optimization/` for verification throughput checks.
- Lean assets: `verification_env/` (expects Lean toolchain accessible via `lake`/`elan`).

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`.
- Unit tests (fast): `python -m pytest test`. Focused runs: `python -m pytest test/test_masking.py`.
- Full pytest (avoids heavy dataset scripts): `python -m pytest`.
- Train 20B: `python train_gspo_fim_20b.py` (assumes data at `data/fim_fresh.jsonl` and Lean env ready).
- Train 120B/local variants: `python train_gspo_fim_120b.py` or `python train_gspo_fim_local.py` with matching hardware/config.

## Coding Style & Naming Conventions
- Language: Python with 4-space indentation; prefer explicit imports.
- Keep functions small and pure where possible; isolate side effects (I/O, subprocess) behind helpers.
- Tests live under `test/` and follow `test_*.py` naming.
- Avoid non-ASCII unless required by domain (Lean code may include Unicode; otherwise stick to ASCII).

## Testing Guidelines
- Framework: pytest. Aim to keep fast unit tests under `test/`; heavier smoke/benchmark scripts live outside or behind `if __name__ == "__main__":`.
- New modules should ship with unit coverage; mock heavy deps (e.g., `unsloth`, `trl`, `datasets`) to keep tests lightweight.
- Use `python -m pytest path/to/test_file.py -k pattern` for targeted debugging.

## Commit & Pull Request Guidelines
- Commit messages: concise imperative summaries (e.g., “Add testable helpers and unit tests for GSPO FIM trainer”).
- Include context in PR descriptions: what changed, why, risks, and how to verify (commands run).
- Link issues/tickets when available; include logs or screenshots for user-facing or training-impactful changes.

## Security & Configuration Tips
- Lean verification relies on `lake` via `elan`; ensure `~/.elan/bin` is on `PATH`.
- Training scripts may write to `outputs_fim_grpo/`; avoid committing large artifacts or datasets.
- Keep API keys or tokens out of the repo; use environment variables or local config files ignored by git.
- Codex-CLI note: running `datasets`/Polars/Unsloth may require full approval (disabling the default sandbox) so `/dev/shm` is writable; otherwise Intel OMP SHM errors can occur. In standard Docker hosts, use `--shm-size` or `--ipc=host` as an alternative.
