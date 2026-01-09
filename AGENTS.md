# Repository Guidelines

## Project Structure & Module Organization
- Core code: `fim_rlvr_lean4/` (Lean verifier, curriculum logic, masking utilities).
- Training entrypoints (Unsloth + TRL GRPO): `train_gspo_fim_20b.py`, `train_gspo_fim_30b.py`, `train_gspo_fim_120b.py`, `train_gspo_fim_mistral3.py`, `train_gspo_fim_nemo.py`.
- Tinker API training: `train_tinker_fim.py`, `tinker_integration/` (see `tinker_integration/AGENTS.md`), `configs/tinker_training.yaml`, `docs/tinker_api_integration_notes.md`.
- Local/smoke trainer: `train_grpo_fim_local.py`.
- Data pipeline and analyses: `data_pipeline/`, `dataset_analysis.md`, `dataset.md`.
- Tests: `test/` for unit tests; additional smoke scripts in `data_pipeline/`.
- Benchmarks: `benchmarks/verification_performance_optimization/` for verification throughput checks.
- Lean assets: `verification_env/` (expects Lean toolchain accessible via `lake`/`elan`).

## Branches & Framework Focus
- This repo is developed on framework-focused branches and `AGENTS.md` is tracked per-branch.
  - **Unsloth-focused branches**: branch name typically contains `unsloth`; training scripts use `unsloth` + `trl` trainers.
  - **verl-focused branches**: branch name typically contains `verl`; training code integrates the `verl` framework (see `verl_logs/` and related run scripts).
- This `AGENTS.md` is written with an **Unsloth-first** mindset, but keep in mind some shared pipelines/scripts (e.g., Lean verification utilities) may be updated across both tracks.
- When making changes, keep them consistent with the current branch’s framework (don’t mix APIs/config patterns unless explicitly requested).

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`.
- Install Tinker deps: `pip install -r tinker_integration/requirements.txt`.
- Unit tests (fast): `python -m pytest test`. Focused runs: `python -m pytest test/test_masking.py`.
- Full pytest (avoids heavy dataset scripts): `python -m pytest`.
- Train (Mistral3): `bash scripts/run_train_mistral3.sh` (creates a per-run `training_logs/run_*/` folder and tees stdout to `train.log`).
- Train (20B/30B/120B): `python train_gspo_fim_20b.py`, `python train_gspo_fim_30b.py`, `python train_gspo_fim_120b.py`.
- Train (Tinker API): `python train_tinker_fim.py --config configs/tinker_training.yaml`.
- Tinker smoke checks: `python test_tinker_connection.py`, `python test_tinker_minimal.py`.

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

## Data, Logging, and Common Env Vars
- Prefer **Parquet** datasets (JSONL is intentionally avoided for performance in current trainers).
- `train_gspo_fim_mistral3.py` data resolution:
  - Looks for a local parquet file in `FIM_DATA_DIR` (default `/data`).
  - If none found, downloads a parquet shard from Hugging Face (requires runtime network access).
  - Override path directly via `FIM_PARQUET_PATH` if needed.
- Tinker API auth:
  - `TINKER_API_KEY` is required for real Tinker API calls (can be set in `.env` or the shell).
- Per-run logs:
  - Set `FIM_LOG_DIR` to isolate logs; `scripts/run_train_mistral3.sh` does this automatically.
  - `training_logs/` is ignored by git.
- Proof-hole safety defaults (recommended):
  - `FIM_EXCLUDE_SORRY=1`: filters out dataset rows containing `sorry`/`admit`.
  - `FIM_NO_SORRIES=1`: passes `--no-sorries` to Lean verification so `sorry` cannot receive reward.

## Security & Configuration Tips
- Lean verification relies on `lake` via `elan`; ensure `~/.elan/bin` is on `PATH`.
- Training scripts may write to `outputs_fim_grpo*/` and per-run `training_logs/`; avoid committing large artifacts or datasets.
- Keep API keys or tokens out of the repo; use environment variables or local config files ignored by git.
- Codex-CLI note: running `datasets`/Polars/Unsloth may require full approval (disabling the default sandbox) so `/dev/shm` is writable; otherwise Intel OMP SHM errors can occur. In standard Docker hosts, use `--shm-size` or `--ipc=host` as an alternative.
- Installation safety: prefer user-led installs for heavy or disruptive packages (e.g., `vllm`, `flash-attn`, `apex`). The assistant can suggest commands, but should not run them unless the user explicitly approves, since these installs can spike RAM/CPU and drop SSH sessions.

## Performance & Context Management for AI Agents
- **Context Preservation**: Avoid re-reading large dataset or log files repeatedly. Summarize key findings and maintain them in your active context.
- **Efficient Log Access**: When reading large logs (especially `logs/tinker_fim/debug_samples.jsonl`), filter for specific fields using `jq` or `grep` rather than reading full entries. **Each JSON object is extremely long** (containing raw completions and verification outputs); avoid reading the full object for a line unless the specific task requires inspecting the raw model content.
