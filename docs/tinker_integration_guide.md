# Tinker Integration Guide

This guide explains how to set up and run the `tinker_integration/` training pipeline in this repo.

## What This Path Does

`train_tinker_fim.py` runs an RLVR loop for Lean 4 proof infilling:

1. Sample model completions from Tinker.
2. Verify completions locally with Lean.
3. Send rewards/advantages back to Tinker with CISPO updates.

Core code lives in `tinker_integration/`.

## Prerequisites

- Python 3.10+
- Lean toolchain available (`lake`/`elan`) for local verification
- Valid `TINKER_API_KEY`
- Parquet dataset path configured in `configs/tinker_training.yaml` or via env vars

## 1. Create a Dedicated Environment

Use a named virtual environment (example: `tinker_env`).

```bash
python3 -m venv tinker_env
source tinker_env/bin/activate
python -m pip install --upgrade pip
pip install -r tinker-requirements.txt
```

Notes:
- The correct package name is `tinker` (not `tinker-api`).
- `tinker-requirements.txt` includes `tinker_integration/requirements.txt` plus runtime extras used by this pipeline.

## 2. Configure Secrets and Runtime Env

Set your API key in shell or `.env`.

```bash
export TINKER_API_KEY="your-api-key"
```

Optional but recommended:

```bash
export FIM_EXCLUDE_SORRY=1
export FIM_NO_SORRIES=1
export FIM_LOG_DIR="logs/tinker_fim"
```

## 3. Validate Integration Before Training

Run smoke checks:

```bash
python test_tinker_connection.py
python test_tinker_minimal.py
```

If these fail, use the troubleshooting section below before starting a long run.

## 4. Launch Training

Default run:

```bash
python train_tinker_fim.py --config configs/tinker_training.yaml
```

Quick sanity run:

```bash
python train_tinker_fim.py --config configs/tinker_training.yaml --max-steps 5
```

Useful overrides:

```bash
python train_tinker_fim.py \
  --config configs/tinker_training.yaml \
  --dataset /path/to/data.parquet \
  --model openai/gpt-oss-120b \
  --group-size 4 \
  --learning-rate 5e-5 \
  --max-steps 200
```

Resume from latest checkpoint:

```bash
python train_tinker_fim.py --config configs/tinker_training.yaml --resume
```

Resume from specific checkpoint:

```bash
python train_tinker_fim.py --config configs/tinker_training.yaml --resume-from checkpoint_100
```

## 5. Monitor and Inspect

Tinker CLI:

```bash
tinker run list
tinker run info <session-id>
```

Local artifacts (default locations depend on config/env):
- Metrics: `logs/tinker_fim/metrics.jsonl`
- Summary: `logs/tinker_fim/summary.json`
- Debug samples: `logs/tinker_fim/debug_samples.jsonl`
- Checkpoints: `checkpoints/` or configured checkpoint dir

When reading `debug_samples.jsonl`, filter specific fields instead of opening full long lines.

## 6. Important Operational Notes

- Keep the local script running. Remote training depends on your local control loop for sampling, Lean verification, and reward submission.
- CISPO requires strict token alignment (`input_tokens`, `target_tokens`, `logprobs`, `advantages`).
- Verify your dataset path is valid and points to Parquet data expected by the env-group builder.

## Troubleshooting

### `No matching distribution found for tinker-api`

Cause: wrong package name.

Fix:

```bash
pip install tinker
```

### `TINKER_API_KEY not found`

Set the key and retry:

```bash
export TINKER_API_KEY="your-api-key"
python test_tinker_connection.py
```

### Lean verification failures for obviously good completions

Check:
- `verification_env/` is present and usable with `lake`
- You are not accidentally allowing `sorry`/`admit` when benchmarking strict verification
- Prompt/verification stitching issues around newline/indent boundaries

### Local environment pollution

Always activate `tinker_env` before running Tinker scripts:

```bash
source tinker_env/bin/activate
which python
```

## Minimal Day-to-Day Workflow

```bash
source tinker_env/bin/activate
export TINKER_API_KEY="your-api-key"
python test_tinker_connection.py
python train_tinker_fim.py --config configs/tinker_training.yaml
```

## Related References

- `tinker_integration/AGENTS.md`
- `docs/tinker_api_integration_notes.md`
- `configs/tinker_training.yaml`
- `train_tinker_fim.py`
