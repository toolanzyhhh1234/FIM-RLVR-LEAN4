# Cloud Instance Setup Guide for FIM-RLVR-LEAN4

This guide outlines the steps to set up a fresh GPU cloud instance (e.g., RunPod, Lambda, AWS) to run the `gpt-oss-120b` training.

## 1. Environment & Dependencies

### Python Environment
Ensure you have a conda environment or venv set up (Python 3.10+ recommended).
https://www.anaconda.com/docs/getting-started/anaconda/install#linux-installer

```bash
# Basic pip upgrades
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt
```
# need to install vLLM with uv as well
uv pip install vllm --torch-backend=auto


### Node.js Setup (Required for MCP Servers)
Some tools (like `@openai/codex` or modern MCP servers) require Node.js v16+. Standard Ubuntu repositories often have v12, which is too old.

```bash
# Check current version
node -v

# If version is < 16, install Node.js 20.x (LTS) via NodeSource:
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

### Lean 4 Setup
The project relies on Lean 4 for verifications. You must install `elan` (the Lean version manager) and the specific Lean version used in this project (v4.15.0).

```bash
# Install elan (interactive by default, use -y for auto)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# Source the env to get 'lake' and 'lean' on PATH
source $HOME/.elan/env

# Install Lean v4.15.0 and set it as default
elan toolchain install leanprover/lean4:v4.15.0
elan default leanprover/lean4:v4.15.0
```

## 2. Repository Setup

Clone the repository and move checking inputs.

```bash
git clone https://github.com/toolanzyhhh1234/FIM-RLVR-LEAN4.git
cd FIM-RLVR-LEAN4

# Ensure you have your FIM data
# If 'data/fim_fresh.jsonl' was not tracked by git due to size, define how to get it.
# (If it's in the repo, you are good).
ls -lh data/fim_fresh.jsonl
```

## 3. Build Verification Environment (CRITICAL)

You must build the Lean environment so that `mathlib` is downloaded and compiled. If you skip this, the verifier will timeout or fail on imports.

```bash
# CRITICAL: You must enter the directory containing lakefile.lean
cd verification_env

# 1. Update dependencies (downloads mathlib source)
lake update

# 2. Download pre-built mathlib cache (Saves ~1 hour of compiling)
lake exe cache get

# Note: We SKIP 'lake build' as our verifier uses 'lake env lean', which only needs the cache.
# (See docs/LEAN_BUILD_EXPLAINED.md for details)

cd ..
```

## 4. Benchmark Verification (New)

Before starting training, ensure your CPU can handle the verification load. We have a benchmark script optimized for 6+ cores.

```bash
# Run the benchmark (default 6 cores)
python3 test/test_lean_6cpu.py --cores 6
```

*Expected Output: At least 1.0 proofs/sec throughput with 6 workers.*

*Note: The first time you run `lake build`, it might take a few minutes to fetch the toolchain.*

## 5. Run Validity Check (Optional but Recommended)
Before starting the heavy training, run the miniF2F validity check to ensure the cloud environment correctly compiles valid proofs.

```bash
python check_minif2f_validity.py
```
*Expected Output: `Validity Rate: 100.00%` on samples.*

## 6. Start Training

Launch the 120B model training.

```bash
# Ensure you are authenticated with HF if accessing gated models (unsloth/gpt-oss-120b usually okay)
huggingface-cli login

# Run the training script
python train_grpo_fim_120b.py
```

### Troubleshooting
*   **CUDA OOM**: If you hit OOM on 96GB, edit `train_grpo_fim_120b.py` and reduce `MAX_SEQ_LENGTH` (e.g. to 1024) or set `gradient_accumulation_steps=8` and `per_device_train_batch_size=1`.
*   **Lean Error**: If `LeanVerifier` fails, ensure `lake` is in your global PATH (`source $HOME/.elan/env`).
