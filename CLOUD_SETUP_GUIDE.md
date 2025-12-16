# Cloud Instance Setup Guide for FIM-RLVR-LEAN4

This guide outlines the steps to set up a fresh GPU cloud instance (e.g., RunPod, Lambda, AWS) to run the `gpt-oss-120b` training.

## 1. Environment & Dependencies

### Python Environment
Ensure you have a conda environment or venv set up (Python 3.10+ recommended).

```bash
# Basic pip upgrades
pip install --upgrade pip

# Install PyTorch (ensure CUDA version matches your driver, typically 12.1 or 12.4 for H100s)
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (Optimized for RL/Fine-tuning)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other RLVR dependencies
pip install --no-deps trl peft asyc accelerate bitsandbytes datasets
```

### Lean 4 Setup
The project relies on Lean 4 for verifications. You must install `elan` (the Lean version manager).

```bash
# Install elan (interactive by default, use -y for auto)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# Source the env to get 'lake' and 'lean' on PATH
source $HOME/.elan/env
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
cd verification_env

# 1. Download the correct Lean version (v4.15.0) automatically
lake build

# 2. Download pre-built mathlib cache (Saves ~1 hour of compiling)
lake exe cache get

# 3. Build again to link everything
lake build

cd ..
```

*Note: The first time you run `lake build`, it might take a few minutes to fetch the toolchain.*

## 4. Run Validity Check (Optional but Recommended)
Before starting the heavy training, run the miniF2F validity check to ensure the cloud environment correctly compiles valid proofs.

```bash
python check_minif2f_validity.py
```
*Expected Output: `Validity Rate: 100.00%` on samples.*

## 5. Start Training

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
