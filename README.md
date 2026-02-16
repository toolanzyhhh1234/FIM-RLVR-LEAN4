# FIM-RLVR-LEAN4

**Fill-in-the-Middle (FIM) + Reinforcement Learning with Verification Rewards (RLVR) for Lean 4.**

### 🚀 Project Summary

**The Big Idea:**
Can we efficiently bootstrap rigorous mathematical reasoning in AI models?
This project hypothesizes that **Fill-in-the-Middle (FIM)** tasks combined with **Reinforcement Learning (RLVR)** are the most effective way to "warm up" models with little prior formal knowledge. By grounding them in a strict formal system, we aim to evolve standard LLMs into reasoning engines capable of **100% verified reliability**.

**For the Non-Expert (The Potential):**
Most AI models today (like broad chatbots) guess answers based on likelihood, which often leads to convincing-sounding errors ("hallucinations").
This project builds an AI that acts like a mathematician: it attempts a logical step, then checks if it works using a strict **"truth machine"** (the Lean 4 compiler—a computer system that automatically validates proofs).
*   If the step is wrong, the AI learns immediately.
*   If the step is right, the logic is mathematical fact.
**The Goal:** Create AI systems that don't just guess, but solve complex problems (like logic puzzles or Math Olympiad questions) with **100% verified reliability**.

**For the Expert (The Setup):**
This repository implements a **FIM + GSPO** pipeline designed to bootstrap formal capabilities from weak priors:
*   **Hypothesis**: FIM is the most cost-effective method to **rapidly adapt** emerging state-of-the-art base models into Lean 4 specialists. Since open-source models improve quickly, we need a way to bootstrap formal capabilities without expensive training from scratch—especially for models with limited initial Lean exposure.
*   **Method**: **GSPO (Group Sequence Policy Optimization)**—a variant of GRPO optimized for stability with MoE models like GPT-OSS—samples multiple solutions for missing proof blocks.
*   **Feedback**: A binary reward signal derived from the **Lean 4 compiler**, verified in parallel.
*   **Research Questions**:
    *   **Curriculum efficacy**: Does starting with FIM and moving to independent proving allow the model to eventually outperform models trained on independent proving from the start?
    *   **Efficiency**: Does FIM speed up the training process (convergence)?
    *   **Transfer**: Does this formal grounding transfer to natural language mathematics (e.g., **IMO Bench**)?

**Explore the details:**
*   [Technical Specification & Architecture](project-spec.md)

## Prerequisites

- Python 3.10+
- [Lean 4 toolchain](https://leanprover.github.io/lean4/doc/setup.html) (v4.15.0)
- CUDA-compatible GPU (recommended for training)

## Status: Transitioning to Tinker API

**The training pipeline is functional, with Tinker integration running cleanly, and we're digging into verification performance.**

**What works:**
- **Lean 4 Verification Pipeline**: Robust, thread-safe verifier with parallel verification (linear speedup with cpu core counts)
- **GRPO Training Loop**: Integrated with Unsloth + TRL (`train_gspo_fim_mistral3.py`, `train_gspo_fim_qwen3-vl-8b.py` on `further-investigation-on-unsloth` branch)
- **Tinker Integration**: No syntax errors and several successful trial runs

**What we learned:**
- **Dense models are inefficient**: High active parameters but performance similar to sparse MoE models of comparable size. Training cost scales poorly.
- **Small models struggle**: <14B models fail to adhere to FIM answer format and reason inefficiently. Would require SFT to bootstrap, adding cost/complexity.
- **Large MoE models excel**: Initial evaluation of `gpt-oss-120b` shows ~60% accuracy out-of-box with proper format compliance.

**Why Tinker:**
- Custom infrastructure for 120B models is prohibitively expensive (cost scales with total parameters for GPU rental)
- Tinker's billing scales with **active parameters**—making ultra-sparse MoE like `gpt-oss-120b` cost-effective
- Easier to iterate and debug without managing distributed training infrastructure

**Current focus**: Applying for Tinker research grant to train `gpt-oss-120b` with CISPO.

**In progress**: Lean 4 verification success rate is currently lower than expected; we are investigating the pipeline in depth to determine whether this is a model capability issue or a pipeline design issue.

## � Training SInfrastructure: Tinker API

This project uses [Tinker API](https://tinker-docs.thinkingmachines.ai/) by Thinking Machines Lab for scalable RL training on large MoE models.

**Why Tinker?**
- **Cost-effective MoE training**: Token-based pricing scales with active parameters, not total parameters
- **Native CISPO support**: Clipped Importance Sampling Policy Optimization—stable for MoE architectures where GRPO fails
- **No infrastructure overhead**: Distributed training handled automatically

**Primary model**: `gpt-oss-120b` (ultra-sparse MoE, ~60% accuracy on our task out-of-box)

📄 **[Full Tinker Integration Plan & Research Grant Proposal](docs/tinker_api_research_grant.md)**

---

## 🚀 Call for Sponsorship & Collaboration

**We are seeking support to scale this research.**

This project is currently running on a small personal budget. To fully validate the FIM-RLVR hypothesis—that verifiable feedback can significantly boost formal reasoning capabilities—we need to scale up to:
- Larger models (e.g., GPT-OSS-20B, GPT-OSS-120B, or any other model supported by sponsors).
- Massive datasets (e.g., full Mathlib, millions of synthetic samples).
- Extensive H100 GPU compute for full training runs.

**If you are interested in the results of this research or would like to sponsor the compute required to push this project to the next level, please reach out!** Your support would be extremely helpful in allowing us to continue iterating and potentially finding strong evidence for the efficacy of verification-driven RL in formal mathematics.

**Contact:** ifchou@student.unimelb.edu.au or open a GitHub issue.

## Open Source Commitment

All artifacts from this project will be open-sourced:
- Training code and Tinker integration
- Verification data and Lean4 compiler results  
- Model weights (LoRA adapters)
- Research logs (`research_logs/`)

---

## Contributing

Contributions and collaborators are welcome! If you share an interest in formal verification + RL, feel free to open a PR or reach out. I'm a student with other commitments, so responses may take a few days—but I genuinely appreciate the interest.

## Usage

### Option 1: Tinker API Training (Recommended for Large Models)

This is the recommended approach for training large MoE models like `gpt-oss-120b`.

#### 1. Setup

```bash
# Install dependencies
pip install -r tinker_integration/requirements.txt

# Setup Lean 4 verification environment
cd verification_env && lake update && cd ..
```

#### 2. Configure Tinker API Key

You can set your API key in a `.env` file (recommended) or as an environment variable:

```bash
# Option A: Create a .env file in the project root
echo 'TINKER_API_KEY=your-api-key-here' >> .env

# Option B: Export as environment variable
export TINKER_API_KEY="your-api-key-here"

# Get your API key from: https://tinker.thinkingmachines.ai/
```

#### 3. Prepare Dataset

Ensure you have a theorem dataset in Parquet format with the required fields:
- `theorem_id` (or `id`): Unique identifier for each theorem
- `prefix`: Code before the masked region
- `suffix`: Code after the masked region  
- `middle`: The ground truth for the masked region

Example datasets are available in `data/`:
```bash
# NuminaMath-LEAN dataset
data/NuminaMath-LEAN/data/train-00000-of-00001.parquet

# LeanDojoBench dataset
data/LeanDojoBench/data/train-00000-of-00001.parquet
```

#### 4. Run Training

```bash
# Basic training with default config
python train_tinker_fim.py --config configs/tinker_training.yaml

# Resume from checkpoint
python train_tinker_fim.py --config configs/tinker_training.yaml --resume

# Override specific settings
python train_tinker_fim.py --config configs/tinker_training.yaml \
    --max-steps 500 \
    --learning-rate 0.0001 \
    --group-size 8

# Dry run (validate config without training)
python train_tinker_fim.py --config configs/tinker_training.yaml --dry-run
```

#### Configuration Options

The configuration file (`configs/tinker_training.yaml`) supports the following options:

| Option | Default | Description |
|--------|---------|-------------|
| `model_name` | `openai/gpt-oss-120b` | Primary model for training |
| `fallback_model` | `Qwen/Qwen3-235B-A22B` | Fallback if primary unavailable |
| `lora_rank` | `16` | LoRA fine-tuning rank |
| `max_steps` | `1000` | Maximum training steps |
| `learning_rate` | `0.00005` | Learning rate for LoRA updates |
| `temperature` | `0.8` | Sampling temperature |
| `group_size` | `4` | Completions per theorem (CISPO) |
| `max_concurrent_verifications` | `8` | Parallel Lean4 verifications |
| `verification_timeout` | `60.0` | Timeout per verification (seconds) |
| `checkpoint_interval` | `10` | Steps between checkpoints |
| `logging_steps` | `10` | Steps between metric logs |

#### Environment Variable Overrides

Most settings can be overridden via environment variables:

```bash
export TINKER_API_KEY="your-api-key"      # Required
export FIM_MODEL_NAME="Qwen/Qwen3-235B-A22B"
export FIM_MAX_STEPS="500"
export FIM_LEARNING_RATE="0.0001"
export FIM_CHECKPOINT_INTERVAL="10"
export FIM_CHECKPOINT_DIR="checkpoints/my_run"
export FIM_LOG_DIR="logs/my_run"
export FIM_DATASET_PATH="data/my_dataset.parquet"
export WANDB_PROJECT="my-project"         # Enable W&B logging
```

#### Command Line Options

```
usage: train_tinker_fim.py [-h] [--config CONFIG] [--resume] [--resume-from CHECKPOINT]
                           [--max-steps N] [--checkpoint-dir DIR] [--checkpoint-interval N] [--log-dir DIR]
                           [--dataset PATH] [--model NAME] [--group-size N]
                           [--learning-rate LR] [--verification-env PATH]
                           [--no-sorries] [--verbose] [--quiet] [--dry-run]

Options:
  --config, -c          Path to YAML configuration file
  --resume              Resume from latest checkpoint
  --resume-from         Resume from specific checkpoint (e.g., checkpoint_100)
  --max-steps           Override max training steps
  --checkpoint-dir      Override checkpoint directory
  --checkpoint-interval Override checkpoint interval (in steps)
  --log-dir             Override log directory
  --dataset             Override dataset path
  --model               Override model name
  --group-size          Override group size for CISPO
  --learning-rate       Override learning rate
  --verification-env    Path to Lean4 verification environment (default: verification_env)
  --no-sorries          Fail verification when file contains sorry
  --verbose, -v         Enable verbose logging
  --quiet, -q           Suppress non-error output
  --dry-run             Validate configuration without starting training
```

---

### Option 2: Local Training (Unsloth + TRL)

For smaller models or local experimentation:

```bash
# Install dependencies (Unsloth, TRL, etc.)
pip install unsloth trl

# Setup Lean 4 environment
cd verification_env && lake update

# Run local training
python3 train_grpo_fim_local.py
```

*Note: Local training is configured for testing with smaller models like `unsloth/Qwen2.5-0.5B-Instruct`.*

---

## Tinker Integration Architecture

The Tinker integration consists of the following components:

```
tinker_integration/
├── __init__.py              # Public exports
├── async_verifier.py        # Async wrapper for LeanVerifier
├── checkpoint.py            # Checkpoint save/load management
├── client.py                # Tinker API client setup
├── config.py                # YAML configuration management
├── env_group_builder.py     # Curriculum-aware environment builder
├── error_handler.py         # Retry logic and error aggregation
├── lean_env.py              # Tinker Env interface for Lean4
├── metrics.py               # Training metrics and logging
├── prompt_formatter.py      # FIM prompt construction
├── training_loop.py         # CISPO training loop
└── requirements.txt         # Tinker-specific dependencies
```

**Key Features:**
- **CISPO Loss**: Clipped Importance Sampling Policy Optimization—stable for MoE models
- **Curriculum Learning**: Progressive difficulty from 10% to 100% proof masking
- **Async Verification**: Parallel Lean4 verification with configurable concurrency
- **Checkpointing**: Save/resume training state including curriculum progress
- **Metrics**: JSONL logging with optional Weights & Biases integration

## License
MIT
