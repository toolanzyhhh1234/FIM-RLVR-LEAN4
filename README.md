# FIM-RLVR-LEAN4

**Fill-in-the-Middle (FIM) + Reinforcement Learning with Verification Rewards (RLVR) for Lean 4.**

### 🚀 Project Summary

### 🚀 Project Summary

**The Big Idea:**
Can we efficiently bootstrap formal mathematical reasoning in general-purpose models?
This project hypothesizes that **Fill-in-the-Middle (FIM)** tasks combined with **Reinforcement Learning (RLVR)** are the most effective way to "warm up" models that have little prior Lean 4 knowledge. Instead of training from scratch, we aim to rapidly align weak models to formal systems, allowing us to finetune them into rigorous theorem provers. We are also investigating whether this formal grounding transfers back to natural language mathematics (e.g., IMO Bench).

**For the Non-Expert (The Potential):**
Most AI models struggle with complex math because they don't "understand" the rules—they just memorize patterns. By training an AI to fill in missing gaps in a strict formal proof (FIM), we force it to understand the logic structure locally. If this works, it means we can take a standard AI and quickly teach it to be rigorous, potentially solving hard problems (like the Mathematical Olympiad) by checking its own work against a mathematical "truth machine."

**For the Expert (The Setup):**
This repository implements a **FIM + GSPO** pipeline pipeline designed to bootstrap formal capabilities:
*   **Hypothesis**: FIM is a superior objective for "warming up" models on formal languages compared to standard next-token prediction, especially for models with sparse pre-training on Lean.
*   **Method**: **GSPO (Group Sequence Policy Optimization)**—a variant of GRPO optimized using MoE models like GPT-OSS—samples candidates for missing tactic blocks.
*   **Feedback**: Verifiable reward signal from the Lean 4 compiler (Parallelized).
*   **Goal**: Demonstrate that RLVR on FIM tasks allows models to generalize better and potentially transfer reasoning tokens to natural language domains.

**Explore the details:**
*   [Technical Specification & Architecture](project-spec.md)
*   [Parallel Verification Walkthrough](walkthrough.md)

## Status: Proof-of-Concept / Verification Mode

**The code is functional and the training loop is working.**

We have successfully implemented:
- **Lean 4 Verification Pipeline**: A robust, thread-safe verifier that compiles generated code against a pinned Lean environment.
- **Parallel Verification**: Optimized logic achieving >2x speedup by verifying multiple candidates simultaneously.
- **GRPO Training Loop**: Integrated with Unsloth and TRL for efficient training.

However, we are currently operating on **limited compute resources** (running verification loops on local hardware with small models like `Qwen2.5-0.5B` to ensure the pipeline logic is sound).

## 🚀 Call for Sponsorship & Collaboration

**We are seeking support to scale this research.**

This project is currently running on a small personal budget. To fully validate the FIM-RLVR hypothesis—that verifiable feedback can significantly boost formal reasoning capabilities—we need to scale up to:
- Larger models (e.g., GPT-OSS-20B, GPT-OSS-120B, or any other model supported by sponsors).
- Massive datasets (e.g., full Mathlib, millions of synthetic samples).
- Extensive H100 GPU compute for full training runs.

**If you are interested in the results of this research or would like to sponsor the compute required to push this project to the next level, please reach out!** Your support would be extremely helpful in allowing us to continue iterating and potentially finding strong evidence for the efficacy of verification-driven RL in formal mathematics.

## Usage

1. **Setup**:
   ```bash
   # Install dependencies (Unsloth, TRL, etc.)
   pip install unsloth trl
   
   # Setup Lean 4 environment
   cd verification_env && lake update
   ```

2. **Run Training**:
   ```bash
   python3 train_grpo_fim.py
   ```
   *Note: Currently configured for local testing with `unsloth/Qwen2.5-0.5B-Instruct`.*

## License
MIT
