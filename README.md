# FIM-RLVR-LEAN4

**Fill-in-the-Middle (FIM) + Reinforcement Learning with Verification Rewards (RLVR) for Lean 4.**

### 🚀 Project Summary

**The Big Idea:**
We are training AI to perform rigorous mathematical reasoning by forcing it to "fill in the blanks" of formal proofs and using a compiler (Lean 4) to verify the results. Instead of just predicting the next word, the AI learns from a feedback loop of absolute truth.

**For the Non-Expert (The Potential):**
Most AI models today are like students who guess the answer and hope it sounds right. This project builds an AI that acts like a mathematician: it attempts a logical step, checks if it works using a strict "truth machine" (the Lean 4 compiler), and learns from its successes. The goal is to create AI systems that don't just hallucinate convincing answers, but can solve complex problems with **100% verified reliability**.

**For the Expert (The Setup):**
This repository implements a **Fill-in-the-Middle (FIM)** training pipeline using **Reinforcement Learning with Verifiable Rewards (RLVR)**.
*   **Method**: We use **GSPO (Group Sequence Policy Optimization)**—a variant of GRPO optimized for stability with MoE models like GPT-OSS—to sample multiple candidate solutions for missing tactic blocks.
*   **Feedback**: These candidates are verified **in parallel** against the Lean 4 kernel.
*   **Objective**: The agent optimizes for a binary reward signal derived purely from successful compilation, grounding the LLM in formal logic without needing human preference labels.

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
