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
*   **Hypothesis**: FIM is a superior objective for "warming up" models on formal languages compared to next-token prediction, enabling rapid alignment to formal systems (Lean 4).
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

## Status: Proof-of-Concept / Verification Mode

**The code is functional and the training loop is working.**

We have successfully implemented:
- **Lean 4 Verification Pipeline**: A robust, thread-safe verifier that compiles generated code against a pinned Lean environment.
- **Parallel Verification**: Optimized logic achieving >2x speedup by verifying multiple candidates simultaneously.
- **GRPO Training Loop**: Integrated with Unsloth and TRL for efficient training.

However, we are currently operating on **limited compute resources** (running verification loops on local hardware with small models like `Qwen2.5-0.5B` to ensure the pipeline logic is sound). We are attempting to scale up to the `gpt-oss-20b` base model within our current budget, though the extent of training validation we can achieve remains uncertain.

## 🚀 Call for Sponsorship & Collaboration

**We are seeking support to scale this research.**

This project is currently running on a small personal budget. To fully validate the FIM-RLVR hypothesis—that verifiable feedback can significantly boost formal reasoning capabilities—we need to scale up to:
- Larger models (e.g., GPT-OSS-20B, GPT-OSS-120B, or any other model supported by sponsors).
- Massive datasets (e.g., full Mathlib, millions of synthetic samples).
- Extensive H100 GPU compute for full training runs.

**If you are interested in the results of this research or would like to sponsor the compute required to push this project to the next level, please reach out!** Your support would be extremely helpful in allowing us to continue iterating and potentially finding strong evidence for the efficacy of verification-driven RL in formal mathematics.

**Contact:** ifchou@student.unimelb.edu.au or open a GitHub issue.

## Contributing

Contributions and collaborators are welcome! If you share an interest in formal verification + RL, feel free to open a PR or reach out. I'm a student with other commitments, so responses may take a few days—but I genuinely appreciate the interest.

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
