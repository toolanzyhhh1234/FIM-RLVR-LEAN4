# FIM-RLVR-LEAN4

**Fill-in-the-Middle (FIM) + Reinforcement Learning with Verification Rewards (RLVR) for Lean 4.**

This project explores using Reinforcement Learning with verifiable feedback from the Lean 4 compiler to improve the ability of Large Language Models (LLMs) to generate valid Lean proofs. specifically focusing on "Fill-in-the-Middle" tasks where the model must reconstruct missing tactic blocks.

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
- Larger models (e.g., Llama-3-70B, DeepSeek-Prover).
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
