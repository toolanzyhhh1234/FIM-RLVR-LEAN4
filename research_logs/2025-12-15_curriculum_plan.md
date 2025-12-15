# Curriculum Plan for Lean 4 FIM-RLVR

## Overview
We aim to train a `gpt-oss-20b` formulation to solve Lean 4 theorems by filling in the missing proofs. We will use a curriculum approach to gradually increase difficulty.

## Stage 0: MVP Sanity Check (Current)
- **Goal**: Verify the RLVR pipeline (Data -> Model -> Reward -> Verifier).
- **Data**: `data/fim_fresh.jsonl` (1000 samples).
- **Format**: Exact line FIM (Random 15% missing).
- **Success Metric**: Non-negative reward average, successful compilations > 5%.

## Stage 1: Syntax & Small Holes
- **Goal**: Teach the model to respect Lean syntax and indentation when filling small gaps.
- **Difficulty**: Easy.
- **Masking**: 
  - Random 1-3 lines.
  - Focus on simple tactic steps (`rw`, `simp`, `apply`).
- **Data Size**: ~10k samples.

## Stage 2: Tactic Block Logic
- **Goal**: Complete coherent functional blocks of the proof.
- **Difficulty**: Medium.
- **Masking**: 
  - 3-10 lines.
  - Masking `have` blocks or `calc` blocks.
- **Data Size**: ~50k samples.

## Stage 3: Full Proof Synthesis (The "Holy Grail")
- **Goal**: Generate the entire proof body given the theorem statement.
- **Difficulty**: Hard.
- **Masking**: 
  - Mask everything between `:= by` and the end of the file/theorem.
- **Strategy**: 
  - This effectively turns FIM into "Conditioned Generation".
  - We might need "Thought Augmentation" (Chain of Thought) here to let the model plan before generating code.

## Thought Augmentation Strategy
To bridge the gap between Stage 2 and 3, we plan to inject `<think>` tokens:
1.  **Generate Thoughts**: Use a teacher model (e.g., DeepSeek-V3/R1 or GPT-4o) to explain the *Gold Proof* given the *Context*.
2.  **Train**: `Context -> <think> Explanation </think> -> Proof`.
3.  **Inference**: Model generates its own thought process before committing to code.
