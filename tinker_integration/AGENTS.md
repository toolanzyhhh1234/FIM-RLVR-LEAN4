# Tinker Integration Agent Guidelines

This document outlines specific guidelines and workflows for the `tinker_integration` module.

## Context for AI Agents
The `tinker_integration/` directory implements a remote training backend using the **Tinker API**. Unlike the local `unsloth`-based trainers in the root directory, this module coordinates with an external service to train very large Mixture-of-Experts (MoE) models (e.g., `gpt-oss-120b`) that cannot fit on local HW.

### Key Architecture
- **Training Loop** (`training_loop.py`): Implements `CISPOTrainingLoop`. It follows an **RLVR** (Reinforcement Learning with Verification Rewards) pattern:
    1.  **Sample**: Request synthetic completions from Tinker's inference API.
    2.  **Verify**: Run local Lean 4 verification (`lean_env.py`) to compute binary rewards.
    3.  **Update**: Send advantage-weighted data back to Tinker for `forward_backward` (CISPO loss) and `optim_step`.
- **Client** (`client.py`): `TinkerTrainingClient` manages authentication, LoRA configuration, and API communication. It supports automatic fallback models if the primary is unavailable.
- **Environment**: Adapts the core `fim_rlvr_lean4` verification logic to work with the async loop required by network calls.
- **State Management**:
    - **Checkpoints** (`checkpoint.py`): Manages saving/loading training state (LoRA weights + optimizer).
    - **Metrics** (`metrics.py`): structured logging of pass rates, rewards, and loss.

### Configuration & Data Flow
- **Config**: Managed by `ConfigManager` in `config.py`. Loads from `configs/tinker_training.yaml` and supports CLI overrides.
- **Data**: The `CurriculumEnvGroupBuilder` (`env_group_builder.py`) dynamically constructs batches of theorems based on difficulty, which are then turned into prompt prompts by `FIMPromptFormatter` (`prompt_formatter.py`).


### Supported Models
- **Primary**: `openai/gpt-oss-120b` (Ultra-sparse MoE, best for logic/reasoning)
- **Fallback**: `Qwen/Qwen3-235B-A22B` (Cost-effective MoE)
- Others: `meta-llama/Llama-3.1-70B` (Dense), `deepseek-ai/DeepSeek-V3.1` (MoE)

### Entry Point
- The main entry point is `../train_tinker_fim.py`, which instantiates `TinkerTrainingClient` and starts `CISPOTrainingLoop`.


## Common Agent Tasks
- **Adding Metrics**: Update `metrics.py`'s `TrainingMetrics` dataclass and `MetricsLogger`.
- **Modifying Curriculum**: Check `env_group_builder.py`. The curriculum logic determines which theorems are sampled.
- **Debugging Timeouts**: Look at `error_handler.py` and the `timeout` settings in `config.py`.
- **Inspecting Outputs**: Check `debug_samples.jsonl` in the log directory for raw model samples and verification results.
- **Monitoring**: Use `tinker run list` and `tinker run info <session-id>` to check remote job status.
- **Handling API Errors**: See `TinkerClientError` in `client.py`.


## Critical Constraints
- **Token Alignment**: For `cispo` loss, `input_tokens`, `target_tokens`, `logprobs`, and `advantages` must have exact matching lengths. Mismatches cause training failures.
- **Clock Cycles**: The training loop runs on a "clock cycle" model. `forward_backward` and `optim_step` should be dispatched concurrently where possible.
- **Keep-Alive**: The local script owns the control loop. If it crashes, the remote training stops. Exceptions must be caught and logged (see `ErrorHandler`).

## Development Guidelines
- **Async/Await**: This module is heavily async (`asyncio`) to handle network latency without blocking the verification of other samples.
- **Error Handling**: Network calls must be robust. `ErrorHandler` and `TinkerClientError` classes exist to manage retries and failures.
- **Testing**:
    - `../test_tinker_connection.py`: Smoke test for API connectivity.
    - `../test_tinker_minimal.py`: Minimal loop run.
    - Use `MockSamplingClient` (in `client.py`) for offline logic testing.
- **Security**: Do not commit API keys. Use `TINKER_API_KEY` environment variable.
- **Dependencies**: See `requirements.txt` in this directory.

## Related Files
- `../AGENTS.md` (Root guidelines)
- `../docs/tinker_api_integration_notes.md` (Detailed API specs, code snippets, and gotchas)

## Performance & Context Management
- **Context Saving**: AI models should aim to summarize and save key insights into their internal context or a scratchpad rather than re-reading large files repeatedly.
- **Log Reading**: When inspecting `logs/tinker_fim/debug_samples.jsonl`, **do NOT read the entire file or even full long entries**. Each JSON line contains raw Lean 4 code and long model responses that can quickly exhaust context windows. Use tools like `grep`, `jq`, or `tail` to isolate specific fields (e.g., `step`, `theorem_id`, `reward`, `verification_success`) without reading the full JSON objects unless the raw completion details are explicitly required.
