# Requirements Document

## Introduction

This specification defines the integration of Tinker API (by Thinking Machines Lab) into the Lean4 FIM + RLVR project. The integration enables cost-effective training of large MoE models (particularly `gpt-oss-120b`) for Lean4 proof infilling tasks using CISPO (Clipped Importance Sampling Policy Optimization) instead of GRPO, which is unstable with MoE architectures.

The core value proposition is replacing expensive self-managed GPU infrastructure with Tinker's token-based pricing model, where costs scale with active parameters rather than total parameters—making ultra-sparse MoE models economically viable for research.

## Glossary

- **Tinker_API**: A cloud-based API by Thinking Machines Lab that provides low-level primitives (`forward_backward()`, `optim_step()`, `sample()`) for fine-tuning large language models with RL algorithms.
- **CISPO**: Clipped Importance Sampling Policy Optimization—an RL algorithm that clips importance sampling weights rather than policy updates, proven stable for MoE models (introduced in MiniMax-M1).
- **GRPO**: Group Relative Policy Optimization—an RL algorithm that uses token-level clipping, which causes expert routing collapse in MoE architectures.
- **FIM**: Fill-in-the-Middle—a task format where the model completes a missing segment given prefix and suffix context.
- **RLVR**: Reinforcement Learning with Verification Rewards—using compiler verification (Lean4) as the reward signal.
- **MoE**: Mixture of Experts—a sparse model architecture where only a subset of parameters (active parameters) are used per forward pass.
- **Lean4_Verifier**: The existing `LeanVerifier` class that compiles Lean4 code and returns binary success/failure.
- **Curriculum_Manager**: The existing `CurriculumManager` class that tracks per-theorem mastery and determines hole ratios.
- **Env**: Tinker's environment interface for RL training, defining `initial_observation()` and `step()` methods.
- **EnvGroupBuilder**: Tinker's interface for creating groups of environments for batch sampling.
- **Trajectory**: A sequence of (observation, action, reward) tuples representing one episode.

## Requirements

### Requirement 1: Tinker Environment Adapter

**User Story:** As a researcher, I want to wrap the Lean4 verification logic in Tinker's `Env` interface, so that I can use Tinker's RL training infrastructure with my existing verification pipeline.

#### Acceptance Criteria

1. THE Lean4FIMEnv SHALL implement Tinker's `Env` interface with `initial_observation()` and `step()` methods
2. WHEN `initial_observation()` is called, THE Lean4FIMEnv SHALL return a FIM-formatted prompt containing prefix, hole marker, and suffix
3. WHEN `step()` receives a model completion, THE Lean4FIMEnv SHALL reconstruct the full proof by concatenating prefix + completion + suffix
4. WHEN `step()` reconstructs a proof, THE Lean4FIMEnv SHALL invoke the existing `LeanVerifier.verify()` method
5. WHEN verification succeeds, THE Lean4FIMEnv SHALL return reward=1.0 and episode_done=True
6. WHEN verification fails, THE Lean4FIMEnv SHALL return reward=0.0 and episode_done=True
7. THE Lean4FIMEnv SHALL accept configurable `max_tokens` for the stop condition (default: 512)
8. THE Lean4FIMEnv SHALL store the ground truth middle segment for optional logging/debugging

### Requirement 2: FIM Prompt Formatting

**User Story:** As a researcher, I want consistent FIM prompt formatting that works with large MoE models, so that the model can reliably understand and complete the infilling task.

#### Acceptance Criteria

1. THE FIM_Prompt_Formatter SHALL construct prompts in the format: `{prefix}[MISSING_BLOCK]\n{suffix}`
2. THE FIM_Prompt_Formatter SHALL prepend a system instruction: "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. Output ONLY the missing code."
3. WHEN the suffix is empty (100% masking), THE FIM_Prompt_Formatter SHALL omit the suffix and adjust the instruction accordingly
4. THE FIM_Prompt_Formatter SHALL be configurable to use alternative prompt templates for experimentation
5. THE FIM_Prompt_Formatter SHALL preserve exact whitespace and newlines from the original proof segments

### Requirement 3: Curriculum-Aware Environment Group Builder

**User Story:** As a researcher, I want to integrate the existing curriculum logic with Tinker's group sampling, so that I can train with mastery-based hole-size progression.

#### Acceptance Criteria

1. THE CurriculumEnvGroupBuilder SHALL implement Tinker's `EnvGroupBuilder` interface
2. WHEN creating environments, THE CurriculumEnvGroupBuilder SHALL query the existing `CurriculumManager` for the appropriate mask ratio
3. THE CurriculumEnvGroupBuilder SHALL create G environments per theorem (configurable, default: 4)
4. WHEN trajectories complete, THE CurriculumEnvGroupBuilder SHALL call `CurriculumManager.update_outcome()` with verification results
5. THE CurriculumEnvGroupBuilder SHALL support the 70/20/10 sampling policy (current/review/challenge levels)
6. THE CurriculumEnvGroupBuilder SHALL accept a dataset of theorems and sample from it according to curriculum state

### Requirement 4: Tinker Training Client Setup

**User Story:** As a researcher, I want to configure Tinker's training client with appropriate model and LoRA settings, so that I can fine-tune large MoE models efficiently.

#### Acceptance Criteria

1. THE Tinker_Training_Client SHALL be configured to use `gpt-oss-120b` as the primary model
2. THE Tinker_Training_Client SHALL use LoRA fine-tuning with configurable rank (default: 16)
3. THE Tinker_Training_Client SHALL use CISPO as the loss function (`loss_fn="cispo"`)
4. THE Tinker_Training_Client SHALL support fallback to alternative models (e.g., `Qwen/Qwen3-235B-A22B`)
5. WHEN initializing, THE Tinker_Training_Client SHALL validate API credentials and model availability
6. IF API credentials are invalid, THEN THE Tinker_Training_Client SHALL raise a descriptive error

### Requirement 5: CISPO Training Loop

**User Story:** As a researcher, I want a training loop that uses CISPO with group sampling, so that I can train MoE models stably without expert routing collapse.

#### Acceptance Criteria

1. THE CISPO_Training_Loop SHALL sample G completions per environment using Tinker's `sample()` primitive
2. THE CISPO_Training_Loop SHALL compute rewards by running Lean4 verification on each completion
3. THE CISPO_Training_Loop SHALL call Tinker's `forward_backward()` with CISPO loss and computed rewards
4. THE CISPO_Training_Loop SHALL call Tinker's `optim_step()` to update LoRA weights
5. THE CISPO_Training_Loop SHALL run for a configurable number of steps (default: 1000)
6. THE CISPO_Training_Loop SHALL log training metrics (loss, reward mean, reward std) every N steps
7. THE CISPO_Training_Loop SHALL checkpoint LoRA weights at configurable intervals

### Requirement 6: Async Verification for Throughput

**User Story:** As a researcher, I want verification to run asynchronously, so that I can maximize training throughput by overlapping verification with model sampling.

#### Acceptance Criteria

1. THE Async_Verifier SHALL wrap `LeanVerifier` with async/await interface
2. WHEN multiple completions need verification, THE Async_Verifier SHALL run them concurrently
3. THE Async_Verifier SHALL respect a configurable concurrency limit (default: 8 parallel verifications)
4. THE Async_Verifier SHALL handle verification timeouts gracefully (default: 60 seconds per proof)
5. IF verification times out, THEN THE Async_Verifier SHALL return reward=0.0 and log the timeout

### Requirement 7: Dataset Integration

**User Story:** As a researcher, I want to load theorem data from existing Parquet datasets, so that I can reuse the data pipeline already built for this project.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL load theorems from Parquet files (existing format)
2. THE Dataset_Loader SHALL extract `theorem_id`, `prefix`, `suffix`, `middle` fields
3. THE Dataset_Loader SHALL support filtering by theorem difficulty or source
4. THE Dataset_Loader SHALL integrate with `CurriculumManager` to track per-theorem state
5. WHEN a theorem is sampled, THE Dataset_Loader SHALL apply dynamic masking using the existing `apply_dynamic_mask()` function

### Requirement 8: Metrics and Logging

**User Story:** As a researcher, I want comprehensive logging of training progress, so that I can analyze curriculum effectiveness and model improvement.

#### Acceptance Criteria

1. THE Metrics_Logger SHALL track pass rate per curriculum level (10%, 20%, ..., 100% masking)
2. THE Metrics_Logger SHALL track promotion events (theorem advancing to harder level)
3. THE Metrics_Logger SHALL track verification latency statistics
4. THE Metrics_Logger SHALL track token usage for cost estimation
5. THE Metrics_Logger SHALL output logs in JSON format for analysis
6. THE Metrics_Logger SHALL support optional integration with Weights & Biases

### Requirement 9: Checkpoint and Resume

**User Story:** As a researcher, I want to save and resume training state, so that I can recover from interruptions and continue long training runs.

#### Acceptance Criteria

1. THE Checkpoint_Manager SHALL save LoRA weights at configurable intervals
2. THE Checkpoint_Manager SHALL save `CurriculumManager` state (per-theorem levels and history)
3. THE Checkpoint_Manager SHALL save training step count and optimizer state
4. WHEN resuming, THE Checkpoint_Manager SHALL restore all state and continue from the last checkpoint
5. THE Checkpoint_Manager SHALL save checkpoints to a configurable local filesystem directory

### Requirement 10: Configuration Management

**User Story:** As a researcher, I want a unified configuration system, so that I can easily adjust hyperparameters and experiment settings.

#### Acceptance Criteria

1. THE Config_Manager SHALL load configuration from YAML files
2. THE Config_Manager SHALL support environment variable overrides for sensitive values (API keys)
3. THE Config_Manager SHALL validate configuration schema on load
4. THE Config_Manager SHALL provide sensible defaults for all optional parameters
5. THE Config_Manager SHALL log the effective configuration at training start

### Requirement 11: Error Handling and Retry

**User Story:** As a researcher, I want robust error handling for API and verification failures, so that training can continue despite transient issues.

#### Acceptance Criteria

1. WHEN Tinker API calls fail with transient errors, THE Error_Handler SHALL retry with exponential backoff (max 3 retries)
2. WHEN Lean verification crashes (not just fails), THE Error_Handler SHALL log the error and return reward=0.0
3. WHEN a theorem consistently fails verification, THE Error_Handler SHALL flag it for review but continue training
4. THE Error_Handler SHALL aggregate error statistics for post-training analysis
5. IF critical errors exceed a threshold, THEN THE Error_Handler SHALL pause training and alert the user


