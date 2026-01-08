# Implementation Plan: Tinker API Integration

## Overview

This implementation plan adapts the existing Lean4 FIM + RLVR project to use Tinker API for training large MoE models with CISPO. The plan preserves existing components (LeanVerifier, CurriculumManager, masking) while adding Tinker-specific adapters and a new training loop.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `tinker_integration/` module directory
  - Add tinker-api to requirements
  - Create `__init__.py` with public exports
  - _Requirements: 4.1, 4.5_

- [x] 2. Implement FIM Prompt Formatter
  - [x] 2.1 Create `tinker_integration/prompt_formatter.py`
    - Implement `PromptTemplate` dataclass
    - Implement `FIMPromptFormatter` class with `format()` method
    - Handle empty suffix case (100% masking)
    - Preserve exact whitespace from input
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.2 Write property test for prompt formatting
    - **Property 1: FIM Prompt Construction Preserves Content**
    - **Validates: Requirements 1.2, 2.1, 2.5**

- [-] 3. Implement Lean4FIMEnv (Tinker Environment)
  - [x] 3.1 Create `tinker_integration/lean_env.py`
    - Implement `Observation`, `StopCondition`, `StepResult` dataclasses
    - Implement `Lean4FIMEnv` class with `initial_observation()` and `step()`
    - Integrate with existing `LeanVerifier` for reward computation
    - Store ground truth middle for debugging
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [ ]* 3.2 Write property test for proof reconstruction
    - **Property 2: Proof Reconstruction Round-Trip**
    - **Validates: Requirements 1.3**

- [x] 4. Implement AsyncVerifier
  - [x] 4.1 Create `tinker_integration/async_verifier.py`
    - Wrap `LeanVerifier` with async/await interface
    - Implement semaphore-based concurrency control
    - Add timeout handling with configurable duration
    - Integrate with MetricsLogger for latency tracking
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 4.2 Write property test for concurrency limit
    - **Property 5: Verification Concurrency Limit**
    - **Validates: Requirements 6.3**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 6. Implement CurriculumEnvGroupBuilder
  - [x] 6.1 Create `tinker_integration/env_group_builder.py`
    - Implement `TheoremDataset` class for Parquet loading
    - Implement `CurriculumEnvGroupBuilder` with `make_envs()` and `update_outcomes()`
    - Integrate with existing `CurriculumManager` for mask ratio selection
    - Use existing `apply_dynamic_mask()` for hole creation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 6.2 Write property test for group size
    - **Property 3: Environment Group Size Consistency**
    - **Validates: Requirements 3.3**

  - [ ]* 6.3 Write property test for curriculum sampling distribution
    - **Property 4: Curriculum Sampling Distribution**
    - **Validates: Requirements 3.5**

  - [ ]* 6.4 Write property test for dataset field extraction
    - **Property 6: Dataset Field Extraction**
    - **Validates: Requirements 7.2**


- [x] 7. Implement ConfigManager
  - [x] 7.1 Create `tinker_integration/config.py`
    - Implement `TrainingConfig` dataclass with defaults
    - Implement `ConfigManager` with YAML loading
    - Add environment variable override support
    - Add schema validation
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 7.2 Write property test for config resolution
    - **Property 9: Configuration Resolution with Overrides**
    - **Validates: Requirements 10.2**

  - [ ]* 7.3 Write property test for config defaults
    - **Property 10: Configuration Defaults**
    - **Validates: Requirements 10.4**

- [-] 8. Implement MetricsLogger
  - [x] 8.1 Create `tinker_integration/metrics.py`
    - Implement `TrainingMetrics` dataclass
    - Implement `MetricsLogger` with JSONL output
    - Add verification latency tracking
    - Add promotion event logging
    - Add token usage tracking
    - Add optional W&B integration
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 8.2 Write property test for JSON output format
    - **Property 7: Metrics Output Format**
    - **Validates: Requirements 8.5**

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement ErrorHandler
  - [x] 10.1 Create `tinker_integration/error_handler.py`
    - Implement `ErrorStats` dataclass
    - Implement `ErrorHandler` with exponential backoff retry
    - Add verification crash handling
    - Add theorem failure flagging
    - Add critical threshold detection
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 10.2 Write property test for retry backoff
    - **Property 11: Retry Exponential Backoff**
    - **Validates: Requirements 11.1**

- [x] 11. Implement CheckpointManager
  - [x] 11.1 Create `tinker_integration/checkpoint.py`
    - Implement `CheckpointManager` with save/load methods
    - Save LoRA weights via Tinker API
    - Save CurriculumManager state to JSON
    - Save training metadata (step count)
    - Store checkpoints on local filesystem (configurable directory)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 11.2 Write property test for checkpoint round-trip
    - **Property 8: Checkpoint State Round-Trip**
    - **Validates: Requirements 9.2, 9.4**

- [x] 12. Implement Tinker Training Client Setup
  - [x] 12.1 Create `tinker_integration/client.py`
    - Implement `create_training_client()` factory function
    - Configure for `gpt-oss-120b` with LoRA
    - Add API credential validation
    - Add fallback model support
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 12.2 Write unit tests for client setup
    - Test credential validation error messages
    - Test model configuration
    - _Requirements: 4.5, 4.6_

- [-] 13. Implement CISPO Training Loop
  - [x] 13.1 Create `tinker_integration/training_loop.py`
    - Implement `CISPOTrainingLoop` class
    - Implement `train()` async method with main loop
    - Implement `_sample_completions()` using Tinker sampler
    - Implement `_update_policy()` with CISPO loss
    - Integrate with MetricsLogger for step logging
    - Integrate with CheckpointManager for periodic saves
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 13.2 Write unit tests for training loop
    - Test reward computation from verification results
    - Test advantage calculation (group-relative baseline)
    - _Requirements: 5.1, 5.2_

- [x] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Create main training entrypoint
  - [x] 15.1 Create `train_tinker_fim.py`
    - Load configuration from YAML
    - Initialize all components
    - Run training loop
    - Handle graceful shutdown
    - _Requirements: All_

  - [x] 15.2 Create example configuration file
    - Create `configs/tinker_training.yaml` with documented options
    - Include sensible defaults for Lean4 FIM task
    - _Requirements: 10.1, 10.4_

- [ ] 16. Integration testing and documentation
  - [ ]* 16.1 Write integration test for end-to-end flow
    - Test 10 training steps with mock Tinker client
    - Verify curriculum updates
    - Verify checkpoint save/load
    - _Requirements: All_

  - [ ] 16.2 Update README with Tinker integration instructions
    - Add setup instructions for Tinker API key
    - Add example training command
    - Document configuration options
    - _Requirements: All_

- [ ] 17. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
