"""
Tinker API Integration for Lean4 FIM + RLVR.

This module provides adapters and utilities for training large MoE models
(particularly gpt-oss-120b) using Tinker API with CISPO loss for Lean4
proof infilling tasks.

Key Components:
- Lean4FIMEnv: Tinker Env adapter for Lean4 verification
- FIMPromptFormatter: Consistent FIM prompt construction
- CurriculumEnvGroupBuilder: Curriculum-aware environment group builder
- AsyncVerifier: Async wrapper for LeanVerifier
- CISPOTrainingLoop: Main training loop with CISPO loss
- ConfigManager: YAML-based configuration management
- MetricsLogger: Training metrics and logging
- CheckpointManager: Checkpoint save/restore
- ErrorHandler: Robust error handling with retry logic
"""

from typing import TYPE_CHECKING

# Version
__version__ = "0.1.0"

# Public API - lazy imports to avoid circular dependencies
# These will be populated as modules are implemented

__all__ = [
    # Core environment
    "Lean4FIMEnv",
    "Observation",
    "StopCondition",
    "StepResult",
    # Prompt formatting
    "FIMPromptFormatter",
    "PromptTemplate",
    # Environment group builder
    "CurriculumEnvGroupBuilder",
    "TheoremDataset",
    # Async verification
    "AsyncVerifier",
    # Training client
    "TinkerTrainingClient",
    "create_training_client",
    "create_training_client_sync",
    "ClientConfig",
    "LoRAConfig",
    "TinkerClientError",
    "TinkerAuthenticationError",
    "TinkerModelNotAvailableError",
    # Training loop
    "CISPOTrainingLoop",
    # Configuration
    "ConfigManager",
    "TrainingConfig",
    # Metrics
    "MetricsLogger",
    "TrainingMetrics",
    # Checkpointing
    "CheckpointManager",
    # Error handling
    "ErrorHandler",
    "ErrorStats",
]


# Lazy imports - components will be available once implemented
def __getattr__(name: str):
    """Lazy import of components."""
    if name in __all__:
        # Import from submodules as they are implemented
        if name in ("Lean4FIMEnv", "Observation", "StopCondition", "StepResult"):
            from .lean_env import Lean4FIMEnv, Observation, StopCondition, StepResult
            return locals()[name]
        elif name in ("FIMPromptFormatter", "PromptTemplate"):
            from .prompt_formatter import FIMPromptFormatter, PromptTemplate
            return locals()[name]
        elif name in ("CurriculumEnvGroupBuilder", "TheoremDataset"):
            from .env_group_builder import CurriculumEnvGroupBuilder, TheoremDataset
            return locals()[name]
        elif name == "AsyncVerifier":
            from .async_verifier import AsyncVerifier
            return AsyncVerifier
        elif name in (
            "TinkerTrainingClient", "create_training_client", 
            "create_training_client_sync", "ClientConfig", "LoRAConfig",
            "TinkerClientError", "TinkerAuthenticationError", 
            "TinkerModelNotAvailableError"
        ):
            from .client import (
                TinkerTrainingClient, create_training_client,
                create_training_client_sync, ClientConfig, LoRAConfig,
                TinkerClientError, TinkerAuthenticationError,
                TinkerModelNotAvailableError
            )
            return locals()[name]
        elif name == "CISPOTrainingLoop":
            from .training_loop import CISPOTrainingLoop
            return CISPOTrainingLoop
        elif name in ("ConfigManager", "TrainingConfig"):
            from .config import ConfigManager, TrainingConfig
            return locals()[name]
        elif name in ("MetricsLogger", "TrainingMetrics"):
            from .metrics import MetricsLogger, TrainingMetrics
            return locals()[name]
        elif name == "CheckpointManager":
            from .checkpoint import CheckpointManager
            return CheckpointManager
        elif name in ("ErrorHandler", "ErrorStats"):
            from .error_handler import ErrorHandler, ErrorStats
            return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
