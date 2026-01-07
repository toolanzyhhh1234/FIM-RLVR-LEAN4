"""
Metrics logging for Tinker API integration.

This module provides comprehensive logging for training progress, verification
latency, curriculum state, and token usage. Supports JSONL output and optional
Weights & Biases integration.

Requirements covered:
- 8.1: Track pass rate per curriculum level (10%, 20%, ..., 100% masking)
- 8.2: Track promotion events (theorem advancing to harder level)
- 8.3: Track verification latency statistics
- 8.4: Track token usage for cost estimation
- 8.5: Output logs in JSON format for analysis
- 8.6: Support optional integration with Weights & Biases
"""

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Metrics for a single training step.
    
    Captures the key statistics at each training step for later analysis.
    
    Attributes:
        step: Training step number.
        timestamp: Unix timestamp when metrics were recorded.
        reward_mean: Mean reward across the batch.
        reward_std: Standard deviation of rewards.
        pass_rate: Fraction of completions that passed verification.
        curriculum_level_distribution: Optional distribution of curriculum levels.
    """
    step: int
    timestamp: float
    reward_mean: float
    reward_std: float
    pass_rate: float
    curriculum_level_distribution: Optional[Dict[str, int]] = None


@dataclass
class VerificationMetrics:
    """
    Metrics for verification events.
    
    Attributes:
        timestamp: Unix timestamp of the verification.
        latency: Time taken for verification in seconds.
        success: Whether verification succeeded.
        event: Optional event type (e.g., "timeout", "error").
        error: Optional error message.
    """
    timestamp: float
    latency: Optional[float] = None
    success: Optional[bool] = None
    event: Optional[str] = None
    error: Optional[str] = None


class MetricsLogger:
    """
    Logs training metrics to JSONL files and optionally W&B.
    
    Provides comprehensive logging for:
    - Training step metrics (reward, pass rate)
    - Verification latency statistics (Requirement 8.3)
    - Curriculum promotion events (Requirement 8.2)
    - Pass rate per curriculum level (Requirement 8.1)
    - Token usage for cost estimation (Requirement 8.4)
    - JSONL output format (Requirement 8.5)
    - Optional W&B integration (Requirement 8.6)
    
    Example:
        >>> logger = MetricsLogger(
        ...     log_dir="./logs",
        ...     wandb_project="lean4-fim",
        ...     wandb_run_name="experiment-1",
        ... )
        >>> 
        >>> # Log training step
        >>> logger.log_training_step(
        ...     step=100,
        ...     reward_mean=0.65,
        ...     reward_std=0.12,
        ...     pass_rate=0.65,
        ... )
        >>> 
        >>> # Log verification latency
        >>> logger.log_verification_latency(latency=2.5, success=True)
        >>> 
        >>> # Log promotion
        >>> logger.log_promotion(theorem_id="thm_001", new_level=0.3)
        >>> 
        >>> # Get summary
        >>> summary = logger.get_summary()
    
    Attributes:
        log_dir: Directory for log files.
        wandb_project: Optional W&B project name.
        wandb_run_name: Optional W&B run name.
    """
    
    def __init__(
        self,
        log_dir: str,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to write log files. Created if doesn't exist.
            wandb_project: Optional W&B project name for cloud logging.
            wandb_run_name: Optional W&B run name. Auto-generated if not provided.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self._metrics_file = self.log_dir / "metrics.jsonl"
        self._verification_file = self.log_dir / "verification.jsonl"
        
        # W&B integration (Requirement 8.6)
        self._wandb = None
        self._wandb_project = wandb_project
        self._wandb_run_name = wandb_run_name
        if wandb_project:
            self._init_wandb(wandb_project, wandb_run_name)
        
        # Aggregated statistics
        self._verification_latencies: List[float] = []
        self._pass_rates_by_level: Dict[float, List[bool]] = defaultdict(list)
        self._promotion_count = 0
        self._token_count = 0
        self._timeout_count = 0
        self._error_count = 0
        self._total_verifications = 0
        self._successful_verifications = 0
        
        logger.info(f"MetricsLogger initialized with log_dir={log_dir}")
    
    def _init_wandb(self, project: str, run_name: Optional[str]) -> None:
        """
        Initialize Weights & Biases integration.
        
        Args:
            project: W&B project name.
            run_name: Optional run name.
        """
        try:
            import wandb
            self._wandb = wandb.init(
                project=project,
                name=run_name,
                resume="allow",  # Allow resuming if run exists
            )
            logger.info(f"W&B initialized: project={project}, run={run_name}")
        except ImportError:
            logger.warning(
                "wandb not installed. Install with 'pip install wandb' "
                "to enable W&B integration."
            )
            self._wandb = None
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self._wandb = None
    
    def _write_jsonl(self, filepath: Path, record: Dict[str, Any]) -> None:
        """
        Write a record to a JSONL file.
        
        Args:
            filepath: Path to the JSONL file.
            record: Dictionary to write as JSON.
        """
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to {filepath}: {e}")
    
    def log_training_step(
        self,
        step: int,
        reward_mean: float,
        reward_std: float,
        pass_rate: float,
        curriculum_level_distribution: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Log metrics for a training step.
        
        Writes to JSONL file and optionally to W&B.
        
        Args:
            step: Training step number.
            reward_mean: Mean reward across the batch.
            reward_std: Standard deviation of rewards.
            pass_rate: Fraction of completions that passed verification.
            curriculum_level_distribution: Optional distribution of curriculum levels.
        """
        metrics = TrainingMetrics(
            step=step,
            timestamp=time.time(),
            reward_mean=reward_mean,
            reward_std=reward_std,
            pass_rate=pass_rate,
            curriculum_level_distribution=curriculum_level_distribution,
        )
        
        # Write to JSONL (Requirement 8.5)
        self._write_jsonl(self._metrics_file, asdict(metrics))
        
        # Log to W&B if enabled (Requirement 8.6)
        if self._wandb:
            wandb_metrics = {
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/pass_rate": pass_rate,
            }
            if curriculum_level_distribution:
                for level, count in curriculum_level_distribution.items():
                    wandb_metrics[f"curriculum/level_{level}"] = count
            self._wandb.log(wandb_metrics, step=step)
    
    def log_verification_latency(self, latency: float, success: bool) -> None:
        """
        Track verification latency statistics.
        
        Implements Requirement 8.3.
        
        Args:
            latency: Time taken for verification in seconds.
            success: Whether verification succeeded.
        """
        self._verification_latencies.append(latency)
        self._total_verifications += 1
        if success:
            self._successful_verifications += 1
        
        record = {
            "timestamp": time.time(),
            "latency": latency,
            "success": success,
        }
        self._write_jsonl(self._verification_file, record)
        
        # Log to W&B if enabled
        if self._wandb:
            self._wandb.log({
                "verification/latency": latency,
                "verification/success": 1 if success else 0,
            })
    
    def log_verification_timeout(self) -> None:
        """
        Log a verification timeout event.
        
        Part of Requirement 8.3 - tracking verification issues.
        """
        self._timeout_count += 1
        self._total_verifications += 1
        
        record = {
            "timestamp": time.time(),
            "event": "timeout",
        }
        self._write_jsonl(self._verification_file, record)
        
        if self._wandb:
            self._wandb.log({"verification/timeout": 1})
    
    def log_verification_error(self, error: str) -> None:
        """
        Log a verification error.
        
        Part of Requirement 8.3 - tracking verification issues.
        
        Args:
            error: Error message describing what went wrong.
        """
        self._error_count += 1
        self._total_verifications += 1
        
        record = {
            "timestamp": time.time(),
            "event": "error",
            "error": error,
        }
        self._write_jsonl(self._verification_file, record)
        
        if self._wandb:
            self._wandb.log({"verification/error": 1})
    
    def log_promotion(self, theorem_id: str, new_level: float) -> None:
        """
        Log a curriculum promotion event.
        
        Implements Requirement 8.2.
        
        Args:
            theorem_id: Identifier of the theorem being promoted.
            new_level: The new curriculum level (mask ratio).
        """
        self._promotion_count += 1
        
        record = {
            "timestamp": time.time(),
            "event": "promotion",
            "theorem_id": theorem_id,
            "new_level": new_level,
        }
        self._write_jsonl(self._metrics_file, record)
        
        if self._wandb:
            self._wandb.log({
                "curriculum/promotions": self._promotion_count,
                "curriculum/latest_promotion_level": new_level,
            })
        
        logger.debug(f"Promotion logged: {theorem_id} -> level {new_level}")
    
    def log_pass_rate_by_level(self, level: float, success: bool) -> None:
        """
        Track pass rate per curriculum level.
        
        Implements Requirement 8.1.
        
        Args:
            level: Curriculum level (mask ratio, e.g., 0.1, 0.2, ..., 1.0).
            success: Whether the completion passed verification.
        """
        self._pass_rates_by_level[level].append(success)
    
    def log_token_usage(self, tokens: int) -> None:
        """
        Track token usage for cost estimation.
        
        Implements Requirement 8.4.
        
        Args:
            tokens: Number of tokens used (input + output).
        """
        self._token_count += tokens
        
        if self._wandb:
            self._wandb.log({"tokens/total": self._token_count})
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log effective configuration.
        
        Called at training start to record the configuration used.
        
        Args:
            config: Configuration dictionary (sensitive values should be masked).
        """
        record = {
            "timestamp": time.time(),
            "event": "config",
            "config": config,
        }
        self._write_jsonl(self._metrics_file, record)
        
        if self._wandb:
            self._wandb.config.update(config)
        
        logger.info("Configuration logged to metrics file")
    
    def log_custom_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Log a custom event with arbitrary data.
        
        Useful for logging events not covered by other methods.
        
        Args:
            event_name: Name of the event.
            data: Dictionary of event data.
        """
        record = {
            "timestamp": time.time(),
            "event": event_name,
            **data,
        }
        self._write_jsonl(self._metrics_file, record)
        
        if self._wandb:
            self._wandb.log({f"custom/{event_name}": 1, **data})
    
    def get_pass_rates_by_level(self) -> Dict[float, float]:
        """
        Get pass rate statistics per curriculum level.
        
        Implements Requirement 8.1.
        
        Returns:
            Dictionary mapping curriculum level to pass rate (0.0 to 1.0).
        """
        result = {}
        for level, outcomes in self._pass_rates_by_level.items():
            if outcomes:
                result[level] = sum(outcomes) / len(outcomes)
            else:
                result[level] = 0.0
        return result
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get verification statistics.
        
        Returns:
            Dictionary containing verification statistics.
        """
        latencies = self._verification_latencies
        return {
            "total_verifications": self._total_verifications,
            "successful_verifications": self._successful_verifications,
            "timeouts": self._timeout_count,
            "errors": self._error_count,
            "success_rate": (
                self._successful_verifications / self._total_verifications
                if self._total_verifications > 0 else 0.0
            ),
            "avg_latency": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "min_latency": min(latencies) if latencies else 0.0,
            "max_latency": max(latencies) if latencies else 0.0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the training run.
        
        Returns:
            Dictionary containing:
                - total_tokens: Total tokens used
                - total_promotions: Number of curriculum promotions
                - avg_verification_latency: Average verification time
                - pass_rates_by_level: Pass rate per curriculum level
                - verification_stats: Detailed verification statistics
        """
        verification_stats = self.get_verification_stats()
        
        return {
            "total_tokens": self._token_count,
            "total_promotions": self._promotion_count,
            "avg_verification_latency": verification_stats["avg_latency"],
            "pass_rates_by_level": self.get_pass_rates_by_level(),
            "verification_stats": verification_stats,
        }
    
    def save_summary(self, filepath: Optional[str] = None) -> str:
        """
        Save summary statistics to a JSON file.
        
        Args:
            filepath: Optional path for the summary file.
                     Defaults to log_dir/summary.json.
        
        Returns:
            Path to the saved summary file.
        """
        if filepath is None:
            filepath = str(self.log_dir / "summary.json")
        
        summary = self.get_summary()
        summary["saved_at"] = time.time()
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {filepath}")
        return filepath
    
    def close(self) -> None:
        """
        Close the metrics logger and finalize W&B run.
        
        Should be called at the end of training to ensure all data is flushed.
        """
        # Save final summary
        self.save_summary()
        
        # Finish W&B run
        if self._wandb:
            try:
                self._wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the logger."""
        self.close()
        return False
