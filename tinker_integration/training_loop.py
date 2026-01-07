"""
CISPO Training Loop for Tinker API integration.

This module implements the main training loop using Tinker's primitives
with CISPO (Clipped Importance Sampling Policy Optimization) loss for
training large MoE models on Lean4 proof infilling tasks.

Requirements covered:
- 5.1: Sample G completions per environment using Tinker's sample() primitive
- 5.2: Compute rewards by running Lean4 verification on each completion
- 5.3: Call Tinker's forward_backward() with CISPO loss and computed rewards
- 5.4: Call Tinker's optim_step() to update LoRA weights
- 5.5: Run for configurable number of steps (default: 1000)
- 5.6: Log training metrics (loss, reward mean, reward std) every N steps
- 5.7: Checkpoint LoRA weights at configurable intervals
"""

import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager
    from .config import TrainingConfig
    from .env_group_builder import CurriculumEnvGroupBuilder
    from .error_handler import ErrorHandler
    from .lean_env import Lean4FIMEnv, StepResult
    from .metrics import MetricsLogger


logger = logging.getLogger(__name__)


@dataclass
class ModelInput:
    """
    Input format for Tinker's forward_backward.
    
    Attributes:
        tokens: List of token IDs (prompt + completion).
        length: Total length of the sequence.
    """
    tokens: List[int]
    length: int


@dataclass
class SampleResult:
    """
    Result from Tinker's sample() call.
    
    Attributes:
        tokens: Generated token IDs.
        logprobs: Log probabilities for each generated token.
    """
    tokens: List[int]
    logprobs: np.ndarray


@dataclass
class Trajectory:
    """
    Complete trajectory for one episode.
    
    Attributes:
        observation_tokens: Token IDs of the initial observation (prompt).
        action_tokens: Token IDs of the model's completion.
        reward: Reward from verification (1.0 for success, 0.0 for failure).
        logprobs: Log probabilities of the action tokens.
    """
    observation_tokens: List[int]
    action_tokens: List[int]
    reward: float
    logprobs: np.ndarray


class CISPOTrainingLoop:
    """
    RLVR training loop using Tinker API with CISPO loss.
    
    Implements the core RL loop for training large MoE models on Lean4
    proof infilling tasks:
    
    1. Sample completions from policy (Requirement 5.1)
    2. Verify with Lean4 and compute rewards (Requirement 5.2)
    3. Update policy with CISPO loss (Requirements 5.3, 5.4)
    4. Log metrics and checkpoint periodically (Requirements 5.6, 5.7)
    
    The loop uses group-relative advantage computation, where the baseline
    is the mean reward within each group of completions for the same theorem.
    This is essential for stable CISPO training.
    
    Example:
        >>> loop = CISPOTrainingLoop(
        ...     training_client=client,
        ...     env_group_builder=builder,
        ...     config=config,
        ...     metrics_logger=metrics,
        ...     checkpoint_manager=checkpointer,
        ... )
        >>> await loop.train()
    
    Attributes:
        client: Tinker TrainingClient for model operations.
        env_builder: CurriculumEnvGroupBuilder for creating environments.
        config: TrainingConfig with hyperparameters.
        metrics: MetricsLogger for tracking progress.
        checkpointer: CheckpointManager for saving state.
        error_handler: Optional ErrorHandler for robust error handling.
        step_count: Current training step.
    """
    
    def __init__(
        self,
        training_client: Any,
        env_group_builder: "CurriculumEnvGroupBuilder",
        config: "TrainingConfig",
        metrics_logger: "MetricsLogger",
        checkpoint_manager: "CheckpointManager",
        error_handler: Optional["ErrorHandler"] = None,
        start_step: int = 0,
    ):
        """
        Initialize the CISPO training loop.
        
        Args:
            training_client: Tinker TrainingClient for model operations.
            env_group_builder: CurriculumEnvGroupBuilder for creating environments.
            config: TrainingConfig with hyperparameters.
            metrics_logger: MetricsLogger for tracking progress.
            checkpoint_manager: CheckpointManager for saving state.
            error_handler: Optional ErrorHandler for robust error handling.
            start_step: Starting step number (for resuming from checkpoint).
        """
        self.client = training_client
        self.env_builder = env_group_builder
        self.config = config
        self.metrics = metrics_logger
        self.checkpointer = checkpoint_manager
        self.error_handler = error_handler
        self.step_count = start_step
        
        # Track training state
        self._running = False
        self._should_stop = False
        self._current_sampler = None
        
        # Statistics
        self._total_tokens_generated = 0
        self._total_verifications = 0
        self._total_successes = 0
    
    async def train(self) -> Dict[str, Any]:
        """
        Run the training loop for configured number of steps.
        
        Implements Requirements 5.1-5.7:
        - Samples completions using Tinker's sample() primitive
        - Computes rewards via Lean4 verification
        - Updates policy with CISPO loss
        - Logs metrics every logging_steps
        - Checkpoints every checkpoint_interval
        
        Returns:
            Dictionary containing training summary statistics.
            
        Raises:
            RuntimeError: If critical error threshold is exceeded.
        """
        self._running = True
        self._should_stop = False
        
        logger.info(
            f"Starting CISPO training loop: "
            f"steps={self.config.max_steps}, "
            f"group_size={self.config.group_size}, "
            f"lr={self.config.learning_rate}"
        )
        
        try:
            for step in range(self.step_count, self.config.max_steps):
                if self._should_stop:
                    logger.info(f"Training stopped at step {step}")
                    break
                
                self.step_count = step
                
                # Execute one training step
                step_metrics = await self._train_step(step)
                
                # Log metrics periodically (Requirement 5.6)
                if step % self.config.logging_steps == 0:
                    self._log_step_metrics(step_metrics, step)
                
                # Checkpoint periodically (Requirement 5.7)
                if step > 0 and step % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint(step)
            
            # Final checkpoint
            if not self._should_stop:
                await self._save_checkpoint(self.step_count)
            
        except Exception as e:
            logger.error(f"Training error at step {self.step_count}: {e}")
            # Try to save emergency checkpoint
            try:
                await self._save_checkpoint(self.step_count, emergency=True)
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            raise
        finally:
            self._running = False
        
        return self._get_training_summary()
    
    async def _train_step(self, step: int) -> Dict[str, Any]:
        """
        Execute a single training step.
        
        Args:
            step: Current step number.
            
        Returns:
            Dictionary containing step metrics.
        """
        # 1. Create environment group (Requirement 5.1)
        envs = self.env_builder.make_envs()
        
        # 2. Get initial observations
        observations = []
        stop_conditions = []
        for env in envs:
            obs, stop = env.initial_observation()
            observations.append(obs)
            stop_conditions.append(stop)
        
        # 3. Sample completions from policy (Requirement 5.1)
        completions = await self._sample_completions(
            observations,
            stop_conditions,
            step
        )
        
        # 4. Execute steps and get rewards (Requirement 5.2)
        results = []
        trajectories = []
        for env, obs, completion in zip(envs, observations, completions):
            result = await self._execute_step(env, completion)
            results.append(result)
            
            # Build trajectory for policy update
            trajectory = Trajectory(
                observation_tokens=obs.tokens,
                action_tokens=completion.tokens,
                reward=result.reward,
                logprobs=completion.logprobs,
            )
            trajectories.append(trajectory)
        
        # 5. Update curriculum based on outcomes
        self.env_builder.update_outcomes(results)
        
        # 6. Compute advantages and update policy (Requirements 5.3, 5.4)
        rewards = np.array([r.reward for r in results])
        await self._update_policy(trajectories, rewards)
        
        # Track statistics
        self._total_verifications += len(results)
        self._total_successes += int(rewards.sum())
        
        return {
            "rewards": rewards,
            "pass_rate": float((rewards > 0.5).mean()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "theorem_id": self.env_builder.get_current_theorem_id(),
            "mask_ratio": self.env_builder.get_current_mask_ratio(),
        }
    
    async def _sample_completions(
        self,
        observations: List[Any],
        stop_conditions: List[Any],
        step: int,
    ) -> List[SampleResult]:
        """
        Sample completions using Tinker's sampling client.
        
        Implements Requirement 5.1: Sample G completions per environment.
        
        Args:
            observations: List of Observation objects with tokenized prompts.
            stop_conditions: List of StopCondition objects with max_tokens.
            step: Current training step (used for checkpoint naming).
            
        Returns:
            List of SampleResult objects with generated tokens and logprobs.
        """
        # Get sampling client from training client
        # This saves current weights and returns a client configured with them
        sampler = await self.client.save_weights_and_get_sampling_client_async(
            f"step_{step}"
        )
        self._current_sampler = sampler
        
        completions = []
        for obs, stop in zip(observations, stop_conditions):
            result = await sampler.sample_async(
                prompt_tokens=obs.tokens,
                max_tokens=stop.max_tokens,
                temperature=self.config.temperature,
            )
            
            # Convert to SampleResult if needed
            if isinstance(result, SampleResult):
                completions.append(result)
            else:
                # Handle different result formats from Tinker API
                completions.append(SampleResult(
                    tokens=result.tokens if hasattr(result, 'tokens') else list(result),
                    logprobs=result.logprobs if hasattr(result, 'logprobs') else np.zeros(len(result.tokens)),
                ))
            
            # Track token usage
            self._total_tokens_generated += len(completions[-1].tokens)
            self.metrics.log_token_usage(len(completions[-1].tokens))
        
        return completions
    
    async def _execute_step(
        self,
        env: "Lean4FIMEnv",
        completion: SampleResult,
    ) -> "StepResult":
        """
        Execute environment step with error handling.
        
        Args:
            env: Lean4FIMEnv instance.
            completion: SampleResult with generated tokens.
            
        Returns:
            StepResult from the environment.
        """
        try:
            result = await asyncio.wait_for(
                self._async_step(env, completion.tokens),
                timeout=self.config.verification_timeout + 10  # Extra buffer
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Environment step timed out")
            if self.error_handler:
                self.error_handler.handle_timeout(
                    self.env_builder.get_current_theorem_id()
                )
            # Return failure result
            from .lean_env import StepResult
            return StepResult(reward=0.0, episode_done=True)
        except Exception as e:
            logger.error(f"Environment step error: {e}")
            if self.error_handler:
                self.error_handler.handle_verification_crash(
                    self.env_builder.get_current_theorem_id() or "unknown",
                    str(e)
                )
            from .lean_env import StepResult
            return StepResult(reward=0.0, episode_done=True)
    
    async def _async_step(
        self,
        env: "Lean4FIMEnv",
        action_tokens: List[int],
    ) -> "StepResult":
        """
        Async wrapper for environment step.
        
        The environment's step() method may be sync or async depending
        on the verifier implementation.
        """
        result = env.step(action_tokens)
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    async def _update_policy(
        self,
        trajectories: List[Trajectory],
        rewards: np.ndarray,
    ) -> None:
        """
        Update policy using CISPO loss.
        
        Implements Requirements 5.3 and 5.4:
        - Calls forward_backward() with CISPO loss
        - Calls optim_step() to update LoRA weights
        
        Uses group-relative baseline for advantage computation:
        advantage_i = reward_i - mean(rewards)
        
        Args:
            trajectories: List of Trajectory objects with tokens and logprobs.
            rewards: Array of rewards for each trajectory.
        """
        # Compute advantages (group-relative baseline)
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # Prepare training data for CISPO
        for trajectory, advantage in zip(trajectories, advantages):
            # Build model input with completion tokens
            model_input = self._build_model_input(trajectory)
            
            # Call forward_backward with CISPO loss (Requirement 5.3)
            # Advantages are per-token, so we broadcast the trajectory advantage
            token_advantages = np.full(len(trajectory.action_tokens), advantage)
            
            await self.client.forward_backward_async(
                data=[model_input],
                advantages=token_advantages,
                ref_logprobs=trajectory.logprobs,
            )
        
        # Optimizer step (Requirement 5.4)
        await self.client.optim_step_async(
            learning_rate=self.config.learning_rate
        )
    
    def _build_model_input(self, trajectory: Trajectory) -> ModelInput:
        """
        Build model input from trajectory.
        
        Args:
            trajectory: Trajectory with observation and action tokens.
            
        Returns:
            ModelInput for Tinker's forward_backward.
        """
        # Concatenate prompt and completion tokens
        all_tokens = trajectory.observation_tokens + trajectory.action_tokens
        return ModelInput(
            tokens=all_tokens,
            length=len(all_tokens),
        )
    
    def _log_step_metrics(self, step_metrics: Dict[str, Any], step: int) -> None:
        """
        Log training metrics for a step.
        
        Implements Requirement 5.6.
        
        Args:
            step_metrics: Dictionary of metrics from _train_step.
            step: Current step number.
        """
        self.metrics.log_training_step(
            step=step,
            reward_mean=step_metrics["reward_mean"],
            reward_std=step_metrics["reward_std"],
            pass_rate=step_metrics["pass_rate"],
        )
        
        # Log pass rate by curriculum level
        mask_ratio = step_metrics.get("mask_ratio")
        if mask_ratio is not None:
            for reward in step_metrics["rewards"]:
                self.metrics.log_pass_rate_by_level(
                    level=mask_ratio,
                    success=reward > 0.5,
                )
        
        # Log to console
        logger.info(
            f"Step {step}: "
            f"reward={step_metrics['reward_mean']:.3f}±{step_metrics['reward_std']:.3f}, "
            f"pass_rate={step_metrics['pass_rate']:.1%}, "
            f"theorem={step_metrics.get('theorem_id', 'N/A')}, "
            f"mask_ratio={step_metrics.get('mask_ratio', 'N/A')}"
        )
    
    async def _save_checkpoint(self, step: int, emergency: bool = False) -> None:
        """
        Save training checkpoint.
        
        Implements Requirement 5.7.
        
        Args:
            step: Current step number.
            emergency: If True, this is an emergency checkpoint after an error.
        """
        checkpoint_type = "emergency" if emergency else "regular"
        logger.info(f"Saving {checkpoint_type} checkpoint at step {step}")
        
        try:
            extra_metadata = {
                "total_tokens": self._total_tokens_generated,
                "total_verifications": self._total_verifications,
                "total_successes": self._total_successes,
                "emergency": emergency,
            }
            await self.checkpointer.save(step, extra_metadata=extra_metadata)
            logger.info(f"Checkpoint saved at step {step}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if not emergency:
                raise
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the training run.
        
        Returns:
            Dictionary containing training summary.
        """
        return {
            "final_step": self.step_count,
            "total_tokens_generated": self._total_tokens_generated,
            "total_verifications": self._total_verifications,
            "total_successes": self._total_successes,
            "overall_pass_rate": (
                self._total_successes / self._total_verifications
                if self._total_verifications > 0 else 0.0
            ),
            "stopped_early": self._should_stop,
        }
    
    def stop(self) -> None:
        """
        Request graceful stop of training.
        
        The training loop will complete the current step and then stop.
        """
        logger.info("Stop requested - will stop after current step")
        self._should_stop = True
    
    @property
    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary containing current statistics.
        """
        return {
            "step_count": self.step_count,
            "total_tokens_generated": self._total_tokens_generated,
            "total_verifications": self._total_verifications,
            "total_successes": self._total_successes,
            "is_running": self._running,
        }


async def run_training(
    training_client: Any,
    env_group_builder: "CurriculumEnvGroupBuilder",
    config: "TrainingConfig",
    metrics_logger: "MetricsLogger",
    checkpoint_manager: "CheckpointManager",
    error_handler: Optional["ErrorHandler"] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run CISPO training.
    
    Sets up signal handlers for graceful shutdown and runs the training loop.
    
    Args:
        training_client: Tinker TrainingClient.
        env_group_builder: CurriculumEnvGroupBuilder.
        config: TrainingConfig.
        metrics_logger: MetricsLogger.
        checkpoint_manager: CheckpointManager.
        error_handler: Optional ErrorHandler.
        resume_from: Optional checkpoint name to resume from.
        
    Returns:
        Training summary dictionary.
    """
    start_step = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_step = await checkpoint_manager.load(resume_from)
        logger.info(f"Resumed at step {start_step}")
    
    # Create training loop
    loop = CISPOTrainingLoop(
        training_client=training_client,
        env_group_builder=env_group_builder,
        config=config,
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
        error_handler=error_handler,
        start_step=start_step,
    )
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, requesting graceful stop")
        loop.stop()
    
    # Register signal handlers (Unix only)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (ValueError, OSError):
        # Signal handling may not work in all environments
        pass
    
    # Run training
    return await loop.train()
