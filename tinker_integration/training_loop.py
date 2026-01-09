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
class SampleResult:
    """
    Result from Tinker's sample() call.
    
    Attributes:
        tokens: Generated token IDs.
        logprobs: Log probabilities for each generated token.
    """
    tokens: List[int]
    logprobs: List[float]


@dataclass
class Trajectory:
    """
    Complete trajectory for one episode.
    
    Attributes:
        prompt_tokens: Token IDs of the prompt.
        completion_tokens: Token IDs of the model's completion.
        reward: Reward from verification (1.0 for success, 0.0 for failure).
        logprobs: Log probabilities of the completion tokens.
    """
    prompt_tokens: List[int]
    completion_tokens: List[int]
    reward: float
    logprobs: List[float]


class CISPOTrainingLoop:
    """
    RLVR training loop using Tinker API with CISPO loss.
    
    Implements the core RL loop for training large MoE models on Lean4
    proof infilling tasks using the correct Tinker API signatures.
    """
    
    def __init__(
        self,
        service_client: Any,
        training_client: Any,
        tokenizer: Any,
        env_group_builder: "CurriculumEnvGroupBuilder",
        config: "TrainingConfig",
        metrics_logger: "MetricsLogger",
        checkpoint_manager: "CheckpointManager",
        error_handler: Optional["ErrorHandler"] = None,
        start_step: int = 0,
        debug_log_dir: Optional[str] = None,
    ):
        """
        Initialize the CISPO training loop.
        
        Args:
            service_client: Tinker ServiceClient for creating samplers.
            training_client: Tinker TrainingClient for model operations.
            tokenizer: Tokenizer from training_client.get_tokenizer().
            env_group_builder: CurriculumEnvGroupBuilder for creating environments.
            config: TrainingConfig with hyperparameters.
            metrics_logger: MetricsLogger for tracking progress.
            checkpoint_manager: CheckpointManager for saving state.
            error_handler: Optional ErrorHandler for robust error handling.
            start_step: Starting step number (for resuming from checkpoint).
            debug_log_dir: Directory for debug logs (samples, verifications).
        """
        self.service_client = service_client
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.env_builder = env_group_builder
        self.config = config
        self.metrics = metrics_logger
        self.checkpointer = checkpoint_manager
        self.error_handler = error_handler
        self.step_count = start_step
        
        # Debug logging
        self.debug_log_dir = debug_log_dir or config.log_dir
        self._debug_log_file = None
        self._init_debug_log()
        
        # Initialize prompt formatter with tokenizer for proper chat template
        from .prompt_formatter import FIMPromptFormatter
        self.prompt_formatter = FIMPromptFormatter(tokenizer=tokenizer)
        
        # Track training state
        self._running = False
        self._should_stop = False
        self._current_sampling_client = None
        self._sampler_checkpoint_path = None
        
        # Statistics
        self._total_tokens_generated = 0
        self._total_verifications = 0
        self._total_successes = 0
    
    def _init_debug_log(self):
        """Initialize debug log file for samples and verifications."""
        import os
        os.makedirs(self.debug_log_dir, exist_ok=True)
        debug_path = os.path.join(self.debug_log_dir, "debug_samples.jsonl")
        self._debug_log_file = open(debug_path, "a")
        logger.info(f"Debug log initialized: {debug_path}")
    
    def _log_debug(self, data: Dict[str, Any]):
        """Write a debug entry to the log file."""
        import json
        import time
        data["timestamp"] = time.time()
        self._debug_log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._debug_log_file.flush()
    
    async def train(self) -> Dict[str, Any]:
        """
        Run the training loop for configured number of steps.
        
        Returns:
            Dictionary containing training summary statistics.
        """
        import tinker
        from tinker import types
        
        self._running = True
        self._should_stop = False
        
        # Set up Adam optimizer params
        adam_params = types.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        
        # Sampling params
        sampling_params = tinker.SamplingParams(
            max_tokens=getattr(self.config, 'max_completion_tokens', 512),
            temperature=self.config.temperature,
            top_p=0.95,
            top_k=50,
        )
        
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
                step_metrics = await self._train_step(step, adam_params, sampling_params)
                
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
            import traceback
            traceback.print_exc()
            # Try to save emergency checkpoint
            try:
                await self._save_checkpoint(self.step_count, emergency=True)
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            raise
        finally:
            self._running = False
        
        return self._get_training_summary()

    
    async def _train_step(
        self, 
        step: int, 
        adam_params: Any,
        sampling_params: Any,
    ) -> Dict[str, Any]:
        """Execute a single training step."""
        import tinker
        from tinker import types
        from tinker.types.tensor_data import TensorData
        import torch
        
        # 1. Create environment group
        envs = self.env_builder.make_envs()
        
        # Capture theorem info for logging
        current_theorem_id = self.env_builder.get_current_theorem_id()
        current_mask_ratio = self.env_builder.get_current_mask_ratio()
        logger.debug(f"Step {step}: theorem={current_theorem_id}, mask_ratio={current_mask_ratio}")
        
        # 2. Save weights and create sampling client
        save_future = await self.training_client.save_weights_for_sampler_async(
            name=f"step_{step:06d}"
        )
        save_result = save_future.result()
        sampling_path = save_result.path
        self._sampler_checkpoint_path = sampling_path
        
        sampling_client = self.service_client.create_sampling_client(
            model_path=sampling_path
        )
        self._current_sampling_client = sampling_client
        
        # 3. Sample completions for each environment
        all_trajectories: List[Trajectory] = []
        all_rewards: List[float] = []
        
        for env in envs:
            # Build FIM prompt using the formatter (matches Unsloth pipeline)
            # Determine task type based on suffix
            task_type = "full" if not env.suffix.strip() else "fim"
            prompt_text = self.prompt_formatter.format(env.prefix, env.suffix)
            prompt_tokens = self.tokenizer.encode(prompt_text)
            
            # Sample G completions for this environment
            model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
            
            sample_future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,  # One sample per env, env_builder creates group_size envs
                sampling_params=sampling_params,
            )
            sample_result = sample_future.result()
            
            # Process each sample
            for seq in sample_result.sequences:
                completion_tokens = seq.tokens
                completion_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(completion_tokens)
                
                self._total_tokens_generated += len(completion_tokens)
                
                # Decode completion
                completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                
                # Extract code from response using proper tags
                extracted_code = self.prompt_formatter.extract_code_from_response(
                    completion_text, task_type=task_type
                )
                
                # If extraction failed, try using raw completion (model may not have followed format)
                if extracted_code is None:
                    extracted_code = self.prompt_formatter.strip_markdown_fences(completion_text)
                    tag_ok = False
                else:
                    extracted_code = self.prompt_formatter.strip_markdown_fences(extracted_code)
                    tag_ok = True
                
                # Log sample for debugging (randomly sample ~10% to avoid spam)
                import random
                if random.random() < 0.1 or step < 3:  # Always log first 3 steps
                    logger.debug(f"[Step {step}] Sample ({len(completion_tokens)} tokens):")
                    logger.debug(f"  Task type: {task_type}")
                    logger.debug(f"  Tag extraction: {'OK' if tag_ok else 'FAILED'}")
                    logger.debug(f"  Prompt (last 200 chars): ...{prompt_text[-200:]}")
                    logger.debug(f"  Raw completion: {completion_text[:300]}...")
                    logger.debug(f"  Extracted code: {extracted_code[:200] if extracted_code else 'None'}...")
                
                # Run verification with extracted code
                try:
                    result, verification_output = await self._verify_completion_with_code(
                        env, extracted_code or ""
                    )
                    reward = result.reward
                except Exception as e:
                    logger.warning(f"Verification error: {e}")
                    reward = 0.0
                    verification_output = str(e)
                
                # Log full debug info to file
                full_code = env.prefix + (extracted_code or "") + env.suffix
                self._log_debug({
                    "step": step,
                    "theorem_id": current_theorem_id,
                    "mask_ratio": current_mask_ratio,
                    "task_type": task_type,
                    "tag_extraction_ok": tag_ok,
                    "prompt": prompt_text,
                    "raw_completion": completion_text,
                    "extracted_code": extracted_code,
                    "full_code_sent_to_lean": full_code,
                    "reward": reward,
                    "verification_success": reward > 0.5,
                    "verification_output": verification_output,
                    "num_completion_tokens": len(completion_tokens),
                })
                
                self._total_verifications += 1
                if reward > 0.5:
                    self._total_successes += 1
                
                trajectory = Trajectory(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reward=reward,
                    logprobs=completion_logprobs,
                )
                all_trajectories.append(trajectory)
                all_rewards.append(reward)
        
        # 4. Update curriculum based on outcomes
        rewards_array = np.array(all_rewards)
        outcomes = [r > 0.5 for r in all_rewards]
        self.env_builder.update_outcomes_from_list(outcomes)
        
        # 5. Compute advantages (group-relative baseline)
        mean_reward = rewards_array.mean()
        advantages = rewards_array - mean_reward
        
        # Skip update if all advantages are zero
        if np.allclose(advantages, 0.0):
            logger.debug(f"Step {step}: Skipping update - all advantages are zero")
            return {
                "rewards": rewards_array,
                "pass_rate": float((rewards_array > 0.5).mean()),
                "reward_mean": float(mean_reward),
                "reward_std": float(rewards_array.std()),
                "skipped": True,
            }
        
        # 6. Build training datums
        datums: List[types.Datum] = []
        
        for trajectory, advantage in zip(all_trajectories, advantages):
            # Full sequence: prompt + completion
            all_tokens = trajectory.prompt_tokens + trajectory.completion_tokens
            
            # input_tokens are all but last, target_tokens are all but first
            input_tokens = all_tokens[:-1]
            target_tokens = all_tokens[1:]
            
            # Pad logprobs and advantages for prompt tokens
            ob_len = len(trajectory.prompt_tokens) - 1
            padded_logprobs = [0.0] * ob_len + trajectory.logprobs
            padded_advantages = [0.0] * ob_len + [float(advantage)] * (len(input_tokens) - ob_len)
            
            # Ensure lengths match
            if len(padded_logprobs) != len(input_tokens):
                # Truncate or pad as needed
                padded_logprobs = padded_logprobs[:len(input_tokens)]
                padded_logprobs.extend([0.0] * (len(input_tokens) - len(padded_logprobs)))
            
            if len(padded_advantages) != len(input_tokens):
                padded_advantages = padded_advantages[:len(input_tokens)]
                padded_advantages.extend([0.0] * (len(input_tokens) - len(padded_advantages)))
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            )
            datums.append(datum)
        
        # 7. Forward-backward with CISPO loss
        fb_future = self.training_client.forward_backward(
            data=datums,
            loss_fn="cispo",  # Use CISPO for MoE stability
        )
        
        # 8. Optimizer step
        optim_future = self.training_client.optim_step(adam_params)
        
        # Wait for both
        fb_result = fb_future.result()
        optim_result = optim_future.result()
        
        # Extract loss from result
        loss = fb_result.metrics.get("loss:sum", 0.0) if fb_result.metrics else 0.0
        
        return {
            "rewards": rewards_array,
            "pass_rate": float((rewards_array > 0.5).mean()),
            "reward_mean": float(mean_reward),
            "reward_std": float(rewards_array.std()),
            "loss": loss,
            "theorem_id": self.env_builder.get_current_theorem_id(),
            "mask_ratio": self.env_builder.get_current_mask_ratio(),
            "skipped": False,
        }

    
    async def _verify_completion_with_code(
        self, env: "Lean4FIMEnv", extracted_code: str
    ) -> tuple["StepResult", str]:
        """
        Verify extracted code using the environment's verifier.
        
        Args:
            env: The Lean4FIMEnv instance.
            extracted_code: The extracted code from model response.
            
        Returns:
            Tuple of (StepResult, verification_output_string).
        """
        from .lean_env import StepResult
        
        try:
            # Reconstruct full code: prefix + extracted_code + suffix
            full_code = env.prefix + extracted_code + env.suffix
            
            # Verify with Lean4
            success, output = env.verifier.verify(full_code)
            
            result = StepResult(reward=1.0 if success else 0.0, episode_done=True)
            return result, output
            
        except asyncio.TimeoutError:
            logger.warning("Verification timed out")
            if self.error_handler:
                self.error_handler.handle_timeout(
                    self.env_builder.get_current_theorem_id()
                )
            result = StepResult(reward=0.0, episode_done=True)
            return result, "TIMEOUT"
        except Exception as e:
            logger.error(f"Verification error: {e}")
            if self.error_handler:
                self.error_handler.handle_verification_crash(
                    self.env_builder.get_current_theorem_id() or "unknown",
                    str(e)
                )
            result = StepResult(reward=0.0, episode_done=True)
            return result, f"ERROR: {e}"
    
    def _log_step_metrics(self, step_metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics for a step."""
        self.metrics.log_training_step(
            step=step,
            reward_mean=step_metrics["reward_mean"],
            reward_std=step_metrics["reward_std"],
            pass_rate=step_metrics["pass_rate"],
        )
        
        # Log to console
        logger.info(
            f"Step {step}: "
            f"reward={step_metrics['reward_mean']:.3f}+/-{step_metrics['reward_std']:.3f}, "
            f"pass_rate={step_metrics['pass_rate']:.1%}, "
            f"theorem={step_metrics.get('theorem_id', 'N/A')}, "
            f"mask_ratio={step_metrics.get('mask_ratio', 'N/A')}"
        )
    
    async def _save_checkpoint(self, step: int, emergency: bool = False) -> None:
        """Save training checkpoint."""
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
        """Get summary statistics for the training run."""
        # Close debug log file
        if self._debug_log_file:
            self._debug_log_file.close()
            
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
        """Request graceful stop of training."""
        logger.info("Stop requested - will stop after current step")
        self._should_stop = True
    
    @property
    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "step_count": self.step_count,
            "total_tokens_generated": self._total_tokens_generated,
            "total_verifications": self._total_verifications,
            "total_successes": self._total_successes,
            "is_running": self._running,
        }
