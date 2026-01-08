"""
Lean4 FIM Environment for Tinker API integration.

This module implements Tinker's Env interface for Lean4 proof infilling,
wrapping the existing LeanVerifier to provide RL environment semantics.

Requirements covered:
- 1.1: Implement Tinker's Env interface with initial_observation() and step()
- 1.2: Return FIM-formatted prompt containing prefix, hole marker, and suffix
- 1.3: Reconstruct full proof by concatenating prefix + completion + suffix
- 1.4: Invoke existing LeanVerifier.verify() method
- 1.5: Return reward=1.0 and episode_done=True on verification success
- 1.6: Return reward=0.0 and episode_done=True on verification failure
- 1.7: Accept configurable max_tokens for stop condition (default: 512)
- 1.8: Store ground truth middle segment for optional logging/debugging
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, Any


@dataclass
class Observation:
    """
    Tokenized observation for the model.
    
    This represents the input that will be fed to the language model
    for completion generation.
    
    Attributes:
        tokens: List of token IDs representing the FIM prompt.
    """
    tokens: List[int]


@dataclass
class StopCondition:
    """
    Defines when to stop generation.
    
    Controls the sampling behavior of the model, specifying maximum
    output length and optional stop strings.
    
    Attributes:
        max_tokens: Maximum number of tokens to generate (default: 512).
        stop_strings: Optional list of strings that trigger early stopping.
    """
    max_tokens: int = 512
    stop_strings: Optional[List[str]] = None


@dataclass
class StepResult:
    """
    Result of taking an action in the environment.
    
    Returned by the step() method after processing a model completion.
    For Lean4 verification, episodes are single-step (always done after step).
    
    Attributes:
        reward: 1.0 if verification succeeded, 0.0 otherwise.
        episode_done: Always True for Lean4 verification (single-step episodes).
        next_observation: None for single-step episodes.
        next_stop_condition: None for single-step episodes.
    """
    reward: float
    episode_done: bool
    next_observation: Optional[Observation] = None
    next_stop_condition: Optional[StopCondition] = None


class TokenizerProtocol(Protocol):
    """Protocol defining the tokenizer interface we expect."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        ...


class VerifierProtocol(Protocol):
    """Protocol defining the verifier interface we expect."""
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        """
        Verify Lean4 code.
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        ...


class Lean4FIMEnv:
    """
    RL Environment for Lean4 proof infilling.
    
    Implements Tinker's Env interface to wrap Lean4 verification as an
    RL environment. Each episode consists of:
    1. initial_observation(): Returns FIM prompt as tokenized observation
    2. step(): Processes model completion, verifies with Lean4, returns reward
    
    The environment is single-step: after step() is called, the episode ends.
    
    Example:
        >>> from fim_rlvr_lean4.lean_verifier import LeanVerifier
        >>> from tinker_integration.prompt_formatter import FIMPromptFormatter
        >>> 
        >>> verifier = LeanVerifier("./verification_env")
        >>> formatter = FIMPromptFormatter()
        >>> env = Lean4FIMEnv(
        ...     prefix="theorem foo : 1 + 1 = 2 := by\\n  ",
        ...     suffix="",
        ...     ground_truth_middle="norm_num",
        ...     tokenizer=tokenizer,
        ...     verifier=verifier,
        ...     prompt_formatter=formatter,
        ... )
        >>> obs, stop = env.initial_observation()
        >>> # Model generates completion tokens...
        >>> result = env.step(completion_tokens)
        >>> print(f"Reward: {result.reward}")
    
    Attributes:
        prefix: Code before the hole (preserved exactly).
        suffix: Code after the hole (preserved exactly).
        ground_truth: The original middle segment for debugging.
        tokenizer: Tokenizer for encoding/decoding.
        verifier: LeanVerifier instance for proof checking.
        max_tokens: Maximum tokens for model generation.
        formatter: FIMPromptFormatter for prompt construction.
    """
    
    def __init__(
        self,
        prefix: str,
        suffix: str,
        ground_truth_middle: str,
        tokenizer: Any,
        verifier: Any,
        max_tokens: int = 512,
        prompt_formatter: Optional["FIMPromptFormatter"] = None,
    ):
        """
        Initialize the Lean4 FIM environment.
        
        Args:
            prefix: Code before the hole. Whitespace is preserved exactly.
            suffix: Code after the hole. Whitespace is preserved exactly.
            ground_truth_middle: The original middle segment (for debugging/logging).
            tokenizer: Tokenizer with encode() and decode() methods.
            verifier: LeanVerifier instance (or compatible) for proof verification.
            max_tokens: Maximum tokens for generation (default: 512).
            prompt_formatter: Optional FIMPromptFormatter. If None, creates default.
        """
        self.prefix = prefix
        self.suffix = suffix
        self.ground_truth = ground_truth_middle
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_tokens = max_tokens
        
        # Lazy import to avoid circular dependency
        if prompt_formatter is None:
            from .prompt_formatter import FIMPromptFormatter
            self.formatter = FIMPromptFormatter()
        else:
            self.formatter = prompt_formatter
        
        # Track episode state
        self._episode_started = False
        self._episode_done = False
    
    def initial_observation(self) -> Tuple[Observation, StopCondition]:
        """
        Return the FIM prompt as tokenized observation.
        
        Constructs the prompt using the formatter and tokenizes it.
        This is the starting point for each episode.
        
        Returns:
            Tuple of (Observation, StopCondition):
                - Observation contains tokenized FIM prompt
                - StopCondition specifies max_tokens for generation
        
        Raises:
            RuntimeError: If called after episode has already started.
        """
        if self._episode_started and not self._episode_done:
            raise RuntimeError("Episode already in progress. Call step() first.")
        
        # Reset episode state
        self._episode_started = True
        self._episode_done = False
        
        # Format the prompt (Requirement 1.2)
        prompt_text = self.formatter.format(self.prefix, self.suffix)
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt_text)
        
        return (
            Observation(tokens=tokens),
            StopCondition(max_tokens=self.max_tokens)
        )
    
    def step(self, action: List[int]) -> StepResult:
        """
        Process model completion and return verification reward.
        
        Takes the model's generated tokens, decodes them, reconstructs
        the full proof, and verifies with Lean4.
        
        Args:
            action: Token IDs of the model's completion.
        
        Returns:
            StepResult with:
                - reward=1.0 if verification succeeded (Requirement 1.5)
                - reward=0.0 if verification failed (Requirement 1.6)
                - episode_done=True (single-step episodes)
                - next_observation=None
                - next_stop_condition=None
        
        Raises:
            RuntimeError: If called before initial_observation() or after episode ended.
        """
        if not self._episode_started:
            raise RuntimeError("Must call initial_observation() before step().")
        if self._episode_done:
            raise RuntimeError("Episode already done. Create new environment for next episode.")
        
        # Mark episode as done
        self._episode_done = True
        
        # Decode completion (Requirement 1.3)
        completion = self.tokenizer.decode(action, skip_special_tokens=True)
        
        # Reconstruct full proof (Requirement 1.3)
        full_code = self.prefix + completion + self.suffix
        
        # Verify with Lean4 (Requirement 1.4)
        success, output = self.verifier.verify(full_code)
        
        # Return reward based on verification result (Requirements 1.5, 1.6)
        return StepResult(
            reward=1.0 if success else 0.0,
            episode_done=True,
            next_observation=None,
            next_stop_condition=None
        )
    
    def get_ground_truth(self) -> str:
        """
        Get the ground truth middle segment.
        
        Useful for debugging and logging to compare model output
        with the expected completion.
        
        Returns:
            The original middle segment that was masked.
        """
        return self.ground_truth
    
    def get_fim_prompt(self) -> str:
        """
        Get the FIM prompt as text (not tokenized).
        
        Returns:
            The formatted FIM prompt string.
        """
        return self.formatter.format(self.prefix, self.suffix)
    
    def step_with_text(self, completion_text: str) -> StepResult:
        """
        Process a text completion (already decoded) and return verification reward.
        
        This is a convenience method that skips the tokenization step,
        useful when the completion has already been decoded.
        
        Args:
            completion_text: The decoded completion text.
        
        Returns:
            StepResult with reward based on verification.
        """
        if not self._episode_started:
            # Auto-start episode if not started
            self._episode_started = True
        
        if self._episode_done:
            raise RuntimeError("Episode already done. Create new environment for next episode.")
        
        # Mark episode as done
        self._episode_done = True
        
        # Reconstruct full proof
        full_code = self.prefix + completion_text + self.suffix
        
        # Verify with Lean4
        success, output = self.verifier.verify(full_code)
        
        # Return reward based on verification result
        return StepResult(
            reward=1.0 if success else 0.0,
            episode_done=True,
            next_observation=None,
            next_stop_condition=None
        )
    
    def get_debug_info(self) -> dict:
        """
        Get debug information about the environment state.
        
        Returns:
            Dictionary containing prefix, suffix, ground_truth, and episode state.
        """
        return {
            "prefix": self.prefix,
            "suffix": self.suffix,
            "ground_truth": self.ground_truth,
            "max_tokens": self.max_tokens,
            "episode_started": self._episode_started,
            "episode_done": self._episode_done,
        }
