"""
Error handling with retry logic and aggregation for Tinker API integration.

This module provides robust error handling for API and verification failures,
enabling training to continue despite transient issues.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional, Any, Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

T = TypeVar("T")


@dataclass
class ErrorStats:
    """Aggregated error statistics for post-training analysis."""
    
    transient_errors: int = 0
    verification_crashes: int = 0
    timeouts: int = 0
    theorem_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transient_errors": self.transient_errors,
            "verification_crashes": self.verification_crashes,
            "timeouts": self.timeouts,
            "theorem_failures": dict(self.theorem_failures),
        }


class ErrorHandler:
    """
    Handles errors with retry logic and aggregation.
    
    Features:
    - Exponential backoff retry for transient API errors
    - Verification crash handling with theorem flagging
    - Critical threshold detection to pause training
    - Error statistics aggregation for analysis
    
    Example:
        handler = ErrorHandler(max_retries=3, base_delay=1.0)
        
        # With retry for API calls
        result = await handler.with_retry(api_call, arg1, arg2)
        
        # Handle verification crash
        reward = handler.handle_verification_crash("theorem_123", "Lean crashed")
        
        # Get statistics
        stats = handler.get_stats()
    """
    
    # Default threshold for flagging theorems for review
    THEOREM_FLAG_THRESHOLD = 5
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        critical_threshold: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the error handler.
        
        Args:
            max_retries: Maximum number of retry attempts for transient errors
            base_delay: Base delay in seconds for exponential backoff
            critical_threshold: Number of errors before pausing training
            logger: Optional logger instance (creates default if not provided)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.critical_threshold = critical_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.stats = ErrorStats()
        self._paused = False
    
    async def with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> Optional[T]:
        """
        Execute function with exponential backoff retry.
        
        Retries the function up to max_retries times with exponentially
        increasing delays between attempts.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result if successful, None if all retries fail
            
        Raises:
            RuntimeError: If critical error threshold is exceeded
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Handle both sync and async functions
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                last_error = e
                self.stats.transient_errors += 1
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
        
        self.logger.error(f"All retries failed: {last_error}")
        self._check_critical_threshold()
        return None
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds: base_delay * 2^attempt
        """
        return self.base_delay * (2 ** attempt)
    
    def handle_verification_crash(self, theorem_id: str, error: str) -> float:
        """
        Handle verification crash (not just failure).
        
        Logs the error, tracks statistics, and flags theorems with
        consistent failures for manual review.
        
        Args:
            theorem_id: Identifier of the theorem that crashed
            error: Error message or description
            
        Returns:
            reward=0.0 (crashes are treated as failures)
            
        Raises:
            RuntimeError: If critical error threshold is exceeded
        """
        self.stats.verification_crashes += 1
        self.stats.theorem_failures[theorem_id] += 1
        self.logger.error(f"Verification crash for {theorem_id}: {error}")
        
        # Flag theorems with consistent failures
        failure_count = self.stats.theorem_failures[theorem_id]
        if failure_count >= self.THEOREM_FLAG_THRESHOLD:
            self.logger.warning(
                f"Theorem {theorem_id} flagged for review "
                f"({failure_count} failures)"
            )
        
        self._check_critical_threshold()
        return 0.0
    
    def handle_timeout(self, theorem_id: Optional[str] = None) -> float:
        """
        Handle verification timeout.
        
        Args:
            theorem_id: Optional identifier of the theorem that timed out
            
        Returns:
            reward=0.0 (timeouts are treated as failures)
        """
        self.stats.timeouts += 1
        if theorem_id:
            self.stats.theorem_failures[theorem_id] += 1
            self.logger.warning(f"Verification timeout for {theorem_id}")
        else:
            self.logger.warning("Verification timeout")
        
        return 0.0
    
    def _check_critical_threshold(self) -> None:
        """
        Check if critical error threshold exceeded.
        
        Raises:
            RuntimeError: If threshold exceeded and not already paused
        """
        total_errors = (
            self.stats.transient_errors +
            self.stats.verification_crashes
        )
        
        if total_errors >= self.critical_threshold and not self._paused:
            self._paused = True
            self.logger.critical(
                f"Critical error threshold ({self.critical_threshold}) exceeded. "
                "Training paused. Review errors before continuing."
            )
            raise RuntimeError("Critical error threshold exceeded")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get error statistics for analysis.
        
        Returns:
            Dictionary containing:
            - transient_errors: Count of transient API errors
            - verification_crashes: Count of verification crashes
            - timeouts: Count of verification timeouts
            - flagged_theorems: List of theorem IDs flagged for review
        """
        return {
            "transient_errors": self.stats.transient_errors,
            "verification_crashes": self.stats.verification_crashes,
            "timeouts": self.stats.timeouts,
            "flagged_theorems": self.get_flagged_theorems(),
        }
    
    def get_flagged_theorems(self) -> List[str]:
        """
        Get list of theorems flagged for manual review.
        
        Returns:
            List of theorem IDs with >= THEOREM_FLAG_THRESHOLD failures
        """
        return [
            tid for tid, count in self.stats.theorem_failures.items()
            if count >= self.THEOREM_FLAG_THRESHOLD
        ]
    
    def reset_stats(self) -> None:
        """Reset all error statistics."""
        self.stats = ErrorStats()
        self._paused = False
    
    def resume(self) -> None:
        """Resume training after being paused due to critical threshold."""
        if self._paused:
            self.logger.info("Training resumed after critical threshold pause")
            self._paused = False
    
    @property
    def is_paused(self) -> bool:
        """Check if training is paused due to critical errors."""
        return self._paused
