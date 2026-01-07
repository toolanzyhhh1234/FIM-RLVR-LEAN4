"""
Async Verifier for Lean4 proof verification.

This module provides an async wrapper around LeanVerifier with concurrency
control and timeout handling, enabling high-throughput verification during
RL training.

Requirements covered:
- 6.1: Wrap LeanVerifier with async/await interface
- 6.2: Run multiple verifications concurrently
- 6.3: Respect configurable concurrency limit (default: 8)
- 6.4: Handle verification timeouts gracefully (default: 60 seconds)
- 6.5: Return reward=0.0 and log timeout on timeout
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import MetricsLogger


class VerifierProtocol(Protocol):
    """Protocol defining the verifier interface we expect."""
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        """
        Verify Lean4 code.
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        ...


class AsyncVerifier:
    """
    Async wrapper for LeanVerifier with concurrency control.
    
    Provides an async interface for Lean4 proof verification, enabling
    concurrent verification of multiple proofs while respecting resource
    limits. This is essential for high-throughput RL training where
    multiple completions need verification in parallel.
    
    Key features:
    - Semaphore-based concurrency control (Requirement 6.3)
    - Configurable timeout per verification (Requirement 6.4)
    - Optional metrics logging for latency tracking
    - Graceful handling of timeouts and errors (Requirement 6.5)
    
    Example:
        >>> from fim_rlvr_lean4.lean_verifier import LeanVerifier
        >>> 
        >>> lean_verifier = LeanVerifier("./verification_env")
        >>> async_verifier = AsyncVerifier(
        ...     lean_verifier=lean_verifier,
        ...     max_concurrent=8,
        ...     timeout_seconds=60.0,
        ... )
        >>> 
        >>> # Single verification
        >>> success = await async_verifier.verify(lean_code)
        >>> 
        >>> # Batch verification
        >>> results = await async_verifier.verify_batch([code1, code2, code3])
    
    Attributes:
        verifier: The underlying LeanVerifier instance.
        semaphore: Asyncio semaphore for concurrency control.
        timeout: Timeout in seconds for each verification.
        executor: ThreadPoolExecutor for running blocking verification.
        metrics: Optional MetricsLogger for latency tracking.
    """
    
    def __init__(
        self,
        lean_verifier: Any,
        max_concurrent: int = 8,
        timeout_seconds: float = 60.0,
        metrics_logger: Optional["MetricsLogger"] = None,
    ):
        """
        Initialize the async verifier.
        
        Args:
            lean_verifier: LeanVerifier instance (or compatible) for proof checking.
            max_concurrent: Maximum number of concurrent verifications (default: 8).
                           This limits resource usage and prevents overwhelming
                           the Lean compiler.
            timeout_seconds: Timeout per verification in seconds (default: 60.0).
                            Proofs that take longer are considered failed.
            metrics_logger: Optional MetricsLogger for tracking verification latency.
        
        Raises:
            ValueError: If max_concurrent < 1 or timeout_seconds <= 0.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        
        self.verifier = lean_verifier
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout_seconds
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.metrics = metrics_logger
        
        # Track statistics
        self._total_verifications = 0
        self._successful_verifications = 0
        self._timeout_count = 0
        self._error_count = 0

    async def verify(self, full_code: str) -> bool:
        """
        Verify Lean4 code asynchronously.
        
        Runs the verification in a thread pool to avoid blocking the event loop,
        while respecting the concurrency limit via semaphore.
        
        Args:
            full_code: Complete Lean4 source code to verify.
        
        Returns:
            True if verification succeeds, False otherwise.
            Returns False on timeout or error (Requirement 6.5).
        """
        async with self.semaphore:
            self._total_verifications += 1
            start_time = time.monotonic()
            
            try:
                # Run blocking verification in thread pool
                loop = asyncio.get_event_loop()
                success, output = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self.verifier.verify,
                        full_code
                    ),
                    timeout=self.timeout
                )
                
                # Track success
                if success:
                    self._successful_verifications += 1
                
                # Log latency if metrics logger available
                latency = time.monotonic() - start_time
                if self.metrics:
                    self.metrics.log_verification_latency(latency, success)
                
                return success
                
            except asyncio.TimeoutError:
                # Requirement 6.5: Return reward=0.0 (False) and log timeout
                self._timeout_count += 1
                if self.metrics:
                    self.metrics.log_verification_timeout()
                return False
                
            except Exception as e:
                # Handle unexpected errors gracefully
                self._error_count += 1
                if self.metrics:
                    self.metrics.log_verification_error(str(e))
                return False
    
    async def verify_batch(self, codes: List[str]) -> List[bool]:
        """
        Verify multiple codes concurrently.
        
        Submits all verifications concurrently, but the semaphore ensures
        that at most max_concurrent verifications run at any time.
        
        Args:
            codes: List of Lean4 source codes to verify.
        
        Returns:
            List of boolean results, one per input code.
            Order is preserved (results[i] corresponds to codes[i]).
        """
        tasks = [self.verify(code) for code in codes]
        return await asyncio.gather(*tasks)
    
    async def verify_with_output(self, full_code: str) -> Tuple[bool, str]:
        """
        Verify Lean4 code and return both success status and compiler output.
        
        Useful for debugging when you need to see the actual error messages
        from the Lean compiler.
        
        Args:
            full_code: Complete Lean4 source code to verify.
        
        Returns:
            Tuple of (success, output):
                - success: True if verification succeeded, False otherwise
                - output: Compiler output (stderr + stdout), or error message
        """
        async with self.semaphore:
            self._total_verifications += 1
            start_time = time.monotonic()
            
            try:
                loop = asyncio.get_event_loop()
                success, output = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self.verifier.verify,
                        full_code
                    ),
                    timeout=self.timeout
                )
                
                if success:
                    self._successful_verifications += 1
                
                latency = time.monotonic() - start_time
                if self.metrics:
                    self.metrics.log_verification_latency(latency, success)
                
                return success, output
                
            except asyncio.TimeoutError:
                self._timeout_count += 1
                if self.metrics:
                    self.metrics.log_verification_timeout()
                return False, f"Verification timed out after {self.timeout} seconds"
                
            except Exception as e:
                self._error_count += 1
                if self.metrics:
                    self.metrics.log_verification_error(str(e))
                return False, f"Verification error: {str(e)}"
    
    def get_stats(self) -> dict:
        """
        Get verification statistics.
        
        Returns:
            Dictionary containing:
                - total: Total number of verifications attempted
                - successful: Number of successful verifications
                - timeouts: Number of timeouts
                - errors: Number of errors (excluding timeouts)
                - success_rate: Ratio of successful to total (0.0 if no verifications)
        """
        total = self._total_verifications
        return {
            "total": total,
            "successful": self._successful_verifications,
            "timeouts": self._timeout_count,
            "errors": self._error_count,
            "success_rate": (
                self._successful_verifications / total if total > 0 else 0.0
            ),
        }
    
    def reset_stats(self):
        """Reset verification statistics to zero."""
        self._total_verifications = 0
        self._successful_verifications = 0
        self._timeout_count = 0
        self._error_count = 0
    
    def shutdown(self):
        """
        Shutdown the thread pool executor.
        
        Call this when done with the verifier to clean up resources.
        After shutdown, the verifier cannot be used again.
        """
        self.executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - shuts down executor."""
        self.shutdown()
        return False
