"""
Tests for ErrorHandler module.

Tests cover:
- Exponential backoff retry logic
- Verification crash handling
- Theorem failure flagging
- Critical threshold detection
- Error statistics aggregation
"""

import asyncio
import pytest
import logging
from unittest.mock import MagicMock, AsyncMock

from tinker_integration.error_handler import ErrorHandler, ErrorStats


class TestErrorStats:
    """Tests for ErrorStats dataclass."""
    
    def test_default_values(self):
        """Test that ErrorStats initializes with correct defaults."""
        stats = ErrorStats()
        assert stats.transient_errors == 0
        assert stats.verification_crashes == 0
        assert stats.timeouts == 0
        assert len(stats.theorem_failures) == 0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = ErrorStats()
        stats.transient_errors = 5
        stats.verification_crashes = 2
        stats.timeouts = 1
        stats.theorem_failures["theorem_1"] = 3
        
        result = stats.to_dict()
        
        assert result["transient_errors"] == 5
        assert result["verification_crashes"] == 2
        assert result["timeouts"] == 1
        assert result["theorem_failures"] == {"theorem_1": 3}


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        handler = ErrorHandler()
        
        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
        assert handler.critical_threshold == 10
        assert not handler.is_paused
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        logger = logging.getLogger("test")
        handler = ErrorHandler(
            max_retries=5,
            base_delay=2.0,
            critical_threshold=20,
            logger=logger,
        )
        
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0
        assert handler.critical_threshold == 20
        assert handler.logger is logger
    
    def test_calculate_delay(self):
        """Test exponential backoff delay calculation."""
        handler = ErrorHandler(base_delay=1.0)
        
        # Delay should be base_delay * 2^attempt
        assert handler._calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert handler._calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert handler._calculate_delay(2) == 4.0   # 1 * 2^2 = 4
        assert handler._calculate_delay(3) == 8.0   # 1 * 2^3 = 8
    
    def test_calculate_delay_custom_base(self):
        """Test delay calculation with custom base delay."""
        handler = ErrorHandler(base_delay=0.5)
        
        assert handler._calculate_delay(0) == 0.5   # 0.5 * 2^0 = 0.5
        assert handler._calculate_delay(1) == 1.0   # 0.5 * 2^1 = 1.0
        assert handler._calculate_delay(2) == 2.0   # 0.5 * 2^2 = 2.0


class TestWithRetry:
    """Tests for the with_retry method."""
    
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test successful execution on first attempt."""
        handler = ErrorHandler()
        
        async def success_func():
            return "success"
        
        result = await handler.with_retry(success_func)
        
        assert result == "success"
        assert handler.stats.transient_errors == 0
    
    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test successful execution after retries."""
        handler = ErrorHandler(base_delay=0.01)  # Fast retries for testing
        
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"
        
        result = await handler.with_retry(fail_then_succeed)
        
        assert result == "success"
        assert call_count == 3
        assert handler.stats.transient_errors == 2
    
    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test when all retries fail."""
        handler = ErrorHandler(max_retries=2, base_delay=0.01, critical_threshold=100)
        
        async def always_fail():
            raise ValueError("Persistent error")
        
        result = await handler.with_retry(always_fail)
        
        assert result is None
        # Initial attempt + 2 retries = 3 errors
        assert handler.stats.transient_errors == 3
    
    @pytest.mark.asyncio
    async def test_sync_function_support(self):
        """Test that sync functions are also supported."""
        handler = ErrorHandler()
        
        def sync_func():
            return "sync_result"
        
        result = await handler.with_retry(sync_func)
        
        assert result == "sync_result"


class TestVerificationCrashHandling:
    """Tests for verification crash handling."""
    
    def test_handle_verification_crash_returns_zero(self):
        """Test that crash handling returns reward=0.0."""
        handler = ErrorHandler(critical_threshold=100)
        
        reward = handler.handle_verification_crash("theorem_1", "Lean crashed")
        
        assert reward == 0.0
    
    def test_handle_verification_crash_increments_stats(self):
        """Test that crash handling updates statistics."""
        handler = ErrorHandler(critical_threshold=100)
        
        handler.handle_verification_crash("theorem_1", "Error 1")
        handler.handle_verification_crash("theorem_1", "Error 2")
        handler.handle_verification_crash("theorem_2", "Error 3")
        
        assert handler.stats.verification_crashes == 3
        assert handler.stats.theorem_failures["theorem_1"] == 2
        assert handler.stats.theorem_failures["theorem_2"] == 1
    
    def test_theorem_flagging(self):
        """Test that theorems are flagged after threshold failures."""
        handler = ErrorHandler(critical_threshold=100)
        
        # Fail theorem_1 5 times (threshold)
        for i in range(5):
            handler.handle_verification_crash("theorem_1", f"Error {i}")
        
        flagged = handler.get_flagged_theorems()
        
        assert "theorem_1" in flagged
    
    def test_theorem_not_flagged_below_threshold(self):
        """Test that theorems are not flagged below threshold."""
        handler = ErrorHandler(critical_threshold=100)
        
        # Fail theorem_1 4 times (below threshold of 5)
        for i in range(4):
            handler.handle_verification_crash("theorem_1", f"Error {i}")
        
        flagged = handler.get_flagged_theorems()
        
        assert "theorem_1" not in flagged


class TestTimeoutHandling:
    """Tests for timeout handling."""
    
    def test_handle_timeout_returns_zero(self):
        """Test that timeout handling returns reward=0.0."""
        handler = ErrorHandler()
        
        reward = handler.handle_timeout("theorem_1")
        
        assert reward == 0.0
    
    def test_handle_timeout_increments_stats(self):
        """Test that timeout handling updates statistics."""
        handler = ErrorHandler()
        
        handler.handle_timeout("theorem_1")
        handler.handle_timeout()  # Without theorem_id
        
        assert handler.stats.timeouts == 2
        assert handler.stats.theorem_failures["theorem_1"] == 1


class TestCriticalThreshold:
    """Tests for critical threshold detection."""
    
    def test_critical_threshold_raises_error(self):
        """Test that exceeding critical threshold raises RuntimeError."""
        handler = ErrorHandler(critical_threshold=3)
        
        # Accumulate errors up to threshold
        handler.stats.transient_errors = 2
        
        with pytest.raises(RuntimeError, match="Critical error threshold exceeded"):
            handler.handle_verification_crash("theorem_1", "Error")
    
    def test_critical_threshold_pauses_training(self):
        """Test that exceeding threshold sets paused state."""
        handler = ErrorHandler(critical_threshold=3)
        handler.stats.transient_errors = 2
        
        try:
            handler.handle_verification_crash("theorem_1", "Error")
        except RuntimeError:
            pass
        
        assert handler.is_paused
    
    def test_resume_after_pause(self):
        """Test resuming training after pause."""
        handler = ErrorHandler(critical_threshold=3)
        handler.stats.transient_errors = 2
        
        try:
            handler.handle_verification_crash("theorem_1", "Error")
        except RuntimeError:
            pass
        
        assert handler.is_paused
        
        handler.resume()
        
        assert not handler.is_paused


class TestStatistics:
    """Tests for error statistics."""
    
    def test_get_stats(self):
        """Test getting error statistics."""
        handler = ErrorHandler(critical_threshold=100)
        
        handler.stats.transient_errors = 5
        handler.stats.verification_crashes = 2
        handler.stats.timeouts = 1
        
        # Flag a theorem
        for i in range(5):
            handler.handle_verification_crash("flagged_theorem", f"Error {i}")
        
        stats = handler.get_stats()
        
        assert stats["transient_errors"] == 5
        assert stats["verification_crashes"] == 7  # 2 + 5 from flagging
        assert stats["timeouts"] == 1
        assert "flagged_theorem" in stats["flagged_theorems"]
    
    def test_reset_stats(self):
        """Test resetting error statistics."""
        handler = ErrorHandler(critical_threshold=100)
        
        handler.stats.transient_errors = 5
        handler.stats.verification_crashes = 2
        handler._paused = True
        
        handler.reset_stats()
        
        assert handler.stats.transient_errors == 0
        assert handler.stats.verification_crashes == 0
        assert not handler.is_paused
