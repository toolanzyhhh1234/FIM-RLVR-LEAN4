"""
Tests for AsyncVerifier.

Tests the async wrapper around LeanVerifier with concurrency control
and timeout handling.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
from typing import Tuple

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

from tinker_integration.async_verifier import AsyncVerifier


class MockVerifier:
    """Mock verifier for testing without actual Lean compilation."""
    
    def __init__(self, delay: float = 0.0, success: bool = True):
        self.delay = delay
        self.success = success
        self.call_count = 0
        self.concurrent_calls = 0
        self.max_concurrent_calls = 0
        self._lock = None
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        """Mock verification with configurable delay and result."""
        self.call_count += 1
        self.concurrent_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.concurrent_calls)
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        self.concurrent_calls -= 1
        
        if self.success:
            return True, "Success"
        else:
            return False, "Error: proof failed"


class SlowVerifier:
    """Verifier that takes a long time (for timeout testing)."""
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        time.sleep(10)  # Sleep for 10 seconds
        return True, "Success"


class ErrorVerifier:
    """Verifier that raises an exception."""
    
    def verify(self, full_code: str) -> Tuple[bool, str]:
        raise RuntimeError("Verification crashed!")


class TestAsyncVerifierInit:
    """Tests for AsyncVerifier initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        mock_verifier = MockVerifier()
        async_verifier = AsyncVerifier(mock_verifier)
        
        assert async_verifier.timeout == 60.0
        assert async_verifier.verifier is mock_verifier
        async_verifier.shutdown()
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_verifier = MockVerifier()
        async_verifier = AsyncVerifier(
            mock_verifier,
            max_concurrent=4,
            timeout_seconds=30.0,
        )
        
        assert async_verifier.timeout == 30.0
        async_verifier.shutdown()
    
    def test_init_invalid_max_concurrent(self):
        """Test that invalid max_concurrent raises ValueError."""
        mock_verifier = MockVerifier()
        
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            AsyncVerifier(mock_verifier, max_concurrent=0)
        
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            AsyncVerifier(mock_verifier, max_concurrent=-1)
    
    def test_init_invalid_timeout(self):
        """Test that invalid timeout raises ValueError."""
        mock_verifier = MockVerifier()
        
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            AsyncVerifier(mock_verifier, timeout_seconds=0)
        
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            AsyncVerifier(mock_verifier, timeout_seconds=-1)


class TestAsyncVerifierVerify:
    """Tests for AsyncVerifier.verify() method."""
    
    @pytest.mark.asyncio
    async def test_verify_success(self):
        """Test successful verification."""
        mock_verifier = MockVerifier(success=True)
        async_verifier = AsyncVerifier(mock_verifier)
        
        result = await async_verifier.verify("theorem test : True := trivial")
        
        assert result is True
        assert mock_verifier.call_count == 1
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_failure(self):
        """Test failed verification."""
        mock_verifier = MockVerifier(success=False)
        async_verifier = AsyncVerifier(mock_verifier)
        
        result = await async_verifier.verify("theorem test : False := sorry")
        
        assert result is False
        assert mock_verifier.call_count == 1
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_timeout(self):
        """Test that slow verification times out and returns False."""
        slow_verifier = SlowVerifier()
        async_verifier = AsyncVerifier(slow_verifier, timeout_seconds=0.1)
        
        result = await async_verifier.verify("theorem test : True := trivial")
        
        assert result is False
        stats = async_verifier.get_stats()
        assert stats["timeouts"] == 1
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_error(self):
        """Test that verification errors return False."""
        error_verifier = ErrorVerifier()
        async_verifier = AsyncVerifier(error_verifier)
        
        result = await async_verifier.verify("theorem test : True := trivial")
        
        assert result is False
        stats = async_verifier.get_stats()
        assert stats["errors"] == 1
        async_verifier.shutdown()


class TestAsyncVerifierBatch:
    """Tests for AsyncVerifier.verify_batch() method."""
    
    @pytest.mark.asyncio
    async def test_verify_batch(self):
        """Test batch verification."""
        mock_verifier = MockVerifier(success=True)
        async_verifier = AsyncVerifier(mock_verifier)
        
        codes = ["code1", "code2", "code3"]
        results = await async_verifier.verify_batch(codes)
        
        assert len(results) == 3
        assert all(r is True for r in results)
        assert mock_verifier.call_count == 3
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_batch_empty(self):
        """Test batch verification with empty list."""
        mock_verifier = MockVerifier()
        async_verifier = AsyncVerifier(mock_verifier)
        
        results = await async_verifier.verify_batch([])
        
        assert results == []
        assert mock_verifier.call_count == 0
        async_verifier.shutdown()


class TestAsyncVerifierConcurrency:
    """Tests for concurrency control."""
    
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Test that concurrency limit is respected."""
        mock_verifier = MockVerifier(delay=0.1)
        max_concurrent = 2
        async_verifier = AsyncVerifier(mock_verifier, max_concurrent=max_concurrent)
        
        # Submit more tasks than max_concurrent
        codes = ["code" + str(i) for i in range(6)]
        await async_verifier.verify_batch(codes)
        
        # The max concurrent calls should not exceed the limit
        assert mock_verifier.max_concurrent_calls <= max_concurrent
        async_verifier.shutdown()


class TestAsyncVerifierStats:
    """Tests for statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        mock_verifier = MockVerifier(success=True)
        async_verifier = AsyncVerifier(mock_verifier)
        
        await async_verifier.verify("code1")
        await async_verifier.verify("code2")
        
        stats = async_verifier.get_stats()
        assert stats["total"] == 2
        assert stats["successful"] == 2
        assert stats["timeouts"] == 0
        assert stats["errors"] == 0
        assert stats["success_rate"] == 1.0
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_stats_reset(self):
        """Test that statistics can be reset."""
        mock_verifier = MockVerifier(success=True)
        async_verifier = AsyncVerifier(mock_verifier)
        
        await async_verifier.verify("code1")
        async_verifier.reset_stats()
        
        stats = async_verifier.get_stats()
        assert stats["total"] == 0
        assert stats["successful"] == 0
        async_verifier.shutdown()


class TestAsyncVerifierContextManager:
    """Tests for async context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        mock_verifier = MockVerifier(success=True)
        
        async with AsyncVerifier(mock_verifier) as async_verifier:
            result = await async_verifier.verify("code")
            assert result is True


class TestAsyncVerifierWithOutput:
    """Tests for verify_with_output method."""
    
    @pytest.mark.asyncio
    async def test_verify_with_output_success(self):
        """Test verify_with_output returns output on success."""
        mock_verifier = MockVerifier(success=True)
        async_verifier = AsyncVerifier(mock_verifier)
        
        success, output = await async_verifier.verify_with_output("code")
        
        assert success is True
        assert output == "Success"
        async_verifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_verify_with_output_timeout(self):
        """Test verify_with_output returns timeout message."""
        slow_verifier = SlowVerifier()
        async_verifier = AsyncVerifier(slow_verifier, timeout_seconds=0.1)
        
        success, output = await async_verifier.verify_with_output("code")
        
        assert success is False
        assert "timed out" in output.lower()
        async_verifier.shutdown()
