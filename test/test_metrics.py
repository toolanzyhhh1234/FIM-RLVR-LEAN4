"""
Tests for MetricsLogger and TrainingMetrics.

Tests JSONL output, verification latency tracking, promotion logging,
token usage tracking, and optional W&B integration.
"""

import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from tinker_integration.metrics import (
    MetricsLogger,
    TrainingMetrics,
    VerificationMetrics,
)


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""
    
    def test_create_training_metrics(self):
        """Test creating TrainingMetrics with required fields."""
        metrics = TrainingMetrics(
            step=100,
            timestamp=1234567890.0,
            reward_mean=0.65,
            reward_std=0.12,
            pass_rate=0.65,
        )
        
        assert metrics.step == 100
        assert metrics.timestamp == 1234567890.0
        assert metrics.reward_mean == 0.65
        assert metrics.reward_std == 0.12
        assert metrics.pass_rate == 0.65
        assert metrics.curriculum_level_distribution is None
    
    def test_create_training_metrics_with_distribution(self):
        """Test creating TrainingMetrics with curriculum distribution."""
        distribution = {"0.1": 5, "0.2": 3, "0.3": 2}
        metrics = TrainingMetrics(
            step=100,
            timestamp=1234567890.0,
            reward_mean=0.65,
            reward_std=0.12,
            pass_rate=0.65,
            curriculum_level_distribution=distribution,
        )
        
        assert metrics.curriculum_level_distribution == distribution


class TestMetricsLoggerInit:
    """Tests for MetricsLogger initialization."""
    
    def test_init_creates_log_dir(self):
        """Test that initialization creates log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs", "nested")
            
            logger = MetricsLogger(log_dir=log_dir)
            
            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)
    
    def test_init_without_wandb(self):
        """Test initialization without W&B."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            assert logger._wandb is None
    
    def test_init_with_wandb_not_installed(self):
        """Test initialization with W&B project but wandb not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock wandb import to raise ImportError
            with patch.dict('sys.modules', {'wandb': None}):
                logger = MetricsLogger(
                    log_dir=tmpdir,
                    wandb_project="test-project",
                )
                
                # Should gracefully handle missing wandb
                assert logger._wandb is None


class TestMetricsLoggerTrainingStep:
    """Tests for log_training_step method."""
    
    def test_log_training_step_writes_jsonl(self):
        """Test that log_training_step writes to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_training_step(
                step=100,
                reward_mean=0.65,
                reward_std=0.12,
                pass_rate=0.65,
            )
            
            # Read the metrics file
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            assert os.path.exists(metrics_file)
            
            with open(metrics_file, "r") as f:
                line = f.readline()
                record = json.loads(line)
            
            assert record["step"] == 100
            assert record["reward_mean"] == 0.65
            assert record["reward_std"] == 0.12
            assert record["pass_rate"] == 0.65
            assert "timestamp" in record
    
    def test_log_training_step_with_distribution(self):
        """Test log_training_step with curriculum distribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            distribution = {"0.1": 5, "0.2": 3}
            logger.log_training_step(
                step=100,
                reward_mean=0.65,
                reward_std=0.12,
                pass_rate=0.65,
                curriculum_level_distribution=distribution,
            )
            
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["curriculum_level_distribution"] == distribution
    
    def test_log_multiple_training_steps(self):
        """Test logging multiple training steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            for step in range(5):
                logger.log_training_step(
                    step=step,
                    reward_mean=0.5 + step * 0.1,
                    reward_std=0.1,
                    pass_rate=0.5 + step * 0.1,
                )
            
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 5
            
            # Verify steps are in order
            for i, line in enumerate(lines):
                record = json.loads(line)
                assert record["step"] == i


class TestMetricsLoggerVerification:
    """Tests for verification logging methods."""
    
    def test_log_verification_latency(self):
        """Test log_verification_latency writes to verification file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_verification_latency(latency=2.5, success=True)
            
            verification_file = os.path.join(tmpdir, "verification.jsonl")
            assert os.path.exists(verification_file)
            
            with open(verification_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["latency"] == 2.5
            assert record["success"] is True
            assert "timestamp" in record
    
    def test_log_verification_latency_updates_stats(self):
        """Test that log_verification_latency updates internal stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_verification_latency(latency=2.0, success=True)
            logger.log_verification_latency(latency=3.0, success=True)
            logger.log_verification_latency(latency=4.0, success=False)
            
            stats = logger.get_verification_stats()
            
            assert stats["total_verifications"] == 3
            assert stats["successful_verifications"] == 2
            assert stats["avg_latency"] == 3.0  # (2+3+4)/3
    
    def test_log_verification_timeout(self):
        """Test log_verification_timeout writes timeout event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_verification_timeout()
            
            verification_file = os.path.join(tmpdir, "verification.jsonl")
            with open(verification_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["event"] == "timeout"
            assert "timestamp" in record
    
    def test_log_verification_timeout_updates_stats(self):
        """Test that log_verification_timeout updates internal stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_verification_timeout()
            logger.log_verification_timeout()
            
            stats = logger.get_verification_stats()
            
            assert stats["timeouts"] == 2
            assert stats["total_verifications"] == 2
    
    def test_log_verification_error(self):
        """Test log_verification_error writes error event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_verification_error("Connection failed")
            
            verification_file = os.path.join(tmpdir, "verification.jsonl")
            with open(verification_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["event"] == "error"
            assert record["error"] == "Connection failed"
            assert "timestamp" in record


class TestMetricsLoggerPromotion:
    """Tests for promotion logging (Requirement 8.2)."""
    
    def test_log_promotion(self):
        """Test log_promotion writes promotion event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_promotion(theorem_id="thm_001", new_level=0.3)
            
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["event"] == "promotion"
            assert record["theorem_id"] == "thm_001"
            assert record["new_level"] == 0.3
            assert "timestamp" in record
    
    def test_log_promotion_updates_count(self):
        """Test that log_promotion updates promotion count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_promotion(theorem_id="thm_001", new_level=0.2)
            logger.log_promotion(theorem_id="thm_002", new_level=0.3)
            logger.log_promotion(theorem_id="thm_001", new_level=0.3)
            
            summary = logger.get_summary()
            
            assert summary["total_promotions"] == 3


class TestMetricsLoggerPassRateByLevel:
    """Tests for pass rate by curriculum level (Requirement 8.1)."""
    
    def test_log_pass_rate_by_level(self):
        """Test log_pass_rate_by_level tracks outcomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            # Log outcomes for level 0.1
            logger.log_pass_rate_by_level(level=0.1, success=True)
            logger.log_pass_rate_by_level(level=0.1, success=True)
            logger.log_pass_rate_by_level(level=0.1, success=False)
            
            # Log outcomes for level 0.2
            logger.log_pass_rate_by_level(level=0.2, success=True)
            logger.log_pass_rate_by_level(level=0.2, success=False)
            
            pass_rates = logger.get_pass_rates_by_level()
            
            assert abs(pass_rates[0.1] - 2/3) < 0.001  # 2 out of 3
            assert abs(pass_rates[0.2] - 0.5) < 0.001  # 1 out of 2


class TestMetricsLoggerTokenUsage:
    """Tests for token usage tracking (Requirement 8.4)."""
    
    def test_log_token_usage(self):
        """Test log_token_usage tracks total tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_token_usage(tokens=100)
            logger.log_token_usage(tokens=200)
            logger.log_token_usage(tokens=150)
            
            summary = logger.get_summary()
            
            assert summary["total_tokens"] == 450


class TestMetricsLoggerConfig:
    """Tests for configuration logging."""
    
    def test_log_config(self):
        """Test log_config writes config event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            config = {
                "model_name": "test/model",
                "max_steps": 1000,
                "api_key": "***MASKED***",
            }
            logger.log_config(config)
            
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["event"] == "config"
            assert record["config"] == config


class TestMetricsLoggerSummary:
    """Tests for get_summary and save_summary methods."""
    
    def test_get_summary(self):
        """Test get_summary returns comprehensive statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            # Log some data
            logger.log_token_usage(tokens=500)
            logger.log_promotion(theorem_id="thm_001", new_level=0.2)
            logger.log_verification_latency(latency=2.0, success=True)
            logger.log_verification_latency(latency=4.0, success=False)
            
            summary = logger.get_summary()
            
            assert summary["total_tokens"] == 500
            assert summary["total_promotions"] == 1
            assert summary["avg_verification_latency"] == 3.0
            assert "verification_stats" in summary
            assert "pass_rates_by_level" in summary
    
    def test_save_summary(self):
        """Test save_summary writes summary to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_token_usage(tokens=1000)
            logger.log_promotion(theorem_id="thm_001", new_level=0.3)
            
            filepath = logger.save_summary()
            
            assert os.path.exists(filepath)
            
            with open(filepath, "r") as f:
                saved_summary = json.load(f)
            
            assert saved_summary["total_tokens"] == 1000
            assert saved_summary["total_promotions"] == 1
            assert "saved_at" in saved_summary
    
    def test_save_summary_custom_path(self):
        """Test save_summary with custom filepath."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            custom_path = os.path.join(tmpdir, "custom_summary.json")
            filepath = logger.save_summary(filepath=custom_path)
            
            assert filepath == custom_path
            assert os.path.exists(custom_path)


class TestMetricsLoggerContextManager:
    """Tests for context manager functionality."""
    
    def test_context_manager(self):
        """Test MetricsLogger as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with MetricsLogger(log_dir=tmpdir) as logger:
                logger.log_token_usage(tokens=100)
                logger.log_training_step(
                    step=1,
                    reward_mean=0.5,
                    reward_std=0.1,
                    pass_rate=0.5,
                )
            
            # After context exit, summary should be saved
            summary_file = os.path.join(tmpdir, "summary.json")
            assert os.path.exists(summary_file)


class TestMetricsLoggerCustomEvent:
    """Tests for custom event logging."""
    
    def test_log_custom_event(self):
        """Test log_custom_event writes custom event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            
            logger.log_custom_event(
                event_name="checkpoint_saved",
                data={"step": 100, "path": "/checkpoints/step_100"},
            )
            
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                record = json.loads(f.readline())
            
            assert record["event"] == "checkpoint_saved"
            assert record["step"] == 100
            assert record["path"] == "/checkpoints/step_100"
