"""
Tests for ConfigManager and TrainingConfig.

Tests configuration loading from YAML, environment variable overrides,
validation, and defaults.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from tinker_integration.config import (
    ConfigManager,
    TrainingConfig,
    ConfigValidationError,
    load_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""
    
    def test_default_values(self):
        """Test that TrainingConfig has sensible defaults."""
        config = TrainingConfig()
        
        # Model settings
        assert config.model_name == "openai/gpt-oss-120b"
        assert config.lora_rank == 16
        
        # Training settings
        assert config.max_steps == 1000
        assert config.learning_rate == 5e-5
        assert config.temperature == 0.8
        assert config.group_size == 4
        
        # Verification settings
        assert config.max_concurrent_verifications == 8
        assert config.verification_timeout == 60.0
        
        # Logging and checkpointing
        assert config.logging_steps == 10
        assert config.checkpoint_interval == 100
        assert config.checkpoint_dir == "checkpoints"
        
        # Curriculum settings
        assert config.curriculum_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert config.promotion_threshold == 5
        assert config.window_size == 8
    
    def test_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            model_name="custom/model",
            max_steps=500,
            learning_rate=1e-4,
        )
        
        assert config.model_name == "custom/model"
        assert config.max_steps == 500
        assert config.learning_rate == 1e-4
        # Defaults still apply for unspecified fields
        assert config.temperature == 0.8
    
    def test_to_dict_masks_api_key(self):
        """Test that to_dict masks sensitive values."""
        config = TrainingConfig(api_key="secret-key-123")
        
        result = config.to_dict(mask_sensitive=True)
        
        assert result["api_key"] == "***MASKED***"
        assert result["model_name"] == "openai/gpt-oss-120b"
    
    def test_to_dict_no_masking(self):
        """Test that to_dict can expose sensitive values."""
        config = TrainingConfig(api_key="secret-key-123")
        
        result = config.to_dict(mask_sensitive=False)
        
        assert result["api_key"] == "secret-key-123"


class TestConfigManagerInit:
    """Tests for ConfigManager initialization."""
    
    def test_init_without_yaml(self):
        """Test initialization without YAML path."""
        manager = ConfigManager()
        
        assert manager.yaml_path is None
        assert manager.config is None
    
    def test_init_with_yaml_path(self):
        """Test initialization with YAML path."""
        manager = ConfigManager("/path/to/config.yaml")
        
        assert manager.yaml_path == "/path/to/config.yaml"


class TestConfigManagerLoad:
    """Tests for ConfigManager.load() method."""
    
    def test_load_with_env_api_key(self):
        """Test loading config with API key from environment."""
        with patch.dict(os.environ, {"TINKER_API_KEY": "test-api-key"}, clear=False):
            manager = ConfigManager()
            config = manager.load()
            
            assert config.api_key == "test-api-key"
            assert config.model_name == "openai/gpt-oss-120b"
    
    def test_load_missing_api_key_raises(self):
        """Test that missing API key raises ConfigValidationError."""
        # Ensure TINKER_API_KEY is not set
        env = {k: v for k, v in os.environ.items() if k != "TINKER_API_KEY"}
        
        with patch.dict(os.environ, env, clear=True):
            manager = ConfigManager()
            
            with pytest.raises(ConfigValidationError, match="TINKER_API_KEY"):
                manager.load()
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
api_key: yaml-api-key
model_name: custom/model
max_steps: 500
learning_rate: 0.0001
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            manager = ConfigManager(yaml_path)
            config = manager.load()
            
            assert config.api_key == "yaml-api-key"
            assert config.model_name == "custom/model"
            assert config.max_steps == 500
            assert config.learning_rate == 0.0001
            # Defaults still apply
            assert config.temperature == 0.8
        finally:
            os.unlink(yaml_path)
    
    def test_env_overrides_yaml(self):
        """Test that environment variables override YAML values."""
        yaml_content = """
api_key: yaml-api-key
model_name: yaml/model
max_steps: 500
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            env_overrides = {
                "TINKER_API_KEY": "env-api-key",
                "FIM_MODEL_NAME": "env/model",
            }
            with patch.dict(os.environ, env_overrides, clear=False):
                manager = ConfigManager(yaml_path)
                config = manager.load()
                
                # Env vars override YAML
                assert config.api_key == "env-api-key"
                assert config.model_name == "env/model"
                # YAML value preserved when no env override
                assert config.max_steps == 500
        finally:
            os.unlink(yaml_path)
    
    def test_load_nonexistent_yaml_raises(self):
        """Test that nonexistent YAML file raises FileNotFoundError."""
        manager = ConfigManager("/nonexistent/path/config.yaml")
        
        with pytest.raises(FileNotFoundError):
            manager.load()


class TestConfigManagerEnvOverrides:
    """Tests for environment variable overrides."""
    
    def test_int_conversion(self):
        """Test that integer env vars are converted correctly."""
        env_overrides = {
            "TINKER_API_KEY": "test-key",
            "FIM_MAX_STEPS": "2000",
            "FIM_GROUP_SIZE": "8",
            "FIM_LORA_RANK": "32",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            manager = ConfigManager()
            config = manager.load()
            
            assert config.max_steps == 2000
            assert isinstance(config.max_steps, int)
            assert config.group_size == 8
            assert config.lora_rank == 32
    
    def test_float_conversion(self):
        """Test that float env vars are converted correctly."""
        env_overrides = {
            "TINKER_API_KEY": "test-key",
            "FIM_LEARNING_RATE": "0.001",
            "FIM_TEMPERATURE": "0.5",
            "FIM_VERIFICATION_TIMEOUT": "120.0",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            manager = ConfigManager()
            config = manager.load()
            
            assert config.learning_rate == 0.001
            assert isinstance(config.learning_rate, float)
            assert config.temperature == 0.5
            assert config.verification_timeout == 120.0
    
    def test_invalid_int_raises(self):
        """Test that invalid integer env var raises ConfigValidationError."""
        env_overrides = {
            "TINKER_API_KEY": "test-key",
            "FIM_MAX_STEPS": "not-a-number",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            manager = ConfigManager()
            
            with pytest.raises(ConfigValidationError, match="Invalid value"):
                manager.load()
    
    def test_invalid_float_raises(self):
        """Test that invalid float env var raises ConfigValidationError."""
        env_overrides = {
            "TINKER_API_KEY": "test-key",
            "FIM_LEARNING_RATE": "not-a-float",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            manager = ConfigManager()
            
            with pytest.raises(ConfigValidationError, match="Invalid value"):
                manager.load()


class TestConfigManagerValidation:
    """Tests for configuration validation."""
    
    def test_negative_max_steps_raises(self):
        """Test that negative max_steps raises validation error."""
        yaml_content = """
api_key: test-key
max_steps: -1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            manager = ConfigManager(yaml_path)
            
            with pytest.raises(ConfigValidationError, match="max_steps"):
                manager.load()
        finally:
            os.unlink(yaml_path)
    
    def test_negative_learning_rate_raises(self):
        """Test that negative learning_rate raises validation error."""
        yaml_content = """
api_key: test-key
learning_rate: -0.001
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            manager = ConfigManager(yaml_path)
            
            with pytest.raises(ConfigValidationError, match="learning_rate"):
                manager.load()
        finally:
            os.unlink(yaml_path)
    
    def test_invalid_curriculum_levels_raises(self):
        """Test that invalid curriculum_levels raises validation error."""
        yaml_content = """
api_key: test-key
curriculum_levels: [0.1, 1.5, 0.3]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            manager = ConfigManager(yaml_path)
            
            with pytest.raises(ConfigValidationError, match="curriculum_levels"):
                manager.load()
        finally:
            os.unlink(yaml_path)


class TestConfigManagerLogEffectiveConfig:
    """Tests for log_effective_config method."""
    
    def test_log_effective_config(self):
        """Test that effective config is logged correctly."""
        with patch.dict(os.environ, {"TINKER_API_KEY": "secret-key"}, clear=False):
            manager = ConfigManager()
            config = manager.load()
            
            # Create mock metrics logger
            mock_logger = MagicMock()
            
            manager.log_effective_config(mock_logger)
            
            # Verify log_config was called
            mock_logger.log_config.assert_called_once()
            
            # Verify API key is masked
            logged_config = mock_logger.log_config.call_args[0][0]
            assert logged_config["api_key"] == "***MASKED***"
    
    def test_log_effective_config_not_loaded(self):
        """Test that logging before load doesn't crash."""
        manager = ConfigManager()
        mock_logger = MagicMock()
        
        # Should not raise, just log warning
        manager.log_effective_config(mock_logger)
        
        # log_config should not be called
        mock_logger.log_config.assert_not_called()


class TestConfigManagerGetConfig:
    """Tests for get_config method."""
    
    def test_get_config_after_load(self):
        """Test get_config returns config after load."""
        with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}, clear=False):
            manager = ConfigManager()
            manager.load()
            
            config = manager.get_config()
            
            assert isinstance(config, TrainingConfig)
    
    def test_get_config_before_load_raises(self):
        """Test get_config raises if not loaded."""
        manager = ConfigManager()
        
        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_config()


class TestLoadConfigConvenience:
    """Tests for load_config convenience function."""
    
    def test_load_config_without_yaml(self):
        """Test load_config without YAML path."""
        with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}, clear=False):
            config = load_config()
            
            assert isinstance(config, TrainingConfig)
            assert config.api_key == "test-key"
    
    def test_load_config_with_yaml(self):
        """Test load_config with YAML path."""
        yaml_content = """
api_key: yaml-key
max_steps: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            config = load_config(yaml_path)
            
            assert config.api_key == "yaml-key"
            assert config.max_steps == 100
        finally:
            os.unlink(yaml_path)
