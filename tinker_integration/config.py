"""
Configuration management for Tinker API integration.

This module provides YAML-based configuration with environment variable overrides
for training large MoE models with CISPO loss.

Requirements covered:
- 10.1: Load configuration from YAML files
- 10.2: Support environment variable overrides for sensitive values
- 10.3: Validate configuration schema on load
- 10.4: Provide sensible defaults for all optional parameters
- 10.5: Log the effective configuration at training start
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Optional YAML support - gracefully handle if not installed
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

if TYPE_CHECKING:
    from .metrics import MetricsLogger


logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


@dataclass
class TrainingConfig:
    """
    Training configuration with sensible defaults.
    
    All fields have defaults to satisfy Requirement 10.4.
    """
    # API settings
    api_key: Optional[str] = None
    axle_api_key: Optional[str] = None
    
    # Model settings
    model_name: str = "openai/gpt-oss-120b"
    lora_rank: int = 16
    fallback_model: Optional[str] = "Qwen/Qwen3-235B-A22B"
    
    # Training settings
    max_steps: int = 1000
    learning_rate: float = 5e-5
    temperature: float = 0.8
    group_size: int = 4
    max_completion_tokens: int = 512
    
    # Verification settings
    max_concurrent_verifications: int = 8
    verification_timeout: float = 60.0
    axle_api_url: str = "https://axle.axiommath.ai/api/v1"
    axle_environment: str = "lean-4.28.0"
    
    # Reward shaping (intermediate signals)
    tag_reward: float = 0.05
    success_reward: float = 1.0
    unsolved_goals_reward: float = 0.0
    
    # Logging and checkpointing
    logging_steps: int = 10
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Curriculum settings
    curriculum_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    promotion_threshold: int = 5
    window_size: int = 8
    
    # Dataset settings
    dataset_path: Optional[str] = None
    
    # W&B integration (optional)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # S3 checkpoint storage (optional)
    s3_bucket: Optional[str] = None
    
    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Args:
            mask_sensitive: If True, mask sensitive values like api_key
            
        Returns:
            Dictionary representation of config
        """
        result = asdict(self)
        if mask_sensitive and result.get("api_key"):
            result["api_key"] = "***MASKED***"
        if mask_sensitive and result.get("axle_api_key"):
            result["axle_api_key"] = "***MASKED***"
        return result



class ConfigManager:
    """
    Manages configuration loading from YAML with environment variable overrides.
    
    Implements Requirements 10.1-10.5:
    - Loads configuration from YAML files (10.1)
    - Supports environment variable overrides (10.2)
    - Validates configuration schema on load (10.3)
    - Provides sensible defaults (10.4)
    - Logs effective configuration at training start (10.5)
    """
    
    # Environment variables that override YAML values
    # Maps env var name -> config field name
    ENV_OVERRIDES: Dict[str, str] = {
        "TINKER_API_KEY": "api_key",
        "AXLE_API_KEY": "axle_api_key",
        "AXLE_API_URL": "axle_api_url",
        "AXLE_ENVIRONMENT": "axle_environment",
        "FIM_MODEL_NAME": "model_name",
        "FIM_MAX_STEPS": "max_steps",
        "FIM_LEARNING_RATE": "learning_rate",
        "FIM_CHECKPOINT_INTERVAL": "checkpoint_interval",
        "FIM_CHECKPOINT_DIR": "checkpoint_dir",
        "FIM_LOG_DIR": "log_dir",
        "FIM_TEMPERATURE": "temperature",
        "FIM_GROUP_SIZE": "group_size",
        "FIM_LORA_RANK": "lora_rank",
        "FIM_DATASET_PATH": "dataset_path",
        "FIM_VERIFICATION_TIMEOUT": "verification_timeout",
        "FIM_MAX_CONCURRENT_VERIFICATIONS": "max_concurrent_verifications",
        "FIM_MAX_COMPLETION_TOKENS": "max_completion_tokens",
        "FIM_TAG_REWARD": "tag_reward",
        "FIM_LEAN_SUCCESS_REWARD": "success_reward",
        "FIM_UNSOLVED_GOALS_REWARD": "unsolved_goals_reward",
        "WANDB_PROJECT": "wandb_project",
        "WANDB_RUN_NAME": "wandb_run_name",
    }
    
    # Fields that should be converted to integers
    INT_FIELDS = {
        "max_steps", "lora_rank", "group_size", "logging_steps",
        "checkpoint_interval", "promotion_threshold", "window_size",
        "max_concurrent_verifications", "max_completion_tokens"
    }
    
    # Fields that should be converted to floats
    FLOAT_FIELDS = {
        "learning_rate", "temperature", "verification_timeout",
        "tag_reward", "success_reward", "unsolved_goals_reward"
    }
    
    # Required fields (must be present after loading)
    REQUIRED_FIELDS: List[str] = []  # api_key checked separately
    
    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            yaml_path: Optional path to YAML configuration file
        """
        self.yaml_path = yaml_path
        self._config: Optional[TrainingConfig] = None
    
    def load(self) -> TrainingConfig:
        """
        Load configuration from YAML with environment variable overrides.
        
        Order of precedence (highest to lowest):
        1. Environment variables
        2. YAML file values
        3. Default values in TrainingConfig
        
        Returns:
            Loaded and validated TrainingConfig
            
        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If YAML path specified but file doesn't exist
        """
        # Start with empty dict - defaults come from dataclass
        config_dict: Dict[str, Any] = {}
        
        # Load from YAML if provided (Requirement 10.1)
        if self.yaml_path:
            config_dict = self._load_yaml(self.yaml_path)
        
        # Apply environment variable overrides (Requirement 10.2)
        config_dict = self._apply_env_overrides(config_dict)
        
        # Validate configuration (Requirement 10.3)
        self._validate(config_dict)
        
        # Create config object with defaults (Requirement 10.4)
        # Only pass non-None values to let dataclass defaults apply
        filtered_dict = {k: v for k, v in config_dict.items() if v is not None}
        self._config = TrainingConfig(**filtered_dict)
        
        return self._config
    
    def _load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Dictionary of configuration values
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ConfigValidationError: If YAML parsing fails
        """
        if not YAML_AVAILABLE:
            raise ConfigValidationError(
                "PyYAML is required for YAML configuration. "
                "Install with: pip install pyyaml"
            )
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Failed to parse YAML config: {e}")
        
        if not isinstance(config_dict, dict):
            raise ConfigValidationError(
                f"YAML config must be a dictionary, got {type(config_dict).__name__}"
            )
        
        return config_dict
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config_dict: Current configuration dictionary
            
        Returns:
            Updated configuration dictionary
        """
        for env_var, config_key in self.ENV_OVERRIDES.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion based on field type
                try:
                    if config_key in self.INT_FIELDS:
                        value = int(value)
                    elif config_key in self.FLOAT_FIELDS:
                        value = float(value)
                    # String fields remain as-is
                except ValueError as e:
                    raise ConfigValidationError(
                        f"Invalid value for {env_var}: {value}. "
                        f"Expected {'integer' if config_key in self.INT_FIELDS else 'float'}. "
                        f"Error: {e}"
                    )
                
                config_dict[config_key] = value
                logger.debug(f"Applied env override: {env_var} -> {config_key}")
        
        return config_dict
    
    def _validate(self, config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors: List[str] = []
        
        # Check required fields
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in config_dict or config_dict[field_name] is None:
                errors.append(f"Required field '{field_name}' is missing")
        
        # API key is required (can come from env var or config)
        if "api_key" not in config_dict and "TINKER_API_KEY" not in os.environ:
            errors.append(
                "TINKER_API_KEY environment variable or 'api_key' in config is required"
            )
        
        # Validate numeric ranges
        if "max_steps" in config_dict:
            if not isinstance(config_dict["max_steps"], int) or config_dict["max_steps"] < 1:
                errors.append("max_steps must be a positive integer")
        
        if "learning_rate" in config_dict:
            lr = config_dict["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("learning_rate must be a positive number")
        
        if "temperature" in config_dict:
            temp = config_dict["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0:
                errors.append("temperature must be a non-negative number")
        
        if "group_size" in config_dict:
            gs = config_dict["group_size"]
            if not isinstance(gs, int) or gs < 1:
                errors.append("group_size must be a positive integer")
        
        if "lora_rank" in config_dict:
            lr = config_dict["lora_rank"]
            if not isinstance(lr, int) or lr < 1:
                errors.append("lora_rank must be a positive integer")
        
        if "verification_timeout" in config_dict:
            vt = config_dict["verification_timeout"]
            if not isinstance(vt, (int, float)) or vt <= 0:
                errors.append("verification_timeout must be a positive number")

        if "tag_reward" in config_dict:
            tr = config_dict["tag_reward"]
            if not isinstance(tr, (int, float)) or tr < 0:
                errors.append("tag_reward must be a non-negative number")

        if "success_reward" in config_dict:
            sr = config_dict["success_reward"]
            if not isinstance(sr, (int, float)) or sr < 0:
                errors.append("success_reward must be a non-negative number")

        if "unsolved_goals_reward" in config_dict:
            ugr = config_dict["unsolved_goals_reward"]
            if not isinstance(ugr, (int, float)) or ugr < 0:
                errors.append("unsolved_goals_reward must be a non-negative number")
        
        if "max_concurrent_verifications" in config_dict:
            mcv = config_dict["max_concurrent_verifications"]
            if not isinstance(mcv, int) or mcv < 1:
                errors.append("max_concurrent_verifications must be a positive integer")
        
        # Validate curriculum_levels if present
        if "curriculum_levels" in config_dict:
            levels = config_dict["curriculum_levels"]
            if not isinstance(levels, list):
                errors.append("curriculum_levels must be a list")
            elif not all(isinstance(x, (int, float)) and 0 < x <= 1 for x in levels):
                errors.append("curriculum_levels must contain values between 0 and 1")
        
        # Raise all errors at once
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    def log_effective_config(self, metrics_logger: "MetricsLogger") -> None:
        """
        Log the effective configuration (masking sensitive values).
        
        Implements Requirement 10.5.
        
        Args:
            metrics_logger: MetricsLogger instance to log to
        """
        if self._config is None:
            logger.warning("Cannot log config: configuration not loaded")
            return
        
        # Get config dict with sensitive values masked
        config_dict = self._config.to_dict(mask_sensitive=True)
        
        # Log to metrics logger
        metrics_logger.log_config(config_dict)
        
        # Also log to standard logger
        logger.info("Effective configuration:")
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
    
    @property
    def config(self) -> Optional[TrainingConfig]:
        """Get the loaded configuration, or None if not loaded."""
        return self._config
    
    def get_config(self) -> TrainingConfig:
        """
        Get the loaded configuration.
        
        Returns:
            The loaded TrainingConfig
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet
        """
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        return self._config


# Convenience function for quick config loading
def load_config(yaml_path: Optional[str] = None) -> TrainingConfig:
    """
    Convenience function to load configuration.
    
    Args:
        yaml_path: Optional path to YAML configuration file
        
    Returns:
        Loaded TrainingConfig
    """
    manager = ConfigManager(yaml_path)
    return manager.load()
