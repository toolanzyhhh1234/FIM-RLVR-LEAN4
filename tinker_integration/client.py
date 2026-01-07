"""
Tinker Training Client Setup for Lean4 FIM + RLVR.

This module provides factory functions and utilities for creating and configuring
Tinker API training clients for large MoE model fine-tuning with CISPO loss.

Requirements covered:
- 4.1: Configure to use `gpt-oss-120b` as the primary model
- 4.2: Use LoRA fine-tuning with configurable rank (default: 16)
- 4.3: Use CISPO as the loss function (`loss_fn="cispo"`)
- 4.4: Support fallback to alternative models
- 4.5: Validate API credentials and model availability on initialization
- 4.6: Raise descriptive error if API credentials are invalid
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class TinkerClientError(Exception):
    """Base exception for Tinker client errors."""
    pass


class TinkerAuthenticationError(TinkerClientError):
    """Raised when API credentials are invalid or missing."""
    pass


class TinkerModelNotAvailableError(TinkerClientError):
    """Raised when the requested model is not available."""
    pass


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.
    
    Attributes:
        rank: LoRA rank (default: 16 per Requirement 4.2)
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA to
    """
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class ClientConfig:
    """
    Configuration for Tinker training client.
    
    Attributes:
        model_name: Primary model to use (default: gpt-oss-120b per Requirement 4.1)
        fallback_model: Fallback model if primary unavailable (Requirement 4.4)
        lora_config: LoRA configuration
        loss_fn: Loss function to use (default: cispo per Requirement 4.3)
        api_key: Tinker API key (can also be set via TINKER_API_KEY env var)
        api_base_url: Base URL for Tinker API
    """
    model_name: str = "openai/gpt-oss-120b"
    fallback_model: Optional[str] = "Qwen/Qwen3-235B-A22B"
    lora_config: LoRAConfig = None
    loss_fn: str = "cispo"
    api_key: Optional[str] = None
    api_base_url: str = "https://api.tinker.thinkingmachines.ai"
    
    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = LoRAConfig()


@runtime_checkable
class TrainingClient(Protocol):
    """
    Protocol defining the expected interface for a Tinker training client.
    
    This protocol allows for both real Tinker clients and mock implementations
    for testing purposes.
    """
    
    async def forward_backward_async(
        self,
        data: List[Any],
        loss_fn: str,
        advantages: Any,
        ref_logprobs: Any,
    ) -> Dict[str, Any]:
        """Compute forward and backward pass with specified loss function."""
        ...
    
    async def optim_step_async(self, learning_rate: float) -> None:
        """Perform optimizer step to update LoRA weights."""
        ...
    
    async def save_weights_and_get_sampling_client_async(
        self, 
        checkpoint_name: str
    ) -> "SamplingClient":
        """Save current weights and return a sampling client."""
        ...
    
    async def save_state_async(self, state_name: str) -> None:
        """Save current training state (weights + optimizer)."""
        ...
    
    async def load_state_with_optimizer_async(self, state_name: str) -> None:
        """Load training state including optimizer state."""
        ...


@runtime_checkable
class SamplingClient(Protocol):
    """Protocol for Tinker sampling client."""
    
    async def sample_async(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Sample completions from the model."""
        ...


class TinkerTrainingClient:
    """
    Wrapper around Tinker API training client.
    
    This class provides a consistent interface for interacting with Tinker's
    training API, handling authentication, model configuration, and error handling.
    
    Implements Requirements 4.1-4.6.
    """
    
    def __init__(
        self,
        config: ClientConfig,
        _tinker_client: Optional[Any] = None,  # For dependency injection in tests
    ):
        """
        Initialize the Tinker training client.
        
        Args:
            config: Client configuration
            _tinker_client: Optional pre-configured Tinker client (for testing)
        """
        self.config = config
        self._client = _tinker_client
        self._model_name = config.model_name
        self._initialized = False
    
    @property
    def model_name(self) -> str:
        """Get the currently configured model name."""
        return self._model_name
    
    @property
    def is_initialized(self) -> bool:
        """Check if the client has been initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """
        Initialize the client and validate credentials.
        
        This method validates API credentials (Requirement 4.5) and checks
        model availability. If the primary model is unavailable and a fallback
        is configured, it will attempt to use the fallback (Requirement 4.4).
        
        Raises:
            TinkerAuthenticationError: If API credentials are invalid (Requirement 4.6)
            TinkerModelNotAvailableError: If no suitable model is available
        """
        if self._initialized:
            return
        
        # Get API key from config or environment
        api_key = self.config.api_key or os.environ.get("TINKER_API_KEY")
        
        if not api_key:
            raise TinkerAuthenticationError(
                "Tinker API key is required. Set TINKER_API_KEY environment variable "
                "or provide api_key in configuration."
            )
        
        # Validate API key format (basic validation)
        if len(api_key) < 10:
            raise TinkerAuthenticationError(
                "Invalid Tinker API key format. API key appears to be too short."
            )
        
        # If we have an injected client (for testing), use it
        if self._client is not None:
            self._initialized = True
            logger.info(f"Using injected Tinker client with model: {self._model_name}")
            return
        
        # Try to import and initialize the real Tinker client
        try:
            await self._initialize_tinker_client(api_key)
        except ImportError:
            # Tinker package not installed - create a placeholder
            logger.warning(
                "Tinker package not installed. Client will operate in mock mode. "
                "Install with: pip install tinker-api"
            )
            self._client = None
            self._initialized = True
            return
        
        self._initialized = True
    
    async def _initialize_tinker_client(self, api_key: str) -> None:
        """
        Initialize the actual Tinker client.
        
        Args:
            api_key: Validated API key
            
        Raises:
            TinkerAuthenticationError: If authentication fails
            TinkerModelNotAvailableError: If model is not available
        """
        # Try to import tinker - this will fail if not installed
        try:
            import tinker
        except ImportError:
            raise  # Re-raise to be caught by caller
        
        try:
            # Try primary model first (Requirement 4.1)
            try:
                self._client = await self._create_lora_client(
                    tinker, 
                    api_key, 
                    self.config.model_name
                )
                self._model_name = self.config.model_name
                logger.info(f"Initialized Tinker client with primary model: {self._model_name}")
                return
            except Exception as e:
                if self.config.fallback_model:
                    logger.warning(
                        f"Primary model {self.config.model_name} unavailable: {e}. "
                        f"Trying fallback model: {self.config.fallback_model}"
                    )
                else:
                    raise TinkerModelNotAvailableError(
                        f"Model {self.config.model_name} is not available and no fallback configured: {e}"
                    )
            
            # Try fallback model (Requirement 4.4)
            try:
                self._client = await self._create_lora_client(
                    tinker,
                    api_key,
                    self.config.fallback_model
                )
                self._model_name = self.config.fallback_model
                logger.info(f"Initialized Tinker client with fallback model: {self._model_name}")
            except Exception as e:
                raise TinkerModelNotAvailableError(
                    f"Neither primary model ({self.config.model_name}) nor fallback model "
                    f"({self.config.fallback_model}) is available: {e}"
                )
                
        except TinkerAuthenticationError:
            raise  # Re-raise our own auth errors
        except Exception as e:
            # Check if it's a tinker authentication error
            if hasattr(tinker, 'AuthenticationError') and isinstance(e, tinker.AuthenticationError):
                raise TinkerAuthenticationError(
                    f"Tinker API authentication failed. Please check your API key. Error: {e}"
                )
            raise
    
    async def _create_lora_client(
        self, 
        tinker_module: Any, 
        api_key: str, 
        model_name: str
    ) -> Any:
        """
        Create a LoRA training client for the specified model.
        
        Args:
            tinker_module: The imported tinker module
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            Configured Tinker training client
        """
        lora_config = self.config.lora_config
        
        # Create LoRA training client (Requirement 4.2)
        client = await tinker_module.create_lora_training_client_async(
            api_key=api_key,
            model=model_name,
            lora_rank=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
        
        return client
    
    async def forward_backward_async(
        self,
        data: List[Any],
        advantages: Any,
        ref_logprobs: Any,
    ) -> Dict[str, Any]:
        """
        Compute forward and backward pass with CISPO loss.
        
        Uses CISPO loss function as specified in Requirement 4.3.
        
        Args:
            data: List of model inputs
            advantages: Advantage values for each sample
            ref_logprobs: Reference log probabilities
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self._ensure_initialized()
        
        if self._client is None:
            # Mock mode - return dummy values
            logger.debug("forward_backward_async called in mock mode")
            return {"loss": 0.0, "mock": True}
        
        return await self._client.forward_backward_async(
            data=data,
            loss_fn=self.config.loss_fn,  # "cispo" per Requirement 4.3
            advantages=advantages,
            ref_logprobs=ref_logprobs,
        )
    
    async def optim_step_async(self, learning_rate: float) -> None:
        """
        Perform optimizer step to update LoRA weights.
        
        Args:
            learning_rate: Learning rate for this step
        """
        self._ensure_initialized()
        
        if self._client is None:
            logger.debug("optim_step_async called in mock mode")
            return
        
        await self._client.optim_step_async(learning_rate=learning_rate)
    
    async def save_weights_and_get_sampling_client_async(
        self,
        checkpoint_name: str
    ) -> Any:
        """
        Save current weights and return a sampling client.
        
        Args:
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Sampling client configured with current weights
        """
        self._ensure_initialized()
        
        if self._client is None:
            logger.debug("save_weights_and_get_sampling_client_async called in mock mode")
            return MockSamplingClient()
        
        return await self._client.save_weights_and_get_sampling_client_async(
            checkpoint_name
        )
    
    async def save_state_async(self, state_name: str) -> None:
        """
        Save current training state.
        
        Args:
            state_name: Name for the saved state
        """
        self._ensure_initialized()
        
        if self._client is None:
            logger.debug(f"save_state_async called in mock mode: {state_name}")
            return
        
        await self._client.save_state_async(state_name)
    
    async def load_state_with_optimizer_async(self, state_name: str) -> None:
        """
        Load training state including optimizer state.
        
        Args:
            state_name: Name of the state to load
        """
        self._ensure_initialized()
        
        if self._client is None:
            logger.debug(f"load_state_with_optimizer_async called in mock mode: {state_name}")
            return
        
        await self._client.load_state_with_optimizer_async(state_name)
    
    def _ensure_initialized(self) -> None:
        """Ensure the client has been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "Client not initialized. Call initialize() before using the client."
            )


class MockSamplingClient:
    """Mock sampling client for testing when Tinker is not installed."""
    
    async def sample_async(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Return mock sample result."""
        from dataclasses import dataclass
        import numpy as np
        
        @dataclass
        class MockSampleResult:
            tokens: List[int]
            logprobs: Any
        
        # Return a simple mock completion
        mock_tokens = [1, 2, 3, 4, 5]  # Placeholder tokens
        mock_logprobs = np.zeros(len(mock_tokens))
        
        return MockSampleResult(tokens=mock_tokens, logprobs=mock_logprobs)


async def create_training_client(
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    api_key: Optional[str] = None,
    loss_fn: str = "cispo",
    auto_initialize: bool = True,
) -> TinkerTrainingClient:
    """
    Factory function to create and configure a Tinker training client.
    
    This is the primary entry point for creating training clients.
    Implements Requirements 4.1-4.6.
    
    Args:
        model_name: Model to use (default: gpt-oss-120b per Requirement 4.1)
        fallback_model: Fallback model if primary unavailable (Requirement 4.4)
        lora_rank: LoRA rank (default: 16 per Requirement 4.2)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        api_key: API key (or set TINKER_API_KEY env var)
        loss_fn: Loss function (default: cispo per Requirement 4.3)
        auto_initialize: Whether to automatically initialize the client
        
    Returns:
        Configured TinkerTrainingClient instance
        
    Raises:
        TinkerAuthenticationError: If API credentials are invalid (Requirement 4.6)
        TinkerModelNotAvailableError: If no suitable model is available
        
    Example:
        >>> client = await create_training_client(
        ...     model_name="openai/gpt-oss-120b",
        ...     lora_rank=16,
        ... )
        >>> # Use client for training
        >>> await client.forward_backward_async(data, advantages, ref_logprobs)
    """
    # Build configuration
    lora_config = LoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    
    config = ClientConfig(
        model_name=model_name or "openai/gpt-oss-120b",
        fallback_model=fallback_model or "Qwen/Qwen3-235B-A22B",
        lora_config=lora_config,
        loss_fn=loss_fn,
        api_key=api_key,
    )
    
    # Create client
    client = TinkerTrainingClient(config)
    
    # Initialize if requested
    if auto_initialize:
        await client.initialize()
    
    return client


def create_training_client_sync(
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    api_key: Optional[str] = None,
    loss_fn: str = "cispo",
) -> TinkerTrainingClient:
    """
    Synchronous factory function to create a Tinker training client.
    
    Note: This creates an uninitialized client. Call initialize() before use.
    For async code, prefer create_training_client() which auto-initializes.
    
    Args:
        model_name: Model to use (default: gpt-oss-120b)
        fallback_model: Fallback model if primary unavailable
        lora_rank: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        api_key: API key (or set TINKER_API_KEY env var)
        loss_fn: Loss function (default: cispo)
        
    Returns:
        Uninitialized TinkerTrainingClient instance
    """
    lora_config = LoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    
    config = ClientConfig(
        model_name=model_name or "openai/gpt-oss-120b",
        fallback_model=fallback_model or "Qwen/Qwen3-235B-A22B",
        lora_config=lora_config,
        loss_fn=loss_fn,
        api_key=api_key,
    )
    
    return TinkerTrainingClient(config)


def validate_api_key(api_key: Optional[str] = None) -> str:
    """
    Validate and return the Tinker API key.
    
    Checks both the provided key and TINKER_API_KEY environment variable.
    Implements Requirement 4.5 (credential validation).
    
    Args:
        api_key: Optional API key to validate
        
    Returns:
        Validated API key
        
    Raises:
        TinkerAuthenticationError: If no valid API key is found (Requirement 4.6)
    """
    key = api_key or os.environ.get("TINKER_API_KEY")
    
    if not key:
        raise TinkerAuthenticationError(
            "Tinker API key is required. Either:\n"
            "  1. Set the TINKER_API_KEY environment variable, or\n"
            "  2. Pass api_key parameter to create_training_client()\n"
            "\n"
            "Get your API key from: https://tinker.thinkingmachines.ai/"
        )
    
    if not isinstance(key, str):
        raise TinkerAuthenticationError(
            f"API key must be a string, got {type(key).__name__}"
        )
    
    # Basic format validation
    key = key.strip()
    if len(key) < 10:
        raise TinkerAuthenticationError(
            "Invalid API key format: key is too short. "
            "Please check your API key from https://tinker.thinkingmachines.ai/"
        )
    
    return key


# Supported models for reference
SUPPORTED_MODELS = {
    "openai/gpt-oss-120b": {
        "type": "MoE",
        "description": "Ultra-sparse MoE, best performer on Lean4 FIM task",
        "recommended": True,
    },
    "Qwen/Qwen3-235B-A22B": {
        "type": "MoE", 
        "description": "Large capacity MoE, cost-effective",
        "recommended": False,
    },
    "meta-llama/Llama-3.1-70B": {
        "type": "Dense",
        "description": "Strong dense baseline",
        "recommended": False,
    },
    "deepseek-ai/DeepSeek-V3.1": {
        "type": "MoE",
        "description": "Reasoning-focused MoE",
        "recommended": False,
    },
}


def list_supported_models() -> Dict[str, Dict[str, Any]]:
    """
    List models supported by Tinker API for this project.
    
    Returns:
        Dictionary mapping model names to their metadata
    """
    return SUPPORTED_MODELS.copy()
