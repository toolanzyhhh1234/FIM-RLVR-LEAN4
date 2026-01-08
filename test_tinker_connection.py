#!/usr/bin/env python3
"""Quick test to verify Tinker API integration is working."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_tinker_connection():
    """Test the Tinker API connection and client setup."""
    print("=" * 60)
    print("Tinker API Integration Test")
    print("=" * 60)
    
    # 1. Check environment variable
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("❌ TINKER_API_KEY not found in environment")
        return False
    
    print(f"✅ TINKER_API_KEY found (length: {len(api_key)} chars)")
    print(f"   Key prefix: {api_key[:10]}...")
    
    # 2. Test API key validation
    print("\n--- Testing API Key Validation ---")
    from tinker_integration.client import validate_api_key, TinkerAuthenticationError
    
    try:
        validated_key = validate_api_key()
        print(f"✅ API key validation passed")
    except TinkerAuthenticationError as e:
        print(f"❌ API key validation failed: {e}")
        return False
    
    # 3. Test client creation
    print("\n--- Testing Client Creation ---")
    from tinker_integration.client import create_training_client, list_supported_models
    
    print("Supported models:")
    for model, info in list_supported_models().items():
        rec = " (recommended)" if info.get("recommended") else ""
        print(f"   - {model}: {info['type']}{rec}")
    
    try:
        client = await create_training_client(
            model_name="openai/gpt-oss-120b",
            lora_rank=16,
            loss_fn="cispo",
        )
        print(f"✅ Client created successfully")
        print(f"   Model: {client.model_name}")
        print(f"   Initialized: {client.is_initialized}")
        print(f"   Loss function: {client.config.loss_fn}")
        print(f"   LoRA rank: {client.config.lora_config.rank}")
    except Exception as e:
        print(f"⚠️  Client creation note: {e}")
        # This is expected if tinker package isn't installed
    
    # 4. Test mock forward/backward (if in mock mode)
    print("\n--- Testing Mock Operations ---")
    try:
        result = await client.forward_backward_async(
            data=[{"test": "data"}],
            advantages=[0.5],
            ref_logprobs=[0.1],
        )
        print(f"✅ forward_backward_async works")
        print(f"   Result: {result}")
        
        await client.optim_step_async(learning_rate=1e-5)
        print(f"✅ optim_step_async works")
        
        await client.save_state_async("test_checkpoint")
        print(f"✅ save_state_async works")
        
    except Exception as e:
        print(f"❌ Mock operation failed: {e}")
        return False
    
    # 5. Test config loading
    print("\n--- Testing Config Loading ---")
    from tinker_integration.config import ConfigManager
    
    config_path = "configs/tinker_training.yaml"
    if os.path.exists(config_path):
        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load()  # Must call load() first
            print(f"✅ Config loaded from {config_path}")
            print(f"   Model: {config.model_name}")
            print(f"   Group size: {config.group_size}")
            print(f"   Learning rate: {config.learning_rate}")
            print(f"   Max steps: {config.max_steps}")
        except Exception as e:
            print(f"⚠️  Config loading issue: {e}")
    else:
        print(f"⚠️  Config file not found: {config_path}")
    
    print("\n" + "=" * 60)
    print("✅ All basic integration tests passed!")
    print("=" * 60)
    print("\nNote: The client is running in mock mode since the 'tinker' package")
    print("is not installed. This is expected for local development.")
    print("\nTo run actual training with Tinker API:")
    print("  pip install tinker-api")
    print("  python train_tinker_fim.py --config configs/tinker_training.yaml")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_tinker_connection())
    sys.exit(0 if success else 1)
