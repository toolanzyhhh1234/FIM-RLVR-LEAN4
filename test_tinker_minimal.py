#!/usr/bin/env python3
"""Minimal test of Tinker API to verify the actual interface."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_tinker_flow():
    """Test the basic Tinker training flow."""
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    import torch
    
    print("1. Creating ServiceClient...")
    service = tinker.ServiceClient()
    print(f"   Session: {service}")
    
    print("\n2. Creating TrainingClient (this takes a few minutes)...")
    # Use a smaller model for testing
    training_client = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3-8B",  # Smaller model for faster testing
        rank=16,
    )
    print(f"   TrainingClient created!")
    
    print("\n3. Getting tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print(f"   Tokenizer: {type(tokenizer)}")
    
    # Test encoding
    test_text = "theorem test : 1 + 1 = 2 := by"
    tokens = tokenizer.encode(test_text)
    print(f"   Encoded '{test_text[:30]}...' -> {len(tokens)} tokens")
    
    print("\n4. Creating SamplingClient...")
    # Use save_weights_for_sampler + create_sampling_client pattern from the example
    # The async version returns a future, need to await then get result
    save_future = await training_client.save_weights_for_sampler_async(name="test_ckpt")
    save_result = save_future.result()  # Get the actual result from the future
    sampling_path = save_result.path
    sampling_client = service.create_sampling_client(model_path=sampling_path)
    print(f"   SamplingClient created! Path: {sampling_path}")
    
    print("\n5. Testing sample()...")
    # Build ModelInput using from_ints (the correct way for raw tokens)
    prompt_text = "Complete this Lean4 proof:\ntheorem add_comm : forall a b : Nat, a + b = b + a := by\n  "
    prompt_tokens = tokenizer.encode(prompt_text)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    
    sampling_params = tinker.SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )
    
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=2,
        sampling_params=sampling_params,
    )
    
    print(f"   Got {len(response.sequences)} samples:")
    for i, seq in enumerate(response.sequences):
        decoded = tokenizer.decode(seq.tokens)
        print(f"   Sample {i+1} ({len(seq.tokens)} tokens): {decoded[:80]}...")
        print(f"            logprobs available: {seq.logprobs is not None}")
    
    print("\n6. Testing forward_backward() with RL loss...")
    # Build training data following the real example pattern
    completion_tokens = response.sequences[0].tokens
    completion_logprobs = response.sequences[0].logprobs
    
    # Full sequence: prompt + completion
    all_tokens = prompt_tokens + completion_tokens
    
    # For RL: input_tokens are all but last, target_tokens are all but first
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    
    # Pad logprobs and advantages for prompt tokens
    ob_len = len(prompt_tokens) - 1  # observation length
    padded_logprobs = [0.0] * ob_len + list(completion_logprobs)
    advantage = 1.0  # Dummy positive advantage
    padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
    
    print(f"   input_tokens: {len(input_tokens)}, target_tokens: {len(target_tokens)}")
    print(f"   padded_logprobs: {len(padded_logprobs)}, padded_advantages: {len(padded_advantages)}")
    
    # Create Datum with loss_fn_inputs
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
        },
    )
    
    print("   Calling forward_backward with importance_sampling...")
    fb_future = training_client.forward_backward(data=[datum], loss_fn="importance_sampling")
    
    print("\n7. Testing optim_step()...")
    adam_params = types.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_future = training_client.optim_step(adam_params)
    
    # Wait for both
    fb_result = fb_future.result()
    optim_result = optim_future.result()
    print(f"   forward_backward result: {fb_result}")
    print(f"   optim_step result: {optim_result}")
    
    print("\n✅ All basic operations work!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_tinker_flow())
    print(f"\nTest {'passed' if success else 'failed'}")
