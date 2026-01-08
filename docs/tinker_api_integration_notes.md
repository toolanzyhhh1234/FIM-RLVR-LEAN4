# Tinker API Integration Notes

This document captures key learnings from integrating the Tinker API for RL training on Lean4 proof infilling.

## Installation

```bash
pip install tinker
```

The package is `tinker` (not `tinker-api`). Current version: 0.7.0

## Key API Patterns

### 1. Client Creation

```python
import tinker

# Create ServiceClient (instant)
service_client = tinker.ServiceClient()

# Create TrainingClient (takes a few minutes - provisions GPU resources)
training_client = await service_client.create_lora_training_client_async(
    base_model="Qwen/Qwen3-8B",  # or "openai/gpt-oss-120b"
    rank=16,
)

# Get tokenizer from training client
tokenizer = training_client.get_tokenizer()
```

### 2. Creating a Sampling Client

```python
# Two-step process: save weights, then create sampler
save_future = await training_client.save_weights_for_sampler_async(name="step_000001")
save_result = save_future.result()  # Must call .result() on the future
sampling_path = save_result.path

sampling_client = service_client.create_sampling_client(model_path=sampling_path)
```

### 3. Sampling Completions

```python
from tinker import types

# Create ModelInput from raw token list
prompt_tokens = tokenizer.encode("Your prompt here")
model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

# Set sampling parameters
sampling_params = tinker.SamplingParams(
    max_tokens=512,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)

# Sample
response = await sampling_client.sample_async(
    prompt=model_input,
    num_samples=4,  # Group size
    sampling_params=sampling_params,
)

# Process results
for seq in response.sequences:
    tokens = seq.tokens
    logprobs = seq.logprobs  # List[float], available if requested
    decoded = tokenizer.decode(tokens)
```

### 4. Training with RL Loss (CISPO/PPO/importance_sampling)

```python
from tinker import types
from tinker.types.tensor_data import TensorData
import torch

# Build training datum
all_tokens = prompt_tokens + completion_tokens
input_tokens = all_tokens[:-1]
target_tokens = all_tokens[1:]

# Pad logprobs and advantages for prompt tokens
ob_len = len(prompt_tokens) - 1
padded_logprobs = [0.0] * ob_len + completion_logprobs
padded_advantages = [0.0] * ob_len + [advantage] * len(completion_tokens)

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
    },
)

# Forward-backward (submit before waiting for efficiency)
fb_future = training_client.forward_backward(
    data=[datum],  # List of Datum
    loss_fn="cispo",  # or "importance_sampling", "ppo", "dro"
)

# Optimizer step
adam_params = types.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8)
optim_future = training_client.optim_step(adam_params)

# Wait for both
fb_result = fb_future.result()
optim_result = optim_future.result()
```

### 5. Supported Loss Functions

| Loss Function | Use Case |
|---------------|----------|
| `cross_entropy` | Supervised learning |
| `importance_sampling` | Basic policy gradient |
| `ppo` | PPO with clipping |
| `cispo` | CISPO - recommended for MoE models |
| `dro` | Direct Reward Optimization |

## Important Gotchas

1. **Async futures need `.result()`**: The `_async` methods return `AwaitableConcurrentFuture` objects. You must call `.result()` to get the actual value.

2. **Local script must keep running**: Tinker handles model training remotely, but your local Python script must stay running to:
   - Sample completions from Tinker
   - Run verification locally (Lean4)
   - Compute rewards and send them back

3. **Clock cycles**: Tinker uses a clock-cycle model. Submit `forward_backward` and `optim_step` together before waiting to use 1 clock cycle instead of 3.

4. **Token alignment**: For RL losses, `input_tokens`, `target_tokens`, `logprobs`, and `advantages` must all have the same length.

## Monitoring

### CLI
```bash
export TINKER_API_KEY=your_key
tinker run list
tinker run info <session-id>
```

### Web Console
https://console.tinker.thinkingmachines.ai

## References

- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Loss Functions](https://tinker-docs.thinkingmachines.ai/losses)
- [RL Training](https://tinker-docs.thinkingmachines.ai/rl)

## Example Training Script

See `test_tinker_minimal.py` for a working example that tests all basic operations.
