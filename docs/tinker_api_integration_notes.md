# Tinker API Integration Notes

This document captures key learnings from integrating the Tinker API for RL training on Lean4 proof infilling.

## Installation


## Quick Start: Launch Training

```bash
export TINKER_API_KEY=your_key
python train_tinker_fim.py --config configs/tinker_training.yaml
python train_tinker_fim.py --config configs/tinker_training.yaml --max-steps 5
```

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

## GPT-OSS / Harmony Response Format

When using `openai/gpt-oss-120b` or similar models, responses use OpenAI's **Harmony format** with multiple channels:

### Channel Structure
```
<|channel|>analysis<|message|>...chain-of-thought reasoning...<|end|>
<|start|>assistant<|channel|>final<|message|>...actual answer...<|return|>
```

- **analysis**: Internal reasoning/CoT (not shown to end users in production)
- **final**: The actual response intended for the user
- **commentary**: Used for tool calls (less common in our use case)

### Tokenizer Considerations

- Use `skip_special_tokens=False` when decoding to preserve channel markers
- With `skip_special_tokens=True`, markers are stripped but channel names (e.g., "analysis") remain as plain text

### Code Extraction Quality

When extracting `<FIM_CODE>` tags from Harmony responses:

1. **Ideal case**: Tags appear in the `final` channel - this is the real answer
2. **Problematic case**: Tags only appear in `analysis` channel - model is discussing format, not answering
3. **Truncation case**: No `final` channel at all - model ran out of tokens during reasoning

**Diagnostic tip**: Check if `<|channel|>final<|message|>` appears before the extracted code tags. If not, the model likely:
- Spent too many tokens on reasoning (increase `max_tokens`)
- Got stuck in analysis mode (try `Reasoning: low` in system prompt)

### System Prompt Reasoning Control

Control reasoning verbosity in the system message:
```
Reasoning: high   # Extensive CoT (may exceed token limits)
Reasoning: medium # Balanced (default)
Reasoning: low    # Minimal CoT (faster, less truncation risk)
```

## FIM Boundary Normalization: Prompt/Verification Consistency

We hit a subtle but important failure mode in Lean FIM: **prompt-time boundary formatting must match verification-time stitching**.

### The underlying problem

For FIM tasks we conceptually build:

```
prefix + <model output> + suffix
```

However, Lean is whitespace/indentation sensitive. If the first suffix line begins with an indented tactic block marker like `· ...` and the model output does not end with a newline, then naive concatenation can produce:

```
... constructor  · ...
```

which is invalid Lean (`·` must start a new tactic line / block).

This was made worse by postprocessing: our tag extractor historically did `strip("\n")`, which removes trailing newlines the model might have emitted.

### Option A (chosen for training): canonicalize the boundary

**Goal:** avoid wasting reward signal on accidental formatting glitches.

Approach:

1. Ensure the prompt guarantees a separator between the hole and the suffix (suffix starts with `\n`).
2. Use the **same normalized suffix** when reconstructing `prefix + extracted_code + suffix` for verification.
3. Prefer extracting `<FIM_CODE>/<FULL_CODE>` from the Harmony **final** channel, falling back to whole response only if needed.

This is what the current Tinker training loop does.

### Option B (alternative / “raw”): fully raw concatenation

**Goal:** model must infer whether to emit a newline/indent based on the raw suffix.

Approach:

1. Show the model the raw prompt with no inserted newline normalization.
2. Verify using raw `prefix + extracted_code + raw_suffix`.
3. To make this workable, you typically must also **preserve trailing newlines** from `<FIM_CODE>` (i.e., avoid stripping them), otherwise you can silently remove the exact newline the model used to separate from suffix.

We are not using this for training right now because it increases variance and can collapse reward into formatting sensitivity.

### Debugging support

We log the ground-truth masked middle segment and a simple similarity score in `logs/tinker_fim/debug_samples.jsonl` to help distinguish:

- “model is close / semantically right but formatting broke” vs
- “model is genuinely wrong”.

## References

- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Loss Functions](https://tinker-docs.thinkingmachines.ai/losses)
- [RL Training](https://tinker-docs.thinkingmachines.ai/rl)
- [OpenAI Harmony Format](https://cookbook.openai.com/articles/openai-harmony)

## Example Training Script

See `test_tinker_minimal.py` for a working example that tests all basic operations.
