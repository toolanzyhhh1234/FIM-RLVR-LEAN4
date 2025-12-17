vLLM is fully compatible with GRPO training and is actively used during the training process, not just for post-training inference. It accelerates the generation phases of reinforcement learning training.

## vLLM Integration in GRPO Training

When you enable `fast_inference=True`, vLLM becomes the generation engine during GRPO training's rollout phases [1](#3-0) :

```python
# vLLM sampling configuration for training
sampling_params = vLLMSamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=512,
)

training_args = GRPOConfig(
    vllm_sampling_params=sampling_params,  # Used during training
    # ... other params
)
```

## How vLLM Works During GRPO Training

The integration happens through several key mechanisms:

1. **Model Loading**: vLLM engine is attached to the model as `model.vllm_engine` when `fast_inference=True` [2](#3-1) 

2. **Generation Phase**: During training, GRPO uses vLLM for generating completions instead of the standard model forward pass [3](#3-2) 

3. **Mode Switching**: The model automatically switches between training and inference modes during RL training [4](#3-3) 

## Key Benefits During Training

- **Faster Rollouts**: vLLM's paged attention and optimized kernels significantly speed up the generation phase
- **Memory Efficiency**: KV cache management reduces memory usage during batch generation
- **Higher Throughput**: Can generate multiple completions in parallel for each prompt

## Example Usage

Here's a complete example showing vLLM being used during GRPO training [5](#3-4) :

```python
from unsloth import FastLanguageModel, PatchFastRL, vLLMSamplingParams
from trl import GRPOConfig, GRPOTrainer

PatchFastRL()

# Load model with vLLM integration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    fast_inference=True,  # Enables vLLM for training
    gpu_memory_utilization=0.5,
)

# Configure GRPO with vLLM sampling
sampling_params = vLLMSamplingParams(
    temperature=1.0,
    max_tokens=512,
)

training_args = GRPOConfig(
    learning_rate=5e-6,
    num_generations=8,
    vllm_sampling_params=sampling_params,  # vLLM used during training
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_functions,
)

trainer.train()  # vLLM accelerates generation during training
```

## Notes

- vLLM is used during the generation/sampling phases of GRPO training, while the actual policy updates still use the training model
- The integration is seamless - you just need to set `fast_inference=True` and optionally configure `vllm_sampling_params`
- vLLM requires LoRA training (not full fine-tuning) when used with GRPO [6](#3-5) 

Wiki pages you might want to explore:
- [API Reference (unslothai/unsloth)](/wiki/unslothai/unsloth#10)

### Citations

**File:** unsloth/models/rl.py (L240-246)
```python
        # Enable training mode
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training()
        output = f(self, *args, **kwargs)
        # Return inference mode
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            self.model.for_inference()
```

**File:** unsloth/models/rl.py (L278-279)
```python
    vllm_sampling_params: Optional[Any] = field(
        default = None,
```

**File:** unsloth/models/vision.py (L712-717)
```python
            if full_finetuning:
                max_lora_rank = max(get_lora_supported_ranks())
                raise NotImplementedError(
                    f"Unsloth: `fast_inference = True` does not yet support `full_finetuning = True`.\n"
                    f"Use LoRA rank `r = {max_lora_rank}` as the closest replacement for full finetuning with Unsloth for RL."
                )
```

**File:** unsloth/models/vision.py (L760-764)
```python
            model.vllm_engine = llm
            model.fast_generate = model.vllm_engine.generate
            model.fast_generate_batches = functools.partial(
                generate_batches, model.vllm_engine
            )
```

**File:** unsloth/models/rl_replacements.py (L246-275)
```python
def grpo_trainer__generate_and_score_completions(function_name, function):
    if function_name != "_generate_and_score_completions":
        return function

    # TRL 0.19.0 did skip_special_tokens = True which should be False
    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    # Left pad prompt before calculation old and ref hidden states
    line_to_replace = 'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size'

    # The new multi-line string that will replace the line above
    replacement_lines = """
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        try:
            # TRL 0.23.1 and below path
            if not has_images:
                # Left pad prompt before calculation old and ref hidden states
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
            self.model.for_training()
        except:
            # TRL 0.24.0 and below path
            if images is None:
                # Left pad prompt before calculation old and ref hidden states
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
        self.model.for_training()"""

    function = function.replace(line_to_replace, replacement_lines)
```

**File:** tests/saving/language_models/test_save_merged_grpo_model.py (L76-83)
```python
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = False,  # False for LoRA 16bit
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.8,  # Reduce if out of memory
    )
```
