# Quantization vs Compute Dtype (Unsloth bnb 4-bit)

- We load the base weights with `load_in_4bit=True`, so the model weights are in bitsandbytes 4-bit (NF4).
- Unsloth prints `Bfloat16 = TRUE` in the banner; this refers to the **compute dtype** used for matmuls (bf16), not the weight storage format.
- LoRA adapters are trained on top of the 4-bit base; their operations use bf16 compute while base weights stay quantized.
- You can verify locally via `model.config.quantization_config` and check `bnb_4bit_compute_dtype`.
