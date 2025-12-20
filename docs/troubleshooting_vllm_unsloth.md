# Troubleshooting: Unsloth + vLLM (gpt-oss) in this repo

This note captures issues encountered while running `train_gspo_fim_20b.py`
with Unsloth + vLLM and how they were resolved or mitigated.

## 1) `uv pip install` fails inside conda env

**Symptom**
```
uv pip install vllm --torch-backend=auto
error: No virtual environment found; run `uv venv` ... or pass `--system`
```

**Fix**
- Keep using the conda env and install into its site-packages:
```
uv pip install --system vllm --torch-backend=auto
```

## 2) Training appears "stuck" at step 0 with high GPU utilization

**Likely cause**
- Very large generation length: `max_completion_length` defaults to 32768.
  With `num_generations=4` this can take minutes per step even on H200.

**Mitigation**
Use shorter rollouts while debugging:
```
FIM_MAX_COMPLETION_LENGTH=512
FIM_NUM_GENERATIONS=1
FIM_MAX_STEPS=1
python train_gspo_fim_20b.py
```

## 3) vLLM fails with MKL threading layer error

**Symptom**
```
MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1
```

**Fix**
```
MKL_THREADING_LAYER=GNU python train_gspo_fim_20b.py
```
Alternative:
```
MKL_SERVICE_FORCE_INTEL=1 python train_gspo_fim_20b.py
```

## 4) vLLM fails with FlashInfer backend error

**Symptom**
```
Selected backend AttentionBackendEnum.FLASHINFER is not valid for this configuration.
Reason: ['sink setting not supported']
```

**Fix**
Force a different attention backend:
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN python train_gspo_fim_20b.py
```
If FlashAttention isn't available:
```
VLLM_ATTENTION_BACKEND=XFORMERS python train_gspo_fim_20b.py
```

## 5) vLLM + bitsandbytes 4-bit fails to load gpt-oss MoE weights

**Symptom**
```
ValueError: Following weights were not initialized from checkpoint:
{model.layers.*.mlp.experts.* , model.layers.*.mlp.router.*}
```

**Cause**
vLLM bitsandbytes loader does not correctly map gpt-oss MoE expert/router
weights for the `*-bnb-4bit` checkpoint in this setup.

**Fix (recommended)**
Disable 4-bit when using vLLM for this model:
```
FIM_LOAD_IN_4BIT=0 python train_gspo_fim_20b.py
```

**Alternative**
Disable vLLM (remove `fast_inference=True`) and use HF generate with 4-bit.

## 6) vLLM (bf16) fails converting gpt-oss to HF model (missing embed_tokens)

**Symptom**
```
AttributeError: 'GptOssModel' object has no attribute 'embed_tokens'
```

**Cause**
Unsloth's vLLM-to-HF conversion expects `embed_tokens`, but vLLM's
`GptOssModel` exposes embeddings under a different attribute name.

**Workarounds**
- Disable vLLM fast inference (`fast_inference=False`) and use HF generate.
- Upgrade Unsloth/Unsloth-Zoo or vLLM once a fix lands.

## 7) Unsloth docs note: gpt-oss RL is not yet vLLM-compatible

**Source in repo**
`unsloth_example_and_docs/gpt_oss_RL_unsloth_docs.md`

**Key note**
The doc states RL for gpt-oss is not yet vLLM compatible, and Unsloth
rewrote Transformers inference to deliver faster RL inference instead.

**Implication**
Using vLLM for gpt-oss RL is expected to be brittle until upstream support
lands; prefer Unsloth's built-in fast inference path.

## 8) Qwen3 MoE + Unsloth vLLM -> HF conversion fails (gate_up_proj missing)

**Symptom**
```
AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'gate_up_proj'
```

**Cause**
Unsloth's vLLM -> HF state_dict conversion assumes LLaMA-style MLP layout
(`gate_up_proj`). Qwen3 MoE uses `mlp.gate` + `mlp.experts`, so the expected
projection attributes are not present. This happens before LoRA is applied.

**Fix (recommended)**
Disable vLLM fast inference so Unsloth does not attempt the vLLM -> HF
conversion path for Qwen3 MoE.

## 9) Qwen3 MoE + vLLM + bitsandbytes 4-bit can fail

**Symptom**
```
NotImplementedError: BitsAndBytesMoEMethod must select appropriate gemm implementation ...
```

**Cause**
vLLM's bitsandbytes MoE path does not select a GEMM implementation for
this model/config.

**Fix (recommended)**
Run in BF16 (disable 4-bit):
```
FIM_LOAD_IN_4BIT=0 python train_gspo_fim_30b.py
```
