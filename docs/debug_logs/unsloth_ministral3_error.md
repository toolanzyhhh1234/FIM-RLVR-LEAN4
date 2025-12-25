# Unsloth model load failure for Ministral-3 14B (Dec 25, 2025)

## Summary
Unsloth failed to load `unsloth/Ministral-3-14B-Reasoning-2512` with:
"Unsloth: No config file found". The repository is public and reachable, but the
installed `transformers` version is too old to parse `model_type: mistral3`.

## Evidence
- Hugging Face config download succeeded (public repo, config exists):
  - `hf_hub_download('unsloth/Ministral-3-14B-Reasoning-2512', 'config.json')`
  - Cached at `/workspace/.hf_home/hub/models--unsloth--Ministral-3-14B-Reasoning-2512/snapshots/d74416dcd8b108f09d9f6740ccf5469332b39438/config.json`
- `AutoConfig.from_pretrained('unsloth/Ministral-3-14B-Reasoning-2512')` failed with:
  - `KeyError 'ministral3'`
- The model config shows:
  - `model_type: "mistral3"`
  - `transformers_version: "5.0.0.dev0"`
- Installed transformers version:
  - `transformers.__version__ == 4.57.3`

## Conclusion
Unsloth is able to reach the repo, but transformers 4.57.3 cannot resolve the
`mistral3` model type. Unsloth then ends up with no usable config and throws
"No config file found".

## Fix
Upgrade transformers to a dev build that supports Mistral-3:

```
uv pip install --upgrade git+https://github.com/huggingface/transformers.git
```

Then re-run:

```
uv run python train_gspo_fim_dense.py
```

## Recorded Versions
- transformers before update: 4.57.3

- transformers after update: 5.0.0.dev0
- updated via: uv pip install --upgrade git+https://github.com/huggingface/transformers.git
- git ref: a7f29523361b2cc12e51c1f5133d95f122f6f45c

## Follow-up: vLLM conflict
- Running `uv pip install --upgrade vllm` downgraded transformers back to 4.57.3 due to vLLM dependency constraints.
- Fix applied: uninstall vllm and reinstall transformers dev.

Commands:
- `uv pip uninstall vllm`
- `uv pip install --upgrade git+https://github.com/huggingface/transformers.git`

## vLLM fast_inference VLM LoRA limitation
- With vLLM (`fast_inference=True`) and a VLM model, Unsloth raised:
  `RuntimeError: Unsloth: Finetuning vision layers is not supported for fast_inference. Only text layers are supported!`
- Root cause: vLLM path in Unsloth disallows vision-layer LoRA on multimodal models.
- Fix: disable vLLM path by setting `fast_inference=False` (defaulted via `FIM_FAST_INFERENCE=0`).

## Qwen3-VL example requirements (official notebook)
- Added `requirements_unsloth_qwen3vl.txt` to mirror the Unsloth Qwen3-VL GRPO example.
- Pinned: `transformers==4.57.0`, `trl==0.22.2`, `xformers==0.0.33.post1`, `datasets>=3.4.1,<4.0.0`.

## xFormers load failure (Torch 2.9.0+cu130 / Python 3.12)
- xFormers warning during `train_gspo_fim_dense.py` startup:
  - Built for PyTorch 2.9.0+cu128 + CUDA 1208 + Python 3.10.19
  - Running on PyTorch 2.9.0+cu130 + CUDA 12.0 + Python 3.12.12
- Result: xFormers C++/CUDA extensions fail to load; Unsloth falls back to SDPA/fast eager.

## Conclusion: GRPO ref_hidden_states error persists across stacks
- Error: `TypeError: grpo_accumulated_loss() missing 1 required positional argument: 'ref_hidden_states'`
- Observed in:
  - Qwen3-VL env (`.venv310`, transformers==4.57.0, trl==0.22.2, unsloth==2025.9.5)
  - Mistral-3 env (`.venv310_mistral3`, transformers==5.0.0.dev0, trl==0.22.2, unsloth==2025.9.5)
- Deleting `unsloth_compiled_cache` regenerates `UnslothGRPOTrainer.py` with the same missing arg.
- Manual cache patch (adding `ref_hidden_states=None`) is overwritten on regeneration.
- Tried vanilla TRL GRPO (non-DR loss) and still hit the same `ref_hidden_states` issue.
- Conclusion: likely an upstream Unsloth/TRL compatibility bug; not stable to patch generated cache.
