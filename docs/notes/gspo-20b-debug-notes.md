# GSPO 20B Training Debug Log (Dec 17, 2025)

## Environment targets
- GPU: A100 40GB (driver CUDA 12.4), single GPU.
- Goal stack: Python 3.10, torch 2.6.0+cu124, flash-attn wheel (no source build), Unsloth with GSPO.

## Key decisions
- Use torch **2.6.0+cu124** (first torch with `torch.int1` required by torchao).
- Use flash-attn wheel: `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` (GitHub release).
- Use Unsloth extra: `unsloth[cu124-ampere-torch260]` from git.
- Keep Dynamo/torch.compile **disabled** for now to avoid bitsandbytes 4-bit compile issues.
- TRL upstream rejects `loss_type="gspo"`, so we run with `loss_type="grpo"` + `importance_sampling_level="sequence"` (GSPO-equivalent path in TRL) and call `PatchFastRL()` so Unsloth patches TRL trainers.

## Steps executed
1) Created env `unsloth-py310`, installed torch 2.6.0+cu124, flash-attn wheel, Unsloth extra, polars.
2) Patched `train_gspo_fim_20b.py`:
   - Import `PatchFastRL()` and invoke it.
   - Make hyperparams env-configurable.
   - Set `loss_type="grpo"` and `importance_sampling_level="sequence"`.
3) Bitsandbytes/Dynamo issue:
   - 4-bit MoE path (`Params4bit`) caused Dynamo unsupported errors.
   - We disabled Dynamo globally for the smoke run via env:
     `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE=0 UNSLOTH_DISABLE_DYNAMO=1`
4) Smoke run command:
   ```
   conda activate unsloth-py310
   TORCHDYNAMO_DISABLE=1 TORCH_COMPILE=0 UNSLOTH_DISABLE_DYNAMO=1 \
   FIM_PARQUET_PATH=data/data/train-00000-of-00001.parquet \
   FIM_MAX_STEPS=1 FIM_NUM_GENERATIONS=2 FIM_MAX_COMPLETION_LENGTH=64 \
   PYTORCH_CUDA_ALLOC_CONF= \
   python train_gspo_fim_20b.py
   ```
   Result: completed 1 step; rewards zero (expected in tiny run); merge to 16-bit succeeded.

## Outstanding issues / TODO
- **Dynamo + 4-bit:** Current workaround is disabling Dynamo. A cleaner fix is to graph-break only the bitsandbytes 4-bit forward (e.g., wrapper helper marked `@torch._dynamo.disable(recursive=False)`) so compile stays on elsewhere.
- **GSPO label:** Unsloth docs mention `loss_type="gspo"`, but stock TRL raises on it. Either:
  - Continue with `loss_type="grpo"` + `importance_sampling_level="sequence"` (works), or
  - Patch TRL/Unsloth local copy to alias `"gspo"` -> `"grpo"` with seq-level IS.
- **Performance:** With Dynamo off, expect slower training; revisit after a selective graph-break patch.

## Artifacts
- Parquet shard: `data/data/train-00000-of-00001.parquet`
- Latest smoke log: `training_logs/2025-12-17-half-a100.md`

## Next steps (suggested)
1) Implement a selective Dynamo disable just around bnb 4-bit matmul (helper function) and re-enable Dynamo globally.
2) Optionally alias `gspo` to `grpo` in a local TRL/Unsloth patch to match docs.
3) Run a short multi-step test with real hyperparams once Dynamo/bnb is stable.
