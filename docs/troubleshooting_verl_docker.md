# Troubleshooting: verl Docker Container Issues

This document records issues encountered when running verl training in Docker containers, along with solutions and lessons learned.

## Issue 1: blinker distutils Uninstall Error

**Symptom:**
```
error: uninstall-distutils-installed-package
Cannot uninstall blinker 1.4
It is a distutils installed project and thus we cannot accurately determine which files belong to it
```

**Cause:**
- `blinker 1.4` was installed via `apt` (system package) using distutils
- pip cannot safely uninstall distutils-installed packages
- Installing `megatron-bridge` requires a newer blinker version

**Solution:**
```bash
rm -rf /usr/lib/python3/dist-packages/blinker* && pip install megatron-bridge
```

**Prevention:**
- Use virtual environments to isolate from system packages
- Or use `pip install --ignore-installed <package>` to install alongside system version

---

## Issue 2: transformer_engine PyTorch ABI Mismatch

**Symptom:**
```
ImportError: transformer_engine_torch.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv
```

**Cause:**
- `transformer_engine_torch` was pre-compiled against a different PyTorch version
- The symbol `_ZNK3c106SymInt22maybe_as_int_slow_pathEv` is from PyTorch's `c10` library
- Installing `megatron-bridge` may have pulled incompatible dependency versions

**Attempted Solutions:**
```bash
# Reinstall transformer-engine (did not work - same error)
pip uninstall -y transformer-engine transformer_engine_torch transformer_engine_cu12
pip install --no-cache-dir transformer-engine[pytorch]
```

**Root Cause:**
The verl Docker container has tightly coupled CUDA extensions:
- PyTorch 2.8.0+cu128
- transformer_engine 2.6.0
- megatron-core 0.14.0

These are pre-built together. Installing additional packages (like megatron-bridge) can break the ABI compatibility.

**Recommendation:**
- **Pull a fresh container** rather than trying to fix corrupted dependencies
- Use `pip install --no-deps <package>` to avoid pulling transitive dependencies
- Check if there's a container image that already includes the packages you need

---

## Issue 3: vLLM KV Cache Memory Error

**Symptom:**
```
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
```

**Cause:**
- `gpu_memory_utilization` set too low (e.g., 0.2 or 0.35)
- 30B MoE model requires significant memory even with `load_format=dummy`
- vLLM needs space for model structure + KV cache blocks

**Solution:**
Increase `gpu_memory_utilization` in rollout config:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # or higher
```

Also reduce KV cache requirements:
```bash
actor_rollout_ref.rollout.max_model_len=1536  # match actual prompt+response length
actor_rollout_ref.rollout.max_num_seqs=64     # reduce concurrent sequences
actor_rollout_ref.rollout.max_num_batched_tokens=2048
```

---

## General Recommendations

1. **Snapshot before installing packages**: Always snapshot the container before installing new packages that have complex dependencies.

2. **Use --no-deps when possible**: If you only need specific functionality from a package:
   ```bash
   pip install --no-deps <package>
   ```

3. **Check container compatibility**: Before using a package, verify it's compatible with the container's pre-built dependencies.

4. **Fresh container over debugging**: For complex ABI issues, pulling a fresh container is often faster than debugging C++/CUDA symbol mismatches.

5. **Virtual environments**: Consider using a venv even inside Docker to isolate experimental installs:
   ```bash
   python -m venv /workspace/venv
   source /workspace/venv/bin/activate
   ```

---

## Container Information

- **Image**: verl Docker image with vLLM, PyTorch 2.8.0+cu128
- **Key pre-built packages**: transformer_engine, megatron-core, vllm, flash-attn
- **Date**: 2025-12-29
