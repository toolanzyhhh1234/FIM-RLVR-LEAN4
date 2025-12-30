# CUDA Runtime Notes (vLLM / Unsloth)

## Summary
vLLM wheels expect **CUDA 12 runtime** libraries (notably `libcudart.so.12`) to
be visible inside the container. Missing `libcudart.so.12` causes vLLM import to
fail, which in turn makes Unsloth crash during its vLLM patching step.

## Symptoms
- `ImportError: libcudart.so.12: cannot open shared object file`
- Followed by `AttributeError: module vllm has no attribute sampling_params`

The AttributeError is a secondary symptom caused by vLLM failing to import.

## Why this happens in containers
- GPU **drivers** are provided by the host. Containers should not install or
  replace driver packages.
- vLLM still needs the **CUDA runtime** (`libcudart.so.12`) inside the container.
- Installing CUDA runtime via `apt` can conflict with host-mounted NVIDIA
  utilities (e.g., `nvidia-smi`) and fail during package removal.

## Recommended fixes
1) **Use a CUDA 12 host** (e.g., H100) so `libcudart.so.12` is available.
2) **Bind-mount** host CUDA runtime libs into the container:
   - Mount `/usr/local/cuda` from host to container.
   - Set:
     ```
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
     ```
3) Avoid installing GPU drivers inside containers.

## Decision note
- Blackwell / CUDA 13 hosts may require extra work (source builds or new wheels).
- A CUDA 12 host (H100) is the most reliable path for vLLM + Unsloth.
