# FIM-RLVR-LEAN4 Hopper Image

This repo publishes a Hopper-focused training image built on CUDA 12.4 (cu124)
with FlashAttention prebuilt wheels and Apex. It targets H100/H200 class GPUs.

## Image
- `toolazy123/fim-rlvr-lean4:hopper`

## Quick start
```bash
docker pull toolazy123/fim-rlvr-lean4:hopper

docker create --runtime=nvidia --gpus all --net=host --shm-size="128g" \
  --cap-add=SYS_ADMIN --name fim-rlvr-hopper \
  -v /root/FIM-RLVR-LEAN4:/workspace/verl \
  toolazy123/fim-rlvr-lean4:hopper sleep infinity

docker start fim-rlvr-hopper
docker exec -it fim-rlvr-hopper bash
```

## Notes
- Hopper only (H100/H200). Blackwell (sm_120) is not supported in this image.
- For vLLM runs, prefer larger `/dev/shm` (64G-128G).
- The image includes vLLM and FlashAttention cu124 wheels to avoid local builds.

## Example (3B GSPO smoke run)
```bash
cd /workspace/verl
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
bash run_test_gspo_3b_math_patched.sh
```
