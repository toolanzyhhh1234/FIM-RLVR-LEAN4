# Hopper Docker Image (cu124)

This doc mirrors the Docker Hub README for the Hopper-focused image.

## Image
- `toolazy123/fim-rlvr-lean4:hopper`

## Build (local)
```bash
docker build -f Dockerfile.verl-enhanced-vllm -t fim-rlvr-lean4:hopper .
```

## Run
```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="128g" \
  --cap-add=SYS_ADMIN --name fim-rlvr-hopper \
  -v /root/FIM-RLVR-LEAN4:/workspace/verl \
  toolazy123/fim-rlvr-lean4:hopper sleep infinity
docker start fim-rlvr-hopper
docker exec -it fim-rlvr-hopper bash
```

## Notes
- Hopper only (H100/H200). Blackwell requires a separate image.
- FlashAttention uses cu124 prebuilt wheels in this image.
