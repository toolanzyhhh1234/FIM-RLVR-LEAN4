Installation
Requirements
Python: Version >= 3.10

CUDA: Version >= 12.8

verl supports various backends. Currently, the following configurations are available:

FSDP and Megatron-LM (optional) for training.

SGLang, vLLM and TGI for rollout generation.

Choices of Backend Engines
Training:

We recommend using FSDP backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in FSDP Workers.

For users who pursue better scalability, we recommend using Megatron-LM backend. Currently, we support Megatron-LM v0.13.1. The guide for using Megatron-LM backend can be found in Megatron-LM Workers.

Inference:

For inference, vllm 0.8.3 and later versions have been tested for stability. We recommend turning on env var VLLM_USE_V1=1 for optimal performance.

For SGLang, refer to the SGLang Backend for detailed installation and usage instructions. SGLang rollout is under extensive development and offers many advanced features and optimizations. We encourage users to report any issues or provide feedback via the SGLang Issue Tracker.

For huggingface TGI integration, it is usually used for debugging and single GPU exploration.

Install from docker image
Start from v0.6.0, we use vllm and sglang release image as our base image.

Base Image
vLLM: https://hub.docker.com/r/vllm/vllm-openai

SGLang: https://hub.docker.com/r/lmsysorg/sglang

Application Image
Upon base image, the following packages are added:

flash_attn

Megatron-LM

Apex

TransformerEngine

DeepEP

Latest docker file:

Dockerfile.stable.vllm

Dockerfile.stable.sglang

All pre-built images are available in dockerhub: verlai/verl. For example, verlai/verl:sgl055.latest, verlai/verl:vllm011.latest.

You can find the latest images used for development and ci in our github workflows:

.github/workflows/vllm.yml

.github/workflows/sgl.yml

Installation from Docker
After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

Launch the desired Docker image and attach into it:

docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash

Project-specific notes for this repo:
- Mount the host repo into the container so changes persist. On the VM the repo is
  /root/FIM-RLVR-LEAN4, and inside the container it should be /workspace/verl.
  Example: -v /root/FIM-RLVR-LEAN4:/workspace/verl
- Lean4 must be installed inside the container (elan + lake) because verification
  runs in-container. Recommended install prefix: /root/.elan and ensure
  /root/.elan/bin is on PATH.
- Megatron + LoRA requires Megatron-Bridge (mbridge). This is not mentioned in the
  upstream install guide because it is only required for PEFT/LoRA via Megatron.
- Set --shm-size based on host RAM. For H200 machines, a common setting is 120g
  (about half of typical system RAM). Increase if you see shared-memory issues.
- Be prepared for source builds when installing Megatron-Bridge: pip may compile
  wheels for packages like causal-conv1d, mamba-ssm, nv-grouped-gemm, and
  transformer_engine_torch (can take a while on fresh nodes).
If you use the images provided, you only need to install verl itself without dependencies:

# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
