1
ARG RELEASE
0 B
2
ARG LAUNCHPAD_BUILD_ARCH
0 B
3
LABEL org.opencontainers.image.ref.name=ubuntu
0 B
4
LABEL org.opencontainers.image.version=22.04
0 B
5
ADD file ... in /
29.03 MB
6
CMD ["/bin/bash"]
0 B
7
ENV NVARCH=x86_64
0 B
8
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471
0 B
9
ENV NV_CUDA_CUDART_VERSION=12.8.90-1
0 B
10
ARG TARGETARCH
0 B
11
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
0 B
12
RUN |1 TARGETARCH=amd64 /bin/sh -c
4.42 MB
13
ENV CUDA_VERSION=12.8.1
0 B
14
RUN |1 TARGETARCH=amd64 /bin/sh -c
61.32 MB
15
RUN |1 TARGETARCH=amd64 /bin/sh -c
186 B
16
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 B
17
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
0 B
18
COPY NGC-DL-CONTAINER-LICENSE / # buildkit
6.72 KB
19
ENV NVIDIA_VISIBLE_DEVICES=all
0 B
20
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
0 B
21
ENV NV_CUDA_LIB_VERSION=12.8.1-1
0 B
22
ENV NV_NVTX_VERSION=12.8.90-1
0 B
23
ENV NV_LIBNPP_VERSION=12.3.3.100-1
0 B
24
ENV NV_LIBNPP_PACKAGE=libnpp-12-8=12.3.3.100-1
0 B
25
ENV NV_LIBCUSPARSE_VERSION=12.5.8.93-1
0 B
26
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8
0 B
27
ENV NV_LIBCUBLAS_VERSION=12.8.4.1-1
0 B
28
ENV NV_LIBCUBLAS_PACKAGE=libcublas-12-8=12.8.4.1-1
0 B
29
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
0 B
30
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
0 B
31
ENV NCCL_VERSION=2.25.1-1
0 B
32
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.25.1-1+cuda12.8
0 B
33
ARG TARGETARCH
0 B
34
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
0 B
35
RUN |1 TARGETARCH=amd64 /bin/sh -c
1.92 GB
36
RUN |1 TARGETARCH=amd64 /bin/sh -c
62.58 KB
37
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/ # buildkit
1.64 KB
38
COPY nvidia_entrypoint.sh /opt/nvidia/ # buildkit
1.49 KB
39
ENV NVIDIA_PRODUCT_NAME=CUDA
0 B
40
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
0 B
41
ENV NV_CUDA_LIB_VERSION=12.8.1-1
0 B
42
ENV NV_CUDA_CUDART_DEV_VERSION=12.8.90-1
0 B
43
ENV NV_NVML_DEV_VERSION=12.8.90-1
0 B
44
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1
0 B
45
ENV NV_LIBNPP_DEV_VERSION=12.3.3.100-1
0 B
46
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-8=12.3.3.100-1
0 B
47
ENV NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1
0 B
48
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8
0 B
49
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-12-8=12.8.4.1-1
0 B
50
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1
0 B
51
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8=12.8.1-1
0 B
52
ENV NV_NVPROF_VERSION=12.8.90-1
0 B
53
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-8=12.8.90-1
0 B
54
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
0 B
55
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
0 B
56
ENV NCCL_VERSION=2.25.1-1
0 B
57
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.25.1-1+cuda12.8
0 B
58
ARG TARGETARCH
0 B
59
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
0 B
60
RUN |1 TARGETARCH=amd64 /bin/sh -c
2.78 GB
61
RUN |1 TARGETARCH=amd64 /bin/sh -c
86.77 KB
62
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
0 B
63
ARG CUDA_VERSION
0 B
64
ARG PYTHON_VERSION
0 B
65
ARG INSTALL_KV_CONNECTORS=false
0 B
66
WORKDIR /vllm-workspace
104 B
67
ENV DEBIAN_FRONTEND=noninteractive
0 B
68
ARG TARGETPLATFORM
0 B
69
ARG GDRCOPY_CUDA_VERSION=12.8
0 B
70
ARG GDRCOPY_OS_VERSION=Ubuntu22_04
0 B
71
SHELL [/bin/bash -c]
0 B
72
ARG DEADSNAKES_MIRROR_URL
0 B
73
ARG DEADSNAKES_GPGKEY_URL
0 B
74
ARG GET_PIP_URL
0 B
75
RUN |9 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
230 B
76
RUN |9 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
323.04 MB
77
ARG PIP_INDEX_URL UV_INDEX_URL
0 B
78
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
0 B
79
ARG PYTORCH_CUDA_INDEX_BASE_URL
0 B
80
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL
0 B
81
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER
0 B
82
RUN |17 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
41.44 MB
83
ENV UV_HTTP_TIMEOUT=500
0 B
84
ENV UV_INDEX_STRATEGY=unsafe-best-match
0 B
85
ENV UV_LINK_MODE=copy
0 B
86
RUN |17 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
12.88 KB
87
RUN |17 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
120 B
88
RUN |17 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
4.6 GB
89
ARG FLASHINFER_GIT_REPO=https://github.com/flashinfer-ai/flashinfer.git
0 B
90
ARG FLASHINFER_GIT_REF=v0.3.1
0 B
91
ARG FLASHINFER_AOT_COMPILE=false
0 B
92
RUN |20 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
1.17 GB
93
COPY examples examples # buildkit
173.15 KB
94
COPY benchmarks benchmarks # buildkit
148.96 KB
95
COPY ./vllm/collect_env.py . # buildkit
8.25 KB
96
RUN |20 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
122 B
97
COPY requirements/build.txt requirements/build.txt # buildkit
299 B
98
RUN |20 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
27.68 MB
99
ARG DEEPGEMM_GIT_REF
0 B
100
COPY tools/install_deepgemm.sh /tmp/install_deepgemm.sh # buildkit
1.44 KB
101
RUN |21 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
7.33 MB
102
COPY tools/install_gdrcopy.sh install_gdrcopy.sh # buildkit
906 B
103
RUN |21 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
573.42 KB
104
COPY tools/ep_kernels/install_python_libraries.sh install_python_libraries.sh # buildkit
1.51 KB
105
ENV CUDA_HOME=/usr/local/cuda
0 B
106
RUN |21 CUDA_VERSION=12.8.1 PYTHON_VERSION=3.12 INSTALL_KV_CONNECTORS=true
550.57 MB
107
ARG TARGETPLATFORM
0 B
108
ARG INSTALL_KV_CONNECTORS=false
0 B
109
ARG PIP_INDEX_URL UV_INDEX_URL
0 B
110
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
0 B
111
ENV UV_HTTP_TIMEOUT=500
0 B
112
COPY requirements/kv_connectors.txt requirements/kv_connectors.txt # buildkit
236 B
113
RUN |6 TARGETPLATFORM=linux/amd64 INSTALL_KV_CONNECTORS=true PIP_INDEX_URL=
116.26 MB
114
ENV VLLM_USAGE_SOURCE=production-docker-image
0 B
115
ENTRYPOINT ["python3" "-m" "vllm.entrypoints.openai.api_server"]
0 B
116
RUN /bin/bash -c pip install
609.1 KB
117
RUN /bin/bash -c wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
764.36 MB
118
RUN /bin/bash -c pip install
72.63 MB
119
RUN /bin/bash -c pip install
46.65 MB
120
RUN /bin/bash -c export NVTE_FRAMEWORK=pytorch
376.66 MB
121
RUN /bin/bash -c pip install
28.99 MB
122
RUN /bin/bash -c pip install
81.28 MB
123
RUN /bin/bash -c pip install
248.84 MB
124
RUN /bin/bash -c apt update
763.99 KB
125
RUN /bin/bash -c wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
400.65 MB
126
RUN /bin/bash -c apt-get install
403.47 MB
127
WORKDIR /home/dpsk_a2a
119 B
128
RUN /bin/bash -c git clone
6 MB
129
ENV CUDA_HOME=/usr/local/cuda
0 B
130
ENV CPATH=/usr/local/mpi/include:
0 B
131
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:/usr/local/cuda/lib64
0 B
132
ENV LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:/usr/local/mpi/lib:/usr/local/cuda/lib64
0 B
133
ENV GDRCOPY_HOME=/home/dpsk_a2a/gdrcopy
0 B
134
WORKDIR /home/dpsk_a2a/deepep-nvshmem
32 B
135
RUN /bin/bash -c NVSHMEM_SHMEM_SUPPORT=0
497.01 MB
136
RUN /bin/bash -c ln -s
151 B
137
WORKDIR /home/dpsk_a2a/DeepEP
32 B
138
ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install
0 B
139
RUN /bin/bash -c NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install python
130.26 MB
140
RUN /bin/bash -c pip3 install
964.57 KB
141
RUN /bin/bash -c pip3 install
36.35 MB
142
RUN /bin/bash -c pip install
1.62 MB
143
RUN /bin/bash -c pip install
4.5 MB
144
RUN /bin/bash -c pip install
277.22 MB
145
RUN /bin/bash -c pip uninstall
491 B