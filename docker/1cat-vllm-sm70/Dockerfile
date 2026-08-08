# syntax=docker/dockerfile:1.7

# Pinned CloudRift 1Cat-vLLM source build for the Volta Qwen3.5 bring-up.
#
# Build from the repository root:
#   docker build -f docker/1cat-vllm-sm70/Dockerfile \
#     -t cloudriftai/1cat-vllm-sm70:1.2.2-cloudrift .
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/opt/venv/bin:/usr/local/cuda-12.8/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=7.0 \
    FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0 \
    MAX_JOBS=8 \
    NVCC_THREADS=1 \
    VLLM_TARGET_DEVICE=cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libibverbs-dev \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

ARG ONECAT_REPO=https://github.com/cloudrift-ai/1Cat-vLLM.git
ARG ONECAT_COMMIT=91aca502d2bb1f05d9208ab2edec9fae53ff0d0b
ARG ONECAT_BASE_REF=cloudrift/v1.2.2
ARG ONECAT_BASE_TAG=v1.2.2
RUN git init /src \
    && git -C /src remote add origin "${ONECAT_REPO}" \
    && git -C /src fetch --depth 16 origin "${ONECAT_COMMIT}" \
    && git -C /src fetch --depth 1 origin "refs/heads/${ONECAT_BASE_REF}:refs/tags/${ONECAT_BASE_TAG}" \
    && git -C /src checkout --detach "${ONECAT_COMMIT}" \
    && test "$(git -C /src rev-parse HEAD)" = "${ONECAT_COMMIT}" \
    && git -C /src merge-base --is-ancestor "${ONECAT_BASE_TAG}" HEAD

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0
RUN cd /src \
    && sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt \
    && sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r requirements/build/cuda.txt \
        -r requirements/cuda.txt \
        -r requirements/common.txt \
        build

RUN cd /src/flash-attention-v100 \
    && python -m build --wheel --no-isolation --outdir /wheels \
    && cd /src \
    && CMAKE_BUILD_TYPE=Release python -m build --wheel --no-isolation --outdir /wheels

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/opt/venv/bin:/usr/local/cuda-12.8/bin:$PATH \
    VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=0 \
    VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0 \
    VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG=1 \
    VLLM_SM70_UNQUANTIZED_MOE_0DOT3_FUNCTIONAL=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=from=builder,source=/wheels,target=/tmp/wheels,ro \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
    && pip install --no-cache-dir \
        --prefer-binary \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        /tmp/wheels/*.whl \
    && pip install --no-cache-dir \
        catboost==1.2.10 \
        cupy-cuda12x==14.1.1 \
    && find /opt/venv/lib/python3.12/site-packages/vllm -type f -name '*.so' \
        -exec strip --strip-unneeded {} + \
    && find /opt/venv/lib/python3.12/site-packages -maxdepth 1 -type f \
        \( -name 'flash_attn_v100_cuda*.so' -o -name 'paged_kv_utils*.so' \) \
        -exec strip --strip-unneeded {} +
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENTRYPOINT ["vllm", "serve"]
