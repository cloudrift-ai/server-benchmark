# Pinned Volta-capable vLLM runtime for the Phase 1 Qwen3.5 bring-up.
#
# Build from the repository root:
#   docker build -f docker/1cat-vllm-sm70/Dockerfile \
#     -t cloudriftai/1cat-vllm-sm70:1.2.2 .
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

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
        curl \
        git \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

ARG ONECAT_VERSION=1.2.2
ARG ONECAT_SHA256=8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6
RUN curl -fsSL \
        -o "/tmp/1cat_vllm-${ONECAT_VERSION}-cp312-cp312-linux_x86_64.whl" \
        "https://github.com/1CatAI/1Cat-vLLM/releases/download/v${ONECAT_VERSION}/1cat_vllm-${ONECAT_VERSION}-cp312-cp312-linux_x86_64.whl" \
    && echo "${ONECAT_SHA256}  /tmp/1cat_vllm-${ONECAT_VERSION}-cp312-cp312-linux_x86_64.whl" | sha256sum -c -
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0
RUN pip install --no-cache-dir \
        --prefer-binary \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "/tmp/1cat_vllm-${ONECAT_VERSION}-cp312-cp312-linux_x86_64.whl" \
    && rm "/tmp/1cat_vllm-${ONECAT_VERSION}-cp312-cp312-linux_x86_64.whl"
RUN pip install --no-cache-dir \
        catboost==1.2.10 \
        cupy-cuda12x==14.1.1
# TileLang 0.1.9 emits a half-only intrinsic while parsing its BF16 helper for SM70. The paired
# 0.1.10 packages contain the upstream FP32-conversion fallback and must move together. Keep the
# packaged FlashQLA prefill path selected for the first supported boot, avoiding TileLang JIT.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        apache-tvm-ffi==0.1.10 \
        tilelang==0.1.10
# The release exposes a `vllm` console script whose version path asks for the upstream
# distribution name. Preserve the signed wheel metadata and add only the missing alias.
RUN cp -a "/opt/venv/lib/python3.12/site-packages/1cat_vllm-${ONECAT_VERSION}.dist-info" \
        "/opt/venv/lib/python3.12/site-packages/vllm-${ONECAT_VERSION}.dist-info" \
    && sed -i 's/^Name: 1cat-vllm$/Name: vllm/' \
        "/opt/venv/lib/python3.12/site-packages/vllm-${ONECAT_VERSION}.dist-info/METADATA"

# The pinned release profiles every PP rank as though it returned a final hidden-state tensor.
# Non-final Qwen ranks return IntermediateTensors, so patch the exact upstream blocks and fail the
# build if the signed wheel contents ever drift.
COPY docker/1cat-vllm-sm70/patch_onecat.py /tmp/patch_onecat.py
RUN python /tmp/patch_onecat.py \
    && python -m py_compile /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py \
    && rm /tmp/patch_onecat.py

WORKDIR /srv
ENTRYPOINT ["vllm", "serve"]
