# Prebuilt per-model serving image: the vllm-emmy image + the warmed cubin cache and the
# HF model snapshot baked in, so cold-start pays zero nvcc compiles and zero HF downloads
# (no HF_TOKEN needed). One built image serves exactly one (model, GPU, config) triple.
#
# Build (the Makefile drives it; MODEL picks the pinned models/<slug>.env):
#   0. `make serve-goldens MODEL=<id>` — assert recorded goldens cover this model on this
#      card, so the warm bakes tuned fork picks rather than cold-greedy ones.
#   1. `make serve-warm MODEL=<id>` on a machine with the target GPU — serves once from
#      the plain vllm-emmy image with ./warm mounted at /opt/emmy, populating
#      warm/cubin (compiled kernels) + warm/hf (the model snapshot).
#   2. `make serve-image MODEL=<id>` — bakes ./warm into this image.
#   3. `make serve-verify MODEL=<id>` — cold-starts the baked image offline (HF_HUB_OFFLINE,
#      no token) and asserts one request adds zero new cubins.
#
# The caches live at /opt/emmy — NOT under /root/.cache/huggingface, which compose /
# recipes bind-mount over (a mount would shadow baked content).
#
# Serve (config baked in; just map the port):
#   docker run --gpus all --ipc=host -p 8000:8000 cloudriftai/vllm-emmy-<slug>:TAG
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV HF_HOME=/opt/emmy/hf \
    EMMY_CUBIN_CACHE=/opt/emmy/cubin \
    EMMY_PACK_DIR=/opt/emmy/pack \
    HF_HUB_OFFLINE=1

# Weights + tokenizer + config (the HF snapshot — per-tensor identical, but resharded by
# reshard_snapshot.py when it shipped a >10 GB consolidated file; LICENSE/NOTICE included,
# so the Apache 2.0 attribution ships with it), then the warmed sm_120/sm_120a cubins.
# The snapshot ships as four layers that merge into /opt/emmy/hf: Docker Hub 503s the
# monolithic ~24 GB blob at upload initiation, so split_hf.sh (run by the Makefile
# before this build) balances it into sub-10 GB parts.
COPY warm/hf_parts/p0 /opt/emmy/hf
COPY warm/hf_parts/p1 /opt/emmy/hf
COPY warm/hf_parts/p2 /opt/emmy/hf
COPY warm/hf_parts/p3 /opt/emmy/hf
COPY warm/cubin /opt/emmy/cubin
# The execution-plan pack (compiler/backend/pack.py) — written by the warm run's first
# boot. A pack-hit boot skips the whole compiler frontend (trace / pipeline / fork
# resolution / codegen), collapsing the ~25 min gemma-4 cold start to ~weight-load time;
# on any mismatch the runner silently falls back to the full (cubin-cached) compile.
COPY warm/pack /opt/emmy/pack

# The pinned serving config — passed by make from models/<slug>.env (the single source; these
# values are part of the cubin cache key, so they must match the warm run exactly).
ARG MODEL
ARG MAX_MODEL_LEN
ARG MAX_NUM_BATCHED_TOKENS
ARG GPU_MEM_UTIL
ARG DECODE_BUCKET
ENV SERVE_MODEL=${MODEL} \
    SERVE_MAX_MODEL_LEN=${MAX_MODEL_LEN} \
    SERVE_MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} \
    SERVE_GPU_MEM_UTIL=${GPU_MEM_UTIL} \
    EMMY_GEN_DECODE_BUCKET=${DECODE_BUCKET}

COPY serve.sh /opt/emmy/serve.sh
RUN chmod +x /opt/emmy/serve.sh

ENTRYPOINT ["/opt/emmy/serve.sh"]
CMD []
