#!/bin/sh
# The frozen gemma-4 generative serve entrypoint — the arg set `emmy serve --generate`
# builds, parameterized by the GEMMA4_* env (baked from config.env). The warm run
# mounts this script into the plain vllm-emmy image and the baked image ships it, so
# the warmed and released invocations are literally the same — cache-key parity.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${GEMMA4_MODEL}" \
    --runner generate \
    --enforce-eager \
    --dtype float16 \
    --max-model-len "${GEMMA4_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${GEMMA4_MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GEMMA4_GPU_MEM_UTIL}" \
    --hf-overrides '{"architectures": ["EmmyGenModel"]}' \
    "$@"
