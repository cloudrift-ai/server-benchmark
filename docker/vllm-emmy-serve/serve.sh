#!/bin/sh
# The frozen generative serve entrypoint — the arg set `emmy serve --generate` builds,
# parameterized by the SERVE_* env (baked from models/<slug>.env). The warm run
# mounts this script into the plain vllm-emmy image and the baked image ships it, so
# the warmed and released invocations are literally the same — cache-key parity.
#
# The compilation-config mirrors emmy/commands/serve.py's generate path: FULL_DECODE_ONLY
# whole-step decode cudagraphs (capture sizes = the power-of-two ladder to max-num-seqs,
# with the decode bucket riding the list) and the forced fused rotary_embedding CustomOp
# (vLLM's dispatch otherwise hands the eager-inside-graph plugin forward_native — a
# ~0.9 ms/step per-layer torch-op soup). --no-enable-prefix-caching matches the
# benchmark protocol (every request does full prefill work). Keep in sync with
# _generate_compile_args in emmy/commands/serve.py.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${SERVE_MODEL}" \
    --runner generate \
    --dtype float16 \
    --max-model-len "${SERVE_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${SERVE_MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${SERVE_GPU_MEM_UTIL}" \
    --no-enable-prefix-caching \
    --hf-overrides '{"architectures": ["EmmyGenModel"]}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256], "custom_ops": ["+rotary_embedding"]}' \
    "$@"
