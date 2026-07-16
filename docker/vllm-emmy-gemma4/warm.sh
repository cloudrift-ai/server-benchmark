#!/bin/bash
# Warm the gemma-4 caches on the TARGET GPU (RTX 5090): serve once from the plain
# vllm-emmy image with ./warm mounted at /opt/emmy, so the model snapshot lands in
# warm/hf and every compiled kernel in warm/cubin — with the image's nvcc (toolkit_tag)
# at -O3 and the pinned config (cache-key parity with the baked image, which runs the
# same serve.sh). Requires HF_TOKEN in the env (the gated download happens here, once).
#
#   BASE_IMAGE=cloudriftai/vllm-emmy:TAG ./warm.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a  # export the GEMMA4_* config so the -e pass-throughs below carry values
source ./config.env
set +a
: "${BASE_IMAGE:?set BASE_IMAGE to the plain vllm-emmy image to warm from}"
: "${HF_TOKEN:?the gated gemma-4 download needs HF_TOKEN}"
PORT="${PORT:-8000}"
NAME=gemma4-warm

mkdir -p warm/hf warm/cubin
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host -p "$PORT":8000 \
    -e HF_TOKEN \
    -e HF_HOME=/opt/emmy/hf \
    -e EMMY_CUBIN_CACHE=/opt/emmy/cubin \
    -e EMMY_GEN_DECODE_BUCKET="$GEMMA4_DECODE_BUCKET" \
    -e GEMMA4_MODEL -e GEMMA4_MAX_MODEL_LEN -e GEMMA4_MAX_NUM_BATCHED_TOKENS -e GEMMA4_GPU_MEM_UTIL \
    -v "$PWD/warm":/opt/emmy \
    -v "$PWD/serve.sh":/opt/emmy/serve.sh:ro \
    --entrypoint /opt/emmy/serve.sh \
    "$BASE_IMAGE"

echo "[warm] waiting for /health (first boot downloads the model + compiles all layers — minutes)..."
for _ in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    if [ -z "$(docker ps -q -f name=$NAME)" ]; then
        echo "[warm] server died:"; docker logs --tail 50 "$NAME"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[warm] timed out waiting for /health"; docker logs --tail 50 "$NAME"; exit 1; }

echo "[warm] issuing one completion (covers prefill + decode kernels)..."
curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$GEMMA4_MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
    | head -c 400; echo

docker stop "$NAME" >/dev/null
docker rm "$NAME" >/dev/null
echo "[warm] done: $(find warm/cubin -name '*.cubin' | wc -l) cubin(s), snapshot at warm/hf"
