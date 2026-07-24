#!/bin/bash
# Warm the gemma-4 caches on the TARGET GPU (RTX 5090): serve once from the plain
# vllm-emmy image with ./warm mounted at /opt/emmy, so the model snapshot lands in
# warm/hf and every compiled kernel in warm/cubin — with the image's nvcc (toolkit_tag)
# at -O3 and the pinned config (cache-key parity with the baked image, which runs the
# same serve.sh). Requires HF_TOKEN in the env UNLESS warm/hf is pre-seeded with the
# model snapshot (the gated download happens here, once). On a multi-GPU box set
# GPU_DEVICE=<index> to pin the 5090 — warming on the wrong card produces a dead cache.
#
#   BASE_IMAGE=cloudriftai/vllm-emmy:TAG [GPU_DEVICE=1] ./warm.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a  # export the GEMMA4_* config so the -e pass-throughs below carry values
source ./config.env
set +a
: "${BASE_IMAGE:?set BASE_IMAGE to the plain vllm-emmy image to warm from}"
if [ ! -d "warm/hf/hub/models--${GEMMA4_MODEL//\//--}" ]; then
    : "${HF_TOKEN:?the gated gemma-4 download needs HF_TOKEN (or pre-seed warm/hf — see ARCHITECTURE.md)}"
fi
PORT="${PORT:-8000}"
GPUS="all"; [ -n "${GPU_DEVICE:-}" ] && GPUS="device=$GPU_DEVICE"
NAME=gemma4-warm

mkdir -p warm/hf warm/cubin warm/pack
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/opt/emmy/hf \
    -e EMMY_CUBIN_CACHE=/opt/emmy/cubin \
    -e EMMY_PACK_DIR=/opt/emmy/pack \
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
echo "[warm] online pass done: $(find warm/cubin -name '*.cubin' | wc -l) cubin(s)"

# Fixpoint passes under the RELEASE environment. The online warm boot is not the boot
# the shipped image performs: the baked image serves offline (HF_HUB_OFFLINE=1, model
# resolved to the snapshot path), and some kernels only materialize there; a fork pick
# can also flip between boots (each variant's source is stable, so the union converges).
# Re-boot offline against the accumulated cache until a boot compiles nothing new.
# The first boot above also wrote the execution-plan pack (warm/pack, keyed on the model
# CONFIG hash — not the id/path, so online and offline boots share it): these offline
# passes boot from it (frozen fork picks → no flip, fast boot), which both validates the
# pack-hit path pre-bake and makes the fixpoint converge on pass 1 whenever the pack took.
for pass in 1 2 3 4 5; do
    before=$(find warm/cubin -name '*.cubin' | sort)
    docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 \
        -e HF_HUB_OFFLINE=1 \
        -e HF_HOME=/opt/emmy/hf \
        -e EMMY_CUBIN_CACHE=/opt/emmy/cubin \
        -e EMMY_PACK_DIR=/opt/emmy/pack \
        -e EMMY_GEN_DECODE_BUCKET="$GEMMA4_DECODE_BUCKET" \
        -e GEMMA4_MODEL -e GEMMA4_MAX_MODEL_LEN -e GEMMA4_MAX_NUM_BATCHED_TOKENS -e GEMMA4_GPU_MEM_UTIL \
        -v "$PWD/warm":/opt/emmy \
        -v "$PWD/serve.sh":/opt/emmy/serve.sh:ro \
        --entrypoint /opt/emmy/serve.sh \
        "$BASE_IMAGE" >/dev/null
    echo "[warm] offline fixpoint pass $pass: waiting for /health..."
    for _ in $(seq 1 360); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
        if [ -z "$(docker ps -q -f name=$NAME)" ]; then
            echo "[warm] server died:"; docker logs --tail 50 "$NAME"; exit 1
        fi
        sleep 10
    done
    curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[warm] pass $pass timed out"; docker logs --tail 50 "$NAME"; exit 1; }
    served=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$served\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" >/dev/null
    docker stop "$NAME" >/dev/null && docker rm "$NAME" >/dev/null
    after=$(find warm/cubin -name '*.cubin' | sort)
    new=$(comm -13 <(echo "$before") <(echo "$after") | wc -l)
    echo "[warm] offline pass $pass added $new cubin(s)"
    [ "$new" -eq 0 ] && break
done
# A non-converged warm is a FAILURE, not a warning: fork picks are bimodal across boots, so
# verify's single boot can happen to hit only cached variants and PASS while customer boots
# recompile at runtime — exactly the failure class the fixpoint exists to contain, gated by a
# coin flip if this exits 0.
[ "$new" -eq 0 ] || { echo "[warm] FAIL: no fixpoint after 5 offline passes — the cubin set is still growing" >&2; exit 1; }

echo "[warm] done: $(find warm/cubin -name '*.cubin' | wc -l) cubin(s), $(find warm/pack -name '*.json' | wc -l) pack file(s), snapshot at warm/hf"
