#!/bin/bash
# Verify the baked image on the target GPU: cold-start fully offline (HF_HUB_OFFLINE is
# baked in; no HF_TOKEN passed) — proving zero downloads — then issue one completion and
# assert the cubin set did not grow: an empty diff = 100% cache hit (zero nvcc compiles).
#
#   IMAGE=cloudriftai/vllm-emmy-gemma4:TAG [GPU_DEVICE=1] ./verify.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
: "${IMAGE:?set IMAGE to the baked gemma4 image to verify}"
PORT="${PORT:-8000}"
GPUS="all"; [ -n "${GPU_DEVICE:-}" ] && GPUS="device=$GPU_DEVICE"
NAME=gemma4-verify

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 "$IMAGE"

before=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")

echo "[verify] waiting for /health (no downloads, no compiles expected)..."
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    if [ -z "$(docker ps -q -f name=$NAME)" ]; then
        echo "[verify] server died:"; docker logs --tail 50 "$NAME"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[verify] timed out"; docker logs --tail 50 "$NAME"; exit 1; }

curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$GEMMA4_MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
    | head -c 400; echo

after=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")
docker rm -f "$NAME" >/dev/null

if [ "$before" != "$after" ]; then
    echo "[verify] FAIL — new cubins compiled at runtime (cache miss):"
    diff <(echo "$before") <(echo "$after") || true
    exit 1
fi
echo "[verify] PASS — served offline with zero new cubins ($(echo "$before" | wc -l) prebuilt)"
