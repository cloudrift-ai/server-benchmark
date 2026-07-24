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
# Cleanup rides an EXIT trap: under `set -e` an inline `docker rm` after a failing
# diagnostic pipeline never runs and leaks the container (seen 2026-07-23).
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 "$IMAGE"

before=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")

# With a baked pack the boot skips the compiler frontend entirely and health arrives in
# ~weight-load time; without one (pack write skipped at warm) the per-layer CPU
# trace/lower/render runs uncached — the 48-layer gemma-4 boot takes ~25 min. Budget 40.
echo "[verify] waiting for /health (no downloads, no compiles; fast if the pack baked)..."
for _ in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    if [ -z "$(docker ps -q -f name=$NAME)" ]; then
        echo "[verify] server died:"; docker logs --tail 50 "$NAME"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[verify] timed out"; docker logs --tail 50 "$NAME"; exit 1; }

# Under HF_HUB_OFFLINE vLLM serves the model under the RESOLVED snapshot path, not the
# repo id — ask the server for its served name rather than assuming $GEMMA4_MODEL.
served=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$served\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
    | head -c 400; echo

after=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")
# When the image ships a pack, the boot must have actually used it — a silent fallback to
# the full compile (key/environment drift) still passes the cubin check but re-pays the
# ~25 min frontend on every customer boot, which is exactly what the pack exists to kill.
# The "pack hit" line is emmy's runner log; emmy.serving.register() attaches a log
# handler under the bare vLLM entrypoint precisely so it reaches docker logs here.
pack_baked=$(docker exec "$NAME" sh -c "find /opt/emmy/pack -name manifest.json 2>/dev/null | head -1")
# grep without -q: under pipefail, -q's early exit can SIGPIPE docker logs on a hit.
if [ -n "$pack_baked" ] && ! docker logs "$NAME" 2>&1 | grep "pack hit" >/dev/null; then
    echo "[verify] FAIL — a pack is baked but the boot did not hit it (fell back to full compile):"
    docker logs "$NAME" 2>&1 | grep -i "\[pack\]" | tail -5 || true
    exit 1
fi

if [ "$before" != "$after" ]; then
    echo "[verify] FAIL — new cubins compiled at runtime (cache miss):"
    diff <(echo "$before") <(echo "$after") || true
    exit 1
fi
echo "[verify] PASS — served offline with zero new cubins ($(echo "$before" | wc -l) prebuilt)${pack_baked:+, pack-hit boot}"
