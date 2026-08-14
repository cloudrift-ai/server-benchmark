#!/bin/bash
# Warm one model's caches on the TARGET GPU: serve once from the plain vllm-emmy image
# with ./warm mounted at /opt/emmy, so the model snapshot lands in warm/hf and every
# compiled kernel in warm/cubin — with the image's nvcc (toolkit_tag) at -O3 and the
# pinned config (cache-key parity with the baked image, which runs the same serve.sh).
# Requires HF_TOKEN in the env UNLESS warm/hf is pre-seeded with the model snapshot (a
# gated download happens here, once). Set HF_HUB_OFFLINE=1 with a pre-seeded snapshot to
# make even the initial boot network-independent. On a multi-GPU box set
# GPU_DEVICE=<index> to pin the target card — warming on the wrong card produces a dead cache.
#
#   MODEL=google/gemma-4-12B-it BASE_IMAGE=cloudriftai/vllm-emmy:TAG [GPU_DEVICE=1] ./warm.sh
set -euo pipefail
# Resolve the pinned config from MODEL via the naming schema — models/<slug>.env is the
# single source the bake reads too, so warm and release cannot drift apart.
cd "$(dirname "$0")"
MODEL="${MODEL:?set MODEL to the HF model id being released}"
SLUG=$(./model_slug.sh "$MODEL")
CONFIG="models/$SLUG.env"
[ -f "$CONFIG" ] || { echo "no pinned config for $MODEL (expected $CONFIG)" >&2; exit 1; }
set -a  # export the SERVE_* config so the -e pass-throughs below carry values
source "$CONFIG"
set +a

# Every cache-key input is card-specific — the live-probed featurization, the golden picks,
# the memory headroom the config was swept for. A warm on the wrong GPU bakes a cache the
# released image can never hit, and the failure only surfaces ~30 min later in verify (or
# worse, never, on a card whose picks happen to coincide). Check it here, cheaply.
if [ -n "${SERVE_GPU:-}" ] && [ -z "${SKIP_GPU_CHECK:-}" ]; then
    # `|| true`: under `set -euo pipefail` a missing nvidia-smi would abort the script here
    # with no message at all, turning a diagnosable "no GPU on this host" into a silent exit.
    live=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$(( ${GPU_DEVICE:-0} + 1 ))p" || true)
    if [ -n "$live" ] && [ "$live" != "$SERVE_GPU" ]; then
        echo "GPU mismatch: config pins '$SERVE_GPU', device ${GPU_DEVICE:-0} is '$live'." >&2
        echo "  Warm and verify must run on the card the config was swept for." >&2
        echo "  Override with SKIP_GPU_CHECK=1 only if you know the picks transfer." >&2
        exit 1
    fi
fi

# An UNPINNED id resolves to the repo's default branch. When a repo publishes several
# variants under one id — EXL3 rungs are one branch per bit rate, and base-model repos carry
# training-checkpoint branches — that default is a DIFFERENT model than the goldens were
# tuned for and the headroom was swept for, and nothing downstream notices: the wrong weights
# boot, verify passes, and the published image is silently the wrong variant. Every other
# cache-key input is checked somewhere; this one is only checkable here, before hours of warm.
# A single-branch repo is the ordinary case and needs no pin. Unreachable refs SKIP the check
# rather than fail it — warming offline off a pre-seeded snapshot is a supported path.
if [ -n "${SERVE_REVISION:-}" ]; then
    echo "[warm] revision pinned: $SERVE_REVISION"
    if ! printf '%s' "$SERVE_REVISION" | grep -Eq '^[0-9a-f]{40}$'; then
        echo "[warm] WARNING: SERVE_REVISION is not a 40-char commit sha. A branch or tag can be" >&2
        echo "         re-cut under the same name, and the offline boot resolves snapshots/<sha>." >&2
    fi
elif [ -z "${SKIP_REVISION_CHECK:-}" ]; then
    curl_args=(-sf --max-time 20)
    [ -n "${HF_TOKEN:-}" ] && curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
    refs=$(curl "${curl_args[@]}" "https://huggingface.co/api/models/$SERVE_MODEL/refs" || true)
    # count / a sample of the names / the default branch's head, so the operator can copy the
    # pin straight out of the message. Repos with hundreds of checkpoint branches are why the
    # sample is truncated.
    info=$(printf '%s' "$refs" | python3 -c 'import json,sys
bs = json.load(sys.stdin).get("branches", [])
print(len(bs))
print(" ".join(b["name"] for b in bs[:8]) + (" ..." if len(bs) > 8 else ""))
print(next((b["targetCommit"] for b in bs if b["name"] == "main"), ""))' 2>/dev/null || true)
    n=$(printf '%s\n' "$info" | sed -n 1p)
    if [ "${n:-0}" -gt 1 ]; then
        echo "unpinned revision: $SERVE_MODEL publishes $n branches under one id" >&2
        echo "  branches: $(printf '%s\n' "$info" | sed -n 2p)" >&2
        echo "  Without SERVE_REVISION the warm bakes whichever of them the default branch points" >&2
        echo "  at — different weights, and for a quantized repo a different bit allocation than" >&2
        echo "  the goldens were tuned for — published under this config's name, silently." >&2
        echo "  Set SERVE_REVISION=<40-char commit sha> in $CONFIG.$(printf '%s\n' "$info" | sed -n 3p | sed '/./s/^/ main is at /')" >&2
        echo "  Override with SKIP_REVISION_CHECK=1 only if the default branch IS the target." >&2
        exit 1
    fi
    if [ -z "${n:-}" ]; then
        echo "[warm] revision unpinned; this repo's branches are unreachable from here (check skipped)"
    fi
fi

: "${BASE_IMAGE:?set BASE_IMAGE to the plain vllm-emmy image to warm from}"
if [ ! -d "warm/hf/hub/models--${SERVE_MODEL//\//--}" ]; then
    : "${HF_TOKEN:?the gated model download needs HF_TOKEN (or pre-seed warm/hf — see ARCHITECTURE.md)}"
fi
PORT="${PORT:-8000}"
GPUS="all"; [ -n "${GPU_DEVICE:-}" ] && GPUS="device=$GPU_DEVICE"
NAME="emmy-warm-$SLUG"

# ---- the shape matrix ------------------------------------------------------------------
# The execution-plan pack is keyed on the SERVING SHAPE (decode bucket / chunk quantum /
# max-num-batched-tokens) and on the precision lane. A boot whose shape+lane has no baked
# pack re-runs the compiler frontend for every program: measured on gemma-4-12B at ~12 min
# against ~2 min for a pack hit (2026-08-01). The documented per-workload knobs are exactly
# such shapes, so an image baked only at the pinned shape makes every deployment that follows
# them pay that cost on every boot — and it dominated the article-reproduction session, where
# 24 of 28 emmy cells missed the pack.
#
# ``SERVE_WARM_SHAPES`` names the EXTRA shapes to warm beside the pinned one, space-separated,
# each ``bucket:prefill:mnbt`` (empty field = leave that knob at the pinned value), optionally
# suffixed ``:fm`` to warm it under the FAST_MATH gate. The pinned shape is always warmed
# first and is the only one whose failure aborts the release. Packs are small (a few MB), so
# the cost of extra shapes is warm TIME, not image size.
#
#   SERVE_WARM_SHAPES="8:2048:2056 64::4096 32:::fm"
runner_env() {
    [ -n "${SERVE_EMBED_HOST:-}" ] && printf -- '-e EMMY_GEN_EMBED_HOST=%s ' "$SERVE_EMBED_HOST"
    [ -n "${SERVE_PREFILL_CAPACITY:-}" ] && printf -- '-e EMMY_GEN_PREFILL_CAPACITY=%s ' "$SERVE_PREFILL_CAPACITY"
    [ -n "${SERVE_M1_TIER:-}" ] && printf -- '-e EMMY_GEN_M1_TIER=%s ' "$SERVE_M1_TIER"
    [ -n "${SERVE_V2_MODEL_RUNNER:-}" ] && printf -- '-e SERVE_V2_MODEL_RUNNER=%s ' "$SERVE_V2_MODEL_RUNNER"
    return 0
}

shape_env() {  # $1 = spec -> echoes the docker -e flags for that shape
    local spec="$1" b p m lane
    IFS=':' read -r b p m lane <<<"$spec"
    runner_env
    printf -- '-e EMMY_GEN_DECODE_BUCKET=%s ' "${b:-$SERVE_DECODE_BUCKET}"
    if [ -n "$p" ]; then
        printf -- '-e EMMY_GEN_PREFILL_BUCKET=%s ' "$p"
    elif [ -n "${SERVE_PREFILL_BUCKET:-}" ]; then
        printf -- '-e EMMY_GEN_PREFILL_BUCKET=%s ' "$SERVE_PREFILL_BUCKET"
    fi
    printf -- '-e SERVE_MAX_NUM_BATCHED_TOKENS=%s ' "${m:-$SERVE_MAX_NUM_BATCHED_TOKENS}"
    [ "$lane" = "fm" ] && printf -- '-e EMMY_FAST_MATH=1 '
    return 0
}

mkdir -p warm/hf warm/cubin warm/pack warm/triton
docker rm -f "$NAME" >/dev/null 2>&1 || true
initial_env=$(shape_env "")
# shellcheck disable=SC2086 — $initial_env is a deliberately word-split flag list
docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE \
    -e HF_HOME=/opt/emmy/hf \
    -e EMMY_CUBIN_CACHE=/opt/emmy/cubin \
    -e EMMY_PACK_DIR=/opt/emmy/pack \
    -e TRITON_CACHE_DIR=/opt/emmy/triton \
    $initial_env \
    -e SERVE_MODEL -e SERVE_MAX_MODEL_LEN -e SERVE_GPU_MEM_UTIL \
    -e SERVE_REVISION -e SERVE_QUANT -e SERVE_CAPTURE_SIZES -e SERVE_EXTRA_ARGS \
    -v "$PWD/warm":/opt/emmy \
    -v "$PWD/serve.sh":/opt/emmy/serve.sh:ro \
    --entrypoint /opt/emmy/serve.sh \
    "$BASE_IMAGE"

if [ "${HF_HUB_OFFLINE:-}" = "1" ]; then
    echo "[warm] initial pass is offline against the pre-seeded snapshot"
fi
echo "[warm] waiting for /health (first boot resolves the model + compiles all layers — minutes)..."
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
    -d "{\"model\": \"$SERVE_MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
    | head -c 400; echo

docker stop "$NAME" >/dev/null
docker rm "$NAME" >/dev/null
echo "[warm] initial pass done: $(find warm/cubin -name '*.cubin' | wc -l) cubin(s)"

# Fixpoint passes under the RELEASE environment. The online warm boot is not the boot
# the shipped image performs: the baked image serves offline (HF_HUB_OFFLINE=1, model
# resolved to the snapshot path), and some kernels only materialize there; a fork pick
# can also flip between boots (each variant's source is stable, so the union converges).
# Re-boot offline against the accumulated cache until a boot compiles nothing new.
# The first boot above also wrote the execution-plan pack (warm/pack, keyed on the model
# CONFIG hash — not the id/path, so initial and fixpoint boots share it): these offline
# passes boot from it (frozen fork picks → no flip, fast boot), which both validates the
# pack-hit path pre-bake and makes the fixpoint converge on pass 1 whenever the pack took.
fixpoint() {  # $1 = label, $2 = shape spec ("" = the pinned shape)
    local label="$1" spec="${2:-}" pass before after new
    local extra; extra=$(shape_env "$spec")
    for pass in 1 2 3 4 5; do
        before=$(find warm/cubin -name '*.cubin' | sort)
        # shellcheck disable=SC2086 — $extra is a deliberately word-split flag list
        docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 \
            -e HF_HUB_OFFLINE=1 \
            -e HF_HOME=/opt/emmy/hf \
            -e EMMY_CUBIN_CACHE=/opt/emmy/cubin \
            -e EMMY_PACK_DIR=/opt/emmy/pack \
            -e TRITON_CACHE_DIR=/opt/emmy/triton \
            $extra \
            -e SERVE_MODEL -e SERVE_MAX_MODEL_LEN -e SERVE_GPU_MEM_UTIL \
            -e SERVE_REVISION -e SERVE_QUANT -e SERVE_CAPTURE_SIZES -e SERVE_EXTRA_ARGS \
            -v "$PWD/warm":/opt/emmy \
            -v "$PWD/serve.sh":/opt/emmy/serve.sh:ro \
            --entrypoint /opt/emmy/serve.sh \
            "$BASE_IMAGE" >/dev/null
        echo "[warm] $label: offline fixpoint pass $pass: waiting for /health..."
        for _ in $(seq 1 360); do
            if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
            if [ -z "$(docker ps -q -f name=$NAME)" ]; then
                echo "[warm] server died:"; docker logs --tail 50 "$NAME"; return 1
            fi
            sleep 10
        done
        curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[warm] $label pass $pass timed out"; docker logs --tail 50 "$NAME"; return 1; }
        served=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
        curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
            -d "{\"model\": \"$served\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" >/dev/null
        docker stop "$NAME" >/dev/null && docker rm "$NAME" >/dev/null
        after=$(find warm/cubin -name '*.cubin' | sort)
        new=$(comm -13 <(echo "$before") <(echo "$after") | wc -l)
        echo "[warm] $label: offline pass $pass added $new cubin(s)"
        [ "$new" -eq 0 ] && return 0
    done
    # A non-converged warm is a FAILURE, not a warning: fork picks are bimodal across boots, so
    # verify's single boot can happen to hit only cached variants and PASS while customer boots
    # recompile at runtime — exactly the failure class the fixpoint exists to contain, gated by a
    # coin flip if this exits 0.
    echo "[warm] FAIL: $label reached no fixpoint after 5 offline passes — the cubin set is still growing" >&2
    return 1
}

fixpoint "pinned" "" || exit 1

# Extra shapes: same fixpoint discipline, but a failure here does NOT sink the release — the
# pinned shape is the contract, and an extra shape that will not converge should degrade to
# "that lane boots cold" rather than block the image. It is reported loudly at the end.
failed_shapes=""
for spec in ${SERVE_WARM_SHAPES:-}; do
    echo "[warm] ---- extra shape $spec ----"
    fixpoint "shape $spec" "$spec" || failed_shapes="$failed_shapes $spec"
done

# The container wrote warm/ as root; the bake's split_hf.sh hardlinks these files as the
# invoking user, which fs.protected_hardlinks forbids on root-owned files. Reclaim ownership
# with a throwaway container (no sudo needed on the host).
docker run --rm -v "$PWD/warm":/w alpine chown -R "$(id -u):$(id -g)" /w

packs=$(find warm/pack -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[warm] done: $(find warm/cubin -name '*.cubin' | wc -l) cubin(s), $packs pack(s) / $(find warm/pack -name '*.json' | wc -l) plan file(s), snapshot at warm/hf"
[ -n "$failed_shapes" ] && echo "[warm] WARNING: extra shape(s) did not converge and will boot cold:$failed_shapes" >&2
exit 0
