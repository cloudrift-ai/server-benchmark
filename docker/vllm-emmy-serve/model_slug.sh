#!/bin/bash
# The image-naming schema: an HF model id -> the docker-repo-safe slug used for both the
# published image name (`cloudriftai/vllm-emmy-<slug>`) and this directory's per-model
# config file (`models/<slug>.env`).
#
#   docker/vllm-emmy-serve/model_slug.sh google/gemma-4-12B-it   # -> gemma-4-12b-it
#
# ONE implementation on purpose: the Makefile expands it to build the tag, warm.sh and
# verify.sh use it to find the config, and tests exercise it through this script. A second
# copy in Python would drift, and a drifted slug means the warm and the bake disagree about
# which config they read — the exact class of bug the cache-key parity contract exists to
# prevent (see ARCHITECTURE.md).
#
# The rules, in order:
#   1. drop the HF org  — `google/gemma-4-12B-it` and `unsloth/gemma-4-12B-it` share a slug.
#      Deliberate: the slug names the MODEL, and two orgs' copies of one model warm to the
#      same kernels. If you ever need to tell them apart, that is a per-model config
#      decision, not a naming-schema one.
#   2. lowercase       — docker repository names may not contain uppercase.
#   3. keep [a-z0-9._-], replace every other run with a single `-`.
#   4. trim leading / trailing separators, which docker also rejects.
set -euo pipefail

model="${1:-}"
[ -n "$model" ] || { echo "usage: model_slug.sh <hf-model-id>" >&2; exit 2; }

slug=$(printf '%s' "${model##*/}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/[^a-z0-9._-]\{1,\}/-/g' -e 's/^[._-]\{1,\}//' -e 's/[._-]\{1,\}$//')

# An id that sanitizes to nothing (or to a lone separator) would silently produce the tag
# `cloudriftai/vllm-emmy-:0.23.0-abc1234`, which docker rejects far downstream. Fail here.
[ -n "$slug" ] || { echo "model_slug.sh: '$model' sanitizes to an empty slug" >&2; exit 1; }

printf '%s\n' "$slug"
