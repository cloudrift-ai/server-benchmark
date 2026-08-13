#!/bin/bash
# The image-naming schema: an HF model id -> the docker-repo-safe slug used for both the
# published image name (`cloudriftai/vllm-emmy-<slug>`) and this directory's per-model
# config file (`models/<slug>.env`).
#
#   docker/vllm-emmy-serve/model_slug.sh google/gemma-4-12B-it   # -> gemma-4-12b-it
#
# ONE implementation on purpose: the Python library owns the rule so `emmy publish` and this
# Make/shell adapter cannot drift. A drifted slug means the warm and bake disagree about which
# config they read — the exact class of bug the cache-key parity contract exists to prevent.
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

[ "$#" -eq 1 ] || { echo "usage: model_slug.sh <hf-model-id>" >&2; exit 2; }

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
python=${PYTHON:-python3}

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" exec "$python" -m emmy.publish "$1"
