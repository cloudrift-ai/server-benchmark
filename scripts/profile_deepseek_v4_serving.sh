#!/usr/bin/env bash
set -euo pipefail

# Capture a mixed-concurrency trace for PP2 utilization and kernel attribution.
# The profiler-enabled server container must mount ARTIFACT_DIR at /artifacts.

CONTAINER=${CONTAINER:-onecat-dsv4-profile-eager}
ARTIFACT_DIR=${ARTIFACT_DIR:-/home/riftuser/onecat-dsv4-0731/optimization/profile-eager}
RESULT_PREFIX=${RESULT_PREFIX:-profile-eager}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
TOKENIZER=/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062

if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
else
    docker_cmd=(sudo docker)
fi

mkdir -p "${ARTIFACT_DIR}"

profiling=0
stop_profile() {
    if [[ ${profiling} -eq 1 ]]; then
        curl --fail-with-body --silent --show-error --request POST "${BASE_URL}/stop_profile"
    fi
}
trap stop_profile EXIT

curl --fail-with-body --silent --show-error --request POST "${BASE_URL}/start_profile"
profiling=1

for concurrency in 1 2 4 8; do
    "${docker_cmd[@]}" exec "${CONTAINER}" vllm bench serve \
        --backend openai \
        --base-url http://127.0.0.1:8000 \
        --endpoint /v1/completions \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --tokenizer-mode deepseek_v4 \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 16 \
        --random-prefix-len 0 \
        --num-prompts "${concurrency}" \
        --max-concurrency "${concurrency}" \
        --request-rate inf \
        --temperature 0.0 \
        --ignore-eos \
        --seed 731 \
        --save-result \
        --save-detailed \
        --result-dir /artifacts \
        --result-filename "${RESULT_PREFIX}_p1024_o16_c${concurrency}.json"
done

stop_profile
profiling=0
trap - EXIT
