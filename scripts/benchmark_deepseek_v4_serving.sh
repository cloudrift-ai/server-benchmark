#!/usr/bin/env bash
set -euo pipefail

# Benchmark the qualified DeepSeek V4 Flash 0731 serving matrix.
# The server container must mount ARTIFACT_DIR at /artifacts.

CONTAINER=${CONTAINER:-onecat-dsv4-zerojit}
ARTIFACT_DIR=${ARTIFACT_DIR:-/home/riftuser/onecat-dsv4-0731/optimization/bench-eager}
RESULT_PREFIX=${RESULT_PREFIX:-eager}
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
TOKENIZER=/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062

if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
else
    docker_cmd=(sudo docker)
fi

mkdir -p "${ARTIFACT_DIR}"

for shape in 32:8 256:32 1024:64 3072:128; do
    IFS=: read -r input_len output_len <<< "${shape}"
    for concurrency in 1 2 4 8; do
        "${docker_cmd[@]}" exec "${CONTAINER}" vllm bench serve \
            --backend openai \
            --base-url http://127.0.0.1:8000 \
            --endpoint /v1/completions \
            --model "${MODEL}" \
            --tokenizer "${TOKENIZER}" \
            --tokenizer-mode deepseek_v4 \
            --dataset-name random \
            --random-input-len "${input_len}" \
            --random-output-len "${output_len}" \
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
            --result-filename "${RESULT_PREFIX}_p${input_len}_o${output_len}_c${concurrency}.json"
    done
done
