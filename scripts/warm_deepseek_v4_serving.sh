#!/usr/bin/env bash
set -euo pipefail

# Exercise the serving shapes used to warm the DeepSeek V4 Flash 0731 Triton cache.
# The server container must mount ARTIFACT_DIR at /artifacts.

CONTAINER=${CONTAINER:-onecat-dsv4-warm-eager}
ARTIFACT_DIR=${ARTIFACT_DIR:-/home/riftuser/onecat-dsv4-0731/optimization/eager-warm}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
TOKENIZER=/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062

if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
else
    docker_cmd=(sudo docker)
fi

mkdir -p "${ARTIFACT_DIR}"

for spec in 32:8:1 256:32:2 1024:64:4 3072:128:8; do
    IFS=: read -r input_len output_len concurrency <<< "${spec}"
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
        --ignore-eos \
        --seed 731 \
        --save-result \
        --save-detailed \
        --result-dir /artifacts \
        --result-filename "warm_p${input_len}_o${output_len}_c${concurrency}.json"
done

curl --fail-with-body --silent --show-error \
    --output "${ARTIFACT_DIR}/warm_tool_call.json" \
    --header "Content-Type: application/json" \
    --data-binary @- \
    "${BASE_URL}/v1/chat/completions" <<'JSON'
{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [
    {"role": "user", "content": "Use the multiply tool to calculate 17 times 19. Return only the tool call."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "description": "Multiply two integers.",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
          },
          "required": ["a", "b"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0.0,
  "max_tokens": 64
}
JSON
