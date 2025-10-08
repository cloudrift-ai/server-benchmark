#!/bin/bash

set -o allexport
# Configuration - using environment variables with defaults
IMAGE_NAME="${IMAGE_NAME:-vllm/vllm-openai:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_benchmark_container}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-200}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

BENCHMARK_RESULTS_FILE="${BENCHMARK_RESULTS_FILE:-vllm_benchmark.txt}"
HF_DIRECTORY="${HF_DIRECTORY:-/hf_models}"
# Disable automatic export
set +o allexport
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Ensure user can run docker without sudo
if ! docker ps >/dev/null 2>&1; then
    echo "Adding user to docker group..."
    sudo usermod -aG docker $USER
    echo "⚠️  You've been added to the docker group. Re-running with newgrp..."
    exec sg docker "$0 $@"
fi

# Model was already downloaded in 2_hf_download.sh
MODEL_PATH="$HF_DIRECTORY/$MODEL_NAME"

# Calculate GPU assignments
GPUS_PER_INSTANCE=$TENSOR_PARALLEL_SIZE
TOTAL_GPUS=$((GPUS_PER_INSTANCE * NUM_INSTANCES))

echo "Config: ${NUM_INSTANCES}x instances, tensor_parallel=${TENSOR_PARALLEL_SIZE}, total_gpus=${TOTAL_GPUS}"

# Generate docker-compose configuration
COMPOSE_FILE="docker-compose.vllm.yml"
NGINX_CONF="nginx.vllm.conf"
if [ $NUM_INSTANCES -gt 1 ]; then
    # Multi-instance with nginx
    ./venv/bin/python utils/generate_compose.py \
        --num-instances $NUM_INSTANCES \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --container-name $CONTAINER_NAME \
        --model-path $MODEL_PATH \
        --model-name "$MODEL_NAME" \
        --hf-directory $HF_DIRECTORY \
        --hf-token "${HUGGING_FACE_HUB_TOKEN:-}" \
        --extra-args "$VLLM_EXTRA_ARGS" \
        --output $COMPOSE_FILE \
        --nginx-conf-output $NGINX_CONF
else
    # Single instance
    ./venv/bin/python utils/generate_compose.py \
        --num-instances $NUM_INSTANCES \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --container-name $CONTAINER_NAME \
        --model-path $MODEL_PATH \
        --model-name "$MODEL_NAME" \
        --hf-directory $HF_DIRECTORY \
        --hf-token "${HUGGING_FACE_HUB_TOKEN:-}" \
        --extra-args "$VLLM_EXTRA_ARGS" \
        --output $COMPOSE_FILE
fi

# Clean up previous deployment
echo "Cleaning up previous deployment..."
docker compose -f $COMPOSE_FILE -p vllm_benchmark down 2>/dev/null || true

# Start all services and wait for health
echo "Starting containers (may take up to 30min for multi-GPU)..."

# Start containers and wait for health in background
docker compose -f $COMPOSE_FILE -p vllm_benchmark up -d --wait --wait-timeout 1800 &
UP_PID=$!

# Give containers a moment to be created, then start streaming logs
sleep 2
docker compose -f $COMPOSE_FILE -p vllm_benchmark logs -f &
LOGS_PID=$!

# Wait for the up command to complete
if wait $UP_PID; then
    kill -KILL $LOGS_PID 2>/dev/null || true
    wait $LOGS_PID 2>/dev/null || true
    echo "✅ Containers healthy"
else
    kill -KILL $LOGS_PID 2>/dev/null || true
    wait $LOGS_PID 2>/dev/null || true
    echo "❌ Container startup failed or timeout"
    echo "Container status:"
    docker compose -f $COMPOSE_FILE -p vllm_benchmark ps
    docker compose -f $COMPOSE_FILE -p vllm_benchmark logs --tail=100
    docker compose -f $COMPOSE_FILE -p vllm_benchmark down
    rm -f $COMPOSE_FILE $NGINX_CONF
    exit 1
fi

# Start benchmark client container
echo "Starting benchmark client container..."
docker compose -f $COMPOSE_FILE -p vllm_benchmark --profile tools up -d benchmark

# Determine benchmark endpoint
if [ $NUM_INSTANCES -gt 1 ]; then
    BENCHMARK_ENDPOINT="http://nginx_lb:8080"
else
    BENCHMARK_ENDPOINT="http://vllm_0:8000"
fi

# Run benchmark
BENCHMARK_CMD="vllm bench serve \
    --model $MODEL_NAME \
    --dataset-name random \
    --random-input-len $RANDOM_INPUT_LEN \
    --random-output-len $RANDOM_OUTPUT_LEN \
    --max-concurrency $MAX_CONCURRENCY \
    --num-prompts $NUM_PROMPTS \
    --ignore-eos \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --percentile-metrics ttft,tpot,itl,e2el \
    --base-url $BENCHMARK_ENDPOINT"

echo "Running benchmark (endpoint: $BENCHMARK_ENDPOINT)..."
# Stream all output to console and temp file, then filter for results file
TEMP_OUTPUT=$(mktemp)
docker exec vllm_benchmark_client bash -c "$BENCHMARK_CMD" 2>&1 | tee "$TEMP_OUTPUT"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Benchmark command failed"
    rm -f "$TEMP_OUTPUT"
    exit 1
fi

# Extract only the results section to the final results file
awk '/^============ Serving Benchmark Result ============$/ {found=1} found' "$TEMP_OUTPUT" > $BENCHMARK_RESULTS_FILE
rm -f "$TEMP_OUTPUT"

# Stop all services
echo "Stopping containers..."
docker compose -f $COMPOSE_FILE -p vllm_benchmark down

# Clean up temporary files
rm -f $COMPOSE_FILE $NGINX_CONF

echo "✅ Benchmark completed"
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
