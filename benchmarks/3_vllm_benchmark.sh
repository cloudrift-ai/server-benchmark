#!/bin/bash

set -o allexport
# Configuration - using required environment variables
# These must be set by the caller (run_remote_benchmark.py)
: "${IMAGE_NAME:?IMAGE_NAME must be set}"
: "${CONTAINER_NAME:?CONTAINER_NAME must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${RANDOM_INPUT_LEN:?RANDOM_INPUT_LEN must be set}"
: "${RANDOM_OUTPUT_LEN:?RANDOM_OUTPUT_LEN must be set}"
: "${MAX_CONCURRENCY:?MAX_CONCURRENCY must be set}"
: "${NUM_PROMPTS:?NUM_PROMPTS must be set}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE must be set}"
: "${NUM_INSTANCES:?NUM_INSTANCES must be set}"
: "${VLLM_EXTRA_ARGS:?VLLM_EXTRA_ARGS must be set}"
: "${BENCHMARK_RESULTS_FILE:?BENCHMARK_RESULTS_FILE must be set}"
: "${HF_DIRECTORY:?HF_DIRECTORY must be set}"
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
    COMPOSE_CMD=(
        ./venv/bin/python utils/generate_compose.py
        --num-instances $NUM_INSTANCES
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE
        --container-name $CONTAINER_NAME
        --model-path $MODEL_PATH
        --model-name "$MODEL_NAME"
        --hf-directory $HF_DIRECTORY
        --hf-token "${HUGGING_FACE_HUB_TOKEN:-}"
    )
    [ -n "$VLLM_EXTRA_ARGS" ] && COMPOSE_CMD+=(--extra-args "$VLLM_EXTRA_ARGS")
    COMPOSE_CMD+=(
        --output $COMPOSE_FILE
        --nginx-conf-output $NGINX_CONF
    )
    "${COMPOSE_CMD[@]}"
else
    # Single instance
    COMPOSE_CMD=(
        ./venv/bin/python utils/generate_compose.py
        --num-instances $NUM_INSTANCES
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE
        --container-name $CONTAINER_NAME
        --model-path $MODEL_PATH
        --model-name "$MODEL_NAME"
        --hf-directory $HF_DIRECTORY
        --hf-token "${HUGGING_FACE_HUB_TOKEN:-}"
    )
    [ -n "$VLLM_EXTRA_ARGS" ] && COMPOSE_CMD+=(--extra-args "$VLLM_EXTRA_ARGS")
    COMPOSE_CMD+=(--output $COMPOSE_FILE)
    "${COMPOSE_CMD[@]}"
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
# Verify network exists and is stable
NETWORK_NAME="vllm_benchmark_default"
for i in {1..10}; do
    if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
        echo "Network $NETWORK_NAME verified"
        break
    fi
    echo "Waiting for network... (attempt $i/10)"
    sleep 1
done

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    echo "❌ Network $NETWORK_NAME not found after waiting"
    docker compose -f $COMPOSE_FILE -p vllm_benchmark down
    rm -f $COMPOSE_FILE $NGINX_CONF
    exit 1
fi

docker compose -f $COMPOSE_FILE -p vllm_benchmark --profile tools up -d benchmark
# Wait for benchmark client to be running
sleep 2

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

# Append docker-compose config and benchmark command to results file
echo "" >> $BENCHMARK_RESULTS_FILE
echo "============ Docker Compose Configuration ============" >> $BENCHMARK_RESULTS_FILE
cat $COMPOSE_FILE >> $BENCHMARK_RESULTS_FILE
echo "" >> $BENCHMARK_RESULTS_FILE
echo "============ Benchmark Command ============" >> $BENCHMARK_RESULTS_FILE
echo "$BENCHMARK_CMD" >> $BENCHMARK_RESULTS_FILE
echo "==================================================" >> $BENCHMARK_RESULTS_FILE

# Stop all services
echo "Stopping containers..."
docker compose -f $COMPOSE_FILE -p vllm_benchmark down

# Keep docker-compose files for inspection (don't delete)
echo "Docker compose files kept at: $COMPOSE_FILE"
[ -f "$NGINX_CONF" ] && echo "Nginx config kept at: $NGINX_CONF"

echo "✅ Benchmark completed"
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
