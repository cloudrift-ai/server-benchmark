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
    sudo -E ./venv/bin/python utils/generate_compose.py \
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
    sudo -E ./venv/bin/python utils/generate_compose.py \
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
sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark down 2>/dev/null || true

# Start all services and wait for health checks
echo "Starting containers (may take up to 30min for multi-GPU)..."

if sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark up -d --wait --wait-timeout 1800; then
    echo "✅ Containers healthy"
else
    echo "❌ Container startup failed or timeout"
    echo "Container status:"
    sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark ps
    echo "Container logs:"
    sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark logs
    sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark down
    rm -f $COMPOSE_FILE $NGINX_CONF
    exit 1
fi

# Install benchmark dependencies in first container
echo "Installing benchmark dependencies..."
sudo -E docker exec ${CONTAINER_NAME}_0 pip install pandas datasets >/dev/null 2>&1

# Determine benchmark endpoint
if [ $NUM_INSTANCES -gt 1 ]; then
    BENCHMARK_ENDPOINT="http://localhost:8080"
else
    BENCHMARK_ENDPOINT="http://localhost:8000"
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
sudo -E docker exec ${CONTAINER_NAME}_0 bash -c "$BENCHMARK_CMD" | awk '/^============/ {found=1} found' > $BENCHMARK_RESULTS_FILE

# Stop all services
echo "Stopping containers..."
sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark down

# Clean up temporary files
rm -f $COMPOSE_FILE $NGINX_CONF

echo "✅ Benchmark completed"
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
