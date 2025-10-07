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

BENCHMARK_RESULTS_FILE="${BENCHMARK_RESULTS_FILE:-vllm_benchmark.txt}"
HF_DIRECTORY="${HF_DIRECTORY:-/hf_models}"
# Disable automatic export
set +o allexport
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Pre-download the model
echo "Pre-downloading model from Hugging Face..."
sudo -E ./venv/bin/python utils/download_model.py --model-name $MODEL_NAME --hg-dir $HF_DIRECTORY/$MODEL_NAME

MODEL_PATH="$HF_DIRECTORY/$MODEL_NAME"

# Calculate GPU assignments
GPUS_PER_INSTANCE=$TENSOR_PARALLEL_SIZE
TOTAL_GPUS=$((GPUS_PER_INSTANCE * NUM_INSTANCES))

echo ""
echo "Configuration:"
echo "  Instances: $NUM_INSTANCES"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  GPUs per instance: $GPUS_PER_INSTANCE"
echo "  Total GPUs used: $TOTAL_GPUS"
echo ""

# Generate docker-compose configuration
COMPOSE_FILE="docker-compose.vllm.yml"
NGINX_CONF="nginx.vllm.conf"

echo "Generating docker-compose configuration..."
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
        --output $COMPOSE_FILE
fi

echo ""
echo "Generated docker-compose.yml:"
cat $COMPOSE_FILE

if [ $NUM_INSTANCES -gt 1 ]; then
    echo ""
    echo "Generated nginx.conf:"
    cat $NGINX_CONF
fi

# Clean up previous deployment
echo ""
echo "Cleaning up previous deployment..."
sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark down 2>/dev/null || true

# Start all services
echo ""
echo "Starting services with docker compose..."
sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark up -d

# Wait for all vLLM instances to be ready
echo ""
echo "Waiting for all vLLM instances to become healthy..."

MAX_WAIT_TIME=1800  # 30 minutes
CHECK_INTERVAL=10   # 10 seconds
MAX_CHECKS=$((MAX_WAIT_TIME / CHECK_INTERVAL))

for check in $(seq 1 $MAX_CHECKS); do
    sleep $CHECK_INTERVAL

    ALL_HEALTHY=true
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        CONTAINER="${CONTAINER_NAME}_${i}"
        HEALTH=$(sudo docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null || echo "unknown")
        if [ "$HEALTH" != "healthy" ]; then
            ALL_HEALTHY=false
            break
        fi
    done

    if [ "$ALL_HEALTHY" = true ]; then
        echo "All vLLM instances are healthy!"
        break
    fi

    elapsed=$((check * CHECK_INTERVAL))
    echo "Waiting... (${elapsed}s / ${MAX_WAIT_TIME}s)"
done

# Exit if timeout exceeded
if [ $check -eq $MAX_CHECKS ]; then
    echo "❌ Timeout: Instances did not become healthy in ${MAX_WAIT_TIME} seconds."
    echo "Container status:"
    sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark ps
    echo ""
    echo "Logs from containers:"
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
    echo ""
    echo "Using nginx load balancer at $BENCHMARK_ENDPOINT"
else
    BENCHMARK_ENDPOINT="http://localhost:8000"
fi

# Run benchmark
echo ""
echo "Running benchmark against: $BENCHMARK_ENDPOINT"
echo "---------------------------------"

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

echo "Running benchmark..."
sudo -E docker exec ${CONTAINER_NAME}_0 bash -c "$BENCHMARK_CMD" | awk '/^============/ {found=1} found' > $BENCHMARK_RESULTS_FILE

# Stop all services
echo ""
echo "Stopping all services..."
sudo -E docker compose -f $COMPOSE_FILE -p vllm_benchmark down

# Clean up temporary files
rm -f $COMPOSE_FILE $NGINX_CONF

echo ""
echo "✅ Benchmark completed."
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
