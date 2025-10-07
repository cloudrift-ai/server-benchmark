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
READY_STRING="Application startup complete."

BENCHMARK_RESULTS_FILE="${BENCHMARK_RESULTS_FILE:-vllm_benchmark.txt}"
HF_DIRECTORY="${HF_DIRECTORY:-/hf_models}"
# Disable automatic export
set +o allexport
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Model: $MODEL_NAME" > $BENCHMARK_RESULTS_FILE
echo "---------------------------------" >> $BENCHMARK_RESULTS_FILE

# Pre-download the model
echo "Pre-downloading model from Hugging Face..." | tee -a $BENCHMARK_RESULTS_FILE
sudo -E ./venv/bin/python $SCRIPT_DIR/download_model.py --model-name $MODEL_NAME --hg-dir $HF_DIRECTORY/$MODEL_NAME | tee -a $BENCHMARK_RESULTS_FILE

MODEL_PATH="$HF_DIRECTORY/$MODEL_NAME"

# Calculate GPU assignments
GPUS_PER_INSTANCE=$TENSOR_PARALLEL_SIZE
TOTAL_GPUS=$((GPUS_PER_INSTANCE * NUM_INSTANCES))

echo "" | tee -a $BENCHMARK_RESULTS_FILE
echo "Configuration:" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Instances: $NUM_INSTANCES" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE" | tee -a $BENCHMARK_RESULTS_FILE
echo "  GPUs per instance: $GPUS_PER_INSTANCE" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Total GPUs used: $TOTAL_GPUS" | tee -a $BENCHMARK_RESULTS_FILE
echo "" | tee -a $BENCHMARK_RESULTS_FILE

# Generate docker-compose.yml
COMPOSE_FILE=$(mktemp --suffix=.yml)
echo "Generating docker-compose configuration..."

cat > $COMPOSE_FILE <<'COMPOSE_HEADER'
version: '3.8'

services:
COMPOSE_HEADER

# Add vLLM service instances
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    GPU_START=$((i * GPUS_PER_INSTANCE))
    GPU_LIST=""
    for g in $(seq $GPU_START $((GPU_START + GPUS_PER_INSTANCE - 1))); do
        if [ -z "$GPU_LIST" ]; then
            GPU_LIST="$g"
        else
            GPU_LIST="$GPU_LIST,$g"
        fi
    done

    PORT=$((8000 + i))

    cat >> $COMPOSE_FILE <<EOF
  vllm_$i:
    image: vllm/vllm-openai:latest
    container_name: ${CONTAINER_NAME}_${i}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['$GPU_LIST']
              capabilities: [gpu]
    volumes:
      - $HF_DIRECTORY:$HF_DIRECTORY
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - OMP_NUM_THREADS=16
      - CUDA_VISIBLE_DEVICES=$GPU_LIST
    ports:
      - "$PORT:8000"
    shm_size: '16gb'
    ipc: host
    command: >
      --disable-log-requests
      --trust-remote-code
      --max-model-len=8192
      --gpu-memory-utilization=0.90
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE
      --model $MODEL_PATH
      --served-model-name $MODEL_NAME
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 300s

EOF
done

# Add nginx load balancer if multiple instances
if [ $NUM_INSTANCES -gt 1 ]; then
    # Create nginx config
    NGINX_CONF=$(mktemp)
    cat > $NGINX_CONF <<'NGINX_HEADER'
events {
    worker_connections 4096;
}

http {
    upstream vllm_backend {
NGINX_HEADER

    # Add upstream servers
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        echo "        server vllm_$i:8000;" >> $NGINX_CONF
    done

    cat >> $NGINX_CONF <<'NGINX_FOOTER'
    }

    server {
        listen 8080;

        location / {
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Increase timeouts for long-running LLM requests
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;

            # Disable buffering for streaming responses
            proxy_buffering off;
        }
    }
}
NGINX_FOOTER

    # Add nginx service to compose file
    cat >> $COMPOSE_FILE <<EOF
  nginx:
    image: nginx:alpine
    container_name: nginx_lb
    ports:
      - "8080:8080"
    volumes:
      - $NGINX_CONF:/etc/nginx/nginx.conf:ro
    depends_on:
EOF

    # Add dependencies on all vLLM instances
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        cat >> $COMPOSE_FILE <<EOF
      - vllm_$i
EOF
    done

    # Clean up nginx config on exit
    trap "rm -f $NGINX_CONF" EXIT
fi

echo "Docker Compose configuration:"
cat $COMPOSE_FILE

# Clean up previous deployment
echo ""
echo "Cleaning up previous deployment..."
sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark down 2>/dev/null || true

# Start all services
echo ""
echo "Starting services with docker-compose..."
sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark up -d

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
    sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark ps
    echo ""
    echo "Logs from containers:"
    sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark logs
    sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark down
    rm -f $COMPOSE_FILE
    exit 1
fi

# Install benchmark dependencies in first container
echo "Installing benchmark dependencies..."
sudo -E docker exec ${CONTAINER_NAME}_0 pip install pandas datasets >/dev/null 2>&1

# Determine benchmark endpoint
if [ $NUM_INSTANCES -gt 1 ]; then
    BENCHMARK_ENDPOINT="http://localhost:8080"
    echo "" | tee -a $BENCHMARK_RESULTS_FILE
    echo "Using nginx load balancer at $BENCHMARK_ENDPOINT" | tee -a $BENCHMARK_RESULTS_FILE
else
    BENCHMARK_ENDPOINT="http://localhost:8000"
fi

# Run benchmark
echo "" | tee -a $BENCHMARK_RESULTS_FILE
echo "Running benchmark against: $BENCHMARK_ENDPOINT" | tee -a $BENCHMARK_RESULTS_FILE
echo "---------------------------------" | tee -a $BENCHMARK_RESULTS_FILE

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

echo "Running benchmark..." | tee -a $BENCHMARK_RESULTS_FILE
sudo -E docker exec ${CONTAINER_NAME}_0 bash -c "$BENCHMARK_CMD" | awk '/^============/ {found=1} found' | tee -a $BENCHMARK_RESULTS_FILE

# Stop all services
echo ""
echo "Stopping all services..."
sudo -E docker-compose -f $COMPOSE_FILE -p vllm_benchmark down

# Clean up compose file
rm -f $COMPOSE_FILE

echo ""
echo "✅ Benchmark completed."
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
