#!/bin/bash

set -e
# Configuration
IMAGE_NAME="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm_benchmark_container"
MODEL_NAME="facebook/opt-125m"  # Replace with your model
BENCHMARK_CMD="python3 benchmarks/benchmark_serving.py --model $MODEL_NAME"
READY_STRING="Application startup complete."
HF_DIRECTORY=""

# Start the vLLM container in the background
echo "Starting vLLM container..."
docker run --runtime nvidia --gpus all \
    -d \
    --name $CONTAINER_NAME \
    -v $HF_DIRECTORY:$HF_DIRECTORY \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "$MODEL_NAME"


# Wait until model is loaded and server is ready
echo "Waiting for vLLM server to start and model to load..."

RETRIES=10
for i in $(seq 1 $RETRIES); do
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "$READY_STRING"; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting... ($i/$RETRIES)"
    sleep $((2 ** i))
done

# Exit if timeout exceeded
if [ $i -eq $RETRIES ]; then
    echo "❌ Timeout: Server did not become ready in time."
    docker logs $CONTAINER_NAME
    exit 1
fi

# Run benchmark inside the container
echo "Running benchmark inside the container..."
docker exec $CONTAINER_NAME $BENCHMARK_CMD

echo "✅ Benchmark completed."