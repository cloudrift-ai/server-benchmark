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
BENCHMARK_CMD="vllm bench serve --model $MODEL_NAME --dataset-name random --random-input-len $RANDOM_INPUT_LEN --random-output-len $RANDOM_OUTPUT_LEN --max-concurrency $MAX_CONCURRENCY --num-prompts $NUM_PROMPTS --ignore-eos --backend openai-chat --endpoint /v1/chat/completions  --percentile-metrics ttft,tpot,itl,e2el"
READY_STRING="Application startup complete."
GPU_NUMBER=$( nvidia-smi --list-gpus | wc -l )

BENCHMARK_RESULTS_FILE="${BENCHMARK_RESULTS_FILE:-vllm_results.txt}"
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

# Clean up any existing container with the same name
echo "Cleaning up any existing containers..."
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Start the vLLM container in the background
echo "Starting vLLM container..."
sudo -E docker run --gpus all \
    -d \
    --name $CONTAINER_NAME \
    -v $HF_DIRECTORY:$HF_DIRECTORY \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    --env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    --env "OMP_NUM_THREADS=16" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --disable-log-requests --trust-remote-code \
    --max-model-len=8192 --gpu-memory-utilization=0.90 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size $GPU_NUMBER \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME


# Wait until model is loaded and server is ready
echo "Waiting for vLLM server to start and model to load..."

# Check every 10 seconds for up to 30 minutes (180 checks)
MAX_WAIT_TIME=1800  # 30 minutes
CHECK_INTERVAL=10   # 10 seconds
MAX_CHECKS=$((MAX_WAIT_TIME / CHECK_INTERVAL))

for i in $(seq 1 $MAX_CHECKS); do
    sleep $CHECK_INTERVAL
    if sudo -E docker logs $CONTAINER_NAME 2>&1 | grep -q "$READY_STRING"; then
        echo "vLLM server is ready!"
        break
    fi
    elapsed=$((i * CHECK_INTERVAL))
    echo "Waiting... (${elapsed}s / ${MAX_WAIT_TIME}s)"
done

# Exit if timeout exceeded
if [ $i -eq $MAX_CHECKS ]; then
    echo "❌ Timeout: Server did not become ready in ${MAX_WAIT_TIME} seconds."
    echo "Last 50 lines of container logs:"
    sudo docker logs $CONTAINER_NAME 2>&1 | tail -50
    exit 1
fi


# Run benchmark inside the container
echo "Running benchmark inside the container..."
sudo -E docker exec $CONTAINER_NAME pip install pandas datasets

echo "" >> $BENCHMARK_RESULTS_FILE
echo "vllm model gpu benchmark" >> $BENCHMARK_RESULTS_FILE
echo "---------------------------------" >> $BENCHMARK_RESULTS_FILE
sudo -E docker exec $CONTAINER_NAME bash -c "$BENCHMARK_CMD" | awk '/^============/ {found=1} found' | tee -a $BENCHMARK_RESULTS_FILE

sudo docker stop $CONTAINER_NAME
echo ""
echo ""
echo "✅ Benchmark completed."
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
