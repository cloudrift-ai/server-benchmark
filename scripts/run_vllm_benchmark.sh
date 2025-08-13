#!/bin/bash

set -o allexport
# Configuration
IMAGE_NAME="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm_benchmark_container"
#MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" # Replace with your model
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct" #"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
BENCHMARK_CMD="python3 benchmarks/benchmark_serving.py --model $MODEL_NAME --dataset-name random --random-input-len 1000 --random-output-len 1000 --max-concurrency 200 --num-prompts 1000 --ignore-eos --backend openai-chat --endpoint /v1/chat/completions  --percentile_metrics ttft,tpot,itl,e2el"
READY_STRING="Application startup complete."
GPU_NUMBER=$( nvidia-smi --list-gpus | wc -l )

BENCHMARK_RESULTS_FILE="benchmark_results.txt"
source ./.env
HF_DIRECTORY="/hf_models"
# Disable automatic export
set +o allexport
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Hugging face download speed test" > $BENCHMARK_RESULTS_FILE
echo "---------------------------------" >> $BENCHMARK_RESULTS_FILE
python $SCRIPT_DIR/download_model.py --model-name $MODEL_NAME --hg-dir $HF_DIRECTORY/$MODEL_NAME | tee -a $BENCHMARK_RESULTS_FILE

# Start the vLLM container in the background
echo "Starting vLLM container..."
docker run --rm --gpus all \
    -d \
    --name $CONTAINER_NAME \
    -v $HF_DIRECTORY:$HF_DIRECTORY \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    --env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    --env "OMP_NUM_THREADS=16" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --disable-log-requests --no-enable-chunked-prefill --trust-remote-code \
    --max-seq-len-to-capture=8192 \
    --max-model-len=8192 --gpu-memory-utilization=0.90\
    --host 0.0.0.0 --port 8000 \
    -tp $GPU_NUMBER \
    --model $HF_DIRECTORY/$MODEL_NAME \
    --served-model-name $MODEL_NAME


# Wait until model is loaded and server is ready
echo "Waiting for vLLM server to start and model to load..."

RETRIES=11
for i in $(seq 1 $RETRIES); do
    sleep $((2 ** i))
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "$READY_STRING"; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting... ($i/$RETRIES)"
done

# Exit if timeout exceeded
if [ $i -eq $RETRIES ]; then
    echo "❌ Timeout: Server did not become ready in time."
    docker logs $CONTAINER_NAME
    exit 1
fi


# Run benchmark inside the container
echo "Running benchmark inside the container..."
docker exec $CONTAINER_NAME pip install pandas datasets

echo "" >> $BENCHMARK_RESULTS_FILE
echo "vllm model gpu benchmark" >> $BENCHMARK_RESULTS_FILE
echo "---------------------------------" >> $BENCHMARK_RESULTS_FILE
docker exec $CONTAINER_NAME $BENCHMARK_CMD | awk '/^============/ {found=1} found' | tee -a $BENCHMARK_RESULTS_FILE

docker stop $CONTAINER_NAME
echo ""
echo ""
echo "✅ Benchmark completed."
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
