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
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"  # Default to 1 if not specified
NUM_INSTANCES="${NUM_INSTANCES:-1}"  # Default to 1 instance
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

# Clean up any existing containers
echo "Cleaning up any existing containers..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    sudo docker rm -f ${CONTAINER_NAME}_${i} 2>/dev/null || true
done

# Calculate GPU assignments for each instance
# If TENSOR_PARALLEL_SIZE=2 and NUM_INSTANCES=2, we get:
#   Instance 0: GPUs 0,1
#   Instance 1: GPUs 2,3
GPUS_PER_INSTANCE=$TENSOR_PARALLEL_SIZE
TOTAL_GPUS=$((GPUS_PER_INSTANCE * NUM_INSTANCES))

echo "" | tee -a $BENCHMARK_RESULTS_FILE
echo "Configuration:" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Instances: $NUM_INSTANCES" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE" | tee -a $BENCHMARK_RESULTS_FILE
echo "  GPUs per instance: $GPUS_PER_INSTANCE" | tee -a $BENCHMARK_RESULTS_FILE
echo "  Total GPUs used: $TOTAL_GPUS" | tee -a $BENCHMARK_RESULTS_FILE
echo "" | tee -a $BENCHMARK_RESULTS_FILE

# Start multiple vLLM instances
echo "Starting $NUM_INSTANCES vLLM instance(s)..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    # Calculate GPU subset for this instance
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
    CONTAINER="${CONTAINER_NAME}_${i}"

    echo "  Starting instance $i on GPU(s) $GPU_LIST, port $PORT..."

    sudo -E docker run --gpus "\"device=$GPU_LIST\"" \
        -d \
        --name $CONTAINER \
        -v $HF_DIRECTORY:$HF_DIRECTORY \
        --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
        --env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
        --env "OMP_NUM_THREADS=16" \
        --env "CUDA_VISIBLE_DEVICES=$GPU_LIST" \
        -p ${PORT}:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --disable-log-requests --trust-remote-code \
        --max-model-len=8192 --gpu-memory-utilization=0.90 \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --model $MODEL_PATH \
        --served-model-name $MODEL_NAME
done

# Wait for all instances to be ready
echo "Waiting for all vLLM instances to start and load models..."

MAX_WAIT_TIME=1800  # 30 minutes
CHECK_INTERVAL=10   # 10 seconds
MAX_CHECKS=$((MAX_WAIT_TIME / CHECK_INTERVAL))

for i in $(seq 1 $MAX_CHECKS); do
    sleep $CHECK_INTERVAL

    ALL_READY=true
    for inst in $(seq 0 $((NUM_INSTANCES - 1))); do
        CONTAINER="${CONTAINER_NAME}_${inst}"
        if ! sudo -E docker logs $CONTAINER 2>&1 | grep -q "$READY_STRING"; then
            ALL_READY=false
            break
        fi
    done

    if [ "$ALL_READY" = true ]; then
        echo "All vLLM instances are ready!"
        break
    fi

    elapsed=$((i * CHECK_INTERVAL))
    echo "Waiting... (${elapsed}s / ${MAX_WAIT_TIME}s)"
done

# Exit if timeout exceeded
if [ $i -eq $MAX_CHECKS ]; then
    echo "❌ Timeout: Instances did not become ready in ${MAX_WAIT_TIME} seconds."
    for inst in $(seq 0 $((NUM_INSTANCES - 1))); do
        CONTAINER="${CONTAINER_NAME}_${inst}"
        echo "Last 20 lines of instance $inst logs:"
        sudo docker logs $CONTAINER 2>&1 | tail -20
    done
    exit 1
fi

# Install benchmark dependencies in all containers
echo "Installing benchmark dependencies..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    CONTAINER="${CONTAINER_NAME}_${i}"
    sudo -E docker exec $CONTAINER pip install pandas datasets >/dev/null 2>&1
done

# Run benchmarks against all instances in parallel
echo "" | tee -a $BENCHMARK_RESULTS_FILE
echo "Running multi-instance benchmark..." | tee -a $BENCHMARK_RESULTS_FILE
echo "---------------------------------" | tee -a $BENCHMARK_RESULTS_FILE

# Split prompts across instances
PROMPTS_PER_INSTANCE=$((NUM_PROMPTS / NUM_INSTANCES))

# Create temp directory for individual results
TEMP_DIR=$(mktemp -d)

# Launch benchmarks in parallel
PIDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((8000 + i))
    CONTAINER="${CONTAINER_NAME}_${i}"
    TEMP_RESULT="${TEMP_DIR}/result_${i}.txt"

    BENCHMARK_CMD="vllm bench serve --model $MODEL_NAME --dataset-name random --random-input-len $RANDOM_INPUT_LEN --random-output-len $RANDOM_OUTPUT_LEN --max-concurrency $MAX_CONCURRENCY --num-prompts $PROMPTS_PER_INSTANCE --ignore-eos --backend openai-chat --endpoint /v1/chat/completions --percentile-metrics ttft,tpot,itl,e2el --base-url http://localhost:8000"

    echo "  Launching benchmark for instance $i (port $PORT, $PROMPTS_PER_INSTANCE prompts)..."
    (sudo -E docker exec $CONTAINER bash -c "$BENCHMARK_CMD" > "$TEMP_RESULT" 2>&1) &
    PIDS+=($!)
done

# Wait for all benchmarks to complete
echo "Waiting for all benchmarks to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All benchmarks completed. Aggregating results..."

# Parse and aggregate results from all instances
# We'll extract key metrics and compute totals/averages

python3 - <<EOF
import re
import sys

results = []
for i in range($NUM_INSTANCES):
    result_file = f"$TEMP_DIR/result_{i}.txt"
    try:
        with open(result_file, 'r') as f:
            content = f.read()

            # Extract metrics using regex
            metrics = {}
            metrics['successful_requests'] = int(re.search(r'Successful requests:\s+(\d+)', content).group(1))
            metrics['benchmark_duration'] = float(re.search(r'Benchmark duration \(s\):\s+([\d.]+)', content).group(1))
            metrics['total_input_tokens'] = int(re.search(r'Total input tokens:\s+(\d+)', content).group(1))
            metrics['total_generated_tokens'] = int(re.search(r'Total generated tokens:\s+(\d+)', content).group(1))
            metrics['output_throughput'] = float(re.search(r'Output token throughput \(tok/s\):\s+([\d.]+)', content).group(1))
            metrics['total_throughput'] = float(re.search(r'Total Token throughput \(tok/s\):\s+([\d.]+)', content).group(1))
            metrics['mean_ttft'] = float(re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['median_ttft'] = float(re.search(r'Median TTFT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['mean_tpot'] = float(re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['median_tpot'] = float(re.search(r'Median TPOT \(ms\):\s+([\d.]+)', content).group(1))

            results.append(metrics)
    except Exception as e:
        print(f"Error parsing result {i}: {e}", file=sys.stderr)
        sys.exit(1)

# Aggregate metrics
total_requests = sum(r['successful_requests'] for r in results)
max_duration = max(r['benchmark_duration'] for r in results)
total_input_tokens = sum(r['total_input_tokens'] for r in results)
total_output_tokens = sum(r['total_generated_tokens'] for r in results)
aggregate_output_throughput = sum(r['output_throughput'] for r in results)
aggregate_total_throughput = sum(r['total_throughput'] for r in results)
avg_mean_ttft = sum(r['mean_ttft'] for r in results) / len(results)
avg_median_ttft = sum(r['median_ttft'] for r in results) / len(results)
avg_mean_tpot = sum(r['mean_tpot'] for r in results) / len(results)
avg_median_tpot = sum(r['median_tpot'] for r in results) / len(results)

# Print aggregated results
print("============ Multi-Instance Benchmark Result ============")
print(f"Number of instances:                 {$NUM_INSTANCES}")
print(f"Successful requests (total):         {total_requests}")
print(f"Benchmark duration (s):              {max_duration:.2f}")
print(f"Total input tokens:                  {total_input_tokens}")
print(f"Total generated tokens:              {total_output_tokens}")
print(f"Aggregate output throughput (tok/s): {aggregate_output_throughput:.2f}")
print(f"Aggregate total throughput (tok/s):  {aggregate_total_throughput:.2f}")
print(f"Average Mean TTFT (ms):              {avg_mean_ttft:.2f}")
print(f"Average Median TTFT (ms):            {avg_median_ttft:.2f}")
print(f"Average Mean TPOT (ms):              {avg_mean_tpot:.2f}")
print(f"Average Median TPOT (ms):            {avg_median_tpot:.2f}")
print("==========================================================")

print("\nPer-Instance Results:")
for i, r in enumerate(results):
    print(f"\nInstance {i}:")
    print(f"  Output throughput: {r['output_throughput']:.2f} tok/s")
    print(f"  Mean TTFT: {r['mean_ttft']:.2f} ms")
    print(f"  Mean TPOT: {r['mean_tpot']:.2f} ms")
EOF

# Save aggregated results
python3 - >> $BENCHMARK_RESULTS_FILE <<EOF
import re
import sys

results = []
for i in range($NUM_INSTANCES):
    result_file = f"$TEMP_DIR/result_{i}.txt"
    try:
        with open(result_file, 'r') as f:
            content = f.read()

            metrics = {}
            metrics['successful_requests'] = int(re.search(r'Successful requests:\s+(\d+)', content).group(1))
            metrics['benchmark_duration'] = float(re.search(r'Benchmark duration \(s\):\s+([\d.]+)', content).group(1))
            metrics['total_input_tokens'] = int(re.search(r'Total input tokens:\s+(\d+)', content).group(1))
            metrics['total_generated_tokens'] = int(re.search(r'Total generated tokens:\s+(\d+)', content).group(1))
            metrics['output_throughput'] = float(re.search(r'Output token throughput \(tok/s\):\s+([\d.]+)', content).group(1))
            metrics['total_throughput'] = float(re.search(r'Total Token throughput \(tok/s\):\s+([\d.]+)', content).group(1))
            metrics['mean_ttft'] = float(re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['median_ttft'] = float(re.search(r'Median TTFT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['mean_tpot'] = float(re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', content).group(1))
            metrics['median_tpot'] = float(re.search(r'Median TPOT \(ms\):\s+([\d.]+)', content).group(1))

            results.append(metrics)
    except Exception as e:
        print(f"Error parsing result {i}: {e}", file=sys.stderr)

total_requests = sum(r['successful_requests'] for r in results)
max_duration = max(r['benchmark_duration'] for r in results)
total_input_tokens = sum(r['total_input_tokens'] for r in results)
total_output_tokens = sum(r['total_generated_tokens'] for r in results)
aggregate_output_throughput = sum(r['output_throughput'] for r in results)
aggregate_total_throughput = sum(r['total_throughput'] for r in results)
avg_mean_ttft = sum(r['mean_ttft'] for r in results) / len(results)
avg_median_ttft = sum(r['median_ttft'] for r in results) / len(results)
avg_mean_tpot = sum(r['mean_tpot'] for r in results) / len(results)
avg_median_tpot = sum(r['median_tpot'] for r in results) / len(results)

print("============ Multi-Instance Benchmark Result ============")
print(f"Number of instances:                 {$NUM_INSTANCES}")
print(f"Successful requests (total):         {total_requests}")
print(f"Benchmark duration (s):              {max_duration:.2f}")
print(f"Total input tokens:                  {total_input_tokens}")
print(f"Total generated tokens:              {total_output_tokens}")
print(f"Aggregate output throughput (tok/s): {aggregate_output_throughput:.2f}")
print(f"Aggregate total throughput (tok/s):  {aggregate_total_throughput:.2f}")
print(f"Average Mean TTFT (ms):              {avg_mean_ttft:.2f}")
print(f"Average Median TTFT (ms):            {avg_median_ttft:.2f}")
print(f"Average Mean TPOT (ms):              {avg_mean_tpot:.2f}")
print(f"Average Median TPOT (ms):            {avg_median_tpot:.2f}")
print("==========================================================")

print("\nPer-Instance Results:")
for i, r in enumerate(results):
    print(f"\nInstance {i}:")
    print(f"  Output throughput: {r['output_throughput']:.2f} tok/s")
    print(f"  Mean TTFT: {r['mean_ttft']:.2f} ms")
    print(f"  Mean TPOT: {r['mean_tpot']:.2f} ms")
EOF

# Clean up temp directory
rm -rf $TEMP_DIR

# Stop all containers
echo ""
echo "Stopping vLLM containers..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    sudo docker stop ${CONTAINER_NAME}_${i} >/dev/null 2>&1
done

echo ""
echo "✅ Multi-instance benchmark completed."
echo "Results:"
cat $BENCHMARK_RESULTS_FILE
