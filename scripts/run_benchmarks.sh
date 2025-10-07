#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Collect system information
echo "Collecting system information..."
$SCRIPT_DIR/collect_system_info.sh

# Create a marker file to indicate system info is ready
touch system_info_ready.marker

source $SCRIPT_DIR/run_vllm_benchmark.sh

# Create a marker file to indicate vLLM benchmark is ready
touch vllm_benchmark_ready.marker

# Run YABS benchmark
curl -sL https://yabs.sh | bash > yabs_results.txt 2>&1

# Create a marker file to indicate YABS is ready
touch yabs_ready.marker




