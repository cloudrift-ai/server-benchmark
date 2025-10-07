#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Collect system information
echo "Collecting system information..."
$SCRIPT_DIR/collect_system_info.sh

source $SCRIPT_DIR/run_vllm_benchmark.sh

# Run YABS benchmark
curl -sL https://yabs.sh | bash




