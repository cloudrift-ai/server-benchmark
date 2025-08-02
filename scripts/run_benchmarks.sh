#!/bin/bash

source ./$SCRIPT_DIR/run_vllm_benchmark.sh

# Run YABS benchmark
curl -sL https://yabs.sh | bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )



