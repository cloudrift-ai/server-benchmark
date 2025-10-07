#!/bin/bash

set -o allexport
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
HF_DIRECTORY="${HF_DIRECTORY:-/hf_models}"
set +o allexport

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Pre-downloading model from Hugging Face..."
echo "Model: $MODEL_NAME" > hf_download_results.txt
echo "---------------------------------" >> hf_download_results.txt

sudo -E ./venv/bin/python $SCRIPT_DIR/download_model.py --model-name $MODEL_NAME --hg-dir $HF_DIRECTORY/$MODEL_NAME | tee -a hf_download_results.txt

echo "Model download complete!"
