#!/bin/bash

echo "Running YABS benchmark..."
curl -sL https://yabs.sh | bash > yabs_results.txt 2>&1

echo "YABS benchmark complete!"
