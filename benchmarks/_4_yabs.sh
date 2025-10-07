#!/bin/bash

echo "Running YABS benchmark..."
curl -sL https://yabs.sh | bash > yabs.txt 2>&1

echo "YABS benchmark complete!"
