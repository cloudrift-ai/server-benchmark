#!/bin/bash

set -e  # Exit on error

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv docker-compose-v2

echo "Creating virtual environment..."
python3 -m venv venv

echo "Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "Setup complete!"