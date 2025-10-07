#!/bin/bash

# Script to collect system information: CPU, GPU, RAM, Disk, etc.
# Output is saved to system_info.txt

OUTPUT_FILE="system_info.txt"

echo "Collecting system information..."

# Clear or create output file
> $OUTPUT_FILE

echo "================================" | tee -a $OUTPUT_FILE
echo "SYSTEM INFORMATION" | tee -a $OUTPUT_FILE
echo "Collected: $(date)" | tee -a $OUTPUT_FILE
echo "================================" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Hostname
echo "=== HOSTNAME ===" | tee -a $OUTPUT_FILE
hostname | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# OS Information
echo "=== OPERATING SYSTEM ===" | tee -a $OUTPUT_FILE
if [ -f /etc/os-release ]; then
    cat /etc/os-release | tee -a $OUTPUT_FILE
else
    uname -a | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# Kernel Version
echo "=== KERNEL VERSION ===" | tee -a $OUTPUT_FILE
uname -r | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# CPU Information
echo "=== CPU INFORMATION ===" | tee -a $OUTPUT_FILE
if command -v lscpu &> /dev/null; then
    lscpu | tee -a $OUTPUT_FILE
else
    cat /proc/cpuinfo | grep -E "model name|processor|cpu MHz|cache size" | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# CPU Count
echo "=== CPU COUNT ===" | tee -a $OUTPUT_FILE
echo "Physical CPUs: $(lscpu | grep "^Socket(s):" | awk '{print $2}')" | tee -a $OUTPUT_FILE
echo "CPU Cores: $(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')" | tee -a $OUTPUT_FILE
echo "Logical CPUs (threads): $(nproc)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Memory Information
echo "=== MEMORY INFORMATION ===" | tee -a $OUTPUT_FILE
free -h | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Detailed Memory
echo "=== DETAILED MEMORY ===" | tee -a $OUTPUT_FILE
if command -v dmidecode &> /dev/null; then
    sudo dmidecode -t memory 2>/dev/null | grep -E "Size:|Speed:|Type:|Manufacturer:" | tee -a $OUTPUT_FILE || echo "dmidecode requires sudo privileges" | tee -a $OUTPUT_FILE
else
    cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable" | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# GPU Information
echo "=== GPU INFORMATION ===" | tee -a $OUTPUT_FILE
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE
    echo "=== NVIDIA-SMI FULL OUTPUT ===" | tee -a $OUTPUT_FILE
    nvidia-smi | tee -a $OUTPUT_FILE
elif command -v lspci &> /dev/null; then
    lspci | grep -i vga | tee -a $OUTPUT_FILE
    lspci | grep -i 3d | tee -a $OUTPUT_FILE
else
    echo "No GPU information available" | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# Disk Information
echo "=== DISK INFORMATION ===" | tee -a $OUTPUT_FILE
df -h | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Disk Details
echo "=== DISK DETAILS ===" | tee -a $OUTPUT_FILE
if command -v lsblk &> /dev/null; then
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# Network Interfaces
echo "=== NETWORK INTERFACES ===" | tee -a $OUTPUT_FILE
if command -v ip &> /dev/null; then
    ip addr show | grep -E "^[0-9]|inet " | tee -a $OUTPUT_FILE
else
    ifconfig | grep -E "^[a-z]|inet " | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# PCI Devices
echo "=== PCI DEVICES ===" | tee -a $OUTPUT_FILE
if command -v lspci &> /dev/null; then
    lspci | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

# Uptime
echo "=== SYSTEM UPTIME ===" | tee -a $OUTPUT_FILE
uptime | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Load Average
echo "=== LOAD AVERAGE ===" | tee -a $OUTPUT_FILE
cat /proc/loadavg | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Docker Version (if available)
echo "=== DOCKER VERSION ===" | tee -a $OUTPUT_FILE
if command -v docker &> /dev/null; then
    docker --version | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE
    echo "Docker Info:" | tee -a $OUTPUT_FILE
    sudo docker info 2>/dev/null | grep -E "Server Version|Operating System|CPUs|Total Memory|Docker Root Dir|Storage Driver" | tee -a $OUTPUT_FILE || echo "Docker info requires sudo privileges" | tee -a $OUTPUT_FILE
else
    echo "Docker not installed" | tee -a $OUTPUT_FILE
fi
echo "" | tee -a $OUTPUT_FILE

echo "================================" | tee -a $OUTPUT_FILE
echo "System information collected successfully!" | tee -a $OUTPUT_FILE
echo "Saved to: $OUTPUT_FILE" | tee -a $OUTPUT_FILE
echo "================================" | tee -a $OUTPUT_FILE

echo "System information collection complete!"
