## Benchmark LLM Inference

This tool is designed to test a server performance for llm inference

## How to Run

Clone the repository:
```bash
git clone https://github.com/cloudrift-ai/server-benchmark.git
cd server-benchmark
```

Install dependencies:
```bash
./setup.sh
```

Run benchmarks automatically via SSH on remote servers:
```bash
./run_remote_benchmark.py
```

Or run individual benchmarks manually:
```bash
./benchmarks/1_system_info.sh
./benchmarks/2_hf_download.sh
./benchmarks/3_vllm_benchmark.sh
./benchmarks/4_yabs.sh
```

## Benchmark Naming Convention

Benchmark scripts follow the naming convention: `<step_index>_<benchmark_name>.sh`
- The script automatically discovers and runs benchmarks in order
- Result files are named: `<benchmark_name>.txt`

### Adding a new benchmark:
1. Create a script: `benchmarks/5_my_benchmark.sh`
2. Ensure it outputs to: `my_benchmark.txt`
3. The script will be automatically discovered and run

### Disabling a benchmark:
To temporarily disable a benchmark, prefix the filename with an underscore:
```bash
mv benchmarks/4_yabs.sh benchmarks/_4_yabs.sh
```
The script will skip any files not matching the `[0-9]*_*.sh` pattern.

### Re-enabling a benchmark:
```bash
mv benchmarks/_4_yabs.sh benchmarks/4_yabs.sh
```

## Model Configuration

Models in `config.yaml` support the following parameters:

### Single-Instance Configuration
```yaml
models:
  - name: "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    tensor_parallel_size: 1  # Number of GPUs per instance (default: 1)
```

### Multi-Instance Configuration
For servers with multiple GPUs, you can run multiple vLLM instances in parallel to maximize throughput:

```yaml
models:
  - name: "Qwen/Qwen2.5-72B-Instruct-AWQ"
    tensor_parallel_size: 2  # GPUs per instance
    num_instances: 2         # Number of parallel instances (optional, default: 1)
```

**Example: 4-GPU Server Configuration**
- `tensor_parallel_size: 2` + `num_instances: 2` = 2 instances using GPU pairs (0-1, 2-3)
- This effectively doubles throughput compared to a single instance with `tensor_parallel_size: 4`

**Benefits of Multi-Instance Setup:**
- Lower inter-GPU communication overhead
- Better latency per request
- Aggregate throughput = sum of all instance throughputs
- Ideal for models that fit comfortably on fewer GPUs than available

**GPU Assignment:**
- Instance 0: GPUs 0 to (tensor_parallel_size - 1)
- Instance 1: GPUs tensor_parallel_size to (2 × tensor_parallel_size - 1)
- And so on...

**Benchmark Behavior:**
- The benchmark splits total prompts evenly across instances
- All instances run benchmarks concurrently
- Results are aggregated to show total throughput and average latencies
