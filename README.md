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
