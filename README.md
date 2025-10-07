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
Run individual benchmarks:
```bash
./benchmarks/run_system_info.sh
./benchmarks/run_hf_download.sh
./benchmarks/run_vllm_benchmark.sh
./benchmarks/run_yabs.sh
```
