# Latest experiment run

This directory records the latest MPK and stock vLLM experiment run. It is a factual run and artifact index only;
measurement values are retained in the raw files and are not reproduced, compared, or interpreted here.

- Run status: succeeded
- Timestamp: 2026-08-14T20:24:20Z
- Run ID: `20260814T202420Z-40df5bcb`
- Experiment row: `serving_mpk_qwen3_8b_a100/a100x1`
- Git revision: `9a485df4229e0529720a3b46e1d2fc482e97a394`
- Staged source ID: `22afeb5776c5039e5b500dbaafc101a81a010c0ea4e3080489ff5e73dd88d7ba`

## System

- Host: `riftvm`
- OS: Ubuntu 24.04.1 LTS, kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs
- Memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; CUDA driver API: `13.0`; CUDA compiler: `12.9`
- Docker client/server: `28.5.1` / `28.5.1`

## Artifacts

- Experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Runner logs: `results/benchmark.log`, `results/benchmark_a100_x_1.log`
- Environment inventories: `results/a100x1_requirements.freeze.txt`, `results/a100x1_mpk.freeze.txt`,
  `results/a100x1_hf_snapshots.txt`
- MPK setup log: `results/a100x1_mpk_install.log`
- MPK baseline raw files: `results/a100x1_mpk_base_r0.txt` through `results/a100x1_mpk_base_r4.txt`
- MPK persistent-kernel raw files: `results/a100x1_mpk_mega_r0.txt` through `results/a100x1_mpk_mega_r4.txt`
- Stock vLLM raw files: `results/a100x1_stock_r0.txt` through `results/a100x1_stock_r4.txt`
- Stock vLLM server logs: `results/a100x1_stock_serve_r0.log` through `results/a100x1_stock_serve_r4.log`
