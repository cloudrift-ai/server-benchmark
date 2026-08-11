# Golden bench 2026

This cross-model experiment supports the Emmy compiler submission. Measured output remains local; each child contains
one reproducible recipe and writes its raw evidence into the benchmark task directory.

## Common kernel corpus

Every `kernels_*` recipe traces layer 0 of `Qwen/Qwen3-0.6B` at static sequence lengths 1 and 512, then tunes and
benchmarks every traced target. The two buckets represent decode and ordinary prefill. Static traces make the realized
shapes explicit and keep the corpus identical from SM70 through SM120 without making early iteration prohibitively
slow. Add a long-prefill bucket only in response to a concrete unanswered result.

Each trace receives an isolated tuning database, online-prior file, and cubin cache. Search uses at most 12 measured
candidates per kernel with patience 4 and seed 0. The selected realization is recompiled and benchmarked at deployable
`-O3` against eager PyTorch, `torch.compile`, and Emmy with 10 warmups and 100 measured iterations.

The common corpus is the only input to the cross-GPU geomean. Model-specific capability cases, including Gemma fused
kernels and SM70 quantized paths, must be reported separately so they cannot bias the headline comparison.

## End-to-end workloads

The serving recipes pin one model/platform pair and a controlled workload grid. `serving_gemma4_rtx5090` contains an
Emmy-vs-stock A/B. The other recipes initially qualify the chosen systems; they need explicit Emmy engine arms before
they can support a claim that Emmy improves end-to-end performance on those systems.

`serving_qwen36_27b_nvfp4_rtx5090` is provisional. Preserve startup logs and verify native SM120 FP4 execution before
treating its measurements as NVFP4 performance; a vLLM fallback implementation is a compatibility result instead.

The B200 kernel and serving runs are optional. Drop them before reducing the V100, RTX, or H200 coverage.
