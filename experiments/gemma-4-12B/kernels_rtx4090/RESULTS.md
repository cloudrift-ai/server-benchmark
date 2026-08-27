# Gemma 4 12B kernels on RTX 4090

## Conclusion

The Emmy runner completed the 2026-07-30 run, but its kernel evidence is incomplete and cannot support a performance
claim for the full golden set. The preserved table contains Emmy measurements for 62 of 160 kernels; 98 entries
are failures. Among the measured subset, Emmy was more than 2% faster than eager PyTorch on 37 kernels, within 2% on
7, and more than 2% slower on 18. Against `torch.compile`, the corresponding counts are 34, 10, and 18.

Those counts describe only the surviving subset. They must not be summarized as a geometric mean or generalized to
the full set because the missing rows are neither random nor negligible. The largest preserved regressions include
`gemma4_12b.mlp_down.lin.dynM` at 0.18x eager and `gemma4_12b.mlp_down_fused.m256.lin` at 0.43x. Several tiny kernels
show 2–4x ratios, but their 1–4 microsecond integer timings are too coarse to carry a broad performance conclusion.

## Protocol and failure

The recipe ran `scripts/bench_golden_set.py` over 160 `gemma4_12b` cases on one NVIDIA GeForce RTX 4090, comparing
eager PyTorch, `torch.compile`, and Emmy. It requested both the standard and `EMMY_FAST_MATH=1` passes. The first pass
completed its remote table with one missing Emmy row. During the second pass, CUDA became unavailable; subsequent
cases repeatedly reported `cudaErrorNoDevice` or a zero detected device count.

The legacy result collector flattened both pass directories to the same two result basenames. Only one JSON/Markdown
pair was retained locally, and the files do not independently identify their precision lane. The preserved pair has
62 measured and 98 failed rows, matching the later device-loss pass. The earlier, nearly complete remote table is not
in the archive, so this run cannot compare standard and fast-math behavior.

The command took 8,645.8 seconds and the runner reported one successful task because the harness emitted its partial
tables and exited successfully. That runner status does not override the 98 failed kernel rows.

## System and provenance

- Hardware: one NVIDIA GeForce RTX 4090 on the pre-allocated host `riftuser@176.124.69.199`.
- Timestamp: `2026-07-30_15-32-41_82c7b31f`.
- Workload status: partial; 62/160 preserved rows have Emmy measurements.
- The legacy run has no assembled experiment record and does not preserve complete host, driver, or source-revision
  provenance. Package-install output and the full runner logs remain in the archive.

## Durable files

- Raw-results archive: `results.tar.gz`.
- Archived root: `2026-07-30_15-32-41_82c7b31f/`.
- Measurements: `rtx4090x1_golden_bench.json` and `rtx4090x1_golden_bench.md` inside the archive.
- Supporting evidence: runner logs, the executed recipe snapshot, and the legacy task manifest are in the archive.
