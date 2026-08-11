# Gemma serving output-equivalence gate

This gate accompanies the matched RTX 5090 serving matrix. It is a semantic-equivalence check, not a task-quality
benchmark. The recipe's `benchmark.output_probe_file` makes the benchmark harness issue these requests after the
throughput workload on each fresh stock and Emmy server, before teardown. The task JSON preserves the raw records
with the exact arm and repeat from `EMMY_BENCH_ARM` and `EMMY_BENCH_PROCESS_REPEAT` and the expanded workload point.

The recipe's aggregate step executes the frozen gate over every result and writes `output-equivalence.json`:

```bash
python scripts/validate_serving_output_equivalence.py \
  experiments/golden-bench-2026/quality_gemma4_rtx5090/prompts.jsonl \
  --results "$RUN_DIR"/*_benchmark.json --report "$RUN_DIR/output-equivalence.json"
```

Every arm must have repeats 0 through 4 and all twelve prompt IDs. Requests are sequential with seed 0, temperature
0, and a frozen per-prompt output cap. The gate requires identical requests, response text, completion-token counts,
and finish reasons for each matched record. Preserve the benchmark JSON and aggregate report. Any missing record or
difference closes the gate and excludes the matched end-to-end speedup table until resolved.

This does not make the stock-vs-Emmy delta a compiler-only result. Stock uses vLLM's native route while Emmy uses
`EmmyGenModel`; therefore the accepted claim is a matched end-to-end serving-system speedup. A compiler-caused e2e
claim additionally requires a reference-kernel or compiled-kernels-off arm within the same `EmmyGenModel` route.
