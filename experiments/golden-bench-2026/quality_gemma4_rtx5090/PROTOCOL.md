# Gemma serving output-equivalence gate

This gate accompanies the matched RTX 5090 serving matrix. It is a semantic-equivalence check, not a task-quality
benchmark. The recipe's `benchmark.output_equivalence_file` makes the benchmark harness issue these requests after the
throughput workload on each fresh stock and Emmy server, before teardown. The task JSON preserves the raw records
with the exact `benchmark.comparison_arm`, `benchmark.process_repeat`, and expanded workload point.

`emmy bench` executes the frozen gate over every result, writes `output-equivalence.json`, and exits nonzero if the
gate fails:

```bash
emmy bench experiments/golden-bench-2026/serving_gemma4_rtx5090
```

Every arm must have repeats 0 through 4 and all twelve prompt IDs. Requests are sequential with seed 0, temperature
0, and a frozen per-prompt output cap. The gate requires identical requests, response text, completion-token counts,
and finish reasons for each matched record. Preserve the benchmark JSON and aggregate report. Any missing record or
difference closes the gate and excludes the matched end-to-end speedup table until resolved.

This does not make the stock-vs-Emmy delta a compiler-only result. Stock uses vLLM's native route while Emmy uses
`EmmyGenModel`; therefore the accepted claim is a matched end-to-end serving-system speedup. A compiler-caused e2e
claim additionally requires a reference-kernel or compiled-kernels-off arm within the same `EmmyGenModel` route.
