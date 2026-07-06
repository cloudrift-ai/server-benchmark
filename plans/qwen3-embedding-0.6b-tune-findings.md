# Whole-model tune findings — Qwen/Qwen3-Embedding-0.6B, RTX 4090 (sm_89), 2026-07-06

- Run: `emmy tune Qwen/Qwen3-Embedding-0.6B --dynamic seq_len@input_ids:1 --dynamic seq_len@attention_mask:2
  --dynamic seq_len@attention_mask:3 --dynamic seq_len@position_ids:1 --clean --bench
  --dump-dir _tune/tune-model-qwen3-06b-full/dump` on `vm-perf-tuning` (driver 580.65.06, CUDA 12.9), branch
  `feature/golden-sweep-rtx4090-findings` working tree.
- **Dynamic whole-model scope**: symbolic `seq_len`, everything tuned/benched at the 512 hint. Post-#269 codegen is
  the accepted baseline (see `plans/golden-sweep-rtx4090-findings.md`).
- Stats: tune wall **8451 s (2 h 21 m)**, 2 structural terminals, **3232 benches** (2185 ok / **87 bench_fail** in
  the DB), prior calibration +0.98 — but **`silly` = 96%** of post-warmup benches ≥2× best (Finding 1/2 fallout).
- Numbers below are the `--bench` **-O3** re-bench unless marked "-O1 (ranking)". The **full-model
  eager/tcompile/emmy table is MISSING** — it crashed on the Finding-1 bug, so this report has per-kernel evidence
  only (a whole-model e2e claim cannot be made from this run).

## Per-kernel -O3 bench (the 13 reproducers; sorted by emmy µs)

| Kernel | Layer op | eager µs | tcompile µs | emmy µs | vs eager |
|---|---|---:|---:|---:|---:|
| k_linear_mean_loop_reduce | pooling head (mean over seq + linear) | 294 | 203 | 10739 | 0.03x |
| k_sdpa_linear_reduce | attention + o_proj (fused, whole-graph terminal) | 196 | 136 | 5901 | 0.03x |
| k_linear | lm/dense projection | 140 | 130 | 465 | 0.30x |
| k_mean_linear_reduce | pooling head variant | 188 | 63 | 177 | 1.07x |
| k_linear_reduce | projection + reduce | 56 | 53 | 165 | 0.34x |
| k_mean_linear_reduce | pooling head variant | 125 | 48 | 113 | 1.11x |
| k_linear_reduce | projection + reduce | 46 | 44 | 105 | 0.43x |
| k_mean | mean pooling | 77 | 5 | 15 | 5.01x |
| k_cat_slice_transpose_unsqueeze_pointwise | rotary prep pointwise | 70 | 2 | 3 | 27.57x |
| k_linear_mean_reduce | pooling-head split alternative | — | — | skipped | KeyError (Finding 1) |
| k_linear_sdpa_reduce ×2 | per-layer attention | — | — | skipped | hung greedy variant (Finding 3) |
| k_mean (2nd) | — | — | — | skipped | hung greedy variant (Finding 3) |

Dominators: `k_linear_mean_loop_reduce` + `k_sdpa_linear_reduce` = ~16.6 ms of a ~17.7 ms total (94%). Both are
fallback variants that won by forfeit (Findings 1–2). The wins are real but small (`k_mean` 5×, the rotary
pointwise 27×, `k_mean_linear_reduce` ~1.1×).

## Finding 1 — bench worker cannot feed split-producer intermediates: `KeyError('linear_196__xn')` (P0)

The single biggest defect of the run, three blast radii from one bug:

- **41 bench_fail rows** on `k_linear_mean_reduce_286d22` (`eval failures`), every one sharing `PLACE@cone=inline`
  — the entire split-cone alternative family for the pooling head died in the bench worker with
  `KeyError('linear_196__xn')` before a single variant was measured. `eval variants --kernel mean_loop` returns
  **no measured rows**: the deployed 10.7 ms `k_linear_mean_loop_reduce` won by forfeit.
- The **full-model bench crashed** with the same KeyError — no eager/tcompile/emmy e2e table exists for this run.
- Two per-kernel reproducers skipped with the same error.

`linear_196__xn` is a `010_split_demoted` **split-producer intermediate** (the `xn` contiguizing producer): the
bench worker builds input arrays for a kernel from the graph's declared inputs, and a kernel consuming another
kernel's output (`..__xn`) isn't fed. The 2-terminal structural search meant terminal #2 (the split branch) was
un-benchable end to end — which also explains the pathological `silly 96%` (the search kept probing a branch whose
every leaf pinned at 2e6 µs).

**Repro**: `emmy run --ir _tune/tune-model-qwen3-06b-full/dump/*_lowering_cuda.kernels/k_linear_mean_reduce*.torch.json --bench`
(fails with the KeyError; no GPU needed to reach it). **Fix suggestion (P0)**: the bench worker's input builder
must materialize split-producer intermediates (run the producer kernel, or synthesize the `xn` buffer from its
shape) — until then no whole-model tune can measure any `PLACE@cone=inline`-adjacent structural branch, and
whole-model `--bench` is broken for split graphs.

## Finding 2 — `k_sdpa_linear_reduce_104572`: 83% of the search space killed by the 4 s compile budget (P1)

The 5.9 ms fused attention+o_proj terminal has **9 measured configs vs 45 bench_fail** (`eval failures`: clusters
of `compile stage exceeded 4.0s budget (4.4–5.1s)`, shared `FM=128 … PLACE@v7=inline`). The pick is the best of 9
survivors — a starved search, not a considered choice. Contrast the healthy **per-layer** attention kernels
(`k_linear_sdpa_reduce_d362d9` / `_e23835`: 238/210 measured configs, picks ≈ 305–309 µs at -O3; their
`misses best` flags are -O1/-O3 inversions — the picks' -O3 beats the -O1 rank-1s, so those are fine).

**Fix suggestion (P1)**: the timeouts overshoot the budget by only 10–25% — a modest budget bump (or a
budget-scaling heuristic on kernel body size) would recover ~45 variants for exactly the kernels that need search
the most. Repro: pin the failing cluster's knobs on the `k_sdpa_linear_reduce` reproducer and compile.

## Finding 3 — hung greedy variants: two reproducers deploy kernels that never complete (P1)

`k_linear_sdpa_reduce` (×2) and one `k_mean` reproducer bench skipped with
`HungKernelError: kernel … did not complete within 1000 ms` — the *greedy-deployed* variant hangs at -O3 while
other variants of the same kernels measure fine. A kernel that hangs under the reproducer's exact re-lowering is
either a codegen bug (mis-guarded loop on the symbolic axis) or a re-lowering mismatch vs the tuned fork. Needs a
dedicated drill-down: re-lower the reproducer with the pick's knobs pinned, `compile --ir cuda`, and inspect the
guard structure. Not further diagnosed this run.

## Finding 4 — `k_linear` family 0.30–0.43× eager (P2, expected post-#269)

The plain projection kernels trail cuBLAS 2.3–3×, consistent with the golden sweep's re-baselined numbers (the
#269 rewrite dropped operand staging / ring pipelining — the accepted-regression roadmap). No new per-kernel
diagnosis is warranted until those reintroductions land; the golden-sweep report carries the mechanism.

## Serving A/B (Step 2b) — emmy plugin DOWN, stock baseline recorded

Installed the `serving` extra post-tune (vllm 0.22.1; pulled torch 2.11.0 into the venv — after all tune
measurements, and both serve runs share the env).

- **Stock vLLM** (`emmy serve … --bench --stock`, 256 prompts × 512 tokens, concurrency 32): **103.7 req/s,
  53109 tok/s, median E2EL 218.9 ms, P99 911.9 ms** — the baseline to beat.
- **Emmy plugin** (`emmy serve … --bench`): **server failed to start.** The per-layer greedy compile dies with
  `ValueError: StageBundle: requires at least one Source` (`passes/lowering/tile/assembly/_slab.py:216` →
  `ir/tile/ir.py:1975`), killing the vLLM engine core. No emmy serving numbers exist for this run.

## Finding 5 — greedy deploy can pick an un-lowerable config and crash `emmy serve` (P0)

The tune *search* encountered the exact same state and handled it ("dropped un-lowerable candidate (ValueError:
StageBundle: requires at least one Source) — pruning branch, continuing search", 3× early in `tune.log`). The
*deploy* path (the plugin's greedy per-layer compile, prior from this tune) picks such a config and has no
prune-and-retry — the ValueError escapes to the engine and the server is dead on arrival. This is the same class
as the 06-19 report's `LoweringError`-no-fallback finding, one exception type wider: the option-0 fallback catches
`LoweringError` but not the assembly's `ValueError`.

**Fix suggestion (P0):** the deterministic-compile retry/blocklist should treat an un-lowerable assembly
(`StageBundle` ValueError, and the `TileGraphOp`-left-behind case from the golden sweep's SPLITK=6 probe) the same
as a `validate(ctx)` failure — blocklist the pick and fall back to the next-ranked / option-0 config. Repro (no
GPU): compile layer 0 with the trained prior loaded — the pick is deterministic.

## Repro / artifacts

- VM: `_tune/tune-model-qwen3-06b-full/{tune.log,dump/}`; dump has per-kernel `.torch.json` reproducers,
  `kernels.html`, `62_kernel_bench.json`. Golden-sweep DB/prior snapshots (pre-`--clean`):
  `_tune/golden-sweep-rtx4090/{autotune-post-sweep.db,prior-post-sweep.json}`.
- Triage: `emmy eval failures` (the KeyError + compile-budget clusters), `emmy eval variants --kernel sdpa`.

## Workflow notes

- **Non-tty tune is silent for hours.** Piped/tmux output suppresses the progress bar AND the per-shape INFO
  lines; a whole-model tune (one target, no `=== n/N ===` headers) printed nothing between minute 1 and minute
  ~140. Health had to be inferred from `ps` CPU%, cubin-cache count, and DB size. *Improvement:* a periodic
  non-tty heartbeat line (benches done / best-so-far / phase).
- **`--clean` destroys cross-hardware node data silently.** The golden sweep's node table (13k rows) lived in the
  same DB the model tune `--clean`ed; only a manual pre-tune `VACUUM INTO` snapshot saved it. *Improvement:*
  `--clean` should print a row-count warning when the node table is non-empty (or spare it — it's keyed by GPU).
- **A poisoned structural branch burns the whole budget.** 41 identical KeyErrors at 2e6 µs each kept the search
  returning to a dead branch (`silly 96%`). *Improvement:* after N identical infra-class failures (KeyError ≠ a
  kernel property), quarantine the branch instead of pinning each leaf.
- **The per-kernel bench table lacks the kernel hash + op label.** Two `k_linear_reduce` / `k_mean_linear_reduce`
  rows are indistinguishable without cross-referencing the dump; labels here were assembled by hand from
  `.torch.json` summaries. *Improvement:* print the hash suffix + a Layer-op column in the `--bench` table.
- **`pip install .[serving]` upgrades torch under the tuned venv** (2.9→2.11 here, pulled by vllm). Done mid-loop
  it would have invalidated the run; only sequencing it after all measurements avoided that. *Improvement:* pin
  or warn — the serving extra should not silently move the torch the compiler was just benched on.
- **Golden-sweep workflow notes**: the `eval failures` error column (added post-06-19) worked exactly as designed
  — both failure clusters were diagnosed without log grepping. The non-power-of-two-SPLITK and eval-golden-dash
  issues from that report did not resurface here (the latter was fixed on this branch).
