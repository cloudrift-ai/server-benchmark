# Golden sweep findings — RTX 4090, gemma4_12b set (2026-07-20)

- **GPU**: NVIDIA GeForce RTX 4090 (vast.ai rental, Taiwan host, CUDA 12.8 image, torch 2.13.0+cu130)
- **Sweep**: `EMMY_FAST_MATH=1 emmy tune --dataset golden --clean --kernel gemma4_12b` — 33 shapes, one invocation,
  cold. Wall time ≈ 11 h (the 2.5 h budget in the skill was measured on the qwen set; the gemma s2048 GEMMs at
  K=3840/N=30720 and the hd512 attention pair run 10–25 min each, and the fm umbrella doubles the enumeration).
- **A/B**: `emmy run --bench --golden NAME --json …` for all 33 shapes × {std, fm} lanes; wins re-run 2× more.
- **Tally**: **1 added** (`attention.hd512` fm entry, −11%), 0 replaced, 0 pruned; std lane: 27 shapes **worse**
  (cold search lands 1.03–1.89× behind the recorded goldens), 6 same; fm lane: greedy reproduces the recorded
  fm entries essentially everywhere (all "same" within noise except the hd512 win).

The recorded goldens (2026-07-13 pinned --ab sweeps) remain the ground truth on this card — this cold sweep did not
beat them anywhere in the std lane. The sweep's value was (a) the fm hd512 discovery, (b) fresh node data for the
prior, and (c) surfacing two deploy-side gaps that make the recorded potential unreachable for default users (below).

## Fork sibling regret

`emmy eval online --dataset nodes`, this card's `-O1` block. The offline column is the cold-start ranking that
decides what a cold sweep measures at all (and thereby censors what the online model ever trains on); the online
column is the CatBoost this sweep trained (calibration +0.74). Worst-fork rows are shape-labeled.

| metric (-O1, 56 forks)                          | offline prior          | online prior (CatBoost) |
| ---                                             | ---                    | ---                     |
| TILE fork regret (median)                       | 2.09x                  | 1.55x                   |
| REDUCE fork regret (median)                     | 1.05x                  | 1.04x                   |
| STAGE fork regret (median)                      | 1.00x                  | 1.00x                   |
| structural PLACE+REDUCE+STAGE+TILE (median)     | 2.24x                  | 1.00x                   |
| worst TILE fork: matmul free=512 red=3840       | 5.84x                  | 1.61x                   |
| 2nd: matmul free=4096 red=3840                  | 4.12x                  | 1.67x                   |
| 3rd: matmul free=3840 red=4096                  | 2.30x                  | 1.87x                   |
| leaf reachability (mean / median / worst)       | 1.79x / 1.37x / 4.27x  | 1.15x / 1.14x / 1.66x   |

Diagnosis: the miss is almost entirely the **offline TILE weights on sm_89 fp16 warp tiles** — STAGE and REDUCE are
clean in both halves, and the online model (trained on this sweep's own benches) halves the regret but inherits the
offline censoring, so it never sees the golden region either. This is why every std matmul A/B came out "worse":
the cold ranking steers the search away from the `w2x2/f4x4-f4x8/k2 + d2/cp/ring` region the goldens live in.
Fix: refit `scripts/golden_knob_heuristics.py` over the updated golden set + this sweep's node store.

## Per-shape outcomes (std lane, -O3 A/B; greedy = isolated re-bench, golden = best pinned row)

`vs cuBLAS` = greedy_us / eager_us (>1.0 = deployed emmy slower than PyTorch). The eager column is the live
`Eager PyTorch` row from the same run.

| shape | greedy µs | best golden µs | greedy/golden | category | eager µs | vs cuBLAS |
|---|--:|--:|--:|---|--:|--:|
| mlp_gate_up.s2048          |  4339.5 |  2299.9 | 1.89x | worse  |  2875.3 | 1.51x |
| q_proj_global.dynM         |   222.1 |   141.3 | 1.57x | worse  |   196.6 | 1.13x |
| q_proj_global              |   220.2 |   141.3 | 1.56x | worse  |   196.7 | 1.12x |
| q_proj_global.s2048        |   812.0 |   534.0 | 1.52x | worse  |   817.2 | 0.99x |
| o_proj_global              |   224.7 |   150.3 | 1.49x | worse  |   223.3 | 1.01x |
| o_proj_global.dynM         |   234.3 |   157.4 | 1.49x | worse  |   222.9 | 1.05x |
| mlp_down.dynM              |   402.5 |   271.2 | 1.48x | worse  |   385.8 | 1.04x |
| mlp_down                   |   402.5 |   272.6 | 1.48x | worse  |   382.0 | 1.05x |
| o_proj_global.s2048        |   836.9 |   570.8 | 1.47x | worse  |   791.4 | 1.06x |
| q_proj.s2048               |   437.8 |   301.7 | 1.45x | worse  |   377.9 | 1.16x |
| o_proj.s2048               |   463.4 |   320.5 | 1.45x | worse  |   421.9 | 1.10x |
| mlp_down.s2048             |  1585.2 |  1096.7 | 1.45x | worse  |  1467.9 | 1.08x |
| kv_proj.s2048              |   228.6 |   166.2 | 1.38x | worse  |   210.9 | 1.08x |
| q_proj                     |   103.9 |    78.8 | 1.32x | worse  |   108.0 | 0.96x |
| o_proj                     |   109.5 |    83.3 | 1.31x | worse  |   114.8 | 0.95x |
| q_proj.dynM                |   103.2 |    79.1 | 1.31x | worse  |   104.5 | 0.99x |
| o_proj.dynM                |   109.9 |    86.5 | 1.27x | worse  |   115.1 | 0.95x |
| kv_proj.dynM               |    59.2 |    47.3 | 1.25x | worse  |    49.6 | 1.19x |
| k_proj_global.s2048        |    57.6 |    46.5 | 1.24x | worse  |    49.6 | 1.16x |
| attention.hd512.s2048      |  1333.2 |  1088.7 | 1.22x | worse  |   938.0 | 1.42x |
| kv_proj                    |    57.9 |    47.8 | 1.21x | worse  |    49.6 | 1.17x |
| mlp_gate_up.dynM           |   878.6 |   738.3 | 1.19x | worse  |   797.5 | 1.10x |
| attention.hd256.s2048      |   355.3 |   304.5 | 1.17x | worse  |   312.7 | 1.14x |
| mlp_gate_up                |   827.3 |   738.3 | 1.12x | worse  |   789.6 | 1.05x |
| attention.hd512            |   131.3 |   117.8 | 1.11x | worse  |    87.9 | 1.49x |
| k_proj_global              |    19.7 |    18.3 | 1.07x | worse  |    16.9 | 1.16x |
| k_proj_global.dynM         |    19.9 |    18.6 | 1.07x | worse  |    17.0 | 1.17x |
| attention.hd256            |    37.7 |    36.7 | 1.03x | same   |    41.2 | 0.92x |
| attention.hd512.dynM       |   126.2 |   126.4 | 1.00x | same   |    88.9 | 1.42x |
| qknorm.k512                |    10.0 |    10.0 | 1.00x | same   |    11.2 | 0.89x |
| qknorm.k256                |     3.7 |     3.7 | 1.00x | same   |     4.8 | 0.77x |
| rms_norm.k3840             |     6.7 |     6.7 | 1.00x | same   |     6.8 | 0.98x |
| rms_norm.k3840.dynM        |     6.7 |     6.7 | 1.00x | same   |     6.8 | 0.99x |

Fm lane: greedy matches the recorded fm entries within noise on every matmul (ratios 0.99–1.01), and the fm greedy
equals the *std* goldens' absolute latency on most matmuls — the fm sweep enumeration is a superset that happily
re-finds the std configs. The one real fm delta is Finding 2.

## Finding 1 — the golden deploy tier misses bare-shape deploys: loudly for hd512 flash, silently for matmuls

The recorded goldens all **pin** cleanly (0 `pin_unmatched` across 66 A/Bs), but a plain deploy does not reach them:

- `attention.hd512` / `.s2048`: deploy prints `deploy: node 'scaled_dot_product_attention' matches golden shape
  ShapeKey(free_prod=4194304, reduce_max=512, is_warp=True, kind='flash') (2 recorded entries), but no offered
  candidate realizes any of them — enumeration drift`, then greedy falls to DB/prior and deploys 131.3 µs where the
  pinned golden benches 105–117 µs. Only these two shapes warn.
- `mlp_gate_up.s2048`: **no warning at all**, yet a default `emmy run --bench -c "<matmul 2048×3840 × 3840×30720
  fp16 snippet>"` deploys 4367 µs (4378 µs with `EMMY_TUNE_DB` pointed at an empty DB, i.e. goldens-tier-only)
  while the recorded golden pins at 2299.9 µs on the same box. The golden tier either never matches the
  bare-snippet ShapeKey or matches and fails silently — either way the recorded 1.25x-vs-cuBLAS potential deploys
  as 0.64x. Repro is the empty-DB command above.

This is the same class as the 5090 m16 drift (fixed for transposed-B decode matmuls via `canonical_row_key` +
`.lin` re-seed, PR #396/#398) and matches the open "cone-fork offer gap" from the repro-parity session.
**Recommendation**: make the golden-tier lookup loud (the matmul silent-miss is strictly worse than the flash
warning), add a regression test that a golden-shaped bare snippet deploys its golden (or names why not), and
re-seed the 4090 goldens for the fused-cone taxonomy once the offer gap is fixed (the 5090 file has the
`norm_linear`/`mlp_geglu` kinds; the 4090 file predates them).

## Finding 2 — attention.hd512 was missing its fm entry (recorded: −11%)

The shape had one entry (std lane, `REDUCE: g2k`, 116.1 µs). The fm sweep's greedy — un-split `REDUCE: ''` +
`TILE@pj` swapped to the f16-accumulate atom — benches **104.1 µs, reproduced 3×** (104.1/104.1/104.2 isolated;
a later pinned replay showed 111.3, within the documented ~9% alt-pipeline jitter). The `.dynM` twin already
carried exactly this family, so this was a bookkeeping gap, not a discovery frontier. Recorded as an additional
same-name entry (fm never replaces std). `attention.hd512.s2048`'s fm delta (−4.3%, −4.3%, +2.8%) is inside the
noise band — not recorded. The remaining gap to torch SDPA (fm eager 93.5 µs) is the hd512 d_v fold's 255-register
O-accumulator ceiling — codegen, not search; unchanged from the July arc.

## Finding 3 — offline TILE weights misprice sm_89 fp16 warp tiles across the board

Every std matmul A/B is "worse" with the *same signature*: greedy's TILE lands one family off (`eval golden` shows
found==golden knob strings for the recorded rows — the enumeration offers them — while the deployed pick differs),
and the offline TILE regret is 2.09x median / 5.84x worst (table above). REDUCE/STAGE are exonerated in both prior
halves. The online model halves the regret but is trained only on what the offline ranking let the sweep measure.
**Recommendation**: refit `scripts/golden_knob_heuristics.py` (offline weights) over the current goldens + this
sweep's node rows (merge via `scripts/merge_node_db.py`), and A/B the candidate weights with
`emmy eval offline --offline-file` before overwriting. A patience bump is NOT the fix — the golden region is
mispriced, not late-found (patience was the default 50).

## Workflow notes

- **Wall time**: ~11 h vs the skill's ~2.5 h budget — the budget line should be per-dataset (gemma s2048 GEMMs and
  hd512 attention are 4–10× the qwen shapes' cost; fm umbrella doubles enumeration). A per-shape `done in N s`
  summary table at the end of tune.log would let the next run budget properly.
- **`--kernel` semantics differ across eval views**: `eval online --dataset nodes --blame --kernel mlp_gate_up`
  matches 0 nodes (the node store wants op labels like `matmul`), so the per-shape blame files came back empty and
  the blame had to be read off the global table. Either accept golden names there or say so in the error.
- **A/B noise**: the first fm A/B for `attention.hd256` showed the golden at 36.8 µs (greedy "won" by 6%); two
  re-runs put the same golden row at 34.4 µs = exactly greedy's config. Without the Step-4 re-runs this would have
  been recorded as a win. The golden-row jitter is real; keep the re-run rule.
- **vast.ai provisioning**: 3 of 5 hosts this weekend never accepted the account SSH key (vast-side injection
  regression; the `--onstart-cmd` self-injection workaround works only on some hosts) and one hung pulling the
  image. Budget ~30 min of provisioning slack.
- The `run --json` schema (`greedy.isolated` + per-pin `record_knobs`) made the A/B analysis fully scriptable —
  no log scraping. Good addition since the last sweep.
