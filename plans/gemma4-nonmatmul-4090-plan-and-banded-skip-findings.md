# Gemma-4 non-matmul kernels for the RTX 4090 — banded tile-skip findings + the rental measurement plan

- **Date / setup**: 2026-07-16, local RTX 4080 (sm_89 — same arch as the 4090, different perf profile: 76 vs 128
  SMs, ~717 vs ~1008 GB/s). Goal: optimize the gemma-4-12B NON-matmul kernels for the 4090, doing everything
  possible locally and leaving the rented 4090 a pure measurement session.
- **Inventory** (fresh layer compiles at HEAD, sliding layer 0 + global layer 5): flash attention
  (`k_scaled_dot_product_attention_reduce`, hd256 sliding / hd512 global), RMSNorm stat (`k_mean`), norm partials in
  the fused edges, rotary pointwise (`k_cat_slice_unsqueeze_pointwise` ×2), layout copies (`k_unsqueeze` ×2). At
  seq 512 everything except attention is at parity or negligible (≤ 50 µs total on the 4080); attention is where
  the 4090 non-matmul story lives.

## What was DONE locally: the sliding-window banded tile-skip (branch `feature/flash-banded-tile-skip`)

The dominant open item from the gemma-4 long-seq arc: at seq > 1024 the 40 sliding layers' flash streamed the FULL
O(seq²) key range — the explicit-mask warp flash (#365) treats the HF-materialized mask as an opaque bias, so
neither the causal end (`kv0_end`) nor any window bound applied. Implemented per the deferred design in the
global-layer findings, plus the fusion-boundary work it turned out to require:

- `SdpaOp.sliding_window` trace stamp (`trace/huggingface.py::stamp_sliding_windows`, called by
  `commands/compile.py` on layer AND whole-model traces): re-asserts the window the trace erases, from
  `config.sliding_window` × `layer_types`. Both reference backends (`SdpaOp.forward`, `torch_ref`) band too, so
  layer accuracy checks are self-consistent — and the `--layer` harness limitation ("traced layer is pure causal
  at every seq") is fixed as a by-product: a stamped layer trace now COMPUTES the true band.
- The band decomposes as a second single-predicate coordinate `Select` (keep `kv > m − W`) chained after the
  causal one — each rides the existing `FragmentMask` realization unchanged; an explicit bias stays loaded beside
  them (it may mask more, e.g. padding — skip bounds stay sound under any mask ⊆ causal ∧ band).
- `_twist.py` derives the stream START off the band predicate exactly as it derives the causal end:
  `kv_start = ⌊max(0, first_row − W + 1)/bn⌋·bn`; `pipelined_kloop`/`staged_kloop` take `k_first` beside `k_end`
  (absolute loop var, rebased slot/phase arithmetic, clamped prologue primes); slice-local under split-KV;
  dropped (exact, un-skipped) under WSPEC.
- **Fusion had to be made order-independent** — the second mask node broke the pair ordering main relied on, and
  the causal mask silently fused into the score producer where flash re-synthesis DROPPED it (kernels attended
  outside the mask, max_diff 3.4). Three guard fixes in `010_merge_loop_ops`: mask epilogues are exempt from the
  score-producer deferral (they must assemble onto the softmax past one another), the QK contraction is barred
  from chasing them, and `_reduce_heavy` discounts mask adds in rowmax-bearing bodies (a 2–3-mask softmax was
  crossing the work threshold and never assembling onto its P@V offer site). Plus a `try_flash` safety decline:
  a mask stranded on the score producer degrades the fuse instead of silently mis-attending.
- Tests: 12 new cases in `tests/compiler/e2e/test_attention_coverage.py` (staged/unstaged/alt, non-aligned W,
  symbolic seq, split-KV composition, additive-bias-plus-stamp with an in-band padding column, structural
  skip-derivation pins). Full suite 2346 passed / lint clean.

### Measured on the 4080 (greedy deploys, layer 0, fp16; flash kernel row — identical config/grid both sides)

| seq | main (causal-only, band DROPPED — wrong semantics) | branch (true banded + skip) | speedup |
| --- | --- | --- | --- |
| 2048 | 498.0 µs | 397.1 µs | **1.25×** |
| 4096 | 1716.2 µs | 834.2 µs | **2.06×** |

Near-linear scaling on the branch (397 → 834 at 2× seq) vs quadratic on main (498 → 1716, 3.4×), while computing
STRICTLY MORE masking (the true sliding semantics main's layer trace silently loses). The ratio keeps growing —
~4× at 8192 (W=1024). Both rows are the greedy `d1/cp` pick; the tuned `alt`/`ring` families stack on top.

## What still needs a rented 4090 (pure measurement; rent when available)

Golden seeding + A/B replays, in priority order. Workflow per `tune-golden` / the manual pinned `--ab` sweep
(3× reproduction, `cublas_us` = live torch row, -O3 only, NO env tile pin on attention runs).

1. **Banded attention goldens (new shapes, this branch)** — seed `gemma4_12b.attention.hd256.sw1024` at s2048
   (+ dynM, + s4096 if the box has headroom). Candidates: the causal hd256 s2048 winners transfer structurally —
   std `d1/cp/alt` nt8, fm `d2/cp/ring` nt2 (the 4090's s2048 fm pick), plus the greedy `d1/cp` form as control.
   Reference: torch SDPA with the banded additive mask (the golden snippet needs the stamp — via a layer-slice
   reproducer or a stamped snippet; F.sdpa has no window arg).
2. **hd512 dynM golden refresh** — the symbolic split-KV landed after the 4090 rows were recorded (5090 dynM went
   0.77× → 0.89×): re-bench `gemma4_12b.attention.hd512.dynM` with `REDUCE=g2k` vs the recorded plain-alt 139.6 µs
   row and update the yaml if the split wins (expected — the static alt+g2k already wins on this card at 116.1).
3. **Re-validate the hd256/hd512 static + s2048 rows post-branch** — the fusion-guard changes alter no schedules,
   but the banded stamp changes what the LAYER kernels look like at seq > 1024; layer-level A/Bs (not snippet
   goldens) should be re-based on the branch.
4. **Layer e2e at 2048/4096** — the sliding-layer flash with skip vs torch SDPA (which pays full O(seq²) plus mask
   materialization on sliding layers at seq > window): this is the first regime where emmy attention should beat
   eager by a growing margin.

## Open non-matmul items after this (codegen, local-able, in value order)

1. **hd512 d_v fold** (8 global layers; 4090 recorded 0.82×/0.79×/0.87× vs SDPA, knob-exhausted): the 255-reg
   O-accumulator ceiling needs a warp-column d-split (`w<um>x<un>` twisted geometry — per-column V fragments +
   smem P sharing or duplicated scores). Its own feature branch; the 2026-07-14 findings hold the full gate map.
   The banded skip does NOT help these layers (full attention).
2. ~~Banded skip × the explicit-mask hd512 form~~ — DONE on this branch: the stamp asserts `is_causal` on FULL
   layers too, so the 8 global layers' causal end-skip applies through the whole-model trace's opaque bias
   (halves their stream on average at seq > 1024).
3. RMSNorm k3840 / qknorm k256/k512, rotary, unsqueeze copies: at parity on both cards; nothing to do.

## Repro

```bash
# the banded layer bench (this branch):
venv/bin/emmy run google/gemma-4-12B --layer 0 --seq-len 2048 --bench
# unit surface:
venv/bin/pytest tests/compiler/e2e/test_attention_coverage.py -k banded -q
```
