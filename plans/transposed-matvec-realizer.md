# Transposed-matvec realizer → decode TPOT parity (planned 2026-07-23)

## Why

Decode TPOT on gemma-4-12B (RTX 5090, 4K c=1) sits at 19.1 ms vs stock vLLM's 17.4. The gap decomposes
(nsys sessions, plans/gemma4-decode-kernel-count.md) into a ~1.0 ms sampler-region host stall SHARED with
stock, ~0.3–0.5 ms of seam residual, and **~1.5–2 ms of bucket-32 padding tax**: stock decodes c=1 as true
M=1 cuBLAS gemvs at full memory bandwidth; emmy pads the token to the 32-row twins, whose m32 mma forms run
10–20 µs/edge behind a pure gemv streaming the same weights (48 layers × 4 edge classes).

The M=1 gemv tier (`EMMY_GEN_M1_TIER`) exists end-to-end — m1 twins, T==1 routing, post→pre chaining, pack
integration, m1 goldens, the degenerate-composition recognizer fix (unit `_um` axis + `Carrier.rename`) —
and its isolated matvecs hit 1.68 TB/s ≥ cuBLAS gemv. It is gated OFF on ONE blocker, root-caused in the
2026-07-23 session: **the serving twins read B through a transposed (k-major) index map**, and the scalar
b-reduce tier has no coalescing-aware partition for that layout — each lane strides N per k step, nothing
coalesces, ~8× bandwidth loss (down matvec 568 µs in-model vs 72 row-major; SASS `IMAD ×N` + `LDG.E.U16`).
The staged mma tiers got their transposed-B treatment in PR #406; the scalar tier never did.

Projected end state: 19.1 − (1.5–2) ≈ **17.1–17.6 ms — TPOT parity to slightly ahead** — while the fm lane
keeps its TTFT wins (472/1839/1985 vs stock's 565/2068/—). Next floor below that is the ~13.7 ms
weight-streaming limit (quantization territory, out of scope).

## Design — a k-major reduce partition as a fork sibling (no shape special-cases)

Current `bN` block-reduce: one CTA per output row n; the block's threads partition **k** interleaved;
B reads `w[n*K + k]` — contiguous per lane only when B is row-major. For k-major B (`w[k*N + n]`) the
coalescing axes swap, so the sibling partitions the block 2-D:

- **warp lanes sweep n** — at every k step the 32 lanes read `w[k*N + n..n+31]`, contiguous across lanes
  (coalesced regardless of per-lane serial k);
- **warps partition k** — each warp folds a k-slice for its n-tile; a smem tree combines the per-warp
  partials (the existing cooperative combine machinery — `Carrier`/`emit_combine` — unchanged);
- grid covers n-tiles × any residual k-split (`g<w>k` composes on top exactly as for `bN`).

Selection is **structural, not a heuristic**: at realize time the reduce materializer already holds B's
index map; when the reduce axis is NOT B's fastest-varying axis, enumerate the transposed sibling (spelling
`bt<N>`, e.g. `REDUCE: 'bt128'`) beside the plain `b<N>` rows — a fork row like any other, golden-recordable,
prior-featurized (one new feature: reduce-axis-stride-of-B). The plain serial and `g<w>k` forms stay.

Why not transpose-fold at bind/capture instead: the same weight serves the staged mma tiers, which WANT the
k-major layout PR #406 tuned for — a second row-major copy doubles 24 GB of resident weights, and an
in-place layout flip regresses the m32/m4096 tiers. The realizer fixes the one tier that is wrong.

### Work items

1. `REDUCE` codec: parse/print `bt<N>`; enumeration offers it only when the B-stride condition holds
   (mirror of how `b<N>` is offered; off spelling unchanged).
2. Reduce materializer: the n-lane × k-warp partition emitter — lane-contiguous B loads (vectorize to
   `ld.global.v4`+ when N-alignment allows), per-warp `Accum` fold, smem tree combine via the carrier.
3. Featurizer: stamp B's reduce-axis stride class so the prior can separate the siblings; blame/ablate
   check after seeding.
4. Tests: (a) unit — transposed snippet compiles under `REDUCE=bt128` pin and the emitted source indexes
   `w[k*N + n]` with lane-major n; (b) accuracy vs eager on the transposed matvec shapes; (c) perf gate in
   `tests/perf` — bt-form ≥ 5× the b-form on a k-major M=1 shape (the 8× loss is the regression being
   locked out); (d) the m1 composition e2e (cut → bt consumer) on the S=1 graph.

## Rollout — goldens, docker, benchmarks, article

5. **Bench + reseed the m1 goldens on the TRANSPOSED forms** (both cards): pinned `--ab REDUCE=bt64/bt128/
   bt256` on the k-major snippets (`torch.matmul(x, w)` with `w[K,N]`) for the four consumer matvecs +
   the cut totals on the fused m1 keys; REPLACE the row-major-benched values and drop the YAML
   transposed-B caveat. Targets: gate_up ≤ ~150 µs (1.6+ TB/s), down ≤ ~80, qkv/qkg ≤ ~40 cold.
6. **Widen the audit + gate**: `capture_twin_graphs` widths gain 1 (the m1 twins) so `eval golden
   --in-model` and the drift gate see the tier; re-run — 0 major gaps required on both cards.
7. **Flip `EMMY_GEN_M1_TIER` default ON** only after twin-level e2e verdicts at BOTH c=1 and c=64 (the
   recorded rule: serving-edge rows need both-concurrency verdicts; m1 routes only T==1, but the boot
   builds the twins — verify boot time + c=8/c=64 untouched).
8. **Rebuild the serving images at the new rev**: `make wheel && make vllm-emmy-image`, then the gemma-4
   prebake cycle on the 5090 (`gemma4-warm` → `gemma4-serve-image` → `gemma4-serve-verify`; config.env
   already pins the article protocol). The baked pack must contain the m1 twins (tier on at warm time).
9. **Run all benchmarks** through the experiments (`emmy bench experiments/gemma-4-12B/... --local/--ssh`):
   serving lanes + llamacpp on the 5090, per-kernel on both cards, accum sweep — and repeat the serving
   lanes against the BAKED image (the reproducibility check: image numbers within noise of venv numbers,
   zero recompiles on boot).
10. **Update the article** (cloudrift-landing, `article/gemma-report`): e2e tables (headline: TPOT parity +
    the fm TTFT sweep + the c=8/c=64 throughput wins), per-kernel charts both cards via
    `render_golden_bench_chart.py`, headline numbers in the intro, repro section pointing at the new image
    tag. Commit and push both repos.

## Exit gates

- 4K c=1 decode TPOT ≤ 17.6 ms (parity with the measured stock 17.4; stretch: below).
- No regression: c=8 ≥ 360 tok/s, c=64 ≥ 1090 (std) / 1130 (fm), rag/TTFT cells hold, 256 c=1 diagnostic
  improves (the m1 tier's native point).
- `eval golden --in-model`: 0 major gaps, DRIFT limited to the known to_4_cast splice; drift gate green.
- `gemma4-serve-verify` passes: baked-image boot compiles zero cubins, serves offline, tokenless.
- Article tables carry only same-box, same-run lane comparisons (the single-machine rule).
