# Decode parity closers — the last ~1.6 ms to stock TPOT (planned 2026-07-24)

## Where the m1 tier landed

The transposed-matvec realizer took the M=1 gemv tier from catastrophic (110 ms) to bucket-parity-plus:
**c=1 TPOT 17.92 (256) / 18.98 (4K)** vs the bucket-32 path's 18.0/19.1 and stock's 16.3/17.4. All five
edge classes deploy at their k-major floors (qkv/qkg ``g8k/b128t`` ~15 µs, o_proj 8.1/14.2, gate_up
``g16k/b256t`` 142, down ``g32k/b256t`` 76); c=64/c=8 sanity clean (m1 routes T==1 only; the bucket-64
config keeps its 1092–1138 tok/s). The isolated-kernel gain over m32 (~1.5–2 ms) shrank to ~0.1 ms
e2e — **the in-graph structure eats it**: each ``g<w>k`` matvec is a partial→finalize→next dependency
chain (10 extra sync points/layer), and the twins run ~19 tiny kernels/layer at ~2–3 µs serialized wall
each. The residual vs stock decomposes: ~1.0–1.4 ms chain/serialization, ~0.3 ms the attn_out seam copy,
plus the shared ~1.0 ms sampler stall both arms pay (not part of the gap).

## WS1 — recover the split-chain loss (~1.0–1.4 ms)

Two arms, cheapest first; both A/B-able with the existing ``_tune/ab_m1bt.sh`` harness (fresh pack per
golden edit — the pack key excludes goldens):

1. **In-graph re-tune of the m1 rows** (hours, zero compiler work): the recorded rows optimize isolated
   latency; in-graph, one less chain link can beat ~5 µs of kernel time. Sweep the split ways per edge
   (``g4k``/``g8k``/``g16k`` × ``b128t``/``b256t``, plus bare ``b<n>t`` for the short-K edges where
   240-CTA latency-bound may still win in-graph) with pinned ``--ab`` per shape, then A/B the FULL STEP:
   record per-candidate ``head_c1`` TPOT from a targeted serving rerun, not isolated µs — the isolated
   optimum is exactly what mispriced the current rows. Keep the recorded rows honest: per-edge winners by
   e2e verdict.
2. **Finalize-into-consumer fusion** (compiler, the WS2-delegation pattern): the deferred ``g<w>k``
   finalize is a tiny cross-partition reduce over a [w, N] workspace; the NEXT kernel in every m1 twin
   (rope pointwise after qkv, the down partial after gate_up's combine, the norm after o_proj) can open
   with it exactly like ``ZeroPrologue`` rides a predecessor. One realizer + a ``__fin`` suffix naming
   convention; kills one sync point per matvec (5/layer). Gate on the same structural conditions as the
   delegation pass (single consumer, no graph-boundary crossing).

## WS2 — per-layer kernel-count fusion (~0.5 ms)

The m1 twins' fusable neighbor pairs, in measured-cost order (each tiny kernel ≈ 2–3 µs serialized):

- **stat + cone** (the cut halves): one kernel — the stat is 1 CTA, the cone 4; the merged kernel
  computes the stat in CTA 0's first warp and cone-writes from all CTAs behind a grid-sync-free
  ordering (the cone only needs the stat's rsqrt — recompute it per CTA instead: the stat read is
  3840 halves, 7 µs of redundant reads vs a full kernel boundary. Redundant-statistic is already the
  split-K pattern — reuse it).
- **the RoPE pointwise chain** (3 ops) and **the qk-norm trio**: standard neighbor fusion the bucket
  twins already enjoy at m32 — verify why the m1 build splits them (likely the unit-axis demotion
  breaking the fusion pass's shape match) and fix THAT rather than adding rules.
- Target: ≤ 10 kernels/layer (from ~19).

## WS3 — the attn_out seam copy (~0.3 ms)

The post twin's ``attn_out`` input still round-trips through the protective upload copy; aliasing vLLM's
attention output tensor onto the twin's input backing needs the tensor pinned across steps (the fragile
part that deferred it twice). Do it LAST, behind an env gate (``EMMY_GEN_ALIAS_ATTN``), with the
capture-replay-matches-live test extended to assert the aliased path — revert-friendly.

## Golden seeding & verification (per the usual method)

- Every WS1/WS2 winner lands as a golden row via manual pinned ``--ab`` (never tuner sweeps), on the
  SERVING orientation (plain-matmul ``.t`` snippets for matvecs), values from e2e-verdict runs; both
  cards (5090 first, 4090 mirror on the cp lane).
- After each batch: ``emmy eval golden --in-model`` — 0 major gaps, DRIFT only the known to_4_cast
  splice; the drift gate green (EXPECTED_GAPS burn-down where new aux keys appear); full ``make test``.
- Flip ``EMMY_GEN_M1_TIER`` default ON when the tier beats the bucket path at c=1 AND the c=64/c=8
  cells hold — then re-warm/rebake the gemma-4 image at that rev (the fixed pack key + ``.pack_hit``
  verify must pass end-to-end) and rerun the docker bench trio.

## Benchmarks & article

- Rerun the serving lanes' affected cells (emmy + fm, all six points) through the experiments protocol;
  per the single-machine rule everything on the local 5090 box, fresh packs, empty online.
- Update the article (cloudrift-landing, commit+push): the e2e tables' emmy/fm columns, the TPOT
  narrative (bucket-parity → stock-parity as the workstreams land), per-kernel chart refresh if new
  rows shift the catalog, and the headline numbers + intro "XXX speedup" once the final table stands.
- Exit gates: c=1 TPOT ≤ 17.4 (stock parity, stretch: below), no regression c=8 ≥ 360 / c=64 ≥ 1090,
  fm TTFT wins hold (472/1839/1985 class), audits green both cards, baked-image verify PASS.
