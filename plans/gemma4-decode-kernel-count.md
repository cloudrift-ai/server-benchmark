# Decode kernel-count reduction — closing the 4K/4K c=1 TPOT gap (planned 2026-07-22)

## Problem statement

At 4K-in/4K-out c=1 the request is ~99% decode, and the whole loss to stock vLLM is per-STEP structure, not
kernel speed: TPOT 20.06 vs 17.41 ms (5090, fm, seed 0 — this session's A/B, stock baseline reproduced
exactly across sessions). Emmy's decode-step kernel time (~13.4-13.5 ms, bucket-32 twins) already sits at
the ~13.7 ms weight-streaming floor (22.8 GB @ ~1.66 TB/s; WS2 of the TPOT session proved 97% of the
access-pattern ceiling per kernel), so the ~2.65 ms/step gap is overhead: emmy deploys ~19-20 graph nodes
per layer against stock's ~8-9 (split-K partial+finalize pairs, memset nodes, the GeGLU cut's five
kernels, separate q/kv projections, separate norm sweeps), each node costing ~1-2 µs of dispatch plus
drain/refill tails. Two root causes, two workstream families:

- **fusions stock has that emmy lacks** (weight-concat QKV / gate_up, one launch each) — pure launch-count
  wins at zero kernel-quality cost;
- **fixed per-mechanism costs of emmy's own (per-edge-measured) multi-kernel forms** — the memset nodes,
  the finalize launches — which nothing in the per-edge evidence prices.

Every candidate must be judged on the twins' whole-program e2e line (3×), never per-kernel tables or
isolated snippets — µs-class kernels are launch-bound in every isolated harness (stat-sink session
lesson). Golden work follows the manual pinned `--ab` method; no tuner sweeps.

**Exit gate:** decode TPOT ≤ 18.5 ms on the 5090 serving A/B (stretch: parity 17.4); twin post32+pre32
e2e −1.5 ms/step combined; audits MATCH / DRIFT 0 / GAP 0 on both cards; fm-never-loses holds.

## WS1 — sibling-matmul concat (QKV one launch, gate/up one launch) [biggest, first]

**STATUS 2026-07-22: compiler side LANDED on this branch** — `decomposition/035_merge_sibling_linears` (N-way,
insertion-order concat = ABI), `ConstantOp.source_parts` through every loader/plan/serving bind path, plus two
exposed-bug fixes: the recognize stat-free-cone empty-row fallback (PLANAR demote) and 64-bit symbolic-grid
`_gid`/strides (int32 overflow at batch·seq·N·coop ≥ 2³¹). 5090 goldens for the merged keys SEEDED same day
(manual `--ab`; audit MATCH 74 / DRIFT 0 / GAP 0 — see plans/golden-seed-merged-sibling-5090-findings.md); 4090
keys baselined in the drift gate pending a 4090 session. Remaining WS1 work: 4090 seeding, fm-lane twins, qknorm
sink-anchor re-check, twin e2e + serving A/B. Watch: on the fp32 SYMBOLIC path
(Qwen embedding trunk) the merge lets the input norm fuse into the merged matmul (fan-out drops to 1) and the fused
computed-A form demotes to the scalar coop tier — correct but slow; the f16 gemma path has the mma tier. Decide by
serving A/B per protocol.

Stock concatenates q/k/v (and gate/up) weights at load and runs ONE gemm each. Emmy traces the HF module
structure and deploys q (N=4096) and kv (N=2048) separately — with split-K that is 2 partials + 2
finalizes where stock has one launch; the m32 GeGLU cut likewise runs its two N=15360 channel matmuls as
separate launches.

*The move (general rule, no shape naming):* a frontend/optimization pass that merges SIBLING contractions
sharing the same A operand and the same contraction axis, whose B operands are constants — concat B along
N (both layouts: canonical K-major concat columns, `.lin` N-major concat rows), one matmul into one
workspace, consumers retargeted to N-offset views (the flat-memory reshape-fold machinery already
re-decomposes offset reads; the WS1.x fold precedent). Weight concat itself is a load-time `ConstantOp`
`load_ops` chain entry (the const-folding path), so no runtime cost.

- Targets per layer: q+kv → one N=6144 K=3840 matmul (−3 launches with split-K); cut gate half + up half
  → one N=30720 matmul (−1 launch, and the combine reads two halves of one workspace). Do NOT force
  cross-family merges (e.g. into o_proj/down) — the rule's gate is shared-A + constant-B only.
- Evidence: new golden shapes (m32/m64 × both layouts × both cards) — seed via manual `--ab`, whole-twin
  verify. The N=6144 and N=30720 shapes are new keys; expect the big-N split behavior (g4a class) to
  transfer but MEASURE, not assume.
- Watch: qknorm sweeps read q/k slices of the concat workspace — the sink's flat-affine gate must still
  bind (row map now has an N-offset anchor; the binding requires anchor 0 → the sink golden may need
  re-seeding on the merged shape, or the gate extended to constant anchors ≡ 0 mod n).
- Estimate: −4-5 launches/layer ≈ −0.5-0.8 ms/step, plus better tail occupancy from bigger N.

## WS2 — kill the memset nodes (zero from the preceding kernel) [small, cheap, second]

**STATUS 2026-07-22: LANDED** — `lowering/cuda/005_delegate_zero_init` + `ZeroPrologue` stmt +
`CudaOp.zero_prologues` planner plumbing. Decode twins audit (mocked 5090): pre32 8 launches / post32 7 /
pre32-global 6 / post32-global 7, with post twins delegating 2 zero-inits each and ZERO remaining MEMSET nodes.
First-launch and symbolic-accumulator memsets are kept by design. The sink-site re-A/B this WS re-opens
(qknorm/m64 margins) is still pending — fold into the next golden pass.

Every atomic/aux buffer (`g4a` outputs, cut channel workspaces on sm_89, the sink's `__sq`) pays a
per-launch MEMSET node (~1.3 µs isolated, partially overlapped in-stream). The zero can ride the
PRECEDING launch in the same stream instead — for split forms, the partial launches strictly before the
finalize/atomic consumer.

*The move:* a `ZeroPrologue(dst)` leaf stmt (render: one designated CTA writes zeros — these buffers are
≤ a few KB) injectable into the partial's epilogue at `030_split_reduce` time when the finalize's
`zero_outputs` names a buffer the partial precedes; same for 025's `__sq` (the o_proj partial zeroes it).
`zero_outputs` then drops the delegated names. Stream ordering guarantees happen-before; CUDA-graph nodes
shrink by one per site.

- Correctness watch: the split is one partial LAUNCH regardless of `g<w>` width (all its CTAs precede the
  finalize), so delegation is safe there; the FIRST launch of a capture has no preceding kernel — keep
  its memset node.
- Estimate: ~3-5 nodes/layer removed ≈ −0.3-0.5 ms/step; also re-opens the qknorm/m64 sink sites whose
  margin the memset ate (re-A/B them after this lands).

## WS3 — norm→matmul fusion at decode M (after WS1) [A/B-gated]

Today the pre-twin input norm stays a separate kernel because its output fans out to q AND kv — fusing
into one consumer would recompute for the other. WS1's concat makes the fan-out 1, so the fused
norm→qkv computed-A form (the `norm_linear` golden kind, d2/sync rows already seeded at m32 for q/kv)
becomes offerable on the merged shape: kills `k_mean_b3bbda` (~3.4-3.7 µs) and one more launch.

- The known hazard: computed-A loses the `d2/tma/ring` weight transports (1.12 vs 1.61 TB/s at large M).
  At M=32 the seeded fused rows are competitive but NOT clearly ahead of separate norm+matmul — this is
  a per-shape A/B on the concat shape (fused d2/sync vs separate), decided by twin e2e, seeded only where
  it wins. The redundant-statistic split-K rows (#386) are the fused form's split arm.
- The post-side epilogue norms are NOT in scope: the stat-sink already moved their reduction, and the
  sweep cannot join the finalize kernel (the stat completes only after all finalize CTAs).

## WS4 — last-CTA fused finalize (stream-k style) [structural end-game, likely out of scope]

A semaphore + in-kernel fold lets the partial's last-arriving CTA run the finalize + projection —
removing every deferred-finalize LAUNCH (~3-4/layer after WS1) and giving the stat-sink a complete-value
site even on atomic forms. Big lift: new sync primitive, workspace lifetime changes, watchdog
interactions. Only start if WS1-3 leave the exit gate unmet and the twin evidence prices the remaining
finalize launches ≥ ~0.5 ms/step.

## Ordering, verification, risks

1. WS2 first if staffing is tight (smallest blast radius), else WS1 → WS2 → WS3 A/B → gate check → WS4.
2. Per WS: unit tests (hardware-free where possible) → snippet accuracy → twin e2e 3× both cards →
   golden seeding (manual `--ab`) → `eval golden --in-model` (DRIFT 0 required) → serving A/B (4K/4K c=1
   AND c=8 — c=8 amortizes per-step overhead, so wins there are smaller; do not let c=1 gains regress
   c=8/c=64 where emmy currently leads).
3. Serving boot protocol: fresh `EMMY_PACK_DIR`, empty online (`{}`), `EMMY_GEN_DECODE_BUCKET=32`; the
   fresh-pack fm boot exceeds the 1800 s bench cap — warm with a no-bench boot first; kill leftover
   `VLLM::EngineCore` by PID (never `pkill -f` with a pattern that matches your own shell).
4. Risks: WS1's consumer-retarget must not break the twin capture boundaries (q/k/v views feed vLLM's
   attention); the concat shapes invalidate existing per-projection goldens for the merged edges — keep
   the old rows (other models still trace unmerged) and seed the new keys alongside.
