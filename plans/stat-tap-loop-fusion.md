# Plan: tap-only loop-level stat fusion — invert PLACE@stat's resting state

> **PREREQUISITE (added 2026-07-23):** `multi-output-node-foundation.md` lands first. With first-class multi-output
> nodes, every `AuxOutputOp` mention below becomes "output slot 1 of the producer node", and WS1/WS2's aux-node
> bookkeeping simplifies accordingly (the foundation's phase 3 rebases this plan).

## Status / context

`025_sink_row_reduce` + `_sink.py` realize `PLACE@stat=sink` by attaching a norm's row statistic (`Σx²`) to its
producer's store site AFTER the producer settles. Because the merge happens post-schedule, the pass must
reverse-engineer facts fusion once knew and threw away: the write→row correspondence is reconstructed from flat-address
arithmetic (`flat_index_expr`, the mixed-radix bijection proof, `_same_row_base`'s dense-anchor write-group proof), the
producer is probed defensively (`RuleSkipped` waits for 010/030 to settle it; atomic partials / mma epilogues / nested
stores refuse permanently), and the fold is a bespoke stored stmt (`RowAccum`, with `row = flat / n` baked in) because
no loop structure survives to carry an ordinary `Accum`.

This plan inverts the resting state, per the design discussion (2026-07-23): **fuse the statistic into the producer at
`merge_loop_ops` time as a tap** — an ordinary accumulate at the write site while the row index is still a live loop
variable — and realize the escape (`PLACE@stat=fuse`) by **cutting the tap back out** at tile lowering, 020-style.
Sinking-in needs eligibility gates; cutting-out is always legal, so the fused state is safe to make canonical. The
`PLACE@stat` fork, its evidence-only status, and the measured wins (post-attn m32: −1.4 µs/site 5090, −5.1/−6.2 4090 —
see `gemma4-stat-sink-findings.md`) must all survive unchanged.

Scope guard: **tap-only**. The sweep (projection) kernel never merges into the producer — it needs the completed row
stat, so any wide schedule requires the split anyway, and whole-B merge would force multi-real-output kernels and
mixed-role recognition. Fission the norm at loop level: stat → tap in producer; sweep → its own `LoopOp` reading
`T` + `T__sq`. Fan-out on `T` (the residual skip) does not block the tap — the tap consumes nothing, `T` stays the
producer's output.

## The target loop IR form

Stored form after tap fusion (existing stmt kinds only — the canonical tap is an **atomic-accumulate `Write`**, not a
carried `Accum`, so the host's loop roles stay unambiguous: `n` remains a pure map axis for `T`):

```
Loop m:
    Loop n:
        ... producer compute → v ...
        Write T[m, n] = v                    # T stays a real output (residual reads it)
        sq = mul(v, v)                       # the consumer's per-cell term, on the pre-store value
        Write T__sq[m] += sq                 # the TAP: atomic-accumulate, index = live row var
```

plus a separate sweep `LoopOp` (`out[m,n] = T[m,n] · rsqrt(T__sq[m]/N) · w[n]`) and the `AuxOutputOp` node for
`T__sq` (existing machinery — planned/allocated, `zero_outputs` memset, no launch of its own).

`RowAccum` stops being a stored IR-input concept. Its hierarchical rendering (warp shfl → smem stage → ~1 atomic per
block) survives as **tier 3 of a derived lowering** of the tap, selected by what the schedule did to the tapped axis —
the same derive-never-store rule cooperative reduces follow:

1. tapped axis in-thread → serial register accumulate + plain store (no atomics, no zero-init);
2. tapped axis within one CTA → smem block fold + one plain store per row;
3. row spans CTAs (the realistic matmul grid) → today's `RowAccum` rendering + per-launch zero-init.

## Workstream 1 — the tap as a derived lowering (generalize `RowAccum`)

- **1a.** Define the tap's stored form: atomic-accumulate `Write` with a structural marker sufficient for recognition
  to peel it (decision point below). The destination index is the live row expression — the `flat / n` arithmetic,
  `SinkBinding.row_coeffs`, and the static-shape gates all dissolve. Symbolic row extents become expressible for free.
- **1b.** Materializer: derive the fold tier from the settled schedule (the three tiers above). Reuse `RowAccum`'s
  existing rendering code as tier 3; tiers 1–2 are new but small (tier 2 is the coop-reduce smem fold re-targeted at a
  side buffer). Zero-init (`zero_outputs`) is emitted ONLY for tier 3 — tiers 1–2 erase the memset floor that ate the
  qknorm/m64 sites (the findings' top follow-up, solved structurally here instead of by editing partials).
- **1c.** Carrier generality while we're here: key the tap on the fold op (`add` today; `max`/`min` have atomics too)
  instead of hardcoding add — cheap now, bespoke later. Multi-channel (LayerNorm's `Σx, Σx²`) = N taps into an
  N-column aux buffer; the product-monoid shape `Contraction.folds` already models.

**Decision point (resolve in 1a before anything else):** how the peel identifies a tap. Options: (i) purely structural
— an atomic accumulate `Write` to an aux-named buffer whose index omits the innermost map axis; (ii) a lightweight
`Tap` grouping stmt wrapping term-chain + write. (i) keeps the IR vocabulary closed; (ii) makes the peel and the
cut-out realizer trivial and self-describing. Lean (ii) only if (i)'s structural definition turns out ambiguous
against real bodies (split-K partials also write atomically — the aux/naming contract must distinguish).

## Workstream 2 — the loop-level tap fusion rule

New rule in `loop/fusion/` (after `010_merge_loop_ops` in scan order, e.g. `015_tap_row_stat.py`):

- **2a.** Structural binding on the RAW loop form (the sibling of `bind_sinkable_stat`, minus all address algebra):
  consumer `B` contains a reduce loop over producer `A`'s output buffer with a single scalar `Load`, a pure `Assign`
  chain, one additive `Accum`, and a trailing non-reduce sweep loop reading the same buffer. The index correspondence
  comes from the same σ-solve `splice_graph` already does — anchored at `A`'s `Write` instead of a `Load`.
- **2b.** Rewrite: fission `B` (stat / sweep), splice the term chain + tap after `A`'s write, emit
  `A' + AuxOutputOp(T__sq) + sweep-LoopOp` — the same three-fragment graph 025 emits today, built constructively.
  Producer eligibility is by construction (the write site is in the same body); the `RuleSkipped` waiting dance,
  settled-`TileOp` probing, and the `__sq`-exists check all disappear.
- **2c.** Guards and interplay: `A` must be an in-graph `LoopOp` (not a graph input — the input-norm refusal carries
  over by construction); `A` being a graph output is FINE here (unlike merge — `T` stays materialized); check the
  blowup metrics stay quiet (the tap adds O(numel) work — well under `_BLOWUP_FACTOR`); extend `_CUT_WS_RE` /
  naming contracts so a later decided cut-out is not re-fused by `merge_loop_ops` on the restarted scan.
- **2d.** Rule ordering: the tap rule must fire only after `B` has fully assembled (its own reduce+sweep merged) —
  gate on the bound shape, which is only matchable once assembly is done, so no explicit ordering hook is needed;
  verify against the traced decomposition order on TinyLlama + gemma-4.

## Workstream 3 — recognition peel + the inverted `PLACE@stat` realizers

- **3a.** `010_recognize` peels taps before classifying the host body, so a tapped matmul/pointwise recognizes
  EXACTLY as its untapped self — same structural nodes, same fork keys, same golden identity. The peeled taps ride
  the `TileOp` as decoration and re-attach at materialization (WS1b). This is the one genuinely new recognizer
  surface; keep it a pure pre-pass (strip → classify → stamp back).
- **3b.** `PLACE@stat` fork stays, offered where the tap is present. `sink` rows = tap retained (evidence-only,
  unchanged clauses in `greedy_decide` / `evidence_row_vouches`). **`fuse` (option-0) = the cut-out realizer**: a
  020-style fragment that extracts the tap, reconstructs the local-stat norm `LoopOp` (tap term + `Reduction` +
  sweep re-welded, un-mapped) and the untapped producer, both re-entering `010` — landing on today's coop norm
  schedule bit-identically. This reconstruction is the price of the inverted resting state; it must round-trip
  exactly (test: fused → option-0 reproduces the current deployment's kernels and golden MATCH row for row).
- **3c.** Golden/evidence key migration: today the `linear_norm` golden kind records `{PLACE@stat: sink, …}` at the
  NORM's fork. After inversion the decision keys at the tapped PRODUCER's fork. Map or re-record the seeded goldens
  (post-attn m32 both cards); update `LinearNormGoldenConfig` keying; the permanence test in
  `tests/compiler/test_golden_configs.py` must hold across the move. This is the churn-heaviest item — budget it.
- **3d.** Retire `025_sink_row_reduce` + `_sink.py` once 3b/3c land (the realizer inverts; nothing calls the old
  binding). `RowAccum` the stored stmt shrinks to the tier-3 rendering entry point (or is renamed accordingly).

## Workstream 4 — schedule-transform interplay

- **4a.** `030_split_reduce`: a tap cannot ride a partial (no thread holds the complete value — atomic partials).
  Relocate taps to the FINALIZE, the same rule 030 already applies to `Map`-wrapper projections. This replaces
  025's permanent refusal of split producers with a constructive move — split-K sites gain the sink option they
  never had.
- **4b.** `020_cut_edge` on a tapped host: cutting the cone out of a tapped fused-norm→linear edge must carry or
  re-site the tap coherently (simplest: the cut-out realizer (3b) runs first when both stamps demand it; verify the
  stamp combinations that co-occur on real forks).
- **4c.** mma `RegStore` epilogue hosts (the old v2 arm): with the tap as decoration, the epilogue arm is "attach
  tier-3 fold after the fragment store" — likely small now, but keep it OUT of this plan's gate; decode sites are
  covered by finalize/pointwise hosts today.

## Verification (gates, in order)

1. Unit: tap-fusion binding + rewrite (gelu producer, f4 write group, per-head qknorm row map, residual fan-out on
   `T`, input-norm refusal, graph-output producer); peel round-trip in 010; option-0 reconstruction bit-parity.
2. `emmy eval golden --in-model` on both cards: **MATCH 105 / DRIFT 0 / GAP 0** must hold through the migration (3c).
3. Accuracy: twin runs + snippet forms (plain / residual / grouped per-head) PASS vs eager — same reassociation
   class as today (f32 tree over pre-round values).
4. Perf retention: post-attn m32 twin e2e keeps its win on both cards (5090 ≈ −1.4 µs/site, 4090 ≈ −5–6 µs/site);
   re-A/B the qknorm and m64 sites — tiers 1–2 (no memset) may flip them from LOSS to win, which would be the first
   new capability unlocked.
5. `make test` + `make lint`; the golden-permanence test extended across the key migration.

## Non-goals

- Whole-B merge / multi-real-output kernels / mixed-role (`Map`-with-fold-channels) recognition — same end state,
  strictly more surface; shelved.
- Dependent stat chains (softmax's `Σ exp(x − max)`): algebraically un-sinkable — the per-cell term depends on
  another statistic's final value. Note it in the tap rule's docstring so it reads as impossibility, not TODO.
- Loop-level fork-aware fusion: the tap fuses unconditionally BECAUSE the cut-out escape is always legal; no fork
  machinery moves into the loop dialect.

## Risks

- **3c golden migration** is the likeliest breakage point (fork-key churn → silent DRIFT). Mitigate: land WS1+WS2
  behind the peel with option-0 forced first (pure refactor, kernels unchanged), migrate keys as a separate commit
  with the eval-golden gate on both cards.
- The peel's tap-identification ambiguity (WS1 decision point) against atomic split-K partials — resolve the naming/
  marker contract before writing the peel.
- Dump/kname churn: tapped producers change `EMMY_DUMP_DIR` artifacts and reproducer slicing; check
  `<kname>.torch.json` provenance still slices the tap to the norm's ops.
