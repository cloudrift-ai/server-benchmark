# Staged transposed-B: serving `.lin` golden re-tune findings (RTX 5090 + RTX 4090, 2026-07-20)

Session goal: fix the serving-layout (class-1) golden gap — the `.lin` (`F.linear`, `trans_b`) matmul forks ran
gmem-direct only because both cp.async and TMA declined a transposed B, leaving them 1.3–2.75× behind cuBLAS while
their canonical staged twins sat at ~1.0×. This session landed staged transposed-B support and re-tuned every `.lin`
golden on both cards.

## The fix (branch `feature/staged-transposed-b-serving`)

A transposed B (`(N, K)` gmem, K contiguous) stages into an **N-major slab** — `tile_n × bk`, the operand's own
orientation, K stride-1 in gmem and smem alike, i.e. exactly the A operand's geometry. The fill needs no new
machinery (chunk contiguity and row-base alignment hold automatically: B's row stride K is a multiple of `bk_elems`,
which the chunk width divides), and the drain is the plain no-`.trans` ldmatrix — the staged
`LdmatrixLoad(b_trans=True)` path flash's K slab already rides; the `096` pairing peephole covered the N-adjacent
x4-pair form already. Changes: the three `b_trans` rejects in `tile/_schedule.py` widen (warp cp.async + TMA;
alignment gates move to the K chunk since both operands' inner span is then `bk_elems`), `_atom._slab_operands`
flips B to A's geometry under `b_trans` with an `Operand.trans` layout stamp the drain reads (sync compute-fill
slabs stay K-major — the stamp, not `c.b_trans`, drives the drain), and the B swizzle mode derives from `bk_elems`
on the async transports. Scalar tier still declines (no transposed plain-`Load` drain; pin-only tier). Staged output
is bit-identical to gmem-direct; accuracy PASS across cp/tma × f16acc × split-K × `p2` × dynM; suite 2580 passed.

## Workflow note: tuner replaced by manual pinned `--ab`

`emmy tune` was started first (scratch DB/prior, `--dataset golden --kernel .lin`, FAST_MATH superset) but was
killed on user instruction — on shape 1 it spent 290 s / 229 benches with 83% of warmup benches ≥2× off the best
and only 1 post-warmup bench; the per-shape wall time would have dwarfed the value. The replacement was the manual
method (see `manual-golden-sweep-method`): per shape, one `emmy run --bench --golden NAME --ab <cand> --ab …` with
a hand-built candidate grid, 3 reps, record from the ab/golden rows' median (never the greedy row). Candidate grid
per shape: the canonical staged twin's knobs verbatim (std + fm lanes), a cp.async transport variant, and the
recorded gmem-direct tile with staging added. The grid is tiny (3–5 candidates) because the hypothesis was sharp:
the canonical twin's tile should transfer to the trans_b fork once staging realizes there. It did — on the 5090
**every std winner is the canonical twin's tile**; the old `.lin` tiles with staging bolted on trail by 1.5–3×
(they were tuned in a gmem-direct-only world and encode its trade-offs).

## RTX 5090 results (local, 3 reps, spread ≤2%)

All 8 `.lin` shapes re-recorded; std lane replaces the gmem-direct anchors, `[fm]` added beside std where the
f16-accumulate lane wins or ties (7 of 8).

| shape | old µs (x) | new std µs (x) | new fm µs (x) | winner knobs (std) |
|---|---|---|---|---|
| o_proj.m16.lin | 21.4 (0.65) | 9.1 (1.54) | 8.9 (1.57) | w1x8/f2x2/k2 g8k d2/tma/ring |
| o_proj_global.m16.lin | 37.6 (0.77) | 15.7 (1.85) | 15.6 (1.86) | w1x8/f2x2/k4 g4k d2/tma/ring |
| mlp_down.m16.lin | 233.8 (0.36) | 73.8 (1.15) | 73.7 (1.15) | w1x8/f4x1/k4 g4k d2/tma/ring |
| o_proj.dynM.lin | 198.5 (0.49) | 94.3 (1.03) | 73.9 (1.31) | w4x2/f2x4/k2 g4k d2/tma/ring |
| o_proj_global.dynM.lin | 306.1 (0.52) | 179.3 (0.89) | 136.8 (1.16) | w4x2/f2x4/k2 g2k d2/tma/ring |
| mlp_down.dynM.lin | 596.9 (0.49) | 334.8 (0.88) | 249.0 (1.18) | w2x2/f4x8/k2 g8k d2/tma/ring |
| k_proj_global.m256.lin | 22.4 (0.53) | 12.4 (0.95) | — (fm loses: 21.1) | w2x2/f2x4/k2 g4k d2/tma/ring |
| q_proj_global.m256.lin | 217.7 (0.52) | 101.7 (1.11) | 81.4 (1.39) | w2x2/f2x4/k2 g4k d2/tma/ring |

(x = cuBLAS/emmy off the recorded `cublas_us`.) Observations:

- **TMA beats cp.async by 8–25% on every 5090 shape** — consistent with the canonical twins (all d2/tma/ring).
- The decode (m16) shapes flip from the card's worst class to clear wins (up to 1.85× over cuBLAS): the tall-N
  `w1x8` tiles + the weight-stream prefetch ring is exactly what gmem-direct could not express.
- The prefill dynM std shapes land at 0.88–1.03× (fm 1.16–1.31×) — mirroring their canonical twins' std ratios
  (0.9–1.0×), i.e. the layout gap is fully closed; what remains is the ordinary std-vs-fm atom gap.
- fm loses only on k_proj_global.m256 (N=512): the f16acc tile family (f4x8, tile_n 256) over-splits narrow N.
- Deploy verified: `run --bench --golden` greedy now resolves the recorded configs from the golden tier (e.g.
  mlp_down.m16.lin greedy = recorded knobs at 72.6 µs live).

## RTX 4090 results (riftuser@176.124.69.204, sm_89 — cp.async only)

All 23 `.lin` shapes re-recorded (69 shape-runs, 0 failures; spreads ≤1% except two ~6–7% rows). Every winner is
staged `d2/cp/ring` (TMA stays sm_90-gated); every shape roughly halves its latency vs the old gmem-direct entries.

| shape | old µs (x) | new std µs (x) | new fm µs (x) |
|---|---|---|---|
| q_proj.m32.lin | 39.4 (0.49) | 12.5 (1.55) | 12.5 (1.55) |
| kv_proj.m32.lin | 21.2 (0.55) | 8.9 (1.30) | 7.8 (1.49) |
| o_proj.m32.lin | 40.9 (0.50) | 12.8 (1.59) | 12.6 (1.62) |
| mlp_down.m32.lin | 498.9 (0.29) | 139.8 (1.05) | 251.1 (0.58, lane-internal 491→251) |
| q_proj_global.m32.lin | 75.2 (0.48) | 25.9 (1.39) | 25.8 (1.40) |
| k_proj_global.m32.lin | 14.7 (0.39) | 5.9 (0.97) | 5.5 (1.04, lane-internal 14.7→5.5) |
| o_proj_global.m32.lin | 78.0 (0.61) | 22.0 (2.18) | 21.9 (2.19) |
| q_proj.m256.lin | 115.4 (0.48) | 77.5 (0.71) | 58.3 (0.95) |
| kv_proj.m256.lin | 63.8 (0.49) | 41.3 (0.75) | — (loses) |
| o_proj.m256.lin | 118.9 (0.44) | 80.7 (0.65) | 61.2 (0.86) |
| mlp_down.m256.lin | 566.3 (0.38) | 310.3 (0.70) | 303.0 (0.72) |
| q_proj_global.m256.lin | 218.9 (0.49) | 127.0 (0.84) | — (loses) |
| k_proj_global.m256.lin | 25.7 (0.43) | 16.9 (0.66) | — (loses) |
| o_proj_global.m256.lin | 228.5 (0.45) | 162.1 (0.63) | 107.3 (0.95) |
| mlp_ch.m256.lin | 518.7 (0.42) | 243.5 (0.90) | — (loses) |
| q_proj.lin.dynM | 197.3 (0.53) | 104.7 (1.01) | 80.8 (1.30) |
| kv_proj.lin.dynM | 106.7 (0.47) | 57.1 (0.87) | 44.7 (1.11) |
| o_proj.lin.dynM | 207.0 (0.52) | 111.0 (0.97) | 81.9 (1.32) |
| mlp_down.lin.dynM | 784.1 (0.50) | 439.6 (0.90) | 342.2 (1.15) |
| q_proj_global.lin.dynM | 409.1 (0.50) | 225.5 (0.90) | 151.9 (1.33) |
| k_proj_global.lin.dynM | 39.4 (0.45) | 21.0 (0.84) | 17.8 (0.99) |
| o_proj_global.lin.dynM | 419.0 (0.49) | 239.1 (0.86) | 195.3 (1.05) |
| mlp_ch.dynM.lin | 978.4 (0.40) | 515.1 (0.76) | 513.0 (0.76) |

Observations:

- **The 5090's decode-tile discovery generalizes**: the tall-N `w1x8` tiles win every m32 shape except the
  narrow-N `k_proj_global` (N=512), where the old `w1x4` tile + staging wins — same narrow-N exception as the
  5090's fm-loses-on-k_proj pattern.
- The decode (m32) shapes flip to wins (1.05–2.18× std); the m32 fm lane mostly ties std (bandwidth-bound).
- **m256/prefill std stays behind cuBLAS (0.63–0.90×)** — this is the card's known std-lane prefill pattern, not
  a layout residue: the fm entries land at 0.72–0.95×, tracking the canonical twins' ratios. On sm_89 the
  big-tile f16acc family is where wins live (PR #350), and the `.lin` forks now inherit it.
- `d2/cp/ring/p2` (the smem→register double-buffer) wins two std dynM shapes (q_proj, o_proj) — mirroring their
  canonical twins' recorded configs exactly.
- `mlp_ch` (the N=15360 gate/up split channel) is the weakest family both layouts (0.76–0.90× vs the canonical
  twin's 0.89×) — a shape-level std residual, in the same bucket as the 5090 plan's step 3.
- Deploy verified on-card: `run --bench --golden mlp_down.m32.lin` greedy resolves the recorded staged config
  from the tier (137.7 µs live, 1.00× vs eager; the old entry read 498.9).

## Plan: remaining 5090 laggards (post-.lin state: 29 of 143 (name, lane) pairs behind >2%)

Ordered by lever size. Each step has a measurable exit criterion; ratios quoted are recorded cuBLAS/emmy.

### 0. Live re-bench triage (no code, ~1 h)

The 29-laggard list reads the recorded YAML; small shapes swing ±5–10% live and one entry is a known-crude pin.
Run `emmy run --bench --golden NAME` (3×) over the 29 names and re-record any entry whose live median moves >5%
— in particular `mlp_geglu.dynM.cut` (0.51×, a crude pinned cut recorded before the fragment goldens existed; a
plain re-bench should land it near the 4090's pattern). Exit: the laggard list re-ranked on live numbers, stale
records purged.

### 1. Computed-A async-B staging — class 2, the big lever (0.50–0.93×, 7 shapes)

The fused norm→linear / gate⊗up forms run the mandatory `sync` compute-fill, whose **transposed-B channels fill
per-cell** (strided gather, no prefetch) — the measured 1.12 TB/s vs the 1.61 TB/s a clean `d2/tma/ring` sibling
reaches on the same weight stream. This session built exactly the missing piece for the plain tier: the N-major
`(tile_n × bk)` B slab with K stride-1 and the `Operand.trans` drain. Wire it into the sync transport:

- `_atom._sync_operands`: the `if c.b_trans:` branch currently builds a per-cell `SyncOperand` copy fill; instead
  build an A-geometry `Operand` (`shape=(tile_n, bk_elems)`, `tile_is_row=True`, `trans=True`, swizzle from
  `bk_elems`) on `async_operands` — the cp.async fills then fly UNDER the compute fill, one slab per fold channel
  (the gate⊗up node), exactly like the canonical-B path today.
- `_schedule._resolve_sync_stage`: drop the `not c.b_trans` term from the `depth` clause so the asymmetric B-only
  `d2` ring is enumerable on transposed-B fused edges (stays a fork sibling; d1 remains option-0). Budget math is
  already per-channel.
- Watch the two known cliffs: smem occupancy quantization (the d2 ring lost on M=512 gate⊗up before — keep d1/d2
  as measured siblings, never hardwire) and TMA-B contraindication on 2-channel nodes (WS1a: d2 halves occupancy
  on gate⊗up — cp.async only is fine for the first pass).
- Exit: `norm_kv_proj_global.m32` (0.50×), `norm_kv_proj.m32` (0.60×), `norm_q_proj.m32` (0.68×),
  `norm_q_proj_global.m32` (0.89×), `mlp_geglu.m32` (0.92×) re-tuned by the same manual pinned `--ab` method;
  target ≥0.9× on the norm_* decode shapes (they anchor a fused-vs-split loss today — even parity flips the
  tuned decode form back to fused). Accuracy: staged sync output must stay bit-identical to the per-cell fill.

### 2. hd512 flash attention — class 3 (0.87× static, 0.89× dynM)

Standing cold-reachability/enumeration gap from the golden sweep (hd512 = 1 KV head, k_eq_v; symbolic split-KV
not built). Separate targeted session: (a) audit which warp-flash forms the hd512 shape can even enumerate
(the sweep's "cold-unreachable from tier" note); (b) port the hd256 winners' levers — tile-skip, split-KV,
alternating staging with the shape-dependent transport preference (hd256→cp, hd128→tma); (c) if the dynM form
needs symbolic split-KV, scope it first — it is the one item here that is new lowering, not tuning. Exit: hd512
static ≥0.95×, dynM ≥0.9×, goldens re-recorded. (hd256.dynM std at 0.96× rides along for free if the transport
preference generalizes.)

### 3. Canonical matmul tail (0.81–0.97×, ~8 shapes)

`mlp_gate_up_split.m256` / `mlp_ch.dynM` / `kv_proj.m256` at 0.81× lead; then `q_proj.m256` 0.91×,
`k_proj_global[.dynM]` 0.93×, the s2048 prefill GEMMs at 0.96–0.97×. Not layout-related — ordinary std-lane
tuning/codegen residuals (every one already has an fm entry at or above parity). Method: manual pinned `--ab`
neighborhoods around the big-tile f16acc family's std siblings (w2x2–w4x2 × f4x{4,8} × k{2,4} × d2/tma-vs-cp),
one session, record >3% wins only. The `.lin` dynM std residue (`mlp_down.dynM.lin` 0.88×,
`o_proj_global.dynM.lin` 0.89×) belongs to this same bucket — it now mirrors the canonical twins' std ratios, so
it moves (or doesn't) with them; no serving-specific work left there.

### 4. rms_norm.k3840.m32 — class 4 outlier (0.73×, 5.6 µs)

Launch/occupancy overhead dominates a 5.6 µs kernel; the rest of the memory-bound kinds sit at 0.93–0.97× (the
bandwidth floor — accept). Cheap probes only, timeboxed: `REDUCE` b128/b64 vs b256, row-per-warp vs row-per-block
grid shaping. If nothing clears ~0.9×, record the loss as the anchor and stop — the absolute gap is 1.5 µs.

### Standing caveats

- The layout-blind ShapeKey caveat is now SYMMETRIC: staged configs realize on both layouts, so a stale twin on
  either side deploys cross-layout with a foreign µs. Keep layout twins current together; the real fix is still a
  layout signal in the stamped `S_*` features + ShapeKey.
- The `.lin` retune used knob-transfer from canonical twins rather than a tuner sweep; if the tuner's post-warmup
  stall (229 benches / 1 post on shape 1) reproduces elsewhere it deserves its own investigation.
