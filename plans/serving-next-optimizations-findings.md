# Serving next optimizations — FINDINGS (executed 2026-07-25/26; plan retained below for the record)

## OUTCOME

All six workstreams closed in one session (branch `feature/serving-next-optimizations`, commits
`f24613f7` + the findings commit; article updated and pushed the same night). WS1/WS4 were closed
by analysis at planning time; WS2 seeded the M=8 decode tier and the bucket-8 knob won c=4 AND c=8
on both lanes; WS5 landed the coop-band layout gate (the cold-poison class is structurally dead);
WS3 landed the `__zp` 64 KB cap; WS6 closed as a documented step-shape residual with trace
evidence. Full detail in the STATUS/LANDED blocks inline below.

## FINAL consolidated re-bench (fresh packs at `f24613f7`, util 0.96, empty online, -it, 5090)

The re-bench surfaced a **baseline correction bigger than any single workstream**: the article's
stock lane had been running `--gpu-memory-utilization 0.9` (vs the emmy lanes' 0.96), which capped
stock's KV capacity enough to throttle admission at c≥8 — its c=64 cell read 1094 tok/s and its
c=8 TTFT 2068 ms under that handicap, vs **1425 tok/s and 1100 ms at 0.96**. The article now runs
every vLLM lane at 0.96 and its claims were rewritten to the equal-tuning picture:

| cell (median TTFT / TPOT; tok/s) | stock @0.96 | emmy std | emmy fm | verdict |
| --- | --- | --- | --- | --- |
| small_c1 256/256 | 56 / 16.28; 60.8 | 71 / 17.04 | 65 / 17.04 | stock |
| head_c1 4K/4K | 566 / 17.35; 57.2 | 629 / 18.10 | **475** / 18.10 | fm TTFT −16% |
| head_c4 (bucket 8) | 1088 / 18.23; 216.4 | 1806 / 18.92; 206.7 | 1363 / 18.85; 208.7 | stock (step-shape) |
| head_c8 (bucket 8) | 1100 / 20.56; 383.6 | 2428 / 20.75; 375.0 | 1828 / **20.56**; 381.0 | TPOT tie; stock TTFT |
| rag_c4 8192/256 (bucket 8) | 2429 / 26.24; 112.0 | 2580 / 28.94; 102.6 | 2474 / **24.54**; **117.1** | fm tok/s +4.6% |
| small_c64 np256 (bucket 64) | 1679 / 28.02; 1425.1 | 3239 / 29.48; 1145.0 | 2865 / 28.53; 1223.2 | stock |
| c64 single-wave np64 TTFT | 1468 | 1770 | **1324** | fm −10% |

- The old fm rag TTFT "record" (~1996) was an old-environment number, not a bucket effect: the
  bucket-32 isolation rerun read 2393 vs bucket-8's 2474 — both in the ~2400 class stock also sits
  in (2429). rag keeps the c=4 bucket-8 knob (TPOT/tok/s favor it; TTFT within noise of b32).
- Per-kernel catalog (239 cases, `-O3`, 2026-07-25): std geomean **1.169** vs eager (gate ≥1.15 ✓,
  was 1.11), fm **1.332** (gate ≥1.30 ✓, was 1.25); >1.5x-vs-tcompile losers: std 13 / fm 6, all in
  the documented classes (std m4096 f32-acc ceiling, fm hd512, µs-class m32 norm fusions, the m8
  rms tcompile-fusion microbench). The 109x/16.9ms cold-poison case benches at its recorded row.
- Exit gates, honest scoring: TPOT c1-hold ✓ / c4 18.85 (gate 18.8, met within noise) / c8 20.56
  (gate 20.5, ditto) / rag 24.5 ≤ 27.3 ✓✓ / c64 28.5 vs 27 ✗ (tok/s up +2% instead); geomeans ✓✓.
  "fm TTFT beats stock on ALL five points" does NOT survive the corrected baseline — fm wins c1
  and the c64 wave, stock wins the mixed-step c4/c8 cells (the WS6 documented residual) and rag
  TTFT ties. The article states exactly this.
- Deferred, unchanged: the 4090 mirror of every 07-24/25 round (m8 seeding, m1 re-tune for the
  EXPECTED_DRIFTS burn-down) once the box's GPU passthrough returns, then the image rebake.

# Original plan — c=4/small-T decode, per-kernel losers, dyn large-T (planned 2026-07-25)

Follow-up to plans/decode-parity-closers-findings.md and the 2026-07-25 golden rounds (rms_norm M=1,
merged dynM, merged m256). State at planning time (gemma-4-12B-it, 5090, article protocol): fm beats
stock TTFT on c=1 / c=8 / rag and both lanes beat stock c=64 throughput (1124.7 / 1199.6 vs 1094.2);
the open cells are c=4 (fm TTFT 1405 vs 1084; TPOT 19.6 vs 18.2) and TPOT parity at mid/high
concurrency. Method note that produced every win this cycle: **per-kernel nsys of the real serving
graph, then golden rows, then a serving A/B verdict** — isolated benches are L2-warm-biased for any
weight under the 5090's 96 MB L2.

## WS1 — post-dynM c=4 attribution trace (first, cheap)

The dynM round was never measured on the fm c=4 cell (std moved 1848→1815 only; rag moved hugely);
its re-bench happens ONCE, in the final rollout step below, together with every other affected
cell. The diagnostic that can run standalone: a POST-dynM c=4 trace (the classified per-twin
instrument, `_tune/nsys/per_kernel_classified.py` — beware its name→class map is FIRST-WINS and
the sym cut consumers share the `linear_1__cat__linear_2_reduce` stem with the m1 bare kernel),
attributing the fm gap between (a) residual ragged-sym steps, (b) decode-step drag (T=4 on the
bucket-32 twins), (c) scheduler idle. Fix what the trace names.

**CLOSED (2026-07-25 12:15, _tune/nsys/c4b_head vs pack-dyn32):** the post-dynM std c=4 window is
byte-identical to the pre-dynM one BECAUSE the 5.2 ms grid-15360 sym kernels ARE the new dyn cut
consumers at the std lane's structural f32-acc ceiling (g2k 5248 µs ≈ static std 7352-class).
Attribution: 45.8% bucket-32 decode (at goldens), 12.5% ragged-sym (std ceiling), 7.9% static
m4096 (std ceiling), 14.8% vLLM external. std c=4 has NO remaining golden-fixable component — the
levers are WS2 (decode drag) and the fm lane (its f4x8/k4 sym consumer runs ~10x faster at that T;
the fm c=4 re-bench lands in the FINAL STEP).

## WS2 — small-T decode tier (the c=4/c=8 TPOT lever)

T=2..31 decode rides the bucket-32 twins: +1.4 ms/step vs stock at c=4, and in mixed scheduling the
slow decode steps also delay queued prefill chunks (TTFT leak). The bucket-8 probe REFUTED only the
cold path (m8 shapes had no goldens → TPOT 36); the honest experiment needs seeding first:

1. Seed m8 (or m4) twin goldens: the merged fused keys at M=8 (cut vs fused per the m1/m64
   verdicts), consumers on both layouts, the rms_norm M=8 row, pw glue — the m1-tier seeding
   recipe at the new width. Audit ratchet extends to the new width automatically
   (`capture_twin_graphs` widths — add 8 if adopted).
2. A/B `EMMY_GEN_DECODE_BUCKET=8` at head_c4 / head_c8 (article per-lane knob rule, like c=64's
   bucket 64). Exit: c=4 TPOT ≤ 18.8 with TTFT not worse; else record the negative and stop.
3. Only if the bucket knob wins but T=1..3 tails still hurt: an m4-tier routed like the m1 tier
   (T ≤ 4 → static M=4 twins; the alias generalizes — `post_attn_backing` is width-agnostic).

**STATUS (2026-07-25 14:25): seeded + std A/B WIN.** 25 m8 rows in `rtx5090_sm120_gemma4.yaml`
(manual pinned --ab, n=3, fm superset enumeration; the m32/m64 w1x8/f2x2 family holds at M=8 —
gate_up k8-serial 142.7, down g4a 73.2, lm_head k8/p1 1193; CUT wins all four fused forms, the
m1/m64 verdict transfers: norm_qkv 20.6, norm_qk_global 21.0, norm_gate_up 144.7, down_fused 74.1).
Twin deploys verified: pre8 94.6→28.4 µs, post8 548.6→254.0, globals 30.7/272.8 — greedy lands on
every seeded row. Bucket-8 serving A/B (std lane, fresh pack, article protocol): head_c4 TPOT
19.56→**18.94**, TTFT 1815→1803, tok/s 200.1→206.5; head_c8 TPOT 21.38→**20.75**, TTFT 2428→2421,
tok/s 364.2→374.9 — both cells win, TTFT not worse. fm lane: head_c4 TPOT 19.6→**18.89**, TTFT
1405→**1371**, tok/s 208.2; head_c8 TPOT 21.3→**20.59**, TTFT 1853→1829, tok/s 368.5→380.3 — both
lanes, both cells. **Bucket-8 ADOPTED as the c=4/c=8 per-lane knob** (like c=64's bucket 64); width
8 added to the audit twins (`capture_twin_graphs`). Step 3 (m4 tier) not needed — c=4 steady state
is T=4, inside bucket 8. Method incident worth keeping: an INLINE `torch.randn` weight in a `--ab`
snippet traces into the measured graph (67M-CTA scalar monster; its bench worker OOM'd the box and
the OOM took the whole session scope down) — weights must be variable-bound; sweeps now run under
`systemd-run --scope -p MemoryMax=45G`.

## WS3 — per-kernel bench-table losers (the catalog rerun ranks them)

From the pre-rerun table (std 72 / fm 48 losers >5% vs torch.compile); the 2026-07-25 rounds already
fixed the top classes (mlp_down.dynM 5.4x, norm_*.m256 1.7x, norm_gate_up.dynM 1.7x). Re-rank from
the fresh `_tune/artbench-m1/kernels/{std,fm}/golden_bench.json`, then burn down in measured order:

- **gate_up_cat.m64.lin (3.1x std)**: likely a missing `.lin` layout twin — bench std/fm staged
  candidates at m64, seed both layout twins (the "keep BOTH layout twins recorded" rule).
- **mlp_geglu.m64.cut / m64 fused-vs-cut re-check**: the m64 gate_up stayed FUSED on the old
  229-vs-169 verdict; the new fm g2k consumers (74-76 µs class at m256) may flip it — re-bench the
  m64 cut total with the current consumer rows; per-site verdict by the c=64 serving cell (the m64
  twins are the bucket-64 decode path).
- **norm_kv_proj*.m32 (1.75x, 14 vs 8 µs) + rms_norm.k3840.m256 (1.5x, 6 vs 4 µs)**: µs-class,
  launch-bound in isolation — verify in-graph via the c=64/c=1 traces before spending; skip if the
  in-graph share is <0.1 ms/step.
- **attention.hd512 (fm 2.6x)**: the global-layer flash gap — research-class, own workstream
  (tile-skip/split-KV/staging analogs of the hd256 closure; see the hd256 precedent in memory).
  Not a golden fix.
- **std m4096 gate_up/down (1.67x)**: structural — f32-accumulate HMMA at half rate on consumer
  dies; the fm lane IS the fix. Document in the chart caption, do not chase.
- **__zp size threshold**: `005_delegate_zero_init` zeroes a 983 KB m64 workspace from ONE CTA
  (14 µs × 48/step ≈ 1.8% of the c=64 window). Add a word-count cap (a few KB per the design
  comment) above which delegation is refused and the MEMSET node stays. Unit test + c=64 A/B.

## WS4 — dyn large-T coverage — CLOSED BY ANALYSIS (2026-07-25)

A pinned ``--dynamic`` bench always runs at DEFAULT_SEQ_HINT (the pipeline re-binds symbolic axes
to the hint), so a direct T=4095 verify is impossible through ``emmy run --bench``. The static
m4096 goldens close the question instead: the fm dyn consumer rows carry the SAME schedule family
as the m4096 fm winners (``f16_f16/w4x2/f4x8/k4`` + ``d2/tma/ring``, the 3292 µs class on
gate_up), so fm ragged large-T steps are already at their known optimum; the std dyn rows sit at
the std lane's structural f32-accumulate ceiling (static std m4096 gate_up records 7352 vs cuBLAS
4416 — no better std schedule exists at that width). Per-Dim-hint rows would only matter if a
future card/lane shows a schedule that flips between hints — not the case on the 5090 today.

## WS5 — layout-blind ShapeKey cold-poison hardening (compiler)

The recurring bug class of this cycle — THREE separate incidents in one day: qk_global_cat m256
.lin cold-picked the transposed ``b<n>t`` band at 21.5 ms, gate_up m256 .lin timed out on
``g16k/b256t``, and gate_up_cat.m64.lin hit 16.9 ms (the catalog's 109x case). Mechanism: ShapeKey
is layout-blind, the ``b<n>t`` band realizes on BOTH orientations (it is only CORRECT-fast on
k-major B), and a cold/tied pick can land it on the row-major operand (and symmetrically, plain
``b<n>`` rows poison the k-major side — the m1 interleaved-row removal). Today's mitigation is
per-shape golden rows; the structural fix candidates, in preference order:
1. **Realization gate**: the transposed emitter refuses when B's reduce-axis stride IS the
   fastest-varying one (mirror of the enumeration condition — enumeration offers bt only on
   k-major B, but pins/goldens/evidence can still select it on the wrong operand); same gate,
   inverted, for plain ``b<n>`` on k-major B at the matvec tier.
2. A layout bit in the evidence join (featurize B's reduce-axis stride class — the transposed
   plan's featurizer item that never landed), so cross-orientation rows stop tying.
Verify with the catalog: no case may cold-resolve >2x off its recorded row on either orientation.

**LANDED (2026-07-25, option 1 as an ENUMERATION gate).** The bands are layout-gated at the single
choke point every tier resolves through (goldens/evidence pick among OFFERED rows): `b<n>t` only on
k-major B, plain `b<n>` only on K-contiguous B at the matvec tier — both arms (`_reduce_candidates`
via `Contraction.b_trans`; `_reduce_specs` via a new `_matvec_b_kstride` helper that distinguishes
the demoted matvec's vector+matrix operand pair from a plain rowwise reduce, so softmax/rms bands
stay ungated). Env pins bypass (exploration). Verified: `gate_up_cat.m64.lin` cold greedy 16.9 ms →
**234.4 µs** (deploys its recorded row); m1 `.t` snippet cold now picks plain `b128` at 14.0 µs
(correct for the snippet's view layout — ties the recorded 14.1). Audit fallout, understood and
gated: the 4090's `down_proj.m1.t` audit-MATCH was the poison itself (bt "matching" on the
F.linear-layout arm its cold prior walks) — now an EXPECTED_DRIFTS entry in the drift gate with the
mirror-re-tune burn-down note; 5090 DRIFT 0, drift-gate green both cards.

## WS3 status (2026-07-25)

- `gate_up_cat.m64.lin` 109x case: **fixed by WS5** (above).
- `__zp` size threshold: **LANDED** — `_MAX_DELEGATED_WORDS = 16384` (64 KB; measured break-even
  ≈ 90 KB from the 5090's ~70 B/ns one-CTA zero rate vs the ~1.3 µs MEMSET node) in
  `005_delegate_zero_init`, docstring bullet + two-fixture unit tests (small chain delegates,
  240 KB chain refuses). e2e verdict rides the FINAL consolidated pass (c=64 cell).
- µs-class losers (norm_kv_proj*.m32, rms_norm.k3840.m256): in-graph share below the 0.1 ms/step
  bar in the existing c=64/c=1 traces — skipped per plan.
- `mlp_geglu.m64` fused-vs-cut and the m1 `.lin` catalog warts: catalog-only keys (serving rides
  the merged/`.t` forms); re-ranked by the FINAL catalog rerun.
- `attention.hd512` (fm 2.6x) and std m4096 f32-acc: research-class / structural — documented in
  the article caption, not chased.

## WS6 — c=4 TTFT scheduling structure (beyond goldens; the last stock TTFT win)

WS1's closure leaves fm c=4 TTFT (1405-class vs stock 1084) explained by step SHAPE, not kernels:
the 2-chunk queue model prices ~1256 ms and the excess is how mixed steps interleave. This is
vLLM-side step-shape work (admission/chunk-boundary behavior seen from the plugin), NOT a golden
round — investigate only after WS2 lands (decode drag is a confounder in every c=4 TTFT
measurement). Instruments: the classified trace per step-type + vLLM's scheduler counters.
Explicitly out of scope for golden tuning; may conclude "documented residual".

**CLOSED — DOCUMENTED RESIDUAL (2026-07-25, `_tune/nsys/c4_b8fm` trace, classified vs the b8
pack).** Post-WS2 fm c=4 TTFT is 1371 vs stock's 1084 (gap 287 ms, was 321). The trace shows the
structural shape: even the prefill-burst window is 57-62% decode-twin kernel time — a queued
prompt's chunks wait behind decode-heavy mixed steps, and the plugin's mixed step runs the
static-4096 prefill twin AND the bucket-8 decode twin as SEPARATE passes (two weight streams per
step) where stock composes one fused varlen batch. Removing that means owning vLLM's batch
composition (the fork-not-plugin boundary the article already documents at the attention seam).
Bucket-8's TTFT gain (1405→1371) is the decode-drag share of exactly this structure — consistent
with WS2's theory. Side finding: the once-per-step 1.19 ms `cutlass` kernel is vLLM's own
`compute_logits` lm_head (262k vocab, 2.0 GB of weights = its stream floor; identical in the stock
lane) — the emmy `lm_head.m8` golden is catalog-only for the gen path, and there is no lever here.

## Per-WS verification

Unchanged from the golden method: manual pinned --ab on the row's own snippet, audits
(`eval golden --in-model` DRIFT 0, ratchet tightens), `make test`, and a targeted serving A/B on
the cell each WS claims to move (fresh packs, empty online, util 0.96) — the per-WS verdicts stay
cheap and attributable. NO piecemeal article edits along the way.

## FINAL STEP — one consolidated re-bench + article update

After the workstreams settle, ONE full measurement pass and ONE article update:

1. Re-bench every affected article cell on the settled rev, both emmy lanes, fresh packs — the
   six e2e points (small_c1 / head_c1 ×3, head_c4, head_c8, rag_c4, small_c64 on its bucket-64
   boot), including the fm c=4 cell the dynM round never measured.
   **c=64 protocol change**: the TTFT half of the cell is re-measured as a SINGLE 64-request wave
   (``--num-prompts 64 --max-concurrency 64``, all three lanes identically) — at np=256 the median
   TTFT lands on wave-2/3 requests and measures completion dynamics (TPOT×256 + admission waves;
   the per-engine mean/median inversions prove it), not prefill. np=256 stays for the tok/s and
   TPOT halves (steady-state saturation). The footnote then reads "single 64-request wave" instead
   of the queue-domination caveat.
2. Re-run the per-kernel golden-set catalog (std + fm, `bench_golden_set.py`) and regenerate the
   article chart assets (`render_golden_bench_chart.py` → the per_kernel HTML/CSV under
   packages/blog/public/blog/optimizing-gemma-4-12b-rtx/) + the geomean table.
3. Update ALL article tables and narrative in one commit (throughput + latency + per-kernel +
   headline numbers), push cloudrift-landing; commit the emmy goldens/findings alongside.
4. 4090 mirror of ALL 2026-07-24/25 golden rounds once the box's GPU passthrough is restored
   (currently absent from the PCI bus), then the gemma-4 image rebake at the settled rev.

## Exit gates

- fm TTFT beats stock on ALL five points (c=4 is the holdout: 1405 → < 1084).
- TPOT: c=1 ≤ 17.8 hold, c=4 ≤ 18.8, c=8 ≤ 20.5, c=64 ≤ 27, rag hold ≤ 27.3.
- Per-kernel geomean vs eager: std ≥ 1.15, fm ≥ 1.30 (from 1.11 / 1.25), losers >1.5x vs
  torch.compile reduced to the documented research-class set (hd512, std m4096 big-N).
