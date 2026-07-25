# Serving next optimizations — c=4/small-T decode, per-kernel losers, dyn large-T (planned 2026-07-25)

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

## WS4 — dyn large-T coverage (the one-row-per-key limit)

The new dynM consumer rows are tuned at the 512 hint; ragged steps near T≈4096 reuse them (the rag
trace's grid-15360 5.2 ms class predates the fix, but the std g2k row is unverified at T≈4095).
Verify each dyn consumer at T=4095 with pinned --ab; if the 512-hint schedule is >15% off the
m4096-class winner there, the honest fix is per-Dim-hint rows (a second dyn row keyed by hint —
schema work: today one dyn key carries ONE deployable row set benched at DEFAULT_SEQ_HINT).

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
