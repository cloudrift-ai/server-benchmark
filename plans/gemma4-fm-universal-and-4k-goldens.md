# Make FAST_MATH universally ≥ std: 4K-prefill golden coverage + fm-lane evidence hygiene (RTX 5090)

Successor to the M=64 session (`plans/gemma4-m64-decode-goldens-findings.md`). Premise: fm should never lose
to std — the f16acc family is the recorded LARGE-M winner throughout the existing set (`f16_f16/w4x2/f4x8/k4
g2k` at 1.3–1.6x cuBLAS on s512/s2048), yet the 4K/4K serving A/B shows fm as a net LOSS on the long
workload. That is an evidence-coverage failure, not a kernel gap — the shapes the 4K benchmark actually runs
have no fm (often no) goldens, and the fm lane falls to the prior exactly where std falls to measured
twins.db evidence.

Baselines (2026-07-21, all same-day, `serve-ab.sh` protocol: twins.db + empty online + per-lane packs,
seed 0; 4K/4K = mml 8448, mnbt 4096, c=8, 16 prompts):

| workload | stock | emmy std | emmy fm | verdict |
| --- | --: | --: | --: | --- |
| decode c=32 TPOT med | 18.14 | 20.66 | **19.68** | fm wins |
| decode c=64 TPOT med | 20.14 | 22.49 | **21.74** | fm wins |
| 4K/4K out tok/s | 372.3 | 236.7 | 182.0 | **fm LOSES std by 23%** |
| 4K/4K TTFT mean (s) | 2.4 | 26.1 | 43.2 | **fm LOSES std by 65%** |

Step-width routing of the 4K run (`gen_runner.forward_*_device`): T ≤ 32 → static m32 decode twins (fm
seeded, wins); T == 4096 exactly → static M=4096 prefill-chunk twin (NO goldens either lane — std rides
twins.db evidence from the 2026-07-20 run, fm rides the PRIOR); 32 < T < 4096 (partial chunks + mixed
prefill/decode steps) → symbolic `.dynM` programs (evidence anchored at M=512; `ShapeKey.from_matmul`
EXCLUDES the symbolic M — `free_prod=N`, `free_max=0` — so there is ONE bucket per fork across all runtime
Ms and the M=512-measured row always wins the us sort: a "4K-anchored dynM row" is not expressible today).

## WS1 — measure first: step-width histogram + per-width fm-vs-std kernel diff

1. Instrument one 4K/4K run per lane with a step-width counter (a temp env-gated log line in
   `vllm_model_gen.forward` or parse from an `-v` boot — do NOT land instrumentation): how many steps ride
   decode / static-4096 / symbolic, and the T distribution of the symbolic ones. This decides WS3's shape
   list and tells us how much of the 23% sits on each path.
2. Capture the M=4096 prefill-chunk twins (`capture_gen_twins.py --prefill-bucket 4096 --decode-bucket 0
   --no-symbolic`, sliding + global) and the symbolic twins; `emmy run --bench` each under BOTH lanes with
   the serving evidence env. The per-kernel fm-vs-std diff table is the work list. Expect the fm laggards
   to be the prior-picked projections/MLP at M=4096 and the dynM fm picks at mid widths.

## WS2 — seed the static M=4096 prefill-chunk goldens, std + [fm], both layouts

The m64 recipe verbatim (manual pinned `--ab`, 3x medians, canonical + `.lin`): q/kv/o projections (+global
widths), mlp_down, and the MLP edge's winning form at M=4096 — test fused vs `PLACE@cone=cut` (m256 cut won
2.4x; at M=4096 the cut + `mlp_gate_up_split.m4096` halves should win again; if so seed the cut row + split
halves, m256 convention). Anchors: the s2048/m2048 std rows and the `w4x2/f4x8/k4 g2k` fm family (also try
`w4x4`/`w8x2` — at M=4096 more M warp-units may pay, cf. the m64 `(2,8)` lesson; enumerate any winner not in
`_WARP_UNITS`/`_WARP_REGS`). Include the qknorm row counts the 4096-chunk produces (k256 r65536, k512 r4096
— check cold first; seed only laggards) and rms/pw at m4096 if cold-off. Verify: twin re-bench deploys from
tier, `eval golden --in-model` stays MATCH (audit runs at buckets 32/256 — the m4096 set needs the twin
re-bench as its deploy proof, same as m64 did).

Exit gate: M=4096 chunk twins (pre+post, sliding+global) at or under their std times in the fm lane, and
std twins at or under today's twins.db-evidence times (the std lane gets its first *goldens* here too —
portable, unlike the box-local twins.db rows).

**M=64 completeness check (both faces of the regime).** The BATCHED face (decode bucket 64 — padded
64-row decode steps) is fully seeded as of the m64 session: static goldens canonical + `.lin`, std + [fm]
where fm wins, cut + split halves, lm_head/rms/qknorm/pw — verify it still deploys after this session's
YAML edits (twin re-bench, the usual no-shadowing stash A/B). The UNBATCHED face (a single sequence
putting T≈64 tokens through a prefill/mixed step) does NOT touch those goldens — it rides the symbolic
`.dynM` programs, covered by WS3's M=64 grid point. If WS3 lands option (b) (over-width static routing),
T≈64 steps pad onto the prefill-chunk twin instead and the check moves there; either way, close the gap
BY MEASUREMENT: bench a T=64 exact-width twin capture under both lanes and confirm neither lane regresses
vs its M=64 static-twin time by more than the padding overhead.

## WS3 — the symbolic mid-width path: make fm safe where it cannot be seeded

Because dynamic keys are M-blind, pick ONE of these, by measurement (WS1 decides how much this path matters):

- **(a) Universal-config re-record (no code):** re-tune every `.dynM` row (fm AND std) A/B-ing its config
  at M ∈ {64, 512, 2048, 4096} and record the config with the best worst-case vs std (minimax regret; keep
  the M: 512 anchor convention for the recorded us). M=64 is in the grid deliberately: it is the UNBATCHED
  face of the m64 regime — a T≈64 prefill chunk or mixed step rides these symbolic programs, not the
  static m64 decode twins (which only serve the padded decode-bucket path), and the c=64 SHORT protocol's
  mixed steps (T ∈ (32, 256)) live in this band too, so the symbolic configs must not fall apart at small
  M any more than at 4K. If a single fm config cannot beat std across the range on some fork, DELETE that
  fm dynM row — an absent fm row means the fm lane deploys the std config there, which is exactly the
  "fm never loses" invariant.
- **(b) Over-width static routing (small code):** route 32 < T ≤ prefill_bucket onto the static chunk twin
  by PADDING (pad → run → slice, the decode-twin convention) instead of exact-match only
  (`gen_runner.py:596` `t == self._prefill_bucket` → `t <= ...`). Kills the symbolic path for serving
  prefill entirely (mid-width steps cost a full 4096-row chunk — measure the break-even T and gate on it).
  Bigger win if WS1 shows mid widths dominate; changes serving behavior for std too, so A/B both lanes.
- **(c) M-banded dynamic keys (code, riskiest):** add an M-band discriminator to dynamic ShapeKeys. NOT
  recommended this session — it re-keys every existing dynM row and orphans the seeded symbolic tier
  (the m32 golden-orphaning lesson from WS2.1's rejection applies verbatim).

Default expectation: (a) first — it is pure dataset work and the memory notes already say the tuner is
unusable here, manual `--ab` wins. (b) only with WS1 evidence.

## WS4 — the fm-never-loses invariant, recorded and gated

1. Sweep the EXISTING fm rows (decode m16/m32/m64, prefill m256, s512/s2048, fused .lin) with a 3-point
   A/B vs their std siblings at the shapes the serving paths actually run — any fm row that loses its std
   sibling at its own anchor gets re-tuned or dropped. (Known suspects: none at anchors — the suspects are
   all off-anchor, which WS2/WS3 cover.)
2. Add the cheap YAML-level invariant to `test_golden_configs.py`: within a (name, layout) bucket, no [fm]
   row may record a HIGHER emmy_us than the best std row of the same bucket (recorded-at-anchor guard; the
   off-anchor guard is WS2/WS3's coverage, not assertable statically).
3. Re-run the full serving matrix (the four table rows above + c=32) with the final YAML. Exit gates:
   - 4K/4K: fm out tok/s ≥ std AND fm TTFT mean ≤ std; both lanes ≥ today's std 236.7 tok/s.
   - decode c=32 / c=64: fm TPOT ≤ today's (19.68 / 21.74) — no regression from any WS3 choice.
   - `eval golden --in-model`: MATCH, zero DRIFT, both cards.

## Protocol / workflow (carry-forward from the m64 session)

- Per-lane `EMMY_PACK_DIR` (`packs-std` / `packs-fm`) — the pack key lacks the fm lane. Delete stale packs
  after ANY golden edit (the pack bakes the old picks; a stale pack silently serves pre-session kernels).
- fm cold boots can exceed the `--bench` 30-min health cap: boot once without `--bench` to write the pack;
  a timed-out `--bench` leaves a zombie `VLLM::EngineCore` holding ~31 GiB — kill by pid before the next
  arm. Warm the fm cubin cache first via the offline twin benches (cuts the fm boot dramatically).
- M=4096 `--ab` benches are ~10-100x costlier per row than m64 — budget iters accordingly (`--iters 20`),
  and keep the sweep-vs-confirm split (wide sweep at 20, 3x confirm at 50 on winners only).

## Non-goals

- The computed-A async weight-prefetch pipeline (`plans/computed-a-pipeline-and-sdpa-oproj.md`) — the
  structural prefill gap vs stock (TTFT 26 s vs 2.4 s in the STD lane) stays open by design; this plan only
  makes fm stop losing to std and gives both lanes real 4K goldens. Do not conflate the two in the A/B.
- hd512 flash cold-unreachability, M=128 decode goldens, embedding-path varlen flash — separate tracks.
- 4090 parity for the new shapes — seed the 5090 first; port with the established remote flow after.
