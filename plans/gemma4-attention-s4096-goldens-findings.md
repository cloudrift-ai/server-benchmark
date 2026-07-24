# gemma-4-12B attention .s4096 golden seeding + blog-chart regen — findings (2026-07-23/24)

Session goal: regenerate the blog's per-kernel charts (both cards) with a FAST_MATH lane and the 4K-prefill
shape set. The matmul/norm `m4096` tier was already seeded (2026-07-21 session); the gap was **attention
hd256/hd512 at seq 4096** on both cards — seeded here via the tune-golden workflow (fm-superset tune +
pinned/greedy A/B, evidence files backed up before the tune and restored before the chart bench so the bench
stays pure deploy picks).

Boxes: vast.ai RTX 4090 (driver 580.159.03) and RTX 5090 (driver 580.119.02), CUDA 12.8 (nvcc V12.8.93),
PyTorch 2.13.0+cu130, repo at origin/main `b858ed61`. CloudRift had zero 4090/5090 capacity (503 on all 8
candidates; one 4090 rental died in provisioning) — both cards fell back to vast.ai per the runbook.

## RTX 4090 (rtx4090_sm89_gemma4.yaml)

| shape | lane | config | emmy µs | torch SDPA µs | ratio |
| --- | --- | --- | --: | --: | --: |
| attention.hd256.s4096 | std | serial d1/cp/alt, dd f1x8/k16, pj f32 f1x32/k4 | 1205.2 | 1068.8 | 0.89x |
| attention.hd256.s4096 | fm | same, PV atom f16acc | 1077.2 | 1068.8 | 0.99x |
| attention.hd512.s4096 | std (pin-only) | s2048's split-KV g2k | 3514.0 | 3457.0 | 0.98x |
| attention.hd512.s4096 | std (deploy floor) | serial (same tiles) | 4587.5 | 3457.0 | 0.75x |

- The s2048 families transfer wholesale: hd256 keeps serial d1/cp/alt (g2k pins lose: std 1359.7, fm
  1250.0); hd512's split-KV g2k stays the true winner exactly as at s2048.
- **⚠ hd512 g2k OFFER GAP (the session's main compiler finding): at s4096 the split-KV candidate is not
  enumerated un-pinned** (2048-CTA grid → no starvation → no `REDUCE=g2k` offer), so the g2k golden benches
  only via pin and the golden floor logs `no offered candidate realizes any of them — falling through`. A
  SERIAL sibling row was added as the deployable floor (same pattern as `mlp_down.m4096`'s serial sibling).
  The 23% gap (4587.5 → 3514.0) is this shape's open lever; fixing it means offering split-KV past the
  grid-starvation gate (or a shape-conditional offer) in the flash enumeration.
- Golden verify replay (`run --bench --golden`): all pins reproduce, no `pin_unmatched`, no scalar-fallback
  flatten (hd256 std 1285 / fm 1148 / hd512 g2k 3521 — within the ~10% live-rebench band); after the serial
  sibling landed, the floor decides the deploy (fall-through message gone, greedy = serial 4566).

## RTX 5090 (rtx5090_sm120_gemma4.yaml)

| shape | lane | config | emmy µs | torch SDPA µs | ratio |
| --- | --- | --- | --: | --: | --: |
| attention.hd256.s4096 | std | serial d1/cp/alt, dd f1x8/k16, pj f32 f1x32/k4 | 871.5 | 792.8 | 0.91x |
| attention.hd256.s4096 | fm | d2/cp/ring/p2, dd f1x2/k16, PV f16acc f1x32 | 736.5 | 792.8 | **1.08x** |
| attention.hd512.s4096 | std | serial d1/cp/alt (the s2048 family) | 3521.4 | 3672.7 | **1.04x** |

- hd512 serial now BEATS torch SDPA at s4096 (1.04x; at s2048 it was parity) — the split-KV inversion the
  5090 showed at s2048 holds at 4K, so no g2k probe was needed and there is no offer-gap issue here.
- The hd256 fm lane found a genuinely different form (deeper d2 ring staging + f16acc PV) that beats SDPA by
  8% where the std lane sits at 0.91x — the fm tune was not just an atom swap on this card.
- hd512 fm greedy realizes the all-f32 config — no `[fm]` row recorded (would violate nothing, but it would
  be a duplicate std row; `GoldenConfig.fast_math` derives from knobs).

## Blog chart regen (the consuming workflow)

- Chart shape set is now **26 per card**: 13 seq-512 base + 13 4K-tier (`.m4096` matmuls incl.
  `mlp_gate_up_split.m4096` as the gate_up route at that M, `.s4096` attention, `rms_norm.k3840.m4096`,
  `qknorm.k256.m65536` / `qknorm.k512.m65536`), dropping the old `.s2048` tier. Bench:
  `_tune/blog-chart/bench_chart.py` (a NAMES-pinned wrapper of `scripts/bench_golden_set.py`'s
  `run --bench --golden` loop), once per lane; evidence files = the dev box's `autotune.db` + `prior.json`
  restored to pre-tune state, so emmy rows are pure deploy picks.
- 4090 headline (26 shapes): std 10/26 ≥ eager, geomean 0.88x (0.92x excl hd512), best 1.25x, p90 1.07x,
  worst 0.33x (attention.hd512). FAST_MATH: 21/26, geomean 1.11x (1.13x), best 1.40x (q_proj_global), p90
  1.38x, worst 0.63x (k_proj_global). No shape where fm < std beyond noise.
- 5090 headline (26 shapes): std 10/26 ≥ eager, geomean 0.94x (0.96x excl hd512), best 1.12x (o_proj), p90
  1.03x, worst 0.58x (attention.hd512). FAST_MATH: 22/26, geomean 1.19x (1.24x), best 1.67x
  (mlp_down.m4096), p90 1.64x, worst 0.52x (attention.hd512). FM-loses flags: attention.hd512 @s512 (0.52
  vs std 0.58 — the only real fm regression, the shape is already the chart floor) and qknorm.k256.m65536
  (0.92 vs 0.945, within the ~3% noise band). Also notable: rms_norm.k3840.m4096 deployed at 0.69x both
  lanes on this host (the recorded b128 golden realizes; eager is simply faster on this box/driver) and
  qknorm.k512.m65536 measured 1.00x (the recorded 0.78x "emmy-loses guard" did not reproduce on this host).
- Charts render via `emmy.visualize.bar_chart` (3 series: emmy std, emmy FAST_MATH, torch.compile; ratios
  normalized to the std-run eager; tooltips carry raw µs for all four backends) into both cloudrift-landing
  public dirs; article Headline tables gained a FAST_MATH column.

## Workflow notes

- vast.ai: the account-key injection regression persists — self-inject via `--onstart-cmd`, but on one host
  (Hungary, offer 42490702) even the onstart injection never took (SSH permanently denied) → destroyed. Add
  `-o IdentitiesOnly=yes` to every ssh/scp/rsync: one host's MaxAuthTries rejected multi-key agents.
- Host network quality is the real rental lottery: a California host advertising 1.5 Gbps delivered
  ~0.3 MB/s to PyPI (2h of wheels, then the ssh-tethered `make setup` died with the connection — always
  `nohup` remote installs). The replacement (offer 43249409, 14 Gbps) finished the whole venv in 5 min.
- `emmy tune -c <sdpa snippet>` under `EMMY_FAST_MATH=1` measures both regimes in one sweep (gate-on
  enumeration is a superset) — one tune per shape, then per-lane greedy `run --bench --json` A/Bs; record
  from pinned/golden rows, not the greedy row.
