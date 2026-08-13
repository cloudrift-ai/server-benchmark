# Pipeline History: incidents and retired designs

[`ARCHITECTURE.md`](ARCHITECTURE.md) describes how the pipeline works today and states each rule's reason in one
line. This file holds the longer stories behind those rules — the shipped bugs that motivated them and the designs
that were tried and removed. Read it when you want to know *why* a guard exists in the detail the architecture doc
deliberately omits; nothing here is needed to work on the code.

Entry titles are stable — ARCHITECTURE.md references them by name. When you retire a guard, delete or amend its
entry here in the same change.

## Deploy-pick incidents

### Placeholder dtypes read off a rebuilt op

A scalar tile shipped for gemma's `o_proj` at 16× the kernel's measured mma rows. Root cause: within one rule batch,
an earlier apply swapped a consumed node's op for a rebuilt instance still carrying its `(f32, ())` seeding
placeholders. `Match.is_alive` checks node identity, not op content, so the later rule's `_warp_atoms` read
placeholder dtypes off an all-f16 graph and declined the warp tier. Guard now: op `inputs` / `outputs` are refreshed
from the graph at match build AND again at apply time (`Candidate.try_rewrite`) — see "Writing a rule" in Part 1.

### The saturated-score plateau (gemma cold misdeploys, 12–29×)

Cold deploys on a real RTX 4090 shipped degenerate `w1x1` tiles measuring 12–29× off the golden, while the
golden-rank eval reported top-1 on the same shapes. Two compounding defects:

- The offline prior's exponential latency proxy carried a ±80 *quality* clip that sat inside the live score range:
  every good warp tile collapsed onto one `exp(-8)` plateau, and greedy's argmin fell through to emission order.
- The golden-rank metric counted only strictly-better rows, so every row inside a tie plateau reported rank 0 — the
  eval could not see the saturation.

Guards now: the exponential clips only at the float-safety bound (~±700) and **must never saturate over live quality
scores** (the one bounded-value consumer, `FallbackPrior`'s offline multiplier, clamps on its own side); golden rank
is tie-pessimistic (ties count as losses, `prior/fit/rank.dual_rank`), and the optimistic/pessimistic gap is reported
as a saturation canary. This incident is also why goldens became the first evidence tier of a greedy compile: the
per-GPU golden files are the only measured data a fresh machine has, so a cold deploy must consult them before any
model extrapolation. See Parts 3 and 8.

### The `D_pow2_threads` cold-deploy pick

A cold deploy picked a 686-thread block because the offline fit had put an arbitrary large weight on
`D_pow2_threads`. The rank objective is flat in the magnitude of any feature with tiny variance across the golden
candidate pools, so the unpenalized fit chose freely there — invisible in golden-rank metrics, catastrophic at fork
scoring, where an undecided prefix scores the feature 0.0. Guard now: the raw-space L2 penalty in the offline
fitter's loss, whose job is identifiability, not shrinkage (raw-space because the inflated raw weight is an ordinary
O(1) z-space weight). See Part 3.

### The bimodal boot-time kernel set (2026-07, RTX 5090 gemma-4 image)

The released serving image compiled a different kernel set on different boots. The offline `D_*` geometry doesn't
separate an `f2x4` from an `f4x2` fragment or the `bk` variants — 8 exact score ties at the gemma-4 m16
`mlp_down` / `o_proj` forks — and ties broke by enumeration order, which shifts across processes: a per-boot coin
flip. Guard now: every deploy tier (model argmin, reservoir/DB evidence argmins, golden realization) breaks ties
through `knob.canonical_row_key`, pinned by `test_deploy_pick_determinism.py`. See Part 4.

### The mis-calibrated online model (2026-07, RTX 5090 sweeps)

An online prior whose reservoir rows no longer shared feature names with the model (constant predictions,
worse-than-random ranking) still owned deploys, because `fitted` was the only gate. Guard now: the calibration gate —
`Prior.trustworthy` requires the median per-op in-sample Spearman between predictions and reservoir labels to clear
`CALIBRATION_MIN`, else the model is quarantined (keeps training, never deploys). See Part 3.

### The evidence tier silently disabled by one new feature (ninth 4090 sweep)

`mlp_gate_up` misdeployed: the model's `g2k` pick beat a measured-faster fused config the evidence tier was never
allowed to see. The evidence join required strict signature equality, and a scheduler stamp added in #311
(`S_warp_eligible`) was on no perf row recorded before it — one added feature disabled the whole evidence tier
against every existing DB. Guard now: `Prior.sig_groups` is a drift-tolerant join contract for both the reservoir
and DB tiers. See Part 4.

### The unrealizable-golden fallthrough (gemma-4 m64 gate/up, 323×)

A recorded golden entry that no offered candidate realizes doesn't deploy — the compile logs an enumeration-drift
warning and falls through to the tiers below. On the gemma-4 m64 gate/up cone those tiers landed on a per-cell
`fuse`/`b128` scalar config: 54.6 ms against the realizing entry's 169 µs — 323×. The lesson: the fallthrough is a
hazard, not a graceful degradation, so an unrealizable entry is worse than an absent one. Recording rules that
follow: never pin `PLACE@cone` together with a `TILE` (no single offered candidate carries both), and verify a row
deploys, not merely pins — only the in-model drift audit catches a pin-only row. See Parts 3 and 7.

### The missing-floor NaN poisoning (`mlp_down.m4096`)

A shape whose golden entries were ALL pin-only (realizable under `EMMY_KNOBS`, never offered by the enumeration)
gave the deploy nothing to realize: it fell past the golden tier and shipped a 111 ms, 0.03× kernel whose output
NaN-poisoned the downstream accuracy check. Guard now: the pin-only offer audit (`emmy eval golden`) reports
PIN-ONLY per entry and FALL-THROUGH (exit 1) when a shape has no offered-realizable floor sibling. The two audit
views genuinely differ: the 5090 `mlp_down.m4096` split-K row realizes standalone but not on the serving twin's
epilogue-fused down (the offer audit passes it; `--in-model` is the authority), while the 4090
`attention.hd512.s4096` split-KV row fails even standalone (which the offer audit catches at record time — it is
legal only because a serial floor sibling exists beside it). See Part 7.

### The cast-splice drift

The isolated golden-reproduction A/B passed 68/68 while the same model's in-model deploys drifted: a cast spliced
into the serving graph changed the compiled ops so the recorded goldens no longer realized there. This is the blind
spot the in-model drift audit (`emmy eval golden --in-model`) closes — the isolated snippet check and the in-model
compile are different questions, and both must pass. See Part 7.

### The pin that benched greedy against itself

A misspelled hd256 flash pin silently fell back to the planner's own pick; the A/B benched the fallback and
reported a fake 1.00× under the pin's name, which was read as the form refusing the config. Guard now: the
realized-vs-pinned knob check fails a mismatched row (`pin_unmatched`, not benched) right after the pinned compile.
See Part 7, integrity gate 1.

### The transposed-B staging gap (serving, 1.3–2.75×)

The staging transports historically declined a transposed B, so every served `F.linear` fork ran gmem-direct while
the canonical-layout twin staged through smem — a 1.3–2.75× serving gap class. The warp tier now stages either
layout with the same STAGE spellings, but the measured µs still differ per layout, which is why a golden meant to
decide a served model's linear fork must be tuned on the `F.linear` snippet, and why a canonical and a `trans_b`
entry must BOTH stay current under their shared ShapeKey. See Part 7.

## Bench and data-integrity incidents

### The zero-priced un-lowered kernel (#327)

A tune terminal still carrying an un-lowered kernel-bearing node benched as `ok`: the bench sums `CudaOp`s only, so
the un-lowered kernel priced at zero and a cached residual kernel's µs stood in for the whole graph. Guard now:
such a terminal is a `bench_fail` decided before any bench or cache lookup (`tune_async`). See Part 4.

### The linear prior's corner pick (`BR=1` blow-up)

The former linear online prior (`BayesianRidgePrior`) was monotone in every knob, so its greedy optimum was always a
corner of the candidate box — it drove `BR=1` picks that took a 4 µs kernel to 232 µs and produced invalid kernels.
This motivated the switch to a bounded tree ensemble (off-manifold-safe: an un-benched extreme inherits the nearest
leaf's value); CatBoost won the bakeoff (`scripts/prior_bakeoff.py`) on untuned-op generalization (leave-one-op-out
pick ratio ~1.0 vs xgb/lgbm 1.18, rf 1.31). See Part 5.

### Hand-found optima that never reached the store

The fast-math-lane optima were found by manual pinned sweeps and existed only in terminal scrollback — nothing wrote
them to the node store, so the prior never learned from them and later sweeps re-derived them from scratch. Guard
now: bench-to-node recording (`search/bench_record.py`) — every clean pinned/golden/`--ab` measurement from
`run --bench` lands as a parentless leaf row by default. Related detail from the same work: the mma tile-lowering
preserves no `LoopOp` in `.source`, and until the tile-dialect fallback was added to the offer-site recovery, every
tensor-core kernel was silently unrecordable. See Part 6.

### Machine-dependent golden evals

`eval offline` / `eval online` once featurized each golden under the LIVE host's context: a 4090 golden scored on a
5090 host featurized as "sm_89 with 170 SMs" (on a GPU-less host, with the default SM count), so the occupancy
features priced tiles for a card that doesn't exist — reporting rank 0 on shapes the real card misdeployed 12–29×.
The evals now rebuild each golden's context from the card recorded in the golden file (`Context.from_target`),
matching what the offline fitter's case builder always did. See Part 8.

## Retired designs

Removed mechanisms, kept here so old branches, DB rows, and habits can be recognized for what they are:

- **Per-variant fork scoring** — `Fork.score`, `Search.score_of`, the `lazy_score` / `score_tile_geometry` formulas,
  the DB-best `_best_fork` replay, and the `_priority_*` enumeration sorts. All replaced by the single `Prior`
  ranking path (Part 3): forks carry no score, and nothing materializes a `TileOp` just to rank it.
- **The `+∞`-unvisited UCB rule and static tiebreaks in MCTS selection** — the prior's predicted reward is now the
  sole selection signal; a confidently-slow sibling is deprioritized instead of force-benched (Part 5).
- **The `op_effort` "skip already-tuned ops" gate** — it suppressed exactly the prior-driven re-exploration that
  makes re-runs valuable. Replaced by "always re-run, replay from the cache": already-measured variants are served
  from the perf table with no GPU bench (Part 5).
- **ε-greedy golden data collection** (`emmy tune --dataset golden` as the node-store feeder) — search-driven
  collection over-sampled the branches the incumbent prior preferred, and its wall time grew with the golden set.
  Replaced by the budgeted three-slice golden-neighborhood sweep (`scripts/golden_neighbor_bench.py`), whose
  selection is independent of the incumbent and fixed-cost; `tune --explore-eps` survives for interactive tuning
  (Part 6).
- **`BayesianRidgePrior`** — the linear online prior; see "The linear prior's corner pick" above.
- **The `WSPEC` stamped row family** — the warp-specialization producer band is now spelled as `WORK`'s `+p<np>`
  suffix (producer/aux warps are inventory). `EMMY_WSPEC` survives as an env-pin alias, and `ingest_legacy_row`
  folds a legacy row's `WSPEC` key into the synthesized `WORK` entry (Tunable knobs).
- **Worker tokens embedded in `TILE` / `REDUCE` values** — the site-local re-spelling moved the worker inventory
  into `WORK` and shed the embedded tokens. The golden corpus was re-spelled mechanically
  (`scripts/respell_goldens.py`, 715 rows, replay digest-identical); legacy spellings survive only as
  loudly-validated pin aliases.
- **Axis-suffixed schedule keys as the stored spelling** — replaced by short-path-canonical tree-path keys
  (`ir/tile/path.py`). The tune DB, reservoir, and online prior were REGENERATED after the re-key, never migrated —
  no reader special-cases the old spellings, and `tuning_knob_items` renders keys as stored (the old
  `@<axis>`→bare display collapse is gone). The one live exception is the dynamic-attention bare `TILE` — current
  corpus semantics, not legacy debt (Tunable knobs).
- **The v1 JSONL measurement freeze** — refused at load with a re-freeze pointer; freezes are now per-GPU YAML
  directories with content digests (Part 6).
- **`loop/recognize/` as a pass with rules** — flash / online-softmax recognition moved into
  `lowering/tile/010_recognize` (the `_flash` / `_softmax` helpers); the loop dialect carries no pattern
  recognizers.