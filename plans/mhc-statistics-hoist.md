# Price the mHC statistics recomputation out of the elected `post4096` route

Branch `feature/mhc-statistics-hoist`, based on `fix/chain-piece-load-placement` (PR #700 — the elected composed cut
needs its seam-capture fixes to build). Retarget the PR to `main` after #700 merges.

## The finding this plan argues from (GPU-free, deterministic, on this branch)

The evidence-elected `post4096` composed cut re-verifies locally byte-identical (12 pieces, `sha256[:16]
ff45435e954bcbf2`, 96 s: `emmy compile --golden-file twins.yaml --golden
post4096.k_linear_softmax_matmul_mean_reduce_9716a1.f86b6dbe35b7.m4096 --target sm_70 --ir cuda` with the host
tune-DB copy). Its worst piece `…__place_ec1585e895` (block 256, `TILE@n0=f4`, grid 4) computes, per token row, the
sum of squares of the gated mixed hyper-connection carrier:

- `acc12(r) = Σ_{a5<4096} (Σ_{a6<4} gate(r,a6) · mixval(r,a6,a5))²`, where
- `gate(r,a6) = c + σ(base[a6] + scale · acc9(r,a6))` and `acc9(r,a6)` is the mHC gate's fn-projection — a
  16384-step contraction whose integrand carries the normalized stream mix (inner 4-step dot), the residual add,
  and the trunk RMSNorm rsqrt factor read from another piece's workspace.

`acc9` depends only on `(r, a6)`, yet it is stored as a body member of `Fold(a6)` inside `Fold(a5)`, so the piece
re-evaluates it **4096 × 4 = 16,384× per row** — 2³⁰ per-thread serial trips, minutes-plus per launch. Computing it
once per `(r, a6)` costs ~2¹⁶ trips per row: a ~1000× serial-work reduction, algebraically exact.

Why each existing mechanism does not remove it (instrumented `cuttable_seams` / `realize` / pricing over the
deterministic compile — every offer and price below is measured, not inferred):

1. **In-kernel loop-invariant hoisting cannot express the fix.** The subtree is a lift-body member (not an operand
   edge) and reads the enclosing fold's axis `a6`; hoisting it out of `a5` means materializing one value per
   `(r, a6)` — loop fission into a workspace, which is a placement decision, not an emission or normalize move
   (`ir/tile/ARCHITECTURE.md`: a recompute is a Tile-level sharing or seam-offer fact; the `fed_by_body` /
   `sweep_axes` hoist guards exist precisely because moving such folds inside one kernel miscompiled before).
2. **The seam IS offered.** On the piece's own placement fork the ballot holds 11 cut arms, including
   `PLACE@a8` — the plain seam (no providers, no requires) that materializes exactly `acc9`'s contraction into a
   workspace — plus `PLACE@a6` (the whole gated summand) and smaller statistics seams. Capability is NOT the gap:
   provider closure, dependent seams, and the composed cut (#682/#688/#692/#700) all function here.
3. **The ballot contains recomputation-free plans, and the greedy prices them away.** At the deciding fork the
   fused side prices **4.29e-37 µs** and the `PLACE@a8` arm **1.02e-17 µs** (Σ over both pieces); FUSE wins by
   twenty orders of garbage. The offline cold-start prior's score is `exp(-scale · quality)` — an uncalibrated
   proxy the module itself documents as spanning `e**±700` — and the fitted artifact carries **zero `S_*`/`H_*`
   weights**: quality is set entirely by `D_*` schedule-geometry terms (`D_pow2_threads` +136.5 dominates), so a
   2³⁰-trip serial nest with pretty geometry prices as excellent. `S_ext_reduce_prod` exists as a stamp but is
   weightless and nest-blind (flat product over sibling reduce loops). `policy/greedy._resolved_price` states the
   contract: comparing kernel sets sums absolute prices, and calibration "is the prior's to fix".

So this is the pricing gap the serving plan's gap 3 names, not a seam-machinery gap.

## The attack: a monotone serial-work feature + a physical bound at the kernel-set Σ

Make a kernel-set comparison impossible to win by holding a kernel's price below its physical serial-work bound:

- Stamp **`S_ext_serial_cell_work`** (structural, `passes/identity.py` beside `_extents`): the worst per-cell serial
  trip count — max over loop-nest paths of the product of static reduce-loop extents along the path. Nest-aware
  where `S_ext_reduce_prod` is flat; saturating like the other extent products; symbolic extents excluded
  (under-approximation, safe for a lower bound). Additive feature → no `FEATURIZER_VERSION` bump, no refit owed
  (an absent weight scores 0.0 — the documented contract).
- Derive **`D_serial_cell_work`** (`search/features.knob_features`): `log2(1 + stamp / coverage)` where coverage is
  the row's reduce-partition parallelism (the existing `REDUCE` decomposition: cta·coop lanes — the register/ILP
  fold does not reduce per-thread trips; over-dividing would keep the floor a valid lower bound, but cta·coop is
  both honest and available). Serial rows divide by 1.
- Enforce the bound at the ONE consumer of absolute µs — the kernel-set Σ
  (**`policy/greedy._resolved_price`**): a summand whose bound `features.serial_floor_us(knobs)` = per-thread
  serial trips × `SERIAL_TRIP_FLOOR_US` (1e-4 µs = 0.1 ns/trip, conservative for any GPU clock) exceeds the
  enforcement guard `_SERIAL_FLOOR_ENFORCE_US` (1 ms — three orders above launch overhead, three below the bench
  watchdog) is clamped to that bound. A measured µs is never below the bound, so the clamp only lifts model
  garbage. The guard is jurisdiction, not tuning: the bound ignores launch overhead and memory traffic, so at
  ordinary magnitudes the model's ranking must stand (an ungated draft flipped three qwen3emb sdpa corpus replays
  to a cut election by comparing trip counts alone), while a bound past the guard is un-servable whatever those
  effects are. NOT on the prior's scoring surfaces: any µs bound there collapses live-range sibling deltas (the
  `latency_proxy` plateau failure — the shipped fit→prior rank-identity test catches it).

Expected effect at the deciding fork: the fused nest bounds at 2³⁰×1e-4 ≈ 1.1e5 µs; the `PLACE@a8` arm's two
pieces bound at ~6.6 µs each (inside the guard — their model prices stand) — the cut wins by four honest orders,
and the same argument holds at every recomputation fork upstream (the 2³⁶-class fused monster included). Below
the guard NOTHING changes: elections there decide exactly as at main.

Rejected alternatives:

- **A new seam offer** — the seam already exists on the ballot (measured, point 2 above).
- **In-kernel hoist at normalize/emission** — wrong altitude (point 1); the `sweep_axes`/`fed_by_body` guard
  history shows exactly this move miscompiling.
- **Bounding the prior's scoring surfaces (clamped or added)** — tried first; both forms collapse live-range
  sibling deltas at float precision (the proxy is not µs), and the serial-only clamp flipped three sdpa corpus
  elections. The Σ is the one unit-coherent consumer of absolute µs, and sibling ranking never reads it.
- **Refit-only (fitted weight on the new feature)** — a fitted weight has no sign or magnitude guarantee at 2³⁰
  extrapolation (no measurable training row can exist there — that is the point); the floor is the structural
  guarantee, and the next natural refit can still learn the feature.

Known limits, deliberately out of scope: the online prior's µs extrapolation has the same blindness (the host's
online checkpoint does not exist yet, so the offline floor decides today's elections); the tune-harness budget /
knob-space / `--dump-dir` fixes are a separate lane the serving plan already tracks.

## Stages

1. **Stamp `S_ext_serial_cell_work`** in `passes/identity.py` (nest-aware max-path reduce-trip product; the
   `S_ext_` prefix because the extent-free-skeleton invariant owns every extent-dependent key), AND make the
   measured-disqualification match survive featurizer vocabulary growth (`policy/greedy._resolved_price`: a stored
   failed signature also binds as a SUBSET of the candidate's — the stamp derives from the same body the failure
   was measured on). The two land together because the stamp alone was measured to silently disable the
   disqualification tier: with it, the deterministic `post4096` repro fell back to the 11-piece 2^38-trip route;
   with the subset match restored it re-elects the 12-piece plan byte-identical (`ff45435e…`).
   → verify: RED-first unit tests — sibling reduce loops take the max path (≠ flat product), nesting multiplies,
   free loops and symbolic extents excluded, saturation holds (`tests/compiler/passes/test_structural_features.py`);
   vocabulary growth binds, the shared-key-only and shrunk-vocabulary shapes still price (`…/policy/test_greedy.py`);
   `make test-corpus-regen` restamps the derived half with no verdict change; `make test` green; the repro election
   stays byte-identical.
2. **Derive `D_serial_cell_work`** in `search/features.py` (log2, reduce-coverage division, absent-stamp → absent
   feature). → verify: featurizer unit tests (serial row = full trips; `coop`/`g<n>` rows divide; no stamp → no
   key); `make test` green.
3. **Bound the kernel-set Σ** (`serial_floor_us` in `search/features.py`, guarded clamp in
   `policy/greedy._resolved_price`).
   → verify: Σ unit tests — the 2³⁰ fused nest prices its bound and loses to its composed-cut arm, a
   below-guard bound leaves the model's price untouched, a measured µs passes through even past the guard; the
   shipped fit→prior rank-identity tests stay green (prior surfaces untouched); `emmy eval prior --dataset
   nodes` on the checked-in freeze unchanged; the three qwen3emb sdpa `realized` corpus replays keep their fused
   election; the full local repro re-compile elects a plan whose worst piece is no longer the 2³⁰ fused nest
   (trip table re-measured from the emitted CUDA).
4. **Docs + audit**: `pipeline/ARCHITECTURE.md` (featurizer + prior sections) and the serving plan's gap-3 entry;
   AGENTS.md steps 4–19; PR body notes the #700 retarget and the online-prior follow-up.
   → verify: `make lint`, `make test`, plans/ ≤ 10 files.
