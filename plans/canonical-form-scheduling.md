# Canonical-form scheduling: drop the term hash from the pool key

## The invariant to establish

For the same kernel — the same canonical Loop-IR body (`Op.body_identity`) — recognition must yield the same result,
and therefore the same rewrite trajectory: the same term, the same schedule space, the same spelled row vocabulary.
Any case where two source spellings of one computation recognize differently (one gets `CONTRACTION`/`TWISTED`, the
other demotes to `PLANAR`; one closes a computed-operand cone, the other doesn't) is a defect, not a fact to key on.

## Where we are after the identity redesign

`Op.deploy_identity` (the durable join key: golden records, corpus stamps, child receipts) is the canonical
schedule-free lowered body + the io fingerprint, so it already realizes "identity = the computation". `pool_key` is
dissolved: the schedule-pool memo digest is minted at its one site (`lowering/tile/_schedule`) from the deploy
identity plus everything else the enumeration reads — knobs, symbolic-dim hints, pins, the split receipt, and the
spelled key vocabulary (`off`). The term hash (`TileOp.structural_key`, behind `cache_key`) survives only as the
tuning / search-tree / cubin cache key, where over-fineness costs cache misses, never a wrong join.

Two facts still stand between here and "the rewrite trajectory is a function of body and knobs":

- **spelling**: the pool memo stores rows addressed in the term's coordinate frame (tree-path codec keys name fold
  nodes by recognition-assigned axis names), so `off` must stay a frame-guard key term — an α-renamed twin must not
  replay a pool recorded in a different frame;
- **recognition variance**: the term caches what recognition proved, and two same-body terms can differ in it. The
  invariant to establish says any such divergence is a defect; today it is merely undetected.

## The move

Make the term a derived, private artifact of scheduling: schedule `lift(canonical_body)` rather than the
historically built term, with names minted from the canonical walk (the same canonicalization
`Body.structural_key` already performs). Then, by construction:

- recognition is a function of the computation (one canonical input per body-equivalence class);
- the spelled row vocabulary is spelling-stable forever (one final golden re-spelling to get there);
- the `off` frame guard becomes redundant (one frame per body class) and can leave the pool digest.

## What it takes

- Idempotent canonical lift: `lift(lower(term))` lands on one normal-form term per body class (the total lift
  exists; the missing property is canonical naming + factoring of its output).
- Every recognition gap between historical terms and canonical-form terms surfaces at once — expect a bug-fix
  campaign of the "matched-but-unrealizable" / "recognize fallback" family before the switch can be default.
- One final re-spelling of persisted golden rows into the canonical vocabulary (a codec migration with a
  re-spelling story, like the identity re-key executed in the body-identity PR).

## Measured state of greedy compile reuse (2026-08-29, RTX 5090 box, 512-cubed f16 matmul synthetics)

- ONE unique kernel costs ~9s of tile-level greedy compile (enumeration + cut-piece pricing) — the dominant,
  per-unique-shape cost no reuse cache can touch.
- Truly identical sibling layers are ALREADY amortized: 8 identical matmuls cost 10.7s vs 9.1s for one
  (~0.24s marginal per twin; the pool memo hits 7/7 and the pool-id-keyed decision memo replays the pick).
- The structural-pricing memo was the real hole: keyed on the term hash it re-priced every mirror cut piece
  (a depth-i prefix cone vs a depth-i suffix cone — same computation, different term-axis ranges). Re-keyed on
  the exact deploy identity it collapses 31 -> 16 probes and 18.4s -> 10.6s on a 16-layer chain (landed).
- A session-scoped lowered-kernel artifact cache (the "global kernel cache" idea) therefore only pays where the
  SAME kernel shape recurs across DIFFERENT programs in one session; program-level recurrence is already covered
  by the plan cache, and bucketed serving programs mostly differ in shape. Measure real cross-program identity
  overlap on a serve boot before building it; the remaining within-program twin residual (~0.24s: leaf descent +
  materialize + glue) is the bound on what a twin-fill mechanism can win.

## Second follow-up: freeze `Op`

`Op.__setattr__` now makes the io maps immutable in place and invalidates the one identity cache on reassignment —
the cheap version of immutability. The end state is a frozen `Op`: every mutation site (the engine's rebind
stamping `source`/`knobs`, the matcher's `populate_io`, `Sched`'s slice writes) becomes `replace()` + a graph
rebind, and the `__setattr__` hook disappears. Composes naturally with canonical-form scheduling: an op derived
from a canonical form has no reason to mutate after birth.
