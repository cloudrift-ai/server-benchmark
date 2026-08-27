# Scan via an observer on `Fold`

A scan (prefix reduce — cumsum-shaped: accumulate along an axis AND store the running state at every step) has no
spelling today: `Fold` exposes only the final combined state (`out` = the combine's first result), so a loop that both
accumulates and writes per iteration cannot become a tile-IR node. The fix is an **observer** on `Fold` — a pure λ over
the carried state, evaluated after each step's combine, whose results are a tuple of per-step elements written out
through ordinary `OutputSpec`s. No new node kind: the fold keeps its monoid, the observer taps the running state, the
term stays pure algebra with every effect at the kernel boundary.

Design decisions already settled:

- **Inclusive at formation.** The observer sees the state AFTER iteration k's combine. Exclusive scan is derivable by
  shifting `init`/index — no stored flag.
- **Minimal binding**: `λ(k, s₁…sₙ)` — the iteration var plus the combine's result components, positionally. Per-step
  operand values can ride the carried state later if a use case demands it; do not widen the contract now.
- **No new `OutputSpec` field.** A write whose stored value name is one of the observer's results can only live inside
  the synthesized reduce loop after the observer stmts — the sole scope defining the name — and `apply_output_specs`
  has the term beside the specs, so placement is a set-membership check. Store in `OutputSpec` exactly what boundary
  extraction destroyed (`sweep`/`unroll` are a dissolved loop's only record); derive everything the term still holds.
- **Dead observers are ill-formed.** An observed result with no consuming `OutputSpec` is a formation error, not a
  dead-code-elimination job.
- **Outer fold only.** A nested/composed fold (split-K's embedded contraction, flash's score head) stays unobserved;
  asserted at formation.

## 1. The term

`Fold` grows one kw-only field:

```python
# The per-step observer — λ(k, s₁…sₙ) → tuple, evaluated AFTER iteration k's combine
# (inclusive scan), binding the iteration var and the carried state positionally. Part of the
# ALGEBRA (it keys into structural_key); the writes consuming its results stay in
# TileOp.output_specs like any output. None = an ordinary fold.
observe: Lambda | None = field(kw_only=True, default=None)
```

- `__post_init__`: reject at zero axes (no per-step state); validate `observe.params == (axis.name, *combine.results)`;
  reject on a composed fold.
- `_fold_derived_step`: append the observer body after the `Accum` forms, so its results are defined per step.
- `rewrite`: thread the α-rename through `observe` in lockstep with `lift`/`combine`.
- `structural_key`: include `observe` — two folds differing only in observation are different kernels. The write target
  stays excluded, as for every output.
- `defines()`: unchanged — observed names are per-step, never exposed to the enclosing scope; only writes consume them.
- Add an `observed` property (a structural probe like `composed`, not a role — `role` derivation is untouched).

Verify: unit tests on formation asserts, rename round-trip, and structural-key sensitivity to `observe`.

## 2. The boundary

- `extract_output_specs`: a `Write` inside the reduce loop whose value is an observer result extracts to a plain
  `OutputSpec(write=...)` — index template kept verbatim (it reads the fold axis), no new field.
- `apply_output_specs`: a spec whose write value ∈ `observe`'s results reconstitutes INSIDE the synthesized loop after
  the observer stmts; everything else keeps its current placement.

Verify: the existing reconstitution round-trip byte-identity gate, extended with an observed-fold case.

## 3. Schedule gating — same PR as the field, not after

An observer makes intermediate states order-visible, so a `ReducePlan` repartition is semantically visible for the
first time. Gate before any realization exists:

- An observed fold REJECTS GRID split (`035_split_reduce` skips it) and BLOCK coop partitions.
- REG ILP/unroll stays legal (order preserved within one thread's serial stream).
- Parallel-scan legality (block-local scan + carry-in prefix through `combine`) is a later, separate stage; nothing
  here depends on it.

Verify: schedule enumeration over an observed fold offers no split/coop tiers; a pinned split fork errors loudly.

## 4. Recognition + references

- Lift-first recognition of the cumsum shape: a reduce loop that both folds and stores per-iteration lifts to
  `Fold(observe=...)` with the store becoming the boundary spec.
- Mirror the semantics in the eager references (`torch_ref`) — spelled algebra must exist in the reference or CPU CI
  breaks silently (the fp8 lesson, PR #560).

Verify: recognition tests on traced cumsum; reference parity on CPU.

## 5. Realizer

- The materializer expands the per-step store inside the reduce loop — it already walks the synthesized loop, so the
  work is masking: a masked reduce axis must mask the store too. Add a masked-extent corpus case (the masked-M OOB
  latent bug says this path is under-covered).

Verify: realization corpus cases — a plain f32 cumsum, a masked-extent scan, and a scan observing a multi-component
state; accuracy vs eager on GPU.

## Landing order

Steps 1–3 are one PR (the gate must land with the field). Steps 4–5 follow, each independently verifiable. Add a
`GLOSSARY.md` entry for *scan* (prefix reduce; a fold whose running state is observed per step) when the first PR
lands.
