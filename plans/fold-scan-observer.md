# Scan via an observer on `Fold`, on a two-refactor foundation

A scan (prefix reduce — cumsum-shaped: accumulate along an axis AND store the running state at every step) has no
spelling today: `Fold` exposes only the final combined state (`out` = the combine's first result), so a loop that both
accumulates and writes per iteration cannot become a tile-IR node. The fix is an **observer** on `Fold` — a pure λ over
the carried state, evaluated after each step's combine, whose results are a tuple of per-step elements written out
through ordinary `OutputSpec`s. No new node kind: the fold keeps its monoid, the observer taps the running state, the
term stays pure algebra with every effect at the kernel boundary.

Two architectural fixes come FIRST, in the same PR — both are load-bearing for the observer: the schedule-site read
path resolves nodes by object identity with silent misses (the observer adds another derived-stmt producer to exactly
that machinery), and the monoid families are a closed two-arm set hard-coded in `Fold.__post_init__` (the observer's
first real customer — the affine recurrence — is inexpressible in it).

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
- **No general associativity matching.** The Third Homomorphism Theorem is an existence theorem; using it as a
  matcher would need to prove an arbitrary traced loop is also a rightwards fold and then extract the combine — both
  out of reach at compile time. Matching stays program-equality against a FINITE registry of families (below).

## 1. Foundation — site threading

The schedule map is path-keyed (`{codec key → plan}`), but the REVERSE direction — a consumer holding a node object
asking `Sched` for its geometry — resolves by object identity (`ops.py` `_mn_for`: `s.node is node`), and a miss
silently reads as "untiled". Synthesized derived nodes (flash's PV) are addressable only through the one-identity
memo (`Fold._derived_twisted`). Fix the read path:

- **Thread the site down instead of looking it up.** The walker that enumerates `_all_sites()` already knows each
  node's path as it descends; hand consumers `(node, site)` pairs (or pass the site through `Sched.key/get/put/
  tile_of/placed/_mn_for`) and DELETE the identity reverse-lookup. The codec stays the one addressing scheme — this
  changes how a consumer learns its own address, not how addresses are spelled.
- Until every consumer is converted: a real `Fold` that matches no site RAISES instead of falling back to the
  untiled path — a tripwire, so an identity mismatch is loud while the threading lands.

Verify: the corpus replays byte-identical (`make test`; `scripts/digest_kernels.py` for the storage-migration gate);
a deliberately `replace()`d node raises instead of degrading.

## 2. Foundation — the monoid-family registry

Formation today accepts exactly two algebras, hard-coded in `Fold.__post_init__`: componentwise (`component_ops`) or
the twisted exp/LSE family, whose stored combine must byte-equal `exp_combine_states`'s generated program. Keep the
principle — *the family is never stored; the stored combine IS the program, and a family claims a fold only if its
generator would have emitted it* — and make the family a first-class registered object:

```python
class MonoidFamily:
    name: str                                   # established term: "componentwise", "exp", later "affine", "welford"
    def program(self, names) -> Lambda | None   # the canonical S × S → S combine for these state names (the ⊕)
    def step(self, names) -> ...                # the canonical serial update (the ◃ form recognition matches)
    def merge(self, fold) -> tuple[Stmt, ...]   # the derived streaming step (what exp_merge is for exp)
    commutative: bool                           # the ⊕ reorders freely
    observable: bool                            # a per-step observer is meaningful
```

- **A family is (base componentwise monoid, twist ψ).** Associativity is proven at family-AUTHORING time by transport
  of structure (ψ a bijection on the carrier ⇒ the twisted combine is associative by construction); 3HT is the
  authoring discipline — write the ◃ and ▹ steps, and their agreement certifies the combine being registered. Nothing
  algebraic runs at compile time. The registry asserts at import that `step` and `program` agree on a symbolic probe —
  cheap, and catches a mis-authored family the day it is written.
- **Formation** finds the unique family whose `program(names)` equals the stored combine (componentwise first — the
  untwisted fast path); none → reject loudly, exactly as now. `role`, the derived step, and exp's positional
  state-role readings (pivot / denominator / expectation) move INSIDE the family that owns them.
- **Recognition matches the step, not the combine.** Tracing yields the serial `s ◃ x` update, so the matcher
  compares the traced loop body against each family's `step` modulo α-rename and the existing canonicalizers — the
  same program-equality discipline, N registered candidates instead of one hard-coded arm. Exact canonical spelling
  only at first; a variant (FA-2's base-2 rescale is the same twist, textually different) registers only when a real
  trace forces it.
- **Legality becomes a derived read**: each `FoldMove` states what it requires and the family what it provides —
  SHFL butterfly / ATOMIC finalize require a commutative ⊕; SMEM tree / KERNEL finalize require associativity plus an
  order-preserving emitter (make that emitter property explicit); the observer requires order preserved end-to-end.
  This replaces the scattered `component_ops(...) is None` checks.
- **Initial registry**: componentwise (identity twist) and exp/LSE — a pure reorganization of the two existing arms,
  corpus byte-identical. Affine recurrence (`(A, B)` for `x ↦ A·x + B` — associative, non-commutative, observable;
  the delta-rule/SSM class) and Welford (the layernorm-statistics carrier) are later entries that touch zero
  formation code.

Verify: corpus byte-identical after the reorganization; the import-time ◃/⊕ probe; formation rejection messages name
the nearest family.

## 3. The term

`Fold` grows one kw-only field:

```python
# The per-step observer — λ(k, s₁…sₙ) → tuple, evaluated AFTER iteration k's combine
# (inclusive scan), binding the iteration var and the carried state positionally. Part of the
# ALGEBRA (it keys into structural_key); the writes consuming its results stay in
# TileOp.output_specs like any output. None = an ordinary fold.
observe: Lambda | None = field(kw_only=True, default=None)
```

- `__post_init__`: reject at zero axes (no per-step state); validate `observe.params == (axis.name, *combine.results)`;
  reject on a composed fold; reject when the fold's family is not `observable`.
- `_fold_derived_step`: append the observer body after the `Accum` forms, so its results are defined per step.
- `rewrite`: thread the α-rename through `observe` in lockstep with `lift`/`combine`.
- `structural_key`: include `observe` — two folds differing only in observation are different kernels. The write target
  stays excluded, as for every output.
- `defines()`: unchanged — observed names are per-step, never exposed to the enclosing scope; only writes consume them.
- Add an `observed` property (a structural probe like `composed`, not a role — `role` derivation is untouched).

Verify: unit tests on formation asserts, rename round-trip, and structural-key sensitivity to `observe`.

## 4. The boundary

- `extract_output_specs`: a `Write` inside the reduce loop whose value is an observer result extracts to a plain
  `OutputSpec(write=...)` — index template kept verbatim (it reads the fold axis), no new field.
- `apply_output_specs`: a spec whose write value ∈ `observe`'s results reconstitutes INSIDE the synthesized loop after
  the observer stmts; everything else keeps its current placement.

Verify: the existing reconstitution round-trip byte-identity gate, extended with an observed-fold case.

## 5. Schedule gating — same PR as the field, not after

An observer makes intermediate states order-visible, so a `ReducePlan` repartition is semantically visible for the
first time. The gate is a DERIVED read off the family and the move requirements from step 2 — never an ad-hoc check:

- An observed fold REJECTS every order-scrambling move: GRID split (`035_split_reduce` skips the offer) and BLOCK
  coop partitions.
- REG ILP/unroll stays legal (order preserved within one thread's serial stream).
- Parallel-scan legality (block-local scan + carry-in prefix through `combine`) is a later, separate stage; nothing
  here depends on it.

Verify: schedule enumeration over an observed fold offers no split/coop tiers; a pinned split fork errors loudly.

## 6. Recognition + references

- Lift-first recognition of the cumsum shape: a reduce loop that both folds and stores per-iteration lifts to
  `Fold(observe=...)` with the store becoming the boundary spec — the fold half through the step-matching registry
  from step 2.
- Mirror the semantics in the eager references (`torch_ref`) — spelled algebra must exist in the reference or CPU CI
  breaks silently (the fp8 lesson, PR #560).

Verify: recognition tests on traced cumsum; reference parity on CPU.

## 7. Realizer

- The materializer expands the per-step store inside the reduce loop — it already walks the synthesized loop, so the
  work is masking: a masked reduce axis must mask the store too. Add a masked-extent corpus case (the masked-M OOB
  latent bug says this path is under-covered).

Verify: realization corpus cases — a plain f32 cumsum, a masked-extent scan, and a scan observing a multi-component
state; accuracy vs eager on GPU.

## Landing order

ONE PR, built in step order: the two foundations first (each verified byte-identical on its own before the scan work
stacks on top), then the observe field + boundary + gate together (the gate lands with the field), then recognition
and the realizer. Commit per step so each verification gate is a bisectable point. Add `GLOSSARY.md` entries for
*scan* (prefix reduce; a fold whose running state is observed per step) and *monoid family* in the same PR.
