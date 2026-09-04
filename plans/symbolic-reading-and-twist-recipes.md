# Symbolic reading: `Expr` as exact syntax, sympy as semantics, definition-only twist recipes

Drafted 2026-09-03 from a design discussion. **Phases 1 and 2 landed on `feature/symbolic-reading`** (2026-09-04);
phase 0 and phases 3-6 are still open. This file is a working note and must never be referenced from durable docs or
code.

Landed, with two deviations from the draft below:

- `Body.as_expr` needed a second scope rule the draft did not anticipate: resolution is POSITIONAL, and a name read
  before it is bound gets a position-carrying atom. A combine rebinds its accumulators (`acc = add(sa, sb)` whose
  `sa` reads the incoming `acc`), so a name-keyed reading either recurses forever or reads the softmax pivot's
  `m = copy(maximum(m, m__o))` as `m`. Silent and plausible; caught by running `linearize` over `SOFTMAX.program`.
- `factor` refuses on structure only (not a product; a varying denominator). A lone atom is a one-factor product,
  and the draft's "a lone `exp` refuses" is the CALLER's condition — `_hoist_invariant` already declines on
  `not invariant or not varying`. Refusals about usefulness stay in rules.

The blockification consumer that motivated this (a twisted carrier's per-component `alpha*s + beta*s__o` reading,
so a block's inner fold reaches `as_contraction`) is a phase-6 use and is not started.

## Goal

Give the compiler one way to ask what a value cone *means* — a symbolic reading decided by sympy — so that:

1. Twist recipes are written as their mathematics (base monoid, per-element lift, ψ, ψ⁻¹, the stable ⊕ program)
   and nothing else. The per-channel `pattern`s, the `Channel` class, its `name` / `init` fields and Welford's
   phantom `c` parameter go away. `Fold.twist` decides a click by algebra instead of by canonical-form equality
   against a stored spelling, so a second spelling of the same second pass (`Σ(x² − mean²)`, `exp(−m)·Σexp(s)`)
   matches without a new pattern.
2. The two hand-written algebras in tile lowering — `product_spine` with its consumers `_hoist_invariant`
   (`Σ c·x = c·Σ x`) and `_decode_split` (a storage decode times invariant factors) — become uses of the same
   primitive and `product_spine` is deleted.
3. The next targets — dequantization scale folding through contractions, norm→linear cone classification —
   are the same primitive with a different invariance test, and need no new algebra.

Non-goals: no IR is ever built from a normalized expression (kernel identity and the digest gate never see sympy);
the index simplifier in `ir/expr.py` (range-driven `div`/`mod` rules) is untouched.

## Why

- `Fold.twist` matches a dependent reduce's residual lift against `Channel.pattern` by canonical form. The pattern
  is data describing *what the tree spells*, not a property of the twisted monoid, and one recipe accepts exactly
  one spelling per channel. Recipe authors derive patterns by hand; softmax's need the exp law, Welford's needs a
  summation identity. Nothing certifies the stored patterns against ψ (the certification tests check the program
  and the injections only).
- The definition half of a recipe (`lift`, `psi`, `psi_inv`) is read by no code outside the recipe module: it
  exists for the certification tests. The matcher runs entirely on the data half.
- Welford's `c` extra is semantically `1/N`; the fused fold drops it and the recipe never states the relation. A
  tree computing `Σ(x − k·Σx/N)²` for another `k` would match and miscompile. The tree cannot express the
  constraint because the frontend spells `1/N` as a scalar input (`_mean_of_sum` in
  `tests/compiler/passes/test_twisted_rewrite.py` documents this).
- `product_spine` walks a multiply/divide chain and both of its callers then repeat the same three steps
  (classify leaves by axis variance, collect divisors, refuse a varying divisor).

## Context for a fresh agent

Read first: `emmy/compiler/ir/ARCHITECTURE.md` (the term language, `expr.py`, the `Stmt` analysis surface),
`GLOSSARY.md` entries *Twist recipe* and *Componentwise / twisted combine*. Key modules:

- `emmy/compiler/ir/pure/twist.py` — `Recipe`, `Channel`, `SOFTMAX`, `WELFORD`, `Recipe.program`.
- `emmy/compiler/ir/pure/fold.py` — `Fold.twist` / `Fold._twist`: pivot search, score-cone cut, residual match,
  instantiation by renaming.
- `emmy/compiler/pipeline/passes/lowering/tile/_twist.py` — `rewrite_twisted`, `_candidates`, `_hoist_invariant`,
  `_inline`, `_varies`.
- `emmy/compiler/pipeline/passes/lowering/tile/_fromloop.py` — `product_spine`, `_decode_split`, `_Hoist`.
- `emmy/compiler/ir/stmt/base.py` — `Stmt`'s per-stmt analysis surface (`deps`, `defines`, `nested`, `exprs`) and
  the trait convention: conservative default, per-class opt-in, "no isinstance whitelist".
- `emmy/compiler/ir/expr.py` — `Var`, `Literal`, `BinaryExpr`, `FuncCallExpr`, `TernaryExpr`, `CastExpr`;
  `_ExprOps` operator overloading. **`apply_binop` floor-divides for both `/` and `//`**: `Expr` is an integer
  index algebra, so a value cone must be read into `FuncCallExpr` nodes (op-name calls), never `BinaryExpr`.
- `emmy/compiler/ir/elementwise.py` — `ElementwiseImpl`: callable on floats, traits (`commutative`,
  `semiring_product`, `distributes_over`, `decodes`, `identity`).
- Tests touching recipes: `tests/compiler/ir/pure/test_twist.py` (certification), `test_fold.py` (two twisted
  tests), `test_lambda.py` (`SOFTMAX.program(...).components() is None`), `tests/compiler/ir/tile/
  test_scan_observer.py`, `tests/compiler/passes/test_twisted_rewrite.py` (rewrite positives and negatives).
- `scripts/digest_kernels.py` — the kernel-source byte-identity gate; every phase below that touches lowering must
  be proven emission-neutral with it.

Measured facts: sympy 1.14 is in the venv as a torch dependency; torch does not import it; `import sympy` costs
under 0.1 s; a factorization is microseconds. A sympy hard dependency is accepted (decision 9).

## Design

Three layers, one primitive each. Each is useful before the next exists.

**Layer 1 — IR: a value cone as an `Expr` (exact syntax, sympy-free).**
`Stmt.as_expr(name, read, through) -> Expr` joins the per-stmt analysis surface with the conservative default
`Var(name)`: an accumulator, a load, a select, a constant, a `Fold`'s state, or a kind added tomorrow is an atom.
Only `Assign` opts in, as `FuncCallExpr(op.name, args)` when `through(op)` holds. `Body.as_expr(name, through)`
drives the recursion over *this scope's own definitions* (`definitions` is recursive over nested blocks; a value
bound inside an inner loop is not one value). A cone read this way evaluates through `Expr.eval` with no new
code (`FuncCallExpr.eval` resolves the op by name) and exposes its atoms through `free_vars()`.

A small value mixin gives `Var` real-valued operators (`+ - * /`, unary `-`) that build calls by op name, plus
`exp`, `maximum`, `sqrt` as functions. It is the recipe authoring language (layer 4) and the test language.

**Layer 2 — bridge: one closed map from `Expr` to sympy, in `emmy/compiler/ir/symbolic.py`.**
`symbolic(expr)`: `Var` → `Symbol`, `Literal` → number, the integer binary ops → sympy arithmetic with `floor` and
`Mod`, comparisons → relations, `TernaryExpr` → `Piecewise`, `CastExpr` → uninterpreted function,
`FuncCallExpr` → the bridge's own name table (`add subtract multiply divide negative exp`) or an uninterpreted
function of its arguments. Decisions on top, algorithmic sympy only (`expand`, `cancel`, `Poly`,
`as_powers_dict`; never `simplify`):

- `equal(a, b)` — expansion and cancellation.
- `factor(body, name, varies, through)` — one product split into invariant factors (each with whether it divides),
  varying factors (a square listed twice), and the spine: the cone of `name` minus the cones of its atoms.
- `linearize(body, name, varies, through)` — the summand of an additive fold as `Poly(expand(expr), *streamed)
  .terms()`: each monomial in the streamed atoms with its coefficient, which is an expression over invariants by
  construction.

One direction only. Nothing builds IR from a sympy expression: sympy normalizes at construction
(`Mul(Mul(a, b), c)` flattens, `x − x` vanishes, argument order is canonical), which is right for deciding and
wrong for spelling a kernel.

**Layer 3 — folds: `through` is legality.** Each consumer names the ops it is licensed to reassociate, in the
sense `_decode_split` already uses ("only a decode licenses it"). The hoist sees through `semiring_product` and
`divide` only — seeing through `exp` would factor `exp(−m)` out of `exp(s − m)`, the overflow the twist exists to
avoid. The twist sees through the ring and `exp` because it replaces the tree's spelling with the recipe's stable
program wholesale. The default `through` is the ring (`add subtract multiply divide negative`).

**Layer 4 — recipes as one authoring language with two readers.** `lift`, `psi`, `psi_inv` are Python lambdas over
value `Var`s; called with symbols at import they yield `Expr` trees the bridge reads. `advance` / `rescale` /
`combine` are the same kind of lambda (or `def` with locals) but are *flattened to ANF verbatim*, one statement
per distinct node (a node referenced twice emits once), never touched by algebra. Injections stay authored, as one
lambda over `(score, *extras)` returning the singleton's channel states, certified against `ψ ∘ lift` by the
existing test. Seeds are the base ⊕ identities (the existing certification that `ψ⁻¹(seed)` equals them stays).
The state names a recipe adds (Welford's count and mean) come from ψ's result names.

```python
def _advance(g, g_o):
    gn = maximum(g, g_o)
    return gn, exp(g - gn), exp(g_o - gn)


SOFTMAX = Recipe(
    name="softmax",
    base=("maximum", "add", "add"),
    lift=lambda s, v: (s, exp(s), exp(s) * v),
    psi=lambda m, D, O: (m, D * exp(-m), O * exp(-m)),
    psi_inv=lambda m, d, o: (m, d * exp(m), o * exp(m)),
    inject=lambda s, v: (1.0, v),
    advance=_advance,
    rescale=lambda s, s_o, alpha, beta: s * alpha + s_o * beta,
)
```

**The twist match, restated.** `Fold.twist` keeps its first half: the pivot is an operand folding `base[0]` whose
per-element map is alpha-equal to `lift[0]` (Welford's `S` and `T` are both `Σx`; the pivot-first ordering is the
author's disambiguation and stays), and the score role binds by the existing cone cut — which is what keeps the
causal mask inside the score rather than in front of the solver. The residual then goes to `linearize` with the
score, the pivot's final value and the remaining operands as atoms. Each streamed monomial must `equal` a lift
component and becomes that base state; the result is cancelled against each additive `ψᵢ`; the quotient must be
free of base states and is the epilogue (attention's `1/l` comes out here). A click reports the channel, the role
binding and the epilogue. A refusal names the step: a streamed monomial the lift does not produce, a residue that
is not invariant. Construction of the fused fold is unchanged: the pivot's operands plus the extras, the pivot's
lift with the injection appended, the program over the concatenated states, all by renaming and splicing.

## Decisions

1. **`Expr` is syntax, sympy is semantics.** Two expression types, two jobs. `Expr` preserves association and
   temps exactly; sympy decides equality modulo ring identities.
2. **Values are `FuncCallExpr` nodes, never `BinaryExpr`.** `apply_binop`'s `/` is floor division; the index
   simplifier treats calls as opaque, so a value cone can never meet integer rules by accident.
3. **Reading is one-way.** No IR from a sympy expression. The only expression→IR path is the verbatim flatten of a
   recipe's authored program, which is the author's spelling with less typing.
4. **`through` is a legality predicate on the op**, not an op-name set: hoist = `semiring_product or divide`; twist
   = ring + `exp`; default = ring.
5. **`Const` stays an atom.** The hoist wants the constant's name for its epilogue; no consumer needs the value.
   Opt in with one override when one does.
6. **Scope rule.** `Body.as_expr` sees through this scope's definitions only.
7. **No `solve`.** Invariants are free atoms or known constants. Welford's `1/N` is discharged by the frontend
   spelling it as a constant (phase 5), not by the compiler solving for it. Until then the click extends the same
   trust the pattern does today, and the plan says so.
8. **Emission-neutral by construction, proven by digest.** Every phase touching lowering runs
   `scripts/digest_kernels.py` before and after; the realization corpus must not go stale. If a fused fold's
   statements change spelling, that is a defect in the phase, not a reason to regen.
9. **sympy is a declared hard dependency** in the `compile` extra of `pyproject.toml`, imported at module top in
   `ir/symbolic.py`. Tests may use it freely.
10. **The index simplifier is not migrated.** Range-driven `floor`/`Mod` reasoning is not sympy's strength.
11. **Non-ring channels still match.** An uninterpreted atom compares structurally, so when nothing in a cone is
    interpretable the decision collapses to today's alpha-equality. A future `where`-carrying recipe needs no
    pattern field to work when the tree spells its form.
12. **Certification over derivation.** Injections and the stable program are authored and certified, never derived:
    deriving them means building IR from normalized expressions (decision 3).

## Phases

Each phase is one PR with its own digest A/B and the named mini test set only (no directory sweeps; see the
test-budget memory). Line balance under `emmy/` is reported per phase; phases 3 and 4 must be net negative.

**Phase 0 — certify the stored patterns (no compiler change).** Add to `tests/compiler/ir/pure/test_twist.py` a
test that folds each channel's `pattern` at the final pivot over random short streams and compares with `ψᵢ` of
the base fold, binding Welford's `c` to `1/N` explicitly. This is the missing third leg of the certification and
it documents the `1/N` trust in one place. Verify: the new test plus the existing three pass.

**Phase 1 — the value reading.** `Stmt.as_expr` (default), `Assign.as_expr`, `Body.as_expr` (scope rule), the value
`Var` mixin. Tests: a cone reads to the expected `FuncCallExpr` tree; a nested-loop definition stays an atom;
`Expr.eval` of a read cone equals the lambda evaluated by hand (reuse `_eval` from `test_twist.py`). No consumer
yet; `emmy/` grows by the reading only (say so in the PR body). Verify: new tests; `test_fold.py`, `test_lambda.py`.

**Phase 2 — the bridge.** `ir/symbolic.py`: `symbolic`, `equal`, `factor`, `linearize`; sympy declared. Tests: the
attention summand `exp(s − m)·v / l` factors to invariant `l` dividing and varying `e, v` with the two spine
statements; `x·x·(a·b)` lists the square twice; a sum, a varying divisor and a lone `exp` refuse; Welford's
residual linearizes to `{s²: 1, s: −2gc, 1: g²c²}`; an uninterpreted op compares structurally. Verify: new tests.

**Phase 3 — migrate the hoists, delete `product_spine`.** `_hoist_invariant` and `_decode_split` call `factor` with
the products-and-divide predicate; their leaf classification, divisor sets and refusal lines collapse.
`030_stamp_types.py`'s square check stays on the `semiring_product` trait. Verify: `test_twisted_rewrite.py`,
the `_fromloop` decode tests, digest A/B byte-identical, `git diff --stat main -- emmy/` negative.

**Phase 4 — the twist on `linearize`; recipes as definitions.** In order: (a) recipes gain the value-lambda form
and the verbatim flatten while `Channel` still exists, with `Recipe.program` output asserted equal to today's
(`test_scan_observer.py` and `test_lambda.py` pin it); (b) `Fold._twist` replaces the pattern comparison with
linearize → base-state map → cancel, keeping pivot search, score cut and construction; (c) `Channel`, `pattern`,
`name`, `init` and Welford's `c` are deleted, `inject` added, seeds derived; (d) `_candidates` drops the hoisted
spelling for the twist because the epilogue is now the solver's quotient, and the epilogue projection is built
from that quotient's names the way `_hoist_invariant` builds it (the quotient is a product of names or the click
refuses). Verify: `test_twist.py` (rewritten certification: program is the conjugate, injections are `ψ ∘ lift`,
seeds), `test_twisted_rewrite.py` positives *and negatives* (the unrelated max + sum pair, the unsquared
deviation), `test_fold.py`, digest A/B byte-identical on every corpus case with a twisted fold, corpus not stale,
`emmy/` net negative. Add one new positive per recipe for a second spelling (`Σ(x² − mean²)`; `exp(−m)·Σexp(s)`).

**Phase 5 — close the `1/N` trust.** The frontend spells `1/N` for a mean over a known extent as a constant (or as
`1/extent`); the twist reads it as a known rational and the phase-0 binding becomes a check the compiler performs.
Verify: a tree with `k·Σx/N`, `k ≠ 1`, refuses; the Welford positive still clicks.

**Phase 6 — next consumers (separate plans when scheduled).** Dequantization scale folding: `linearize` with
`varies` refined to block invariance for group scales. Norm→linear classification: `linearize` with `rsqrt` as an
atom; the invariant coefficient is the statistic. Both are uses, not extensions, of the primitive.

## Success criteria

- `twist.py` carries no `Channel`, no `pattern`; each recipe is under twenty lines of definitions plus its program.
- `product_spine` is gone; `_hoist_invariant` and `_decode_split` share `factor`.
- Every corpus case with a twisted fold produces byte-identical kernel source before and after phases 3 and 4.
- The rewrite's negative tests still refuse, and one new spelling per recipe clicks without recipe changes.
- The `1/N` assumption is either checked by the compiler (phase 5) or stated in one certification test (phase 0).

## Risks and open questions

- **`Poly` generators with `exp` atoms.** `Poly(expr, s, exp(s), v)` needs the expression polynomial in those
  generators after `expand(power_exp=True)`; `exp(2s)` must be recognized as `exp(s)²` (`powsimp`) before `Poly`.
  Pin this in phase 2's tests before phase 4 depends on it.
- **Role binding of extras.** Today the residual's closing order gives roles positionally. With linearize, an
  extra's role is whichever lift parameter its monomial matches; if two extras are interchangeable (two streamed
  values), pick by operand order and pin the choice in a test.
- **The epilogue as a product of names.** If the quotient is not a product of atom names (a computed invariant such
  as `exp(−m)` from a tree that spelled the unstable form), the click must refuse rather than build the epilogue
  from the expression. That is decision 3 applied; note it in `_decline`.
- **Fixed-arity recipes.** Welford's `combine` is over exactly its carrier; the matcher's arity check on an
  already-fused pivot stays as is.
- **Determinism.** Decisions are booleans and name bindings; sympy's internal ordering never reaches IR. Still,
  memoize `linearize` per canonical lambda so the fixpoint loop in `rewrite_twisted` does not re-expand.

## Out of scope

Deriving the stable program from ψ (stability is not preserved by conjugation); migrating `expr.py`'s simplifier;
a numeric oracle inside the compiler pass (tests only); solving for unknown invariants.
