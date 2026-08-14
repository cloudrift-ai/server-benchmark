# Pass-authoring invariants

Rules that apply to EVERY pass in this tree (`frontend/`, `loop/`, `lowering/`). Per-dialect details live in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) (pass order, knob table, fork semantics). The **tile-lowering** phase
(`lowering/tile/`) is the canonical instance of the invariant below — a **purely algebraic moveset, no
specializations**: it dispatches on stored params of the fold (`axis is None` / `is_contraction` for the schedule
walk; the derived `Fold.role` for the loop annotation the materializer reads), never on a named shape
(matmul / pointwise / attention) —
flash attention is the `TWISTED` fold on the streaming schedule (a twisted monoid is a monoid), selected
structurally, not a distinct kind.

## Quantization is not a concept past the decomposition band

A quantized checkpoint is spelled as generic in-graph algebra at BIRTH (`loader.quant`, immediately post-trace).
The scheme may choose different generic algebra, but it may not mint a scheme-specific op. The generic
`frontend/decomposition/032_fold_constant_subgraphs` rule collapses static computation cones into a bind-time
`ConstantOp` (`source_graph`) before Loop IR. It deliberately leaves storage-decode cones expanded so compressed
device storage is preserved, and leaves layout-only cones to the target-specific `050`/`060` load-layout policy.

The boundary is structural, not a naming guideline. Lowering, shared statement and tile dialects, backends, and search
may contain canonical dtypes, generic ops, and graph algebra. They may NEVER contain a checkpoint format's custom op,
statement, helper, pass branch, schedule feature, environment gate, comment, or name. A new format belongs in the
loader and its birth-time speller must emit only generic algebra. `tests/architecture/test_layering.py` scans every
post-decomposition Python source file for known format names.

## The tile scheduler: one inventory, then a product over sites

Schedule **enumeration and composition** — the step that decides a `TileOp`'s `place` (free axes → grid) and its
`schedule` slices, and offers them as a fork — is driven by the `020_schedule` rule. A row is a joint assignment
across every SITE of the term. The kernel's ONE worker inventory is chosen FIRST; the sites are then a product under
that fixed context:

```
enumerate(term) = [ r for reading in readings(term)                  # collapse / mixed-A — the two tree rewrites
                      for work in inventories(term)                  # w<M>x<N>[+p<n>] | t<N>[x<M>] | ""
                      for r in rows(root_site(reading), work) ]
rows(site, work) = [ merge(v, *child_rows)                           # spelled through ops.Sched.key, site-local
                     for v in values(site, work)                     # the domain: search/space.py, RESOLVED vs work
                     for child_rows in product(rows(c, work) for c in children(site))
                     if legal(site, v, child_rows) ]
fork = build_fork_tree(rows, levels=[WORK, *site keys, RASTER], materialize=…)
```

**`WORK` leads because the codec says so**: `TilePlan.parse(spec, work)` and `ReducePlan.parse(spec, work)` read a
value's unit widths and coop width OFF the inventory, so the dependency runs work → slice. Fixing it at the root also
removes a cycle — a cooperative candidate cannot be parsed without the inventory it would itself determine — and
turns three parent/child coupling rules into "the child resolves against the same `work`, and an unspellable
candidate is simply not in `values(site, work)`".

**The enumeration is memoized per term** (`_schedule._Pool` in `ctx.session_cache`): the rows are a pure function of
`(term, ctx, pins, hints)`, so N same-shape kernels in a graph — and every tune trajectory after the first, since the
pipeline re-runs this rule per trajectory — pay one enumeration. The cache sits BELOW the search policies (greedy and
MCTS share hits without knowing it exists) and holds no ranking and no evidence — only the readings re-bind per op,
so materialization always stamps against that op's own placement and stores. The key folds in the two inputs op
identity deliberately excludes — the symbolic-axis hints and the live schedule pins — and the ctx facts ride the
cache's home: one `Context`, one fact set. Pool rows are read-only mappings and cached slices are asserted
placement-free at build, so a shared pool cannot be corrupted by one consumer for another.

**No site builds `TileOp`s directly, and no term shape gets its own path.** The product over sites is what lets a
term whose operand is a NODE rather than a `Load` be scheduled at all: a materialized operand is not a site, so its
transport is enumerated at the parent's `STAGE`; a computed operand IS a site, so it enumerates its own families and
the parent's `STAGE` narrows to the compute fill.

**Materialization recurses the same tree.** A row decides EVERY site the walk visits, so `_nested_slices` mirrors
`_rows_at` (`_keeps_children` included) and stamps each nested site's slices at its own path key. Stamping the root
alone made a nested key a knob no kernel realized — a row spelling `REDUCE@j` materialized to an empty `schedule`.
Two consequences fall out of "one kernel, one inventory": a row's cooperative claims must AGREE rather than be
folded with a maximum (a `REDUCE` value spells no width — it lives once in `WORK` — so two sites wanting different
bands name a kernel the wire format cannot tell apart, and admitting both produced byte-identical duplicate rows),
and a candidate the enumeration offers must be one materialization can build: `splitk_materialized_b` is asked at
enumeration, not only inside `_splitk_option` where it is enforced with `pinned=True` and would turn an unpinned
offered candidate into a raise.

Three layers own three different questions, and keeping them apart is what stops a rule being stated twice:

- **The candidate DOMAIN** is `search/space.py`. The two families with multiplicative coupling — the scalar and warp
  tile grids — are GENERATED from their bounds (`search/domain.py`'s `Dimension` / `Bound` / `Space`); the families
  with no products to couple (stage spellings, split widths, the coop partitions, the raster orders) stay listed.
  The domain knows integers and products only: it is per-family and term-blind, and does NOT recurse.
- **Per-node LEGALITY** is `lowering/tile/_legality.py`: one function per rule, each returning the refusal REASON or
  `None`, with `enforce(reason, pinned=…)` choosing the severity — an env pin raises it, the unpinned enumeration
  drops the candidate. One predicate, one home; the "pin says yes, enumeration says no" class of bug has nowhere to
  live. The multiplicative rules are `Bound`s (a thread budget, a K-step that must tile a static extent, a 16 B
  inner stride); the categorical ones (operand dtype, transport, a fragment-unrealizable gather epilogue) are plain
  predicates. The smem budget is enforced by the stage RESOLVERS there, which return the largest legal `Stage` or
  decline — a size, not a yes/no. These are also the recursion's downward filter.
- **CHOICE** is the walk itself: which families a SITE offers, the conservative option-0 each leads with, and how a
  row becomes a `TileOp`. Dispatch is TWO stored-param predicates on the node — `node.axis is None` (the register
  strip) and `is_contraction(node)` (tile × stage × reduce), else the reduce partition. **Not `AxisRole`**: the role
  never selected a fourth arm, `TWISTED` is derived by matching the combine's operation family (an operation match
  wearing an algebraic name), and `PLANAR` is the residue. The role stays a loop annotation and a materializer read.

Three properties this shape enforces:

- **`WORK` is chosen once, then validated.** Every site resolves against it, so a combination whose slices disagree
  is never built (a tiled scalar site and a coop reduce are co-representable only when `par_m == 1 and par_n ==
  coop`); `derive_inventory` checks the chosen inventory is the one the resolved slices imply, and
  `ops.seal_workers` stamps it. The producer band `+p<np>` is part of the inventory and is CHOSEN with it.
- **Uniform key sets per fork, `""` a DECIDED empty.** The evidence pick's prefix-consistency depends on it: an
  absent key reads as "free" and would let a gmem-direct leaf inherit a staged row's measurement.
- **Enumeration produces a SET; ranking is the prior's job.** The only ordering obligation is that each family's
  FIRST value is its conservative default (per-family, not global — the reduce tier deliberately leads with its
  cooperative pick). The recursion decides the row set; `build_fork_tree` decides the evidence hierarchy, and the
  two are deliberately not the same shape.

A predicate has ONE home and ONE severity: each legality function returns its refusal, and the caller picks
raise-vs-drop from whether the family is PINNED (an unpinned warp move with an indivisible K-step is dropped;
the same defect in a pin raises). That is the bug class — "the pin says yes and the enumeration says no" — the
single-home rule exists to prevent.

**Term READINGS — the one mechanism ABOVE the product.** Four moves rewrite the term rather than decorating it, and
the criterion that separates a reading from a value is whether the rewrite changes the SITE SET, because that is what
a product cannot absorb. The register strip and split-K do not (`r` and `cta` are spelled TILE / REDUCE values,
applied at materialization); three do, and they are mutually exclusive by shape, so a term has at most TWO readings
(`_schedule._readings`):

- the MONOID-producer composition (`_atomize.bind_prologue_contraction`) — the fused norm→linear / gate⊗up edge, whose
  contraction reads its normalized row off a COMPUTED `a` edge. It ADDS the contraction and the cone's statistic to
  the map form's single reduce site, and its tree is the union's REFERENCE namespace: bare `REDUCE` must mean the
  contraction's K fold, so the map reading spells its statistic at `REDUCE@<axis>` too;
- the COLLAPSE (`Fold.demoted`) — a computed `a` edge spliced back INLINE, REMOVING its site. With no edges the
  bilinear reading declines, so the fold derives `PLANAR` and takes the reduce tiers; this is what carries a stat-free
  cone (`f(x) @ w`) and what a computed-A term with no legal warp row falls back to;
- the mixed-A PROMOTION — a materialized **f32** `a` re-expressed as a one-`Load` cone, so it rides the converting
  compute fill (a copy transport moves raw bytes; only the fill converts, on the slab store). It ADDS one site.

The union carries three obligations: uniform key sets with `""` as a decided empty; NO cross-reading suppression (each
gate is a local predicate on its own term — a 16-bit atom, a resolvable fill, an inventory a value can spell against);
and reading identity surviving into the prior's key space, which `_enumerate` enforces by keying the row → reading map
on `canonical_row_key` and RAISING on a collision (the fix would be an `S_*` stamp, never a new knob key).

**Coverage, as it stands.** The recursion carries the single-site terms — the pointwise cell plus the register-strip
term variant, the reduce partition, and the contraction's tile × stage × reduce × raster product over the scalar and
warp tiers, with split-K routing through the structural `Fold ⊃ Fold` composition `030_split_reduce` consumes — and
the COMPUTED `a` edge with them: the fused cone's contraction offers the warp tier over the MANDATORY resolved `sync`
compute fill (`d1` plus the asymmetric B-only prefetch ring at `d2`), its split-K is the redundant-statistic form (the
k-invariant prologue stays full-row in every partition, only the per-cell cone σ-reindexes), and the cone's own
statistic site is a nested site under the same inventory. The **flash streaming pair** is carried too, and it is why
the enumerator recurses: a `_site_values` entry plus legality predicates, with no emitter of its own. A term the
enumeration cannot schedule yields NO rows and stays unmapped: the guardrail contract, not a failure, since kernels
still compile on the materializer's per-cell path, so what is missing is schedule coverage, never a compile.

**The streaming pair, site by site.** A fold whose DERIVED evaluation contracts (`_schedule._streams` — structural,
never the `TWISTED` role) keeps its children as sites: the hoisted QK score edge and the synthesized P@V. Each
enumerates its own half of `twisted_warp_moves`' free geometry — the key-atom / query-tile pair, since `warps_m` is
the inventory and lives once in `WORK` — and the pair is reconciled at the STREAM (`_legality.twisted_sites_agree`:
the P@V rows are the score's rows, its K-chunk is the streamed key block). That reconciliation is the whole reason
the enumeration recurses: two sites that must agree cannot be a product over one node's families. The stream itself
then decides what only it can see — the K/V transport, sized against the geometry its children chose
(`_legality.resolve_twisted_stage`, including the `split` per-edge groups that also stage Q), and the cross-CTA
split-KV `030_split_reduce` realizes as partial + LSE finalize. The **chain** is the same P@V site under the `""`
inventory (the value axis leaves the grid for a per-thread register vector); the **per-cell / cooperative** forms are
the ordinary reduce partition on the same term, with both children at their decided empty. A form whose derived
evaluation is scheduled folds the stream ITSELF, so it composes with no partition tier but the split-KV — the
alternatives are alternatives, not a product, and a `REDUCE` pin the form cannot honor drops those rows rather than
being ignored (the reduce tiers, which do realize it, carry the term instead).

Two nested sites answer with the decided empty and nothing else. A cone's statistic under a computed `a` edge,
because the parent FORM realizes its partition itself — `_stage.sync_stat_fill` stripes the statistic one row per
warp with the warp's lanes striding the fold, a single hardwired partition (`_schedule._fill_realized`). And the
streaming pair's own `REDUCE` / `STAGE`: their K-step rides `TILE` and their operands ride the stream's `STAGE`, so
those keys are not spelled at all (`_schedule._decided`). That is not an exception to the uniform-key rule — the rule
is that every LEAF spells the same keys, not that every site gets one per family — and it is load-bearing rather than
cosmetic: the featurizer reads one node GROUP per distinct `@<axis>` element and gives each group the reduce geometry
whenever its slice carries a `REDUCE` key at all, so a decided-empty `REDUCE@dd` fabricates a partitioned reduce at a
site that has none and sum-pools its occupancy into the row (measured: `D_threads` and `D_splitk` tripled,
`D_log2_ctas` reading 18 instead of 6, which cost the chain and warp forms their cold deploy). `TILE` keys are never
dropped — that family is what NAMES the node group and what a golden joins against. A value at either site would
stamp a knob no kernel realizes.

A row BUDGET (`_schedule.MAX_ROWS`) bounds one kernel's enumeration and fails LOUDLY when exceeded — never
truncates. The product across sites is generated rather than hand-written, so a widened catalog multiplies where a
flat builder would have added, and a silent truncation would read as "covered everything" while dropping whichever
rows the walk reached last. The widest live term (a static f16 square matmul over both tiers) measures ~133k rows.

The rebuild's acceptance gate — a strict node-id xfail registry that had to shrink to empty — is MET and the
registry is deleted. `scripts/digest_kernels.py` remains the standing gate: it pins each case's rendered kernel
byte-for-byte AND asserts the case's pins actually reached a kernel, so a role that is merely rendering rather than
scheduling cannot pass unnoticed.

Three structural properties the enumeration rests on are asserted rather than trusted, all in
`tests/compiler/passes/test_move_catalog.py`. **Option-0 is conservative PER FAMILY** — position still deploys a
kernel on three prior-free paths, so the leading row must stamp each family's declared OFF, with the reduce tier the
one encoded exception (it leads with its cooperative heuristic pick, and since step 7 that exception has two
spellings, because the band's width moved into `WORK` — so the pair is checked together). **A row is its spelled
knob dict**, so two candidate combinations spelling identically are one row, not two. And **the `WORK` pin's one
non-narrowing branch is tracked**: a pin no candidate matches is offered beside the catalog's own inventories rather
than replacing them. That is the PIN-BLEED rule — one env pin, several kernels in a graph, and this term is not the
one it was written for — so emptying the fork would leave a term unmapped over a pin that was never about it (the
strip site applies the same degrade to a warp `TILE` pin it cannot spell). The reading that used to share the branch,
a coverage gap where narrowing would be right, is gone: the streaming site enumerates its own warp inventories, so a
`w<M>x<N>` pin narrows there like anywhere else.

## No shape-specific pattern matching

A pass must not dispatch on enumerated shapes ("if this is the gated-MLP body do X, if it is the QK^T body do Y").
Each named shape that needs handling is evidence of a missing GENERAL rule; find the per-element formulation that
makes the old and new shapes degenerate cases of one code path. Shape dispatch compounds: every new model
architecture would add a sibling branch to every pass it touches — the combinatorial explosion of compiler
complexity this invariant exists to prevent. It also breeds divergent incidental behavior (per-branch dtype or
layout rules that drift apart) and silently narrows coverage to the shapes someone happened to name.

How to comply:

- **Write the rule per element, not per shape.** Example: `lowering/tile/_atomize.map_cone` /
  `bind_prologue_contraction` classify each ⊗-fold operand independently (plain `Load` stays put; a computed cone is
  bound as the shared A value by value-tree equality, however many fold channels read it). Norm→linear, gate/up +
  SwiGLU, scale→matmul, SDPA P@V, and rotary QK^T are *instances* of that one rule, not branches — and a shape
  nobody designed for (a weight-side decode cone) is covered for free.
- **Gate in the negative.** Enumerating admissible shapes is shape matching by another name. Walk the body and
  report the first thing the transform *fundamentally cannot do*, like `ir/stmt/algebra.classify_fragment_epilogue`
  (the epilogue folds unless it has an ineligible op/dependency) — the eligible set then grows with the renderer
  instead of with a hand-maintained list.
- **Bail conservatively on well-formedness, never on shape identity.** `return None` / `RuleSkipped` for a body
  the rule doesn't fully understand is fine; the conditions must be structural properties (escaping values,
  symbolic extents, mixed dtypes), not "is this the X kernel".
- **Phrase dataflow conditions over cones, don't hand-roll the walk.** `Body.backward_cone` / `forward_cone` /
  `defs_die_at` (`ir/stmt/body.py`) are the shared slicing substrate: a rule asks for a cone and judges its
  *properties* (members, external reads, escapes) — construction never fails, so every bail stays a rule-side
  condition. See the dependence-cones section of `compiler/ir/ARCHITECTURE.md`.
- **When generalizing an existing rule, normalize its incidental divergences** (one dtype rule, one index rule)
  and name the behavioral deltas explicitly in the commit — don't preserve two behaviors behind one entry point.

Loop fusion first keeps a contraction producer materialized when it fans out into the statistic and value paths of a
downstream normalization. The N-way splicer shares repeated equal-coordinate demands, but this fan-out would duplicate
the whole contraction. Reductions that are not contractions remain fusible: softmax intentionally reuses QK inside
max-shift-exp and then reuses the exponentials for normalization and P@V, and retaining either buffer would prevent
flash recognition. The remaining profitability guard follows the same structural rule.
`loop/fusion/010_merge_loop_ops` counts aggregate
arithmetic and reads, and separately counts transcendental executions. The separate count prevents an expensive
pointwise producer such as GELU's `tanh`, originally evaluated once per `(M,K)` element, from moving under a
contraction's `N` loop merely because the contraction's much larger cheap-FMA count hides that duplication. Flash
attention is the structural exception: a merged softmax-then-P@V offer deliberately streams `exp(score)` without
materializing the probability matrix, so the tile recognizer owns that composite and the generic transcendental
brake does not split it. The automatic brake is also limited to unary/single-source activation cones. A multi-source
gated activation such as GeGLU/SwiGLU remains an evidence-controlled placement choice, preserving its GPU golden
coverage until that entire cone has been measured.

Fusion also lets a decomposed contraction's pointwise product reunite with its sole sum-reduction consumer before an
upstream activation-bearing cone is spliced into the product. Otherwise pass order can materialize the full M×K×N
product, whose work-growth then prevents the reduction merge; ordinary softmax/attention reduction order is unchanged.

## Resolve the hardware-atom binding once, structurally, at the tile level

The same invariant applies *across* the tile→kernel boundary: the kernel materializer must not re-recognize structure
the tile IR already holds. The **atomize** step (`lowering/tile/_atomize.py`, called when a warp / register-tiled
option is built — *not* a standalone pass) resolves the algebra→hardware-atom binding once at
RECOGNIZE time (`010_recognize._nodify_contraction` / `_atomize.bind_prologue_contraction`) and feeds it into the
contraction-shaped `Fold`, so materialize reads the operands /
`acc` off the node and only `factorize`s (the projection is peeled off the wrapping zero-axis fold's `lift` — its one
home). Resolving it before the schedule means an atom that **cannot** be
bound (e.g. a non-`Load` operand — a computed-cone / demoted matmul) never gets built in the bilinear shape: its
loads stay INLINE in a fold's lift, so the contraction reading (and with it the placed tiers) declines it instead of
failing several passes later:

- a `CONTRACTION` contraction → the `(a, b, acc, projection)` operand→role facts
  (`_atomize.bind_contraction`): the operands are named by the ⊗ **lift** (the `Assign` the fold accumulates) — B is its
  (n, k)-indexed `Load`, A is the lift's other argument, either a plain `Load` (clean gmem-direct) or, when loop fusion
  has inlined an operand cone, the cone as a zero-axis `Fold` stored INLINE on an operand edge (`_atomize.make_cone`
  — a STAT-FREE computed A, which rides the `sync` compute-fill like the norm→linear cone but carries
  no statistic prologue) — plus the fold accumulator and the projection. The STORED form is ONE `Fold` in the
  BILINEAR shape (`Fold.contraction` builds it, `is_contraction` / `Fold._contraction` read it back — a predicate and
  a derived reading, never a kind): pure algebra — the K `axis` + the shared `a` edge + the product `Channel`s
  `(b_i, acc_i)` read off `operands`, sharing the node's ARITY — and NOTHING else: placement and schedule are not
  node fields at all. The PLACED reading the tiers require is the SCHEDULE SLICE: a `TilePlan` carries the `(m, n)`
  output axes it tiles (bound by `Sched.tile_of` through `TilePlan.at`, from the caller's placement) and derives the
  `Side` geometry from them, so node and slice travel as a pair and a schedule can never ride a stored term. The
  stored term IS the λ form — the flat `(init, combine)` algebra that `Reduction` and the PLANAR demotion consume,
  read directly off the node (the retired `as_fold()` conversion became the identity at the collapse and is gone).
  An **operand is an edge** with two inhabitants — the two things an input can be: MATERIALIZED (a gmem `Load`) or
  COMPUTED (the node itself, stored inline). Tree ownership gives an inline node exactly one consumer — so there is
  no let table, no reference arm, and no resolve step:
  every downstream reader takes the cone's K seam straight off the view's `a` edge (`ops.cone_seam`); `lower`
  flattens it once, at the point of use. **Edge iff closed holds by construction**: operands bind POSITIONALLY to
  lift params, so an operand cannot see the fold's state or its siblings — the closure scan is demoted to the
  validation reading (1q) and lives with its one consumer, the cut (`_cut._captured_values`); closure is the precondition for lifting any subtree into its own kernel (a placement
  cut). Flash's `P = exp(s − m)` is `combine`'s derived singleton-specialization internals — material BELOW the seam
  lattice, never a cut target — and flash's QK score is a hoisted operand edge of the kv stream (step 7): closed by
  construction, reading only the enclosing iteration var, never state. The derived PV contraction's A edge is
  precisely where "no reference arm" becomes visible: `P` is already in a register, and since an edge is a `Load` or
  an inline node and nothing else, pointing at it means wrapping it in the one-stmt node `{o}__p = copy(P)`
  (`_derived_expect_fold`) — the copy IS the reference, and the rename off the accumulator name keeps it stable
  against the twist program's positional temps. That node is the edge directly; there is no empty-body
  `Fold.projection(body=(), operands=(…,))` cone wrapper around it, since with no per-cell work `cone_seam` bridges
  no stats
  either way and both spellings lower identically. The same edge vocabulary applies to **B**: a pure, closed B
  producer can remain inline and fill the Tensor Core B slab directly. This is a generic producer-to-contraction
  fusion over ordinary tensor algebra; storage-format reconstruction must already have decomposed before this band.
  Binding off the lift rather than off "the first (m, k)-indexed `Load`" is load-bearing: a cone-INTERNAL load is
  (m, k)-indexed too, so the positional rule bound gemma's GeGLU combine as `gate @ W` and silently dropped the gelu and
  the up projection. Refusing to bind a stat-free cone at all is equally wrong — it demotes the cell to a PLANAR
  scalar fold, which cost the gemma-4 M=256 post twin 144 ms against 4.3 ms bound. The same rule holds on the **B side**:
  a lift whose B operand is a computed cone never falls through to the positional rule (that binding dropped the fp8
  decode cone's scale from the kernel). A **storage decode times k-invariant multiplicative factors** chain — a
  decode of the (n, k) load ⊗ factors constant along the reduce axis — binds through the **mul-hoist**
  (`_atomize._hoist_k_invariant_factors`):
  the factors commute out of the fold onto the accumulator in the epilogue (`Σ_k a·(s·w) = s·Σ_k a·w`, the split-K
  reassociation category) and the decode is ABSORBED by the B load's own storage dtype — every consumer converts a
  bits-carrier element by dtype (the render's promote on the scalar tier; the gmem-direct fragment load's per-element
  convert on the warp tier, `emmy_mma_load_b_gmem<__nv_fp8_e4m3, __half>`). The chain's leaf is recognized by TRAIT —
  `ElementwiseImpl.decodes` names the storage dtype an op is the decode cast for (the fp8 family today), so a new
  storage format registers one decode op and never touches the binding arm. The warp staged transports carry a
  storage-dtype operand as a RAW BYTE SLAB (each operand slab sized at its OWN element width; ldmatrix is b16-only
  below sm_100a, so the byte slab drains through a cooperative per-lane gather — the gmem fragment loaders' lane map
  pointed at smem, converting to 16-bit fragments under a k16 atom / repacking raw bytes under the fp8 k32 atoms —
  bit-identical to gmem-direct; `resolve_warp_stage`'s byte-slab arm states the 16-divisibility legality, and the
  cp.async byte slab pads its rows by `_stage.BYTE_SLAB_PAD` for the drain's bank spread). The scalar resolver still
  declines 1-byte elements (its fill math is unaudited there), so the scalar tier rides gmem-direct. The arm's boundary is
  the algebra: a k-VARYING (2-D block) scale does not commute and declines; an additive zero-point (affine cone) and a
  codebook (gather) decode are outside the multiplicative form; any other computed B raises, and the recognizer
  demotes the cell to PLANAR (the guardrail contract). The binding now happens ONCE at **recognize time** (`010_recognize._nodify_contraction` — every
  recognized contraction, per-cell scalar included, stores in the bilinear SHAPE — one `Fold` whose operands are
  `(b, a, b_i…)` under a `multiply` lift and an additive combine; an
  unbindable one — a 1-D matvec-shaped output — keeps its loads inline in a fold's lift instead, so it **derives**
  `PLANAR` and takes the reduce tiers at schedule dispatch — no role rewrite. Since the collapse there is ONE stored
  kind and **every role derives from arity** (`Fold.role`, never stored): `FREE` with no axis,
  `TWISTED` off the stored combine's twist family, `CONTRACTION` off the bilinear reading
  (`Fold._contraction`) alone — split-K's outer reduce derives `PLANAR`, and `Fold.composed` is the structural probe
  `030_split_reduce` reads, never a role — `PLANAR` otherwise — so "a contraction"
  below always means a fold that reads as one, and the lowered `Loop`'s annotation falls out of the same read; the
  schedule fork only
  stamps a `tile` onto a `replace()` copy, and `_factor.factorize` reads the facts off
  the placed node instead of `lower()`-ing the contraction and pattern-matching the result. A `STAGE` pin follows the same rule: the
  option builders resolve it against the built node ONCE (the warp / scalar stage resolvers — transport
  eligibility, the slab K-chunk `bk_elems`, the depth clamps) and stamp the resolved `Stage` (or `None`, gmem-direct)
  on the `TileOp`, so the materializer's one staged driver applies it verbatim, deciding nothing. Fitting the smem
  budget is part of resolving: a tile whose single depth-1 slot already exceeds it declines to gmem-direct (the warp
  slab is codec-sized and cannot shrink; the scalar resolver steps `bk_elems` / depth down first), so every offered
  stage row materializes within budget — a resolved-but-unfittable row would only die at `validate(ctx)`, leaving an
  un-lowered `TileOp` in the tune's terminal (issue #327). TMA's box rank follows the flash convention on the matmul
  tiers too: the box's data plane is the operand's trailing 2 gmem dims, and extra LEADING dims ride as extent-1 box
  dims whose origin coordinates are the operand's own index exprs — eligible when those exprs don't move with the
  tile or the K loop (the TMA operand box-rank rule), so a model's `[1, seq, K]` unit-batch view stages exactly like the
  rank-2 snippet twin (the gemma in-model matmuls' TMA lockout). A **transposed B** (the serving `F.linear` layout —
  B given `(N, K)`, K gmem-contiguous, `Fold.b_trans` — derived off the B edge's index) stages on the warp tier
  through an **N-major slab**:
  the B slot takes A's geometry (`tile_n × bk`, K the inner dim — stride-1 in gmem and smem alike, so cp.async chunks
  and the TMA box stay contiguous; `Operand.trans` stamps the layout) and the drain is the plain no-`.trans`
  ldmatrix (`LdmatrixLoad(b_trans=True)` — the same staged path flash's K slab rides). Both operands' inner span is
  then the K chunk, so the eligibility alignment gates on K alone (the warp cp.async / TMA staging gates) and
  the B swizzle mode derives from `bk_elems` like A's. Historically the transports declined transposed B and the
  serving `.lin` forks ran gmem-direct only — the 1.3–2.75× gap class to cuBLAS on the 5090 goldens. The scalar
  tier still declines it (its plain-`Load` drain has no transposed variant; pin-only tier). The **sync compute-fill**
  (the fused computed-A edge) stages its transposed B the same way: every B fold channel — canonical K-major or
  transposed N-major — rides a vectorized `cp.async` `Operand` that flies UNDER the compute fill (the per-cell
  strided copy fill it replaced was the serving fused edges' weight-stream deficit), and the asymmetric B-only `d2`
  prefetch ring is enumerable on transposed-B fused edges too. One staging fact
  is derived at materialization rather than resolved here because it is layout, not eligibility: a slab feeding an
  mma drain is **swizzled** (`_stage.pick_swizzle_atom` picks B32/B64/B128 per operand from the slab's inner row span;
  TMA permutes the 16 B chunks in hardware during the box copy, a cp.async fill applies the identical XOR in software
  on its destination index, each staged `LdmatrixLoad` XORs the address back, and the kernel stays bit-identical to
  its unswizzled sibling — swizzle relocates smem bytes only, which is what keeps the ldmatrix drain free of
  shared-memory bank conflicts; the unswizzled cp path was 4-way/8-way conflict-bound on sm_89, the measured residual
  to cuBLAS there. The fused-edge **sync compute-fill** slabs swizzle the same way — the compute/copy
  fill's slab `Write` carries the mode (the identical flattened-index XOR, one VECTOR store per 16 B run; V scalar
  2 B stores at 16 B thread stride were 8-way store-conflicted), the canonical-B cp.async fill its `Operand`, and
  the drain reads each slab back through its own mode. Unswizzled, the gemma-shape fused edge drained 294.9 M ld
  conflicts / 82.5 M LSU inst on the 5090; the swizzle + vector fill store recovered -29% (std) / -41% (fm).
  Scalar-`Load`-drained slabs stay plain row-major).
- the **MONOID-producer composition** — the fused norm→linear edge and its N-channel form, the gate/up MLP edge —
  binds at recognize time too (`_atomize.bind_prologue_contraction`, structure-only): a projecting zero-axis fold
  over a per-row `PLANAR` statistic whose tail is one or more ⊗-folds of one shared A value nodifies to
  `Fold.projection(fn=projection, operands=(fold,))` — over ONE `role=CONTRACTION` `Fold` whose LIFT multiplies each
  ⊗-channel's B against the one shared inline A cone edge (itself a node tree: the statistic is the cone's inner
  `Fold` operand, the per-cell normalize its own `lift`). Sharing is edge REUSE — the product semiring outputting N
  matrices — and the node schedules and lowers as ONE unit (one `TilePlan`/`Stage`/`ReducePlan` row;
  `Fold.loop` splices the shared cone once and carries the
  N-component product-monoid accumulator set) — a product-monoid fold: components never interact
  per step; the combine — SwiGLU — is projection, riding the wrapping zero-axis fold's `lift`. Channels whose B
  layouts disagree were never legally fusable, so they simply never product (a formation gate, not a node assert).
  It is offered as a fork SIBLING of the
  cooperative reduce form (option-0 stays the coop row; the warp mma rows ride the mandatory `sync` compute-fill;
  dtype / geometry legality stays schedule-side). This retired the pin-only
  `_prologue_warp_option` rescue. The **degenerate M=1** composition (per-token decode: the unit row axis elided,
  `free = ()`) binds too — a synthesized unit free axis keeps the column grid; without it the fused kernel
  schedules at grid 1, ~300× off the memory floor. SSA renames track the stored algebra through the `Fold`
  rewrite handler (`rename_combine` — a verbatim combine left the cooperative combine reading a state name the
  renamed body no longer defined).
- a cooperative / ILP reduce (`PLANAR` / `TWISTED`, or a non-output-tiled `CONTRACTION`) needs **no** binding here — its
  accumulator dtype + the shuffle/tree fold mechanism are **derived** at materialize time (`emit_combine` off the fold
  node's `Reduction` view + `ReduceStage.combine`), never stored. Its one schedule-time staging decision follows the same
  resolve-once-structurally rule: the schedule detects the fused norm→linear shared row when the cooperative
  partition is chosen and stamps a `sync` `Stage` naming it (`smem`) on the `TileOp` — a derived schedule field, not a
  knob — so `_factor._tile_reduce_axis` only applies it, never re-detects.

**The λ-foldMap storage (1m–1p, completed at step 7).** EVERY `Fold` stores pure algebra — there is no `step` field
(deleted at step 7; `Fold = (axis, operands, lift, init, combine, dtypes)`, formation asserts the λ spelling): a
`lift` `Lambda` — `λ(k, v₁…vₙ) → S`, the iteration var first, one param per operand edge bound POSITIONALLY, its
results the element's SINGLETON state
(softmax's is `(x, 1)` — ι spelled in the lift, a literal component a bare float) — plus the TRUE monoid's flat
`(init, combine)` fields (1r: the `Monoid` wrapper class is dissolved — `M(op…)` survives as the free componentwise
pair constructor in `ir/stmt/algebra`, `component_ops`/`degenerate` as free shape-readers on the combine, the rename
lockstep as `rename_combine`, the S×S→S arity check in `Fold.__post_init__`) whose combine threads the fold's REAL
accumulator names (its results). Everything else is DERIVED, never stored: the streaming step is combine specialized
at the singleton (the `Accum` forms
for a componentwise monoid, each landing right after the lift stmt defining its value; the exp family's generated
merge for a twisted one), and NO loop-level algebra annotation exists — `Loop`/`StridedLoop` carry only their
`AxisRole`, and `Fold.from_loop` reconstructs the algebra from the loop BODY alone: the degenerate facts off its
`Accum`s, the twisted spelling by regenerating the exp-family merge over the body-derived `(state, terms)` candidates
and byte-comparing (`_extract_twisted_self`; a split partial extracts against the pre-slice fold — `from_loop(loop,
like)`). The lowering layer's one algebra reader is `passes/lowering/_reduction.Reduction` (the materializer's and
`030_split_reduce`'s view: `combine_states` / `state_merge` / `identities`). The
twist family is selected STRUCTURALLY — the stored combine must BE the exp/LSE generator's program, asserted at
formation; the state-component roles read off the singleton shape: pivot = component 0, literal-1 = denominator,
value name = expectation). The COMPOSED evaluations derive too (step 7): flash's kv stream λ-spells with its QK score
contraction a HOISTED inline-node operand edge (reading the enclosing kv var, never state) and its PV
contraction SYNTHESIZED — and memoized, one identity per stored fold — inside the derived blocked evaluation
(`ir._twisted_derived_step`, byte-identical to the retired in-step spelling); split-K's outer reduce is the
IDENTITY-LIFT composition over its one inline sliced contraction node (combine at that singleton embeds the operand
verbatim — no outer `Accum`s; `Fold.composed` is the one read of the composition, shared by `Fold.role` and
`030_split_reduce`'s structural arm). `Fold.step_stmts()` is the public per-cell read every former `.step` consumer
goes through; `.loop` splices only the operand edges the derived step did not consume. `Fold.from_loop` returns
`None` for a non-λ-representable loop (an effectful / raw-block body — the callers keep the raw-loop-IR projection
escape, an impure-bodied zero-axis fold), and its byte-identity gate compares the derived body/axis/unroll only —
the role annotation is the fold's own derived read, so an unbindable matvec captures a CONTRACTION-shaped loop and
derives `PLANAR` (the 1l
demotion, now a formation fact; `_extract_lift` accepts any PURE prefix). The inverse — un-hoisting a computed
edge back into the lift body — has no implementation at present: its only caller was the scheduler's collapse arm,
and it returns with the enumerator. Kernel identity is the α-INVARIANT TERM HASH (`Fold.structural_key`: canonical
renumbering in first-appearance walk order plus hash-time ANF body-order canonicalization — the stored term is never
reordered, the lowered nest keeps storage order, identity does not; the lowered-nest identity is retired), consumed
by `Op.cache_key`'s TileOp arm and `Graph.structural_key`'s op field. The ZERO-AXIS fold is what `Map` was — no
`fn` / `sources` fields survive it (`lift` / `operands`, built by `Fold.projection`): operands bind positionally to
`lift.params` (one param per operand RESULT COMPONENT — a product operand binds every channel accumulator — so
lowering splices verbatim), and `lift.results` replace the retired `out` last-def convention.

**Effects sit at the kernel boundary (1q).** Every recognized term's `fn` is a STRICT pure `Lambda`: the root-store
`Write`s — and the rms/softmax output-sweep `Loop` around them — left the term for `TileOp.stores`, a tuple of
`Store` decorations (the `Write` held whole for field fidelity, plus the sweep axis/unroll). ONE reconstitution rule
(`ir.effect_tail`, read through `ops.projection_tail`) reassembles the effectful stmt stream wherever the pipeline
consumed it out of the body — the scheduler's tail gates (the transposed band's no-sweep condition, the shared-row
stage detection, the split-K atomic-distributivity gate), the materializer's projection peel and flat-root arm, and
`030_split_reduce`'s projection/cell reads — so the lowered kernels are byte-identical to the stored-`Write` era
(the conversion sites run `split_effects`, whose round-trip gate is the same 1o construction-time byte-identity
pattern; a declining shape keeps the raw spelling). `030`'s split partials nodify their sliced annotated `Loop`
into a `Fold` source and carry the workspace stores as boundary `Store`s; the register strip fans the root store
out per copy. A zero-axis root without an operand edge reassembles its boundary stores after emitting its own body,
so an output sweep still encloses every projection statement that reads its coordinate. A scalar register tile's
per-cell projection copy protects every axis bound inside that reassembled tail (`Body.axis_names`), so an
output-sweep coordinate stays bound by its loop while only per-cell SSA values gain
the cell suffix. If enumeration deliberately leaves a term unmapped because no schedule row is legal, materialization
maps its free axes directly to the scalar grid before lowering; the valid guardrail term therefore remains executable.
The interim `effectful_lambda` is DELETED — what remains impure is exactly the raw-loop-IR kernels
that are not recognized algebra (the un-recognized flat escape cell, `030`'s finalize — `Init` seeds + the
un-annotated `StateMerge` merge `Loop` — the prologue'd split partial, and the coop norm→linear/geglu sibling's
composed contraction tail), formed through the one `Fold.projection`-private `_loop_ir_fn` arm and dying with the
recognizer's growth toward totality. The closure scan (`_cut._captured_values`) demoted to the validation reading of
edge-iff-closed.

**The site-local value grammar + `WORK` (step 7).** Knob KEYS address tree sites (the path codec); since step 7 the
VALUES are site-local too, with the kernel-global worker inventory spelled exactly ONCE in the `WORK` family:
`w<M>x<N>[+p<np>]` (warps; `+p` the producer band the retired per-row `WSPEC` key spelled) / `t<N>x<M>` (the scalar
thread tile) / `t<N>` (the 1-D cooperative width). `TILE` sheds its `a:`/`w`/`n` worker tokens
(`<atom>/f<FM>x<FN>[/k<bk>]` warp | `f<fn>[x<fm>]` scalar — the tier discriminator IS the worker kind) and `REDUCE`
its coop width (`[g<n>[a|k]][/coop[-t]][/r<n>]` — the GRID finalize letter is KEPT: `a`/`k` is the atomic-vs-deferred
finalize MODE, not an axis token). `seal_workers` (`ir/tile/ops.py`) is the one stamp chokepoint: it derives
`TileOp.work` + `knobs['WORK']` from the resolved TILE slices, the coop width off the REDUCE slices, and the producer
band off the resolved `WarpSpec`, failing loudly on cross-site disagreement (one kernel, one inventory; a 1-thread
register-strip inventory stays empty — per-cell launch geometry remains derived). The enumeration itself speaks
TYPED SLICES end to end — the `search/space.py` catalogs hand out `TilePlan` / `ReducePlan` objects built
structurally, and env pins resolve into the same objects ONCE at the top — so a codec string is spelled exactly once
per family, where a row becomes stored state (the fork row and the stamped `TileOp.knobs`). `_materialize`
dispatches warp-vs-scalar on the PARSED plan, and `resolve_site_tile` is the one rule disambiguating an empty site
`TILE` beside a thread `WORK` from the coop tier. The retired embedded-worker spellings RAISE — the worker widths
have one home, so a value carrying its own cannot decode into a second, self-contained reading that silently
disagrees with `WORK` (`values_equal` still bridges the atom aliases and the codec's normal form); the stored golden
corpus was re-spelled mechanically, replay proven digest-identical (the one-shot re-spell script is gone with the
grammar it read).

## The divide rule: `split` an iteration axis

`lowering/tile` carries one one-kernel→graph-fragment rule:

- **`030_split_reduce`** splits the **reduce axis** (the REDUCE codec's `g<w>` cross-CTA shard): the SAME
  computation, its K partitioned across CTAs into a partial + finalize. It runs AFTER its decision — the `g` row was
  chosen FOR the split form — so the partial carries the decided knob row verbatim and the finalize is deliberately
  `_mapped`: both **opt out** of re-recognition, because re-entering would discard the very decision being realized.

The fragment idiom's re-entry semantics are the rule's own: `030` opts its halves OUT of recognition, while a rule
that emits plain un-mapped `LoopOp`s hands them back to `010_recognize` on the pass-scan restart. The shared fixpoint
is what lets such rules compose without knowing about each other.

**Placement routing (phase 4).** `PLACE@<child-path> = cut | fuse` is the per-seam edge property on the recognized
tree — a `PLACE` site is every NON-ROOT node (the child names its parent↔child seam; the cone edge spells `PLACE@a`
through the view-role label), spelled/resolved by the same tree-path codec as the schedule families. Resolution is
TWO-LEVEL and RECURSIVE, decided BEFORE any schedule fork exists (`010_recognize` consults it right after the lift /
prologue bind): a ROUTING golden entry — an ordinary kind entry whose knobs are `PLACE` keys ONLY (the cut set,
never a schedule; the loader rejects a mixed entry, and the schedule golden tier skips routing entries, so the
retired single-namespace hazards — a cut row tying its knob-identical fused twin — cannot return) — or an
authoritative `PLACE` pin picks a cut seam; the realizer (`lowering/tile/_cut.py`) splits the tree there: the child
subtree becomes a plain un-mapped `LoopOp` computing the seam value into a `…__cut_…` workspace over its DERIVED
index space (the enclosing axes its lowered body reads, loop-invariantly nested; a fold child — one that FOLDS AN
AXIS — bridges carrier state as **f32** per the split-reduce workspace rule, while a zero-axis projection child is
the value seam and keeps its leaf operand dtype: in the one-kind IR every node is a `Fold`, so the axis is the
discriminator, not the class), and the parent
consumes a plain workspace `Load` (every edge admits `Load` — the cut terminal). Both pieces re-recognize as fresh
roots on the pass-scan restart and resolve their OWN `(kind, shape)` entries through the full deploy hierarchy —
recursively: the cone piece re-recognizes as the rms_norm shape and its own entry (or a bare pin) cuts the statistic
out, yielding the cascade statistic + scale + plain matmul, every piece joining an EXISTING golden kind's evidence.
Computed-A routing uses the fused-key convention on both sides: the live tree supplies the computed-A fact before a
schedule offer exists, while a persisted `PLACE@a` supplies it for a stat-free activation cone with no second reduce
axis. Keeping one key prevents the routing entry from recursively matching its own materialized producer.
**Fuse is the default by ABSENCE** — no routing entry and no pin leaves recognition byte-untouched (digest-verified),
and cut is evidence/pin-only. Cut legality is structural: single-component CLOSED children only (`_captured_values`
in its demoted validation role — flash's state-capturing `P` is simply not cuttable), and the pure-copy degenerate
(cutting an empty-body root projection's only operand, whose parent would merely copy the workspace out — the
non-terminating case) is refused. Loop fusion brakes on `__cut_` workspace producers — a decided placement is not
fusion's to undo (tune-mode slicing re-enters fusion with the pieces as ordinary pairs). The old `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizers stay retired; their non-default placements return only as
routing entries re-seeded by fresh `--ab` evidence (phase 5 — the 020-era `cut_cone_*` schedule entries stamp the
OLD piece shapes' keys and are re-seeded rather than joined).

The atom spec is subtyped by kind (`ir/atom.py`: `AtomKind` is the fixed mma cell selected by name; `ScalarAtom`
is the plain scalar fma cell). The contraction binder (`bind_contraction`) is loop-addressable so warp-flash can later
reuse it on flash's nested QK^T / PV; flash's score IS a role=CONTRACTION **fold** on a hoisted operand edge of the
kv stream (its PV twin synthesized in the derived blocked evaluation), so warp-flash is
just that node gaining a warp `TilePlan` — no new path.

An atom's logical cell and PTX instruction shape are separate. The Volta `mma_m8n8k4_f16_f32` atom is one logical
16×16×4 warp cell because one instruction performs four independent 8×8×4 operations; its fragment layout maps those
groups onto four output quadrants and carries 2/2/8 A/B/C registers per lane. It accepts only materialized A/B edges:
SM70 has no `ldmatrix`, but materialized f16 A/B edges may use synchronous-copy staging: ordinary vector global loads
and shared stores fill the existing slab ring, and the same cooperative m8n8k4 lane map gathers fragments from shared
memory. The generic staged-loop scheduler still owns `d<n>` slot rotation and `/p<n>` register-fragment pipelining;
blocking copies make deeper shared rings correct but do not promise copy/compute overlap. Computed operand edges,
C-to-A repacking, and flash still decline this atom. Target capability predicates select this family below SM80 and
the established `m16n8k16` families on SM80 and newer; an incompatible atom or copy-transport pin fails instead of
lowering through instructions the target cannot execute.

**The f16-accumulate atom sibling** (`mma_m16n8k16_f16_f16`, C→f16 — atom names follow
`mma_<shape>_<ab_dtype>_<acc_dtype>`, the compressed PTX/CUTLASS D.A.B.C order; the historical acc-unspecified
spellings stay as parse aliases for the f32-accumulate atoms): on the consumer GeForce dies (sm_86/89/120)
f32-accumulate HMMA runs at HALF the f16-accumulate rate, so this atom keeps the whole mma chain on the full-rate f16
accumulator and the lowering promote-folds the packed f16 partials into f32 shadow fragments per K chunk
(`FragmentPromote` — the staged bk slab is the cadence; gmem-direct promotes every `_atom._F16ACC_STEPS` steps plus a
final fold; flash promotes the P@V accumulator per streaming KV block, folded in at the `O·α` rescale point, while the
score node ALWAYS stays f32-accumulate). Precision-gated enumeration, off by default —
the precise `EMMY_F16_MMA_F32_ACC` pin is authoritative on any target, else the `EMMY_FAST_MATH` umbrella offers it on
the consumer-die ccs only (`_F16ACC_CCS`); a `TILE` pin naming the atom (or the flash golden's axis-keyed
`TILE@<pv_k>` spelling) bypasses the gate — pins are authoritative. The realized fork is identified by the `TILE`
codec's atom token and priced by the `MMA_acc_bits` feature; f16 only (mma.sync has no bf16-accumulate form).

**The move catalog** (`search/space.py`) is the permitted-move enumeration the schedule emit forks over, keyed on
`AxisRole`: `scalar_tile_moves()` is the legality-guarded scalar register-tile product (`par × reg`, `block_threads ≤
1024`) with per-cell `""` as the conservative option-0, crossed with the warp / reduce / stage move families
for an unpinned contraction so `compile` / `tune` explores the space (each row → a structural
contraction-fold leaf keyed `TILE@<axis>` in a hierarchical `build_fork_tree`; an env pin wins via `Knob.narrow`).
The producer band is the fourth level (option-0 `""` = uniform SIMT — since step 7 a resolved band
is spelled in `WORK`'s `+p<n>` suffix, never a per-row `WSPEC` key) — offered only on a warp row over a
resolved **TMA** stage without a cross-CTA split, and resolved/thread-budget-gated at materialization
(an ineligible spec degrades to uniform). A computed-A (fused-cone) contraction enumerates its own
warp-only rows (the mandatory resolved `sync` compute-fill stage at BOTH depths
(`d1` + the asymmetric B-only prefetch ring `d2` as fork siblings — the M=512 occupancy loss inverts at decode M,
so the depth is measured per shape), crossed with the shared `RASTER` launch-order candidates (its B stripes
re-stream per M-tile row, exactly the grouped order's L2 reuse — `gn8` measured −8% on the gemma gate_up fused
edge, 5090) and — single-channel nodes only — the **redundant-statistic split-K** rows: the contraction K slices
across CTAs while the k-invariant stat prologue stays full-row in every partition (each recomputes it, cheap
exactly on the small-free decode shapes the split-K CTA-count cap admits), the per-cell cone σ-reindexed to
absolute k and the wrapping zero-axis fold's projection folded into the deferred finalize (the split-K option's
computed-A arm
→ `030_split_reduce`'s structural path). Multi-channel (gate/up) nodes split too: the synthesized fold loop
carries the true N-component identity-family carrier (one additive state per channel), the partial stores each
channel's raw C fragment to its `ws[comp, ksplit, *cell]` slice (the per-acc `RegStore` arm — no ⊗-combine in
the partial), and the deferred finalize folds every component before applying the combine projection once.
Still no scalar / gmem-direct / WSPEC rows; the compute-producer role for the fused edge is the anticipated
`RoleKind` extension. The **flash-form fork**: a `TWISTED` streaming contraction pair (the flash tree) offers its
structurally-different schedules as ONE prior-ranked fork — the warp (fragment-resident) rows over
`twisted_warp_moves()`'s `(warps-per-CTA × key-atoms-per-block × query-tiles-per-warp)` geometry grid (option-0 = the
conservative one-warp / `2·atom_n` block / one tile; the third dimension is the `TILE` codec's `f<FM>x<FN>` reg_m —
each warp streams `fm` independent `(m, l, O)` chains against shared K/V fragments, FA-2's in-flight ILP; the Q@K /
P@V mma `TilePlan`s are the two sites' own values, reconciled at the stream), the scalar
register-vector CHAIN (the FA-2 shared-score form), then the cooperative / per-cell reduce-partition escapes — every
leaf row spelling the same `TILE@dd` / `TILE@pj` key pair plus the BARE `REDUCE` / `STAGE` the stream is primary for,
and its ONE `WORK` inventory (decided-empty where a form doesn't tile). The **flash split-KV** rows are fork siblings
of the un-split warp rows on an under-occupied grid (the occupancy read is the warp row's OWN launch grid — the
shrunk query axis, the value axis gone), and a cross-CTA `REDUCE=g<n>k` pin selects them: the plan stamps
onto each row's `Fold` node and `030_split_reduce` realizes it as a fragment-resident partial (the kv stream windowed to
the CTA's slice, its absolute base/bound on the sliced axis's `Axis.window`; raw `(m, l, O)` state to an f32
`__partial` workspace) plus
an LSE-combine finalize — kernel finalize only (the twisted `e^{Δm}` rescale can't be an atomic). A static kv must be
block-divisible; a **symbolic kv splits too**: the slice width is the bn-aligned runtime `ceil(S/(cta·bn))·bn` (a
composite `Dim`) and each slice stops/masks at its absolute end `min((s+1)·B, S)` (`Fold.bound` — a mid-tensor
slice end reads VALID next-slice keys the extent-only tail masks would keep), an empty last slice contributing the
exact carrier identities; the split partial guards every state write with the symbolic-M `m_guard` (the tail CTA's
clamp-read overhanging query rows would otherwise write into the next head's workspace rows). It pays where the
un-split grid starves the SMs (few heads / short query axis: the 2-head hd256 seq-512 shape runs 33.6 → 11.3 µs
under `g8k`, parity with torch SDPA's internally-split flash; the symbolic hd512 dynM stream 135.2 → 116.7 µs
under `g2k` on the 5090). Any
other non-empty `REDUCE` pin drops the streaming rows and stays the reduce-tier escape, which is the tier that
realizes a partition. `TILE` pins narrow by MATCHING each site's own catalog, codec-canonicalized so `a:scalar` ≡
`""` and `f64x1` ≡ `f64`: an explicit `TILE@<axis>` pin names one site and is authoritative there, while a BARE pin
fans out to every eligible site and cannot say which it meant — so it narrows where it matches and leaves a site it
names nothing at alone (`Knob.narrow`'s no-match-keeps-full-list). That is what lets `TILE=a:scalar` keep the
per-cell tier, `TILE=a:scalar` + `TILE@<pv_k>=f<d>` pin the CHAIN row deterministically, and one bare pin spelling
the f16-accumulate P@V plan (the masked-flash golden form) select that variant while the score keeps its own
f32-accumulate catalog — the pair then reconciling at the stream. A bare pin that named nothing at a site decides
nothing there, precision gates included. Each warp geometry row crosses with its **K/V operand-stage** candidates
(the stream's bare `STAGE` — gmem-direct option-0, then the resolver-gated cp.async AND TMA ring depths — the
batched K/V operands encode as rank-N TMA boxes with leading extent-1 dims, the load's own batch/head index exprs
riding as origin coords; cp.async slabs take the +16 B row pad, TMA slabs stay dense under the hardware swizzle; the
resolved `Stage` rides the `TileOp` and the streaming step becomes the `staged_kloop` drain, K/V slabs kept in each
operand's own layout so staging stays bit-identical to gmem-direct). **Both transports also stage a symbolic
(dynamic-`seq_len`) kv**: TMA rides the runtime globalDim and zero-fills the box overhang past the last key; cp.async
(which has no OOB zero-fill) clamp-reads the tail chunk's key rows to the last valid key. Either way the streaming
drain's tail masks (the same clamp the gmem-direct symbolic path makes) zero those keys' P columns exactly, so the
masked-flash `.dynM` kernel stages at bit-identity to gmem-direct on any sm (the `staged_kloop` ring allocates the
full depth and the last-chunk clamp / loop bound ride the symbolic `Dim`; a producer band over a symbolic kv is not
built). A resolved TMA row additionally rides the `+p` producer band in `WORK` (the matmul tier's legality,
`32·aux ≤ 32·um`; measured occupancy-negative at flash's CTA scale — offered, honest, not the default; the per-edge
`split` groups arm one mbarrier each, so they take no band). The `split` transport granularity — one group per staged
edge, K refilling under softmax + P·V and V under the next step's Q·K, with Q staged through smem too — is a
`stage_moves` member gated by its STRUCTURAL predicate (`_legality.stage_split_groups`: ≥ 2 staged edges consumed at
distinct positions of the derived evaluation, which a contraction's single multiply never is, so the matmul
resolvers decline it). The chain / coop / serial escapes stamp the decided-empty `STAGE: ""`. Staging additionally
requires the K/V (and,
for `split`'s staged Q, the A) BUFFER dtypes to match the atom's operand dtypes — the slab fills byte-copy and cannot
convert, so a wide traced intermediate feeding the stream would deposit garbage; gmem-direct fragment loads convert
per element and keep the warp tier either way. To keep that gate from silently disabling staging on real models,
traced dtype CASTS are first-class: a dtype-changing view splits into a source-shaped elementwise `copy` + a pure
map at the frontend (`optimization/005_split_cast_from_indexmap`). When gate-free loop fusion inlines a closed V map
cone, flash recognition factors it back into a canonical feeder workspace, so the stream still sees an atom-dtype
operand it can stage (the gemma V-norm's f32 `mul` → f16 SDPA edge, the layer-0 findings' biggest lockout). That the
cast is *usually free* — a fan-out-1 pointwise producer absorbing it and
simply writing the narrow dtype — is not something loop fusion can be relied on to arrange: fusion may merge either
way, and on gemma-4 it consistently spliced the cheap cast into its CONSUMERS instead, leaving the wide producer
buffer alive. `optimization/007_sink_narrowing_cast` makes it deterministic, retyping the producer's OUTPUT and
dropping the copy whenever the producer's SOLE consumer is the cast. It is a retype, not a numeric change (an
elementwise op computes in its inputs' promoted precision and rounds on store), and it is what keeps a norm→matmul
edge on the plain mma tier: a mixed-dtype A has no copy transport, so without it the mixed-dtype A demotion diverts the
projection onto the `sync` compute-fill, which has no weight-prefetch ring — measured on gemma-4's gate/up as
1.12 TB/s against the 1.61 TB/s a clean-f16-A `d2/tma` sibling reached on the same 118 MB
weight. **A causal stream tile-skips**: when the score
prologue carries the triangular `Select` (`kv ≤ m` — detected structurally off the predicate, never a kernel identity),
the realizer bounds the stream at the CTA's last query row (`kv_end = min(seq, (grid_m + 1) · um·fm·atom_m)`, hoisted
into the `StridedLoop`'s for-init `end` override; the staged prefetch clamp re-pins onto the last needed chunk). The
bound is CTA-uniform (barriers stay legal) and every skipped step is the carrier's exact identity (`α = 1`,
`P = expf(−1e30 − m_i) = 0`), so the early stop is bit-identical — it halves the streamed keys/mma work on average,
paying wall-clock wherever the grid oversubscribes the SMs (1.67× on hd256 seq-2048) and re-opening the small-CTA flash
forms that previously paid double K/V re-streaming. **A banded stream additionally starts late**: a trace-time
`SdpaOp.sliding_window` stamp (the HF wrapper knows `config.sliding_window` + `layer_types`; the trace itself erases
the window) decomposes to a second coordinate `Select` (`kv > m − W`) beside the causal one — unless the band is
STATICALLY VACUOUS (static seq, `W ≥ S`), which the decomposition drops instead of emitting: a vacuous Select's
predicate constant-folds, the +0 mask term hoists out of the reduce loops, and the mask-chain walk below can no
longer resolve it — the fuse then silently degraded to cut and the softmax·P@V fell to the unstructured sequential
lowering (the gemma-4 layer-0 `seq 512 < window 1024` trace deployed a grid-1 kernel). Flash classification
reads the whole mask CHAIN off the rowmax feed (coord Selects and the explicit additive bias compose; the bias stays
loaded, it may mask more, e.g. padding), re-synthesizes each canonically, and the realizer derives the stream START
off the band predicate exactly as it derives the causal end (`kv_start = ⌊max(0, first_row − W + 1)/bn⌋·bn`, the
kloops' `k_first`). The flash recognizer follows the complete mask chain across the gate-free fusion boundary; a mask
spelling it cannot certify declines the fuse rather than being silently dropped. At seq ≫ W the sliding layers' stream is
O(seq·W), not O(seq²) — 40 of gemma-4's 48 layers at real context lengths.
Two catalog invariants hold: every recorded golden's `WORK`/`TILE`/`STAGE`/`REDUCE` stays a **member** of the
enumerated grids (the permanence test in `tests/compiler/test_golden_configs.py`, site-aware since the step-7
re-spell — membership means the replayed pin resolves to a slice the catalog hands out; a space edit can never
silently orphan a golden into unreachability again, the sixth sweep's `.s512` regression class; the scalar reg grid
carries the golden-informed deep-FM points `f2x6..f2x14`, `f4x6..f4x26` for exactly this reason), and a cross-CTA
split deploy
(`030_split_reduce`) stamps the decided knob row onto its **partial** kernel — the engine merges knobs forward on 1:1
rebinds only, so without the explicit stamp the graph splice dropped them and the deployed split recorded no
schedule identity (the A/B table then couldn't say what greedy deployed).
