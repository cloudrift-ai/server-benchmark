"""Atomize — resolve the algebra→hardware-atom binding structurally.

The warp matmul materializer needs to know which operand is the mma ``a`` vs ``b`` (by
axis-in-index), the fold accumulator, and the projection epilogue.
:func:`bind_contraction` reads them **structurally** off the annotated ``CONTRACTION`` reduce loop
— the operand ``Load``\\ s indexed over the K axis, the fold ``Accum`` target — and returns them as
the ``(a_load, b_load, acc, epilogue)`` facts ``010_recognize._nodify_contraction`` stamps onto the
:class:`~emmy.compiler.ir.tile.ir.Contraction` structural node at RECOGNIZE time (the node
is then the single source of truth — it re-derives ``b_trans`` off ``b`` itself). Reading the
binding **structurally** off the annotated loop — not a stored node kind — is what keeps the ⊗/⊕
algebra a property of the loop, so no per-algebra op-tree node class is needed. The cooperative reduce
needs no binding here — its accumulator dtype + shuffle/tree
mechanism are derived at materialize time (``emit_combine`` off the carrier + ``ReduceStage.combine``).

**Called from ``020_schedule``, not a standalone pass.** The binding is resolved when the tiled
contraction leaf is built (``_warp_option`` / the tiled ``_tile_option``) — so an atom that
**cannot** be bound (e.g. a non-``Load`` operand: a computed-cone / demoted matmul) is rejected at
fork construction, alongside ``_check_warp_static_k``, instead of failing several passes later.
Leading ``_`` so the pass loader skips this module.

**Flash contractions are not recursively atomized.** Flash is a ``TWISTED`` kv
``Fold`` over a ``Q@K`` :class:`~emmy.compiler.ir.tile.ir.Contraction` ``source`` +
a ``P@V`` one in the ``step``, lowered on the scalar tier (block=1) — each contraction carries a
scalar ``TilePlan()`` and factorizes through the one ``_factor`` contraction path. A tensor-core
flash tier would attach an mma ``TilePlan`` to those same nodes and route through that same path (no
bespoke emitter); ``bind_contraction`` stays loop-addressable (it binds the root contraction
structurally), and the recursive-atomize path is unused — the flash tree carries its per-node
geometry."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.tile import Channel, Contraction, Fold, Map, Store
from emmy.compiler.ir.tile.ir import refs_axis
from emmy.compiler.pipeline.pipeline import LoweringError


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs (the materializer's helper)."""
    return {v for e in index for v in e.free_vars()}


def _idx_vars_deep(stmts) -> set:
    """Every free Var name across the index exprs reachable in ``stmts`` (deep)."""
    out: set = set()
    for s in stmts:
        idx = getattr(s, "index", None)
        if idx:
            out |= {v for e in idx for v in e.free_vars()}
        for b in s.nested():
            out |= _idx_vars_deep(list(b))
    return out


def map_cone(body: list, root: str) -> list | None:
    """The backward cone of SSA ``root`` within ``body`` — the fused producer's compute, in body
    order. ``None`` unless every cone stmt is a scalar ``Load`` or a pointwise ``Assign`` (a pure
    MAP cone — a reduce-bearing cone, e.g. an rmsnorm scale, is not compute-fillable per cell)."""
    defs: dict[str, Stmt] = {}
    for st in body:
        for d in st.defines():
            defs[d] = st
    need, cone, seen = [root], [], set()
    while need:
        nm = need.pop()
        st = defs.get(nm)
        if st is None or id(st) in seen:
            continue
        seen.add(id(st))
        if isinstance(st, Load):
            cone.append(st)
            continue
        if isinstance(st, Assign):
            cone.append(st)
            need.extend(st.args)
            continue
        return None  # an Accum / Loop / Select in the cone — not a pure MAP producer
    order = {id(st): i for i, st in enumerate(body)}
    return sorted(cone, key=lambda st: order[id(st)])


def make_cone(cell: list, k_name: str, stat=None, sweep=()) -> Map:
    """Build a computed-A **cone** as a real node tree — the inline node its consuming operand edge
    stores (there is no let table: sharing is the product contraction's channel arity).

    The K seam is decided HERE, once, and lives ON the node: the maximal leading run of cone stmts
    that never index the contraction axis is **row-invariant**, so it joins the per-row statistic
    (``stat``, the :class:`Fold`, plus its scalar ``sweep``) in the cone's SOURCE node — one
    projected reduce, exactly the RMSNorm shape; the k-varying remainder is the cone's ``body``, the
    per-cell normalize. Everything downstream then READS that boundary (``ops.cone_seam``) instead of
    re-scanning stmts: the scheduler sizes the stat smem rows off it, the materializer runs the
    prologue once per tile row and the body per cell."""
    pro: list = []
    rest = list(cell)
    while rest and not refs_axis(rest[0], k_name):
        pro.append(rest.pop(0))
    prologue = Map(body=Body((*sweep, *pro)), sources=() if stat is None else (stat,))
    src = (prologue,) if (pro or sweep or stat is not None) else ()
    return Map(body=Body(tuple(rest)), sources=src)


def bind_contraction(loop: Loop, m_name: str, n_name: str, epilogue: Body) -> tuple[Load | list, Load, str, Body]:
    """Resolve the ``(a_load, b_load, acc, epilogue)`` operand→role facts for a ``CONTRACTION``
    reduce ``loop`` (the lowered ``Accum``-in-``Loop`` form) whose output is indexed by grid axes
    ``m_name`` / ``n_name``, with projection ``epilogue``.

    Reads the facts straight off the loop body — no op-tree node: the operands are named by the ⊗
    **lift** (the ``Assign`` the fold accumulates), B is its (n, k)-indexed ``Load``, and A is the
    lift's other argument — a plain ``Load`` (the clean gmem-direct contraction) or, when loop
    fusion has inlined an operand cone, the cone itself as a ``Map`` NODE (the computed-A form,
    which rides the ``sync`` compute-fill; the caller shapes it into the inline cone node via
    :func:`make_cone`, which also puts the K seam on the node). The fold accumulator is the loop
    body's ``Accum`` target.

    Binding A off the LIFT and not off "the first (m, k)-indexed load" is load-bearing: a
    cone-INTERNAL load is (m, k)-indexed too, so the positional rule bound gemma's GeGLU combine as
    ``linear_1_reduce @ W`` — cp.async-ing the gate projection into the A slab with the gelu and the
    up projection silently dropped. A stat-bearing cone still binds through
    :func:`bind_prologue_contraction` (it carries a statistic prologue and a column loop this shape
    has no notion of); flash's PV binds at fusion. An unbindable body raises, matching the warp
    gmem-direct guard. ``b_trans`` is not returned — the ``Contraction`` node re-derives it off its
    fold's ``b``."""
    k_name = loop.axis.name
    body = list(loop.body)
    loads = [s for s in body if isinstance(s, Load) and k_name in _idx_vars(s.index)]
    a_leaf = next((ld for ld in loads if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in loads if n_name in _idx_vars(ld.index)), None)
    fold = next((s for s in body if isinstance(s, Accum)), None)
    if fold is None:
        raise LoweringError("warp tier: contraction loop has no fold accumulator")
    acc = fold.name
    # Bind A off the ⊗ LIFT, never off "the first (m, k)-indexed load". Once loop fusion has
    # inlined an operand cone, a cone-INTERNAL load is (m, k)-indexed too, and taking it as A
    # silently drops the rest of the cone: gemma's GeGLU combine ahead of down_proj bound
    # ``linear_1_reduce`` as A and emitted ``linear_1_reduce @ W`` — a wrong kernel, cp.async-ing
    # the gate projection into the A slab with the gelu and the up projection gone. The lift names
    # the true operands: B is its (n, k)-indexed load, A is the other argument — a plain ``Load``
    # (the clean gmem-direct contraction) or a computed cone (which rides the sync compute-fill,
    # exactly like the norm→linear cone, just with no statistic prologue).
    lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
    if lift is not None and lift.op.name == "multiply" and len(lift.args) == 2 and b_leaf is not None and b_leaf.names[0] in lift.args:
        a_arg = next(a for a in lift.args if a != b_leaf.names[0])
        if a_leaf is not None and a_arg == a_leaf.names[0]:
            return a_leaf, b_leaf, acc, epilogue
        cone = map_cone(body, a_arg)
        if cone and not any(isinstance(st, Load) and n_name in _idx_vars(st.index) for st in cone):
            return cone, b_leaf, acc, epilogue
        # The lift names a COMPUTED A whose cone declined (an Accum / Select inside it, or an
        # n-indexed load riding it) — falling through to the positional (m, k) rule below would
        # bind a cone-INTERNAL load as A and silently drop the rest of the cone: exactly the
        # wrong-kernel class the lift binding exists to prevent. Raise instead — the recognizer
        # catches LoweringError and demotes the cell to PLANAR, which computes the full body.
        raise LoweringError("warp tier: the ⊗ lift's A operand is a computed cone that does not bind")
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    return a_leaf, b_leaf, acc, epilogue


def _cone_value_key(name: str, defs: dict) -> tuple:
    """The canonical value tree of SSA ``name`` within a k-loop body — Loads keyed by (buffer,
    index), Assigns by (op, child keys), a name defined outside the body by itself. Two folds
    share one A operand iff their lift values have EQUAL keys: fusion duplicates the producer
    cone per consumer (fresh SSA names, interleaved body order), so name/order equality is too
    strict but value-tree equality is exact."""
    st = defs.get(name)
    if st is None:
        return ("free", name)
    if isinstance(st, Load):
        return ("load", st.input, tuple(e.pretty() for e in st.index))
    return ("op", st.op.name, tuple(_cone_value_key(a, defs) for a in st.args))


def bind_prologue_contraction(op, free: tuple) -> tuple[Map, Axis, tuple] | None:
    """Nodify the **reduce-bearing (MONOID) producer cone** composition — the fused norm→linear
    edge: a projecting ``Map`` whose ``source`` is a per-row ``PLANAR`` statistic reduce and whose
    body is that statistic's scalar epilogue followed by a fresh free (column) ``Loop`` over one or
    more ⊗-folds whose shared A cone reads the statistic. One fold channel is the plain norm→linear
    (``rmsnorm(x)·nw @ w``); N channels sharing ONE A value with a pointwise combine tail is the
    fused gate/up MLP edge (``swiglu(x̂@Wg, x̂@Wu)`` — a product-monoid fold; fusion duplicates the
    cone SSA per channel, deduped by value-tree equality). Returns the same
    ``Map(body=projection, source=node)`` factorization the ``Fold`` spelling uses: the
    ``source`` is ONE computed-A product :class:`Contraction` whose A edge holds the cone INLINE —
    a real node tree, ``Map(body=<the per-cell cone>, sources=(<the statistic Fold>,))``, so the
    statistic is addressable and cuttable in its own right instead of hiding inside an operand body —
    with one :class:`Channel` ``(b, acc)`` per ⊗-fold (sharing is the node's arity; the node carries
    a **deferred** ``TilePlan()``); the ``body`` is the PURE
    projection (the combine tail + any stat-free prefix defs it reads) and the root ``Write`` a
    boundary :class:`~emmy.compiler.ir.tile.ir.Store` (1q). Returns ``(node, column axis, stores)``
    — the scheduler adds the axis to the grid and the stores to the ``TileOp`` —
    or ``None`` (not this shape; the reduce ``Map`` form stands alone).

    STRUCTURE-ONLY: no dtype / geometry / pin legality here — those are per-move scheduling guards
    (``_schedule``), so the same node offers whatever tiers the target legally supports."""
    if not isinstance(op, Map) or len(op.sources) != 1 or not isinstance(op.sources[0], Fold):
        return None
    red = op.sources[0]
    if red.role is not AxisRole.PLANAR:
        return None
    if any(isinstance(s, (Map, Fold, Contraction)) for s in red.step_stmts()) or any(
        isinstance(e, (Map, Fold, Contraction)) for e in red.operands
    ):
        return None  # a composed reduce (its partial or an operand edge holds a node) is not the bare statistic shape
    body = list(op.body)
    if not body or not isinstance(body[-1], Loop) or body[-1].is_reduce:
        return None
    stat_epi, nloop = body[:-1], body[-1]
    if not all(isinstance(s, (Load, Assign)) for s in stat_epi):
        return None
    n_ax = nloop.axis
    inner = list(nloop.body)
    if len(inner) < 2 or not isinstance(inner[0], Loop) or not inner[0].is_reduce or not isinstance(inner[-1], Write):
        return None
    kloop, tail_ops, write = inner[0], inner[1:-1], inner[-1]
    if not all(isinstance(s, Assign) for s in tail_ops):
        return None  # the combine tail is a pure pointwise chain (Loads ride the stat-free prefix)
    k_ax = kloop.axis
    kbody = list(kloop.body)
    accums = [st for st in kbody if isinstance(st, Accum)]
    if not accums or any(a.op.name != "add" for a in accums):
        return None
    defs = {st.name: st for st in kbody if isinstance(st, Assign)}
    loads = {st.names[0]: st for st in kbody if isinstance(st, Load)}
    kdefs = {**loads, **defs}

    def _load_vars(nm: str) -> set | None:
        ld = loads.get(nm)
        return {v for e in ld.index for v in e.free_vars()} if ld is not None else None

    # Bind each fold's (B, acc, A-value): the B operand is the (n, k)-indexed load of the ⊗ lift;
    # the other arg is the fold's A value. Every fold must share ONE A value (key equality).
    folds: list[tuple[Load, str]] = []
    a_names: list[str] = []
    for acc in accums:
        lift = defs.get(acc.value)
        if lift is None or lift.op.name != "multiply" or len(lift.args) != 2:
            return None
        b_name = next((a for a in lift.args if (vs := _load_vars(a)) and n_ax.name in vs and k_ax.name in vs), None)
        if b_name is None:
            return None
        a_names.append(next(a for a in lift.args if a != b_name))
        folds.append((loads[b_name], acc.name))
    if len({_cone_value_key(nm, kdefs) for nm in a_names}) != 1:
        return None  # per-fold A values differ — not a shared-operand multi-fold
    cone = map_cone(kbody, a_names[0])
    if cone is None or not cone:
        return None
    for st in cone:
        if isinstance(st, Load) and n_ax.name in {v for e in st.index for v in e.free_vars()}:
            return None  # the cone must be (m, k)-indexed — an n-dependent producer isn't the A tile
    # Every free SSA name the cone reads must be a statistic (the source reduce's carried state or
    # its scalar epilogue) — anything else is a shape this binding doesn't understand.
    stat_defs = {red.out} | {nm for s in stat_epi for nm in s.defines()}
    cone_defs = {nm for st in cone for nm in st.defines()}
    free_refs = {a for st in cone if isinstance(st, Assign) for a in st.args if a not in cone_defs}
    if not free_refs or not free_refs <= stat_defs:
        return None  # a stat-free cone is the demoted option's shape, not ours
    # The statistic prologue must be row-local: its gmem reads may index (m, its own reduce axis)
    # but never the column / contraction axes.
    if _idx_vars_deep([*red.spliced_step(), *stat_epi]) & {n_ax.name, k_ax.name}:
        return None
    # The combine tail: its reads must be the fold accumulators, its own defs, or STAT-FREE prefix
    # defs (a k/n-invariant value like SiLU's ``1.0`` scalar — its backward cone within the prefix
    # is prepended to the node epilogue so the store-side fragment epilogue can re-evaluate it; a
    # stat-DERIVED read would need the reduce's state at the store, which the fragment path has no
    # channel for). The Write stores one scalar from the tail chain (or the bare primary acc).
    derived = {red.out}
    for s in stat_epi:
        if set(s.deps()) & derived:
            derived |= set(s.defines())
    acc_names = {a.name for a in accums}
    tail_defs = {s.name for s in tail_ops}
    prefix_needs: list[str] = []
    for s in tail_ops:
        for a in s.args:
            if a in acc_names or a in tail_defs:
                continue
            if a in stat_defs - derived:
                prefix_needs.append(a)
                continue
            return None
    prefix: list[Stmt] = []
    seen: set[int] = set()
    for nm in prefix_needs:
        for st in map_cone(list(stat_epi), nm) or ():
            if id(st) not in seen:
                seen.add(id(st))
                prefix.append(st)
    if not write.is_scalar or write.values != ((tail_ops[-1].name,) if tail_ops else (folds[0][1],)):
        return None
    # B-layout agreement is a GROUP-FORMATION gate, not a node assert: channels whose B is stored
    # the other way round were never legally fusable (one shared A fragment, one slab orientation),
    # so they simply never group.
    if len({k_ax.name in bl.index[-1].free_vars() for bl, _ in folds}) != 1:
        return None
    # The cone as a NODE TREE: the per-row statistic is its own projected reduce
    # (``Map(body=<the stat's scalar sweep>, sources=(Fold,))``) and the cone's source, the
    # per-cell normalize its body — so the K seam is the node boundary, not a stmt scan.
    # ONE bilinear node with a channel per ⊗-fold: operand sharing is edge reuse (the shared A
    # cone read once per component), and the node schedules / lowers as one unit through its own
    # derived product loop. STORED as the :class:`Contraction` node itself (1s) — pure algebra:
    # the output axes / tile / stage are caller facts, stamped at the point of use.
    node = Contraction(
        k_axis=k_ax,
        a=make_cone(list(cone), k_ax.name, stat=red, sweep=tuple(stat_epi)),
        channels=tuple(Channel(b=bl, acc=acc) for bl, acc in folds),
    )
    return Map(body=Body((*prefix, *tail_ops)), sources=(node,)), n_ax, (Store(write=write),)


__all__ = ["bind_contraction", "bind_prologue_contraction", "make_cone", "map_cone"]
