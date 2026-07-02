"""Atomize — resolve the algebra→hardware-atom binding structurally.

The warp matmul materializer needs to know which operand is the mma ``a`` vs ``b`` (by
axis-in-index), the fold accumulator, and the projection epilogue.
:func:`semiring_binding` reads them **structurally** off the lowered ``CONTRACTION`` reduce loop
— the operand ``Load``\\ s indexed over the K axis, the fold ``Accum`` target — and returns them as
the ``(a_load, b_load, acc, epilogue)`` facts that ``_schedule._contraction_node`` stamps onto the
:class:`~emmy.compiler.ir.tile.ir.Contraction` structural node at fork-emit (the node
is then the single source of truth — it re-derives ``b_trans`` off ``b_load`` itself). Reading the
binding **structurally** off the annotated loop — not a stored node kind — is what keeps the ⊗/⊕
algebra a property of the loop, so no per-algebra op-tree node class is needed. The cooperative reduce
needs no binding here — its accumulator dtype + shuffle/tree
mechanism are derived at materialize time (``emit_combine`` off the carrier + ``ReduceStage.combine``).

**Called from ``_schedule`` (inside ``010_recognize``), not a standalone pass.** The binding is resolved when the tiled
contraction leaf is built (``_warp_option`` / the tiled ``_tile_option``) — so an atom that
**cannot** be bound (e.g. a non-``Load`` operand: a computed-cone / demoted matmul) is rejected at
fork construction, alongside ``_check_warp_static_k``, instead of failing several passes later.
Leading ``_`` so the pass loader skips this module.

**Flash contractions are not recursively atomized.** Flash is a ``TWISTED`` kv
``Reduction`` over a ``Q@K`` :class:`~emmy.compiler.ir.tile.ir.Contraction` ``source`` +
a ``P@V`` one in the ``partial``, lowered on the scalar tier (block=1) — each contraction carries a
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
from emmy.compiler.ir.tile import Contraction, Map, Reduction, TilePlan
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


def bind_contraction(loop: Loop, m_name: str, n_name: str, epilogue: Body) -> tuple[Load, Load, str, Body]:
    """Resolve the ``(a_load, b_load, acc, epilogue)`` operand→role facts for a ``CONTRACTION``
    reduce ``loop`` (the lowered ``Accum``-in-``Loop`` form) whose output is indexed by grid axes
    ``m_name`` / ``n_name``, with projection ``epilogue``.

    Reads the facts straight off the loop body — no op-tree node: the contraction operands are the
    ``Load``\\ s in the loop indexed over the reduce (K) axis; A/B are bound by which output axis
    each one's index carries; the fold accumulator is the loop body's ``Accum`` target. A clean
    gmem-direct contraction has plain-``Load`` operands (a computed-cone / demoted matmul never
    reaches CONTRACTION — recognition leaves it a flat reduce), so an unbindable body (no m/n-bearing
    K-load) raises, matching the warp gmem-direct guard. ``b_trans`` is not returned — the
    ``Contraction`` node re-derives it off ``b_load``."""
    k_name = loop.axis.name
    loads = [s for s in loop.body if isinstance(s, Load) and k_name in _idx_vars(s.index)]
    a_leaf = next((ld for ld in loads if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in loads if n_name in _idx_vars(ld.index)), None)
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    acc = next((s.name for s in loop.body if isinstance(s, Accum)), None)
    if acc is None:
        raise LoweringError("warp tier: contraction loop has no fold accumulator")
    return a_leaf, b_leaf, acc, epilogue


def semiring_binding(node, grid) -> tuple[Load, Load, str, Body]:
    """The root contraction's ``(a_load, b_load, acc, epilogue)`` facts: lower ``node`` to loop-IR,
    find its ``CONTRACTION`` reduce loop, take the projection ``epilogue`` (the stmts after the loop
    — the ``Map`` body, or empty for a bare contraction), and delegate to :func:`bind_contraction`.
    ``node`` is the kernel op, ``grid`` the placement's output axes."""
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    stmts = lower(node)
    ridx = next((i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.role is AxisRole.CONTRACTION), None)
    if ridx is None:
        raise LoweringError("warp tier: no contraction loop to bind")
    epilogue = Body(tuple(stmts[ridx + 1 :]))
    return bind_contraction(stmts[ridx], grid[-2].name, grid[-1].name, epilogue)


def bind_prologue_contraction(op, free: tuple) -> tuple[Contraction, Axis] | None:
    """Nodify the **reduce-bearing (MONOID) producer cone** composition — the fused norm→linear
    edge (``rmsnorm(x)·nw @ w``): a projecting ``Map`` whose ``source`` is a per-row ``PLANAR``
    statistic reduce and whose body is that statistic's scalar epilogue followed by a fresh free
    (column) ``Loop`` over an ⊗-fold contraction whose A cone reads the statistic. Returns the
    computed-A :class:`Contraction` — its A cone **carries the statistic prologue** (the annotated
    stat reduce ``Loop`` + its scalar epilogue ahead of the per-cell map stmts, the k-invariant
    prefix) with a **deferred** ``TilePlan()`` — plus the column axis (the scheduler adds it to the
    grid), or ``None`` (not this shape; the ``Map`` form stands alone).

    STRUCTURE-ONLY: no dtype / geometry / pin legality here — those are per-move scheduling guards
    (``_schedule``), so the same node offers whatever tiers the target legally supports."""
    if not isinstance(op, Map) or not isinstance(op.source, Reduction):
        return None
    red = op.source
    if red.role is not AxisRole.PLANAR or red.source is not None or red.carrier.twist.family != "id":
        return None
    body = list(op.body)
    if not body or not isinstance(body[-1], Loop) or body[-1].is_reduce:
        return None
    stat_epi, nloop = body[:-1], body[-1]
    if not all(isinstance(s, (Load, Assign)) for s in stat_epi):
        return None
    n_ax = nloop.axis
    inner = list(nloop.body)
    if len(inner) != 2 or not isinstance(inner[0], Loop) or not inner[0].is_reduce or not isinstance(inner[1], Write):
        return None
    kloop, write = inner
    k_ax = kloop.axis
    grid = list(free)
    if not grid:
        return None
    m_ax = grid[-1]
    kbody = list(kloop.body)
    accums = [st for st in kbody if isinstance(st, Accum)]
    if len(accums) != 1 or accums[0].op.name != "add":
        return None
    acc = accums[0]
    if write.values != (acc.name,) or not write.is_scalar:
        return None
    defs = {st.name: st for st in kbody if isinstance(st, Assign)}
    lift = defs.get(acc.value)
    if lift is None or lift.op.name != "multiply" or len(lift.args) != 2:
        return None
    loads = {st.names[0]: st for st in kbody if isinstance(st, Load)}

    def _load_vars(nm: str) -> set | None:
        ld = loads.get(nm)
        return {v for e in ld.index for v in e.free_vars()} if ld is not None else None

    b_name = next((a for a in lift.args if (vs := _load_vars(a)) and n_ax.name in vs and k_ax.name in vs), None)
    if b_name is None:
        return None
    a_name = next(a for a in lift.args if a != b_name)
    cone = map_cone(kbody, a_name)
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
    if _idx_vars_deep([*red.partial, *stat_epi]) & {n_ax.name, k_ax.name}:
        return None
    node = Contraction(
        axes=(m_ax, n_ax),
        k_axis=k_ax,
        a_operand=Body((red.loop, *stat_epi, *cone)),
        b_load=loads[b_name],
        acc=acc.name,
        tile=TilePlan(),
        lead_axes=tuple(grid[:-1]),
        epilogue=Body((write,)),
    )
    return node, n_ax


__all__ = ["bind_contraction", "bind_prologue_contraction", "map_cone", "semiring_binding"]
