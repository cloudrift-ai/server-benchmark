"""The geometry-free compute layer — the lift wrapper and its lowering.

A kernel's compute is a :class:`~emmy.compiler.ir.tile.ir.Map` (re-exported here) — a
:class:`~emmy.compiler.ir.stmt.body.Body` of loop-IR stmts holding the per-cell compute. A
reduction is a ``Map`` whose body contains the **annotated reduce ``Loop``** (its
:class:`~emmy.compiler.ir.axis.AxisRole` + :class:`~emmy.compiler.ir.stmt.algebra.Carrier`
stamped by recognition) followed by the post-reduce projection; a contraction is a ``Map`` whose
reduce ``Loop`` is ``CONTRACTION`` (the ``⊗`` lift sits in the loop body). The algebra is read
**structurally** off the annotated loop, never a stored node kind — the ``Monoid`` / ``Semiring``
node wrappers are retired.

This module is the thin lowering of that wrapper to loop IR (:func:`lower` — the body verbatim,
the carriers already dissolved into loose folds at recognition) plus the structural reads
(:func:`axis_role` / :func:`reduce_loop`) and the shared contraction-loop builder
(:func:`contraction_loop`). Stored trees are already resolved — a computed operand is an inline
node on its edge, so there is no name-resolution step ahead of a lowering walk (the old
``resolve`` splice over ``TileOp.bindings`` retired with the let table), and the fused
multi-channel edge lowers through :attr:`Contraction.loop`'s own product derivation (the old
``is_group`` / ``group_loop`` sibling matching retired with it)."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.schedule import ReducePlan
from emmy.compiler.ir.stmt import Assign, Body, Loop, StridedLoop
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.tile.ir import Contraction, Map, Reduction


def cone_seam(cone) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is ``Map(body=<the per-cell normalize>, sources=(<the row-invariant
    prologue>,))``, and the prologue node IS the per-row statistic (its own ``Map`` over the stat
    ``Reduction``) plus any row-invariant cone prefix, placed there when the cone was built
    (``_atomize.make_cone`` splits at the K seam once, structurally).

    ``stats`` are the prologue defs the cell reads — the values bridged through the stat smem rows;
    a prologue whose defs go unread is dropped (nothing to bridge). The ONE seam both sides read:
    the scheduler sizes the stat rows into the sync stage's smem budget, the materializer fills
    them (``sync_stat_fill``)."""
    from emmy.compiler.ir.tile.ir import _deep_defines, _deep_reads  # noqa: PLC0415

    if not isinstance(cone, Map) or not cone.sources:
        return (), tuple(cone.body) if isinstance(cone, Map) else (), ()
    pro = tuple(lower(cone.sources[0]))
    cell = tuple(cone.body)
    stats = tuple(sorted({nm for s in pro for nm in _deep_defines(s)} & _deep_reads(list(cell))))
    return (pro, cell, stats) if stats else ((), cell, ())


def reduce_loop(op):
    """The kernel's outermost **annotated** reduce ``Loop`` (its ``carrier`` set by recognition),
    or ``None`` for a pure pointwise / flat-fallback ``Map`` (no annotated reduce). A
    :class:`~emmy.compiler.ir.tile.ir.Reduction` / :class:`Contraction` synthesizes its loop
    directly (a multi-channel contraction derives the ONE fused product loop — see
    :attr:`Contraction.loop`); a ``Map``
    is read off the top-level body — the annotated reduce loop is a top-level stmt (a
    single-flat-reduce cell); a nested / multi reduce stays un-annotated (flat fallback) and is
    invisible here, so it materializes on the scalar tier."""
    if isinstance(op, (Reduction, Contraction)):
        return op.loop
    if isinstance(op, Map) and op.sources:
        return reduce_loop(op.sources[0])  # a Map projecting over a Reduction / Contraction source
    for s in op.body:
        if isinstance(s, (Loop, StridedLoop)) and s.carrier is not None:
            return s
    return None


def reduce_plan(tile):
    """The tile's reduce partition (:class:`~emmy.compiler.ir.schedule.ReducePlan`), read
    **off the** :class:`~emmy.compiler.ir.tile.ir.Reduction` **node** — when ``tile.op`` is a
    ``Reduction`` (bare, or wrapped via ``Map.source``), else ``None`` (a pure pointwise / scalar
    per-cell ``Map`` has no partition). Every partitioned reduce — a plain / twisted monoid, flash,
    a coop-K / split-K contraction (:func:`nodify_reduce`) — carries its plan on the node; there is
    **no** residual ``TileOp.reduce`` field. The single accessor the materializer / ``030_split_reduce`` read."""
    op = tile.op
    head = op.sources[0] if isinstance(op, Map) and op.sources else op
    red = head if isinstance(head, Reduction) else None
    return red.reduce if red is not None else None


def nodify_reduce(op, plan: ReducePlan):
    """Nodify a kernel op into a :class:`~emmy.compiler.ir.tile.ir.Reduction` node carrying the
    reduce partition ``plan`` **on the node** (not a residual ``TileOp.reduce`` field). A
    :class:`Contraction` node (the recognize-side per-cell contraction) folds through its
    synthesized loop — the coop / ILP K partition treats it as the degenerate carrier of its
    additive fold, any projection riding the wrapping ``Map``. A flat ``Map`` holding an annotated
    reduce ``Loop`` (a split partial) nodifies via :meth:`Reduction.from_loop`, which reconstructs
    the identical annotated loop, so the lowering is byte-identical; a projection tail (a fused
    epilogue) rides a wrapping ``Map`` over the node, a bare reduce becomes the root node.

    Used by the scheduler (the coop / ILP-K contraction) and ``030_split_reduce`` (the split partial) so
    EVERY partitioned reduce reads its plan off a node uniformly — the ``lower(op)``-then-refind
    smell (and the ``TileOp.reduce`` residual) is gone. For the ``Map`` form the reduce ``Loop``
    must be a top-level stmt of ``op.body`` with no prologue ahead of it (true for a split partial /
    bare sum: the reduce is the body's head); a projection tail after it becomes the wrapping
    ``Map`` body."""
    if isinstance(op, Contraction):
        return replace(Reduction.from_loop(op.loop), reduce=plan)
    if isinstance(op, Reduction) and op.role is AxisRole.CONTRACTION:
        # A stored contraction fold: the K partition replaces the (dropped) output tile — the
        # per-cell coop/ILP tier's contract. The derived loop is unchanged (tile is metadata).
        return replace(op, reduce=plan, tile=None)
    head = op.sources[0] if isinstance(op, Map) and op.sources else None
    if isinstance(head, Contraction) or (isinstance(head, Reduction) and head.role is AxisRole.CONTRACTION):
        # A projecting wrapper: nodify the contraction under it, the projection staying put.
        return replace(op, sources=(nodify_reduce(op.sources[0], plan),))
    rloop = reduce_loop(op)
    red = replace(Reduction.from_loop(rloop), reduce=plan)
    body = list(op.body)
    idx = body.index(rloop)
    pre, tail = body[:idx], body[idx + 1 :]
    assert not pre, f"nodify_reduce: unexpected prologue ahead of the reduce loop: {pre}"
    return Map(body=Body(tuple(tail)), sources=(red,)) if tail else red


def axis_role(op) -> AxisRole:
    """The reduce :class:`~emmy.compiler.ir.axis.AxisRole` of a kernel's outermost reduction,
    read **structurally** off the annotated reduce loop (no stored kind tag): a ``CONTRACTION``
    contraction, a ``TWISTED`` (online-softmax / flash) or ``PLANAR`` (plain ``sum`` / ``max`` /
    ``mean``) reduce, or ``FREE`` for a pure pointwise / flat-fallback ``Map``. This is what the
    schedule / materialize passes dispatch on."""
    rl = reduce_loop(op)
    return rl.role if rl is not None else AxisRole.FREE


def lower(op) -> list[Stmt]:
    """Lower the lift wrapper to loop-IR stmts — the ``Map``'s body verbatim. The carriers are
    already dissolved into loose fold ``Accum``\\ s (and the streaming ``merge`` for a twisted
    carrier) at recognition, and the reduce ``Loop``\\ s carry their role/carrier annotations, so
    one ``lower`` call emits the kernel's per-cell body with nothing left to expand. Stored trees
    are already resolved (computed operands are inline nodes), so there is no name-inlining step;
    a multi-channel contraction lowers through its own derived product loop
    (:attr:`Contraction.loop`)."""
    if isinstance(op, Map):
        prefix = [s for src in op.sources for s in lower(src)]
        return [*prefix, *op.body]  # the sources' reduce/contract loop nests, then the projection body
    if isinstance(op, (Reduction, Contraction)):
        return op.lower()
    raise TypeError(f"lower: expected a Map lift wrapper, Reduction, or Contraction, got {type(op).__name__}")


def contraction_loop(lift, fold, operand_bodies, reduce_axis) -> Loop:
    """Build the contraction (matmul) reduce ``Loop`` in the recognizable ``Accum``-in-``Loop``
    form: expand each operand source's stmts (siblings), the ``⊗`` lift ``Assign``
    (``fold.value = lift(operands…)``), and the additive ``fold`` ⊕ (its identity init is the
    ``Loop``'s immediate-``Accum`` prelude — no explicit ``Init``). The loop is stamped
    ``CONTRACTION`` + the degenerate carrier of its additive fold (``fold.as_carrier()``), so the
    K loop carries its combine and the warp / cooperative tiers read the operands structurally off
    the body. Shared by the flash score producer and the scalar register-tile contraction."""
    body: list[Stmt] = []
    names: list[str] = []
    for ob in operand_bodies:
        stmts = list(ob)
        body += stmts
        names.append(stmts[-1].defines()[-1])
    body.append(Assign(name=fold.value, op=lift, args=tuple(names)))
    body.append(fold)
    return Loop(axis=reduce_axis, body=Body(tuple(body)), role=AxisRole.CONTRACTION, carrier=fold.as_carrier())


def pretty(op, indent: str = "") -> list[str]:
    """Structurally pretty-print a kernel op (for dumps) — a
    :class:`~emmy.compiler.ir.tile.ir.Reduction` as a typed header over its synthesized
    loop nest, the ``Map``'s body (its annotated reduce ``Loop`` + projection), or a bare stmt's own
    pretty."""
    if isinstance(op, Reduction):
        head = f"{indent}Reduction[{op.axis.name}] {op.role.name.lower()}"
        return [head, *pretty_body(Body(op.lower()), indent + "    ")]
    if isinstance(op, Map):
        src = [line for s in op.sources for line in pretty(s, indent)]
        return [*src, *pretty_body(op.body, indent)]
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = [
    "Map",
    "axis_role",
    "cone_seam",
    "contraction_loop",
    "lower",
    "nodify_reduce",
    "pretty",
    "reduce_loop",
    "reduce_plan",
]
