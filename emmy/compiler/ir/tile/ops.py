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
multi-channel edge lowers through :attr:`ContractionView.loop`'s own product derivation (the old
``is_group`` / ``group_loop`` sibling matching retired with it)."""

from __future__ import annotations

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.schedule import ReducePlan
from emmy.compiler.ir.stmt import Assign, Body, Loop, StridedLoop
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.tile.ir import ContractionView, Fold, Map, effect_tail


def cone_seam(cone) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is ``Map(body=<the per-cell normalize>, sources=(<the row-invariant
    prologue>,))``, and the prologue node IS the per-row statistic (its own ``Map`` over the stat
    ``Fold``) plus any row-invariant cone prefix, placed there when the cone was built
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


class Sched:
    """Read/write view of one kernel's schedule slices — the ``TileOp.schedule`` dict (1r:
    ``{codec key → resolved TilePlan / ReducePlan / Stage}``) bound to the op tree the keys spell
    against. The ONE accessor pair every reader (materializer, ``030_split_reduce``) and stamper
    (the ``_schedule`` option builders) goes through, so a slice has exactly one home and the key
    spelling is always the tree-path codec's canonical one (:mod:`~emmy.compiler.ir.tile.path`).
    A node that is not a site of the family reads ``None`` and refuses writes loudly."""

    def __init__(self, root, table: dict | None) -> None:
        self.root = root
        self.table = table if table is not None else {}
        self._sites = None

    def _all_sites(self):
        from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415 — path imports ir; keep ops light

        if self._sites is None:
            self._sites = sites(self.root)
        return self._sites

    def key(self, family: str, node) -> str | None:
        """The canonical codec key addressing ``node`` under ``family`` — ``None`` when the node
        is not a site of that family on this tree (nothing to key)."""
        from emmy.compiler.ir.tile.path import spell  # noqa: PLC0415

        try:
            return spell(self.root, family, node, all_sites=self._all_sites())
        except ValueError:
            return None

    def get(self, family: str, node):
        k = self.key(family, node)
        return self.table.get(k) if k is not None else None

    def put(self, family: str, node, value) -> None:
        """Store a resolved slice for ``node`` (drop a ``None`` / empty value — an absent key IS
        the decided-empty)."""
        if value is None:
            return
        k = self.key(family, node)
        if k is None:
            raise ValueError(f"{family} does not apply to this {type(node).__name__} — no site to key the slice on")
        self.table[k] = value

    def tile_of(self, node):
        return self.get("TILE", node)

    def reduce_of(self, node):
        return self.get("REDUCE", node)

    def stage_of(self, node):
        return self.get("STAGE", node)


def sched_of(tile) -> Sched:
    """The :class:`Sched` view of a ``TileOp`` (binds its ``schedule`` dict to its op tree)."""
    return Sched(tile.op, tile.schedule)


def projection_tail(tile) -> list[Stmt]:
    """The kernel's EFFECTFUL projection stmt stream — the root ``Map``'s (pure) body with the
    kernel-boundary ``TileOp.stores`` reconstituted (:func:`~emmy.compiler.ir.tile.ir.effect_tail`).
    The ONE read every scheduler gate that inspects "the tail" goes through (1q), so a converted
    kernel (stores at the boundary) and a raw-loop-IR one (effects still in-body, empty ``stores``)
    answer identically — e.g. the ``b<n>t`` band's no-sweep-``Loop`` condition keeps excluding
    rms/softmax rows after their sweep moved to a ``Store`` decoration."""
    op = tile.op
    body = list(op.body) if isinstance(op, Map) else []
    return effect_tail(body, tile.stores)


def seal_workers(tile) -> None:
    """Derive the kernel's ONE worker inventory (``TileOp.work``) from its resolved ``TILE``
    slices — the 1r factoring of the per-site ``w``/``n`` worker tokens into a single slot,
    FAILING LOUDLY on cross-site disagreement (one kernel, one inventory). Called by every
    option builder / split realizer after the schedule dict is assembled."""
    from emmy.compiler.ir.schedule import derive_workers  # noqa: PLC0415

    tile.work = derive_workers(v for k, v in tile.schedule.items() if k.split("@", 1)[0] == "TILE")


def reduce_loop(op):
    """The kernel's outermost **annotated** reduce ``Loop`` (its ``carrier`` set by recognition),
    or ``None`` for a pure pointwise / flat-fallback ``Map`` (no annotated reduce). A
    :class:`~emmy.compiler.ir.tile.ir.Fold` / :class:`ContractionView` synthesizes its loop
    directly (a multi-channel contraction derives the ONE fused product loop — see
    :attr:`ContractionView.loop`); a ``Map``
    is read off the top-level body — the annotated reduce loop is a top-level stmt (a
    single-flat-reduce cell); a nested / multi reduce stays un-annotated (flat fallback) and is
    invisible here, so it materializes on the scalar tier."""
    if isinstance(op, (Fold, ContractionView)):
        return op.loop
    if isinstance(op, Map) and op.sources:
        return reduce_loop(op.sources[0])  # a Map projecting over a Fold / ContractionView source
    for s in op.body:
        if isinstance(s, (Loop, StridedLoop)) and s.carrier is not None:
            return s
    return None


def reduce_plan(tile):
    """The tile's reduce partition (:class:`~emmy.compiler.ir.schedule.ReducePlan`), read from
    ``TileOp.schedule`` for the PRIMARY :class:`~emmy.compiler.ir.tile.ir.Fold` — when ``tile.op``
    is a ``Fold`` (bare, or wrapped via ``Map.sources``), else ``None`` (a pure pointwise / scalar
    per-cell ``Map`` has no partition). An unstamped fold reads the empty plan (the scalar serial
    fold), matching the retired node field's default. The single accessor the materializer /
    ``030_split_reduce`` read."""
    op = tile.op
    head = op.sources[0] if isinstance(op, Map) and op.sources else op
    if not isinstance(head, Fold):
        return None
    plan = sched_of(tile).reduce_of(head)
    return plan if plan is not None else ReducePlan()


def nodify_reduce(op):
    """Nodify a kernel op into a :class:`~emmy.compiler.ir.tile.ir.Fold` node the reduce
    partition can key on (the plan itself lives in ``TileOp.schedule`` — the caller stores it
    against the returned fold). Returns ``(op2, fold)``. A stored contraction fold (the
    recognize-side per-cell contraction) is already the node — the coop / ILP K partition treats
    it as the degenerate carrier of its additive fold, any projection riding the wrapping ``Map``.
    A flat ``Map`` holding an annotated reduce ``Loop`` (a split partial) nodifies via
    :meth:`Fold.from_loop`, which reconstructs the identical annotated loop, so the lowering is
    byte-identical; a projection tail (a fused epilogue) rides a wrapping ``Map`` over the node, a
    bare reduce becomes the root node.

    Used by the scheduler (the coop / ILP-K contraction) and ``030_split_reduce`` (the split partial) so
    EVERY partitioned reduce keys its plan off a node uniformly. For the ``Map`` form the reduce ``Loop``
    must be a top-level stmt of ``op.body`` with no prologue ahead of it (true for a split partial /
    bare sum: the reduce is the body's head); a projection tail after it becomes the wrapping
    ``Map`` body."""
    if isinstance(op, Fold) and op.role is AxisRole.CONTRACTION:
        return op, op
    head = op.sources[0] if isinstance(op, Map) and op.sources else None
    if isinstance(head, Fold) and head.role is AxisRole.CONTRACTION:
        return op, head
    rloop = reduce_loop(op)
    red = Fold.from_loop(rloop)
    if red is None:  # not λ-representable (a raw-block slice) — the caller keeps the flat spelling
        return op, None
    body = list(op.body)
    idx = body.index(rloop)
    pre, tail = body[:idx], body[idx + 1 :]
    assert not pre, f"nodify_reduce: unexpected prologue ahead of the reduce loop: {pre}"
    return (Map(body=Body(tuple(tail)), sources=(red,)) if tail else red), red


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
    (:attr:`ContractionView.loop`)."""
    from emmy.compiler.ir.stmt import Load  # noqa: PLC0415

    if isinstance(op, Map):
        # A Load source is the CUT TERMINAL (phase 4 — every edge admits Load): the seam value
        # arrives materialized, so the "nest" is the load itself.
        prefix = [s for src in op.sources for s in ((src,) if isinstance(src, Load) else lower(src))]
        return [*prefix, *op.body]  # the sources' reduce/contract loop nests, then the projection body
    if isinstance(op, (Fold, ContractionView)):
        return op.lower()
    raise TypeError(f"lower: expected a Map lift wrapper, Fold, or ContractionView, got {type(op).__name__}")


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
    :class:`~emmy.compiler.ir.tile.ir.Fold` as a typed header over its synthesized
    loop nest, the ``Map``'s body (its annotated reduce ``Loop`` + projection), or a bare stmt's own
    pretty."""
    if isinstance(op, Fold):
        head = f"{indent}Fold[{op.axis.name}] {op.role.name.lower()}"
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
    "Sched",
    "cone_seam",
    "contraction_loop",
    "lower",
    "nodify_reduce",
    "pretty",
    "projection_tail",
    "reduce_loop",
    "reduce_plan",
    "sched_of",
    "seal_workers",
]
