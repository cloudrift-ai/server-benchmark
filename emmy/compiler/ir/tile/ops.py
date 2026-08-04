r"""The geometry-free compute layer — node lowering and the structural reads.

A kernel's compute is a stored :class:`~emmy.compiler.ir.tile.ir.Fold` (a bare reduce), a
:class:`~emmy.compiler.ir.tile.ir.Fold` (a pure pointwise cell, or the projection wrapper over its
source node). :func:`head` is the ONE accessor reaching that node through the projection (zero-axis) fold,
and every structural fact a pass dispatches on — :func:`axis_role`, the reduce ``Axis``, the
operand edges — is a STORED param on it (a fold's role is derived from those params, never
stored). Reading a node fact off a SYNTHESIZED nest is the inversion this module exists to
prevent: :func:`reduce_loop` and :meth:`Fold.lower` are for callers that consume a body.

This module holds the structural reads over a node tree — the cone seam (:func:`cone_seam`), the
iteration-space names (:func:`axis_names`) — plus the tree-path schedule accessor (:class:`Sched`),
kernel identity (:func:`term_key`) and the worker sealing (:func:`seal_workers`). Lowering itself
has ONE spelling and it lives on the node: :meth:`Fold.lower` (a fold flattens through
:attr:`Fold.loop`, a wrapping projection appends its operand nests). Stored trees are already
resolved — a computed operand is an inline node on its edge, so there is no name-resolution step
ahead of a lowering walk."""

from __future__ import annotations

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.schedule import ReducePlan
from emmy.compiler.ir.stmt import Loop, StridedLoop
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile.ir import Fold, deep_defines, deep_reads, effect_tail


def cone_seam(cone) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is ``Fold.projection(body=<the per-cell normalize>, operands=(<the row-invariant
    prologue>,))``, and the prologue node IS the per-row statistic (its own zero-axis ``Fold`` over the stat
    ``Fold``) plus any row-invariant cone prefix, placed there when the cone was built
    (``_atomize.make_cone`` splits at the K seam once, structurally).

    ``stats`` are the prologue defs the cell reads — the values bridged through the stat smem rows;
    a prologue whose defs go unread is dropped (nothing to bridge). The ONE seam both sides read:
    the scheduler sizes the stat rows into the sync stage's smem budget, the materializer fills
    them (``sync_stat_fill``)."""
    if not isinstance(cone, Fold) or cone.axis is not None or not cone.operands:
        return (), tuple(cone.body) if isinstance(cone, Fold) and cone.axis is None else (), ()
    pro = tuple(cone.operands[0].lower())
    cell = tuple(cone.body)
    stats = tuple(sorted({nm for s in pro for nm in deep_defines(s)} & deep_reads(list(cell))))
    return (pro, cell, stats) if stats else ((), cell, ())


class Sched:
    """Read/write view of one kernel's schedule slices — the ``TileOp.schedule`` dict (1r:
    ``{codec key → resolved TilePlan / ReducePlan / Stage}``) bound to the op tree the keys spell
    against. The ONE accessor pair every reader (materializer, ``030_split_reduce``) and stamper
    (the ``_schedule`` option builders) goes through, so a slice has exactly one home and the key
    spelling is always the tree-path codec's canonical one (:mod:`~emmy.compiler.ir.tile.path`).
    A node that is not a site of the family reads ``None`` and refuses writes loudly."""

    def __init__(self, root, table: dict | None, place=None) -> None:
        self.root = root
        self.table = table if table is not None else {}
        #: The kernel's free→grid :class:`~emmy.compiler.ir.schedule.Placement`. A ``TILE`` slice
        #: is geometry over an ``(m, n)`` output pair, and WHICH pair is a function of the site's
        #: position in the tree — so the binding belongs here, on the scheduling structure, and
        #: not at each reader (:meth:`tile_of`).
        self.place = place
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
        """The node's ``TILE`` slice, ALREADY PLACED — its ``(m, n)`` output axes bound
        (:attr:`TilePlan.axes`), so a reader gets the geometry off the slice and never re-derives
        placement of its own. ``axes`` stays ``compare=False`` / ``repr=False``, so binding it here
        cannot reach ``spell()``, a stamped row, a golden or a prior key.

        The pair is a function of the SITE (:meth:`_mn_for`), which is what makes one rule
        possible where there used to be three hand-written ``.at(...)`` calls."""
        plan = self.get("TILE", node)
        if plan is None or self.place is None or plan.axes is not None:
            return plan
        mn = self._mn_for(node)
        return plan.at(*mn) if mn is not None else plan

    def _mn_for(self, node):
        """The ``(m, n)`` output axes a ``TILE`` slice at ``node``'s site tiles, or ``None`` when
        the placement cannot supply them (an unmapped / rank-<2 grid — the caller's untiled path).

        THREE site shapes, one rule each — the geometry the retired ``Contraction`` node used to
        carry as stamped fields, now read off the tree instead:

        - the ROOT contraction tiles the kernel grid's trailing pair;
        - a DERIVED site (flash's synthesized PV) tiles the placement's trailing free pair — it
          lives below the seam lattice, so the grid says nothing about it;
        - any other nested contraction (flash's hoisted QK edge) takes the free m axis and its
          PARENT fold's axis as n — read through a slice partial's window PARENT, so the view
          carries the pre-slice geometry the fragment clamps were built against.
        """
        free, grid = tuple(self.place.free), tuple(self.place.grid)
        site = next((s for s in self._all_sites() if s.node is node), None)
        if site is None:
            return None
        if site.depth == 1:
            return (grid[-2], grid[-1]) if len(grid) >= 2 else None
        if len(free) < 2:
            return None
        if site.derived:
            return (free[-2], free[-1])
        parent = next((s for s in self._all_sites() if s.segments == site.segments[:-1]), None)
        ax = getattr(parent.node, "axis", None) if parent is not None else None
        if ax is None:
            return None
        return (free[-2], ax.window.parent if ax.window is not None else ax)


def sched_of(tile) -> Sched:
    """The :class:`Sched` view of a ``TileOp`` (binds its ``schedule`` dict to its op tree)."""
    return Sched(tile.op, tile.schedule, place=tile.place)


def axis_names(root) -> set[str]:
    """Every ITERATION-SPACE name in ``root``'s tree — the structural nodes' axes plus every loop
    induction variable in their bodies, over the ONE node walk (``path.sites``). An induction
    variable is bound by the enclosing loop nest, not by any value tree, so a subtree reading one
    is never capturing a value.

    The ONE reading that separates the two kinds of free name a λ body can carry: an iteration var
    (bound by the nest, free by construction) and a captured VALUE. The cut's closure predicate
    (``_cut._captured_values``) subtracts this set, and the structural dump shows what remains as
    the λ's capture set."""
    from emmy.compiler.ir.tile.ir import stmt_axis_names  # noqa: PLC0415
    from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415 — path imports ir; keep ops light

    out: set[str] = set()
    for site in sites(root):
        node = site.node
        if not isinstance(node, Fold):
            continue
        if node.axis is None:
            out |= stmt_axis_names(node.body)
        else:
            out.add(node.axis.name)
            out |= stmt_axis_names(node.step_stmts())
    return out


def projection_tail(tile) -> list[Stmt]:
    """The kernel's EFFECTFUL projection stmt stream — the root zero-axis fold's (pure) body with the
    kernel-boundary ``TileOp.stores`` reconstituted (:func:`~emmy.compiler.ir.tile.ir.effect_tail`).
    The ONE read every scheduler gate that inspects "the tail" goes through, so a converted
    kernel (stores at the boundary) and a raw-loop-IR one (effects still in-body, empty ``stores``)
    answer identically — e.g. the ``b<n>t`` band's no-sweep-``Loop`` condition keeps excluding
    rms/softmax rows after their sweep moved to a ``Store`` decoration."""
    op = tile.op
    body = list(op.body) if isinstance(op, Fold) and op.axis is None else []
    return effect_tail(body, tile.stores)


def seal_workers(tile) -> None:
    """Derive and STAMP the kernel's ONE worker inventory (``TileOp.work`` + the ``WORK`` knob —
    the step-7 value-grammar family): the per-site ``w``/``n`` worker tokens factored out of the
    resolved ``TILE`` slices, the cooperative width off the ``REDUCE`` slices (``b512`` →
    ``t512``), and the producer band off the resolved :class:`WarpSpec` (the ``WSPEC`` absorb —
    ``+p<n>``). FAILING LOUDLY on cross-site disagreement (one kernel, one inventory). A 1-thread
    inventory (a bare register strip) keeps ``None`` — the per-cell forms' launch geometry stays
    derived. Called by every option builder / split realizer after the schedule dict is
    assembled."""
    from emmy.compiler.ir.schedule import derive_inventory  # noqa: PLC0415

    coop = max(
        (v.coop for k, v in tile.schedule.items() if k.split("@", 1)[0] == "REDUCE" and hasattr(v, "coop")),
        default=1,
    )
    ws = getattr(tile, "workers", None)
    work = derive_inventory(
        (v for k, v in tile.schedule.items() if k.split("@", 1)[0] == "TILE"),
        coop=coop,
        producer=getattr(ws, "aux_warps", 0) if ws is not None else 0,
    )
    tile.work = work
    tile.knobs["WORK"] = work.spell() if work is not None else ""


def head(op):
    """The kernel's compute NODE — a :class:`~emmy.compiler.ir.tile.ir.Fold` /
    a bilinear ``Fold``, bare or under its projection (zero-axis) fold — or
    ``None`` for a pure pointwise cell / the raw-loop-IR escape (a zero-axis fold with no operands).

    The ONE accessor for "which node is this kernel about", replacing the hand-spelled
    ``op.operands[0] if op.axis is None and op.operands else op`` ternary at every reader. Every
    node-level fact the scheduler dispatches on — the :class:`~emmy.compiler.ir.axis.AxisRole`, the
    reduce ``Axis``, the operand edges — is a STORED param on what this returns, so a scheduling
    read never needs :func:`reduce_loop` (which synthesizes a whole nest) to reach it."""
    node = op.operands[0] if isinstance(op, Fold) and op.axis is None and op.operands else op
    return node if isinstance(node, Fold) and node.axis is not None else None


def reduce_loop(op):
    """The kernel's outermost **annotated** reduce ``Loop`` (its ``role`` stamped by recognition),
    or ``None`` for a pure pointwise / flat-fallback zero-axis ``Fold`` (no annotated reduce). A
    :class:`~emmy.compiler.ir.tile.ir.Fold` / a bilinear ``Fold`` synthesizes its loop
    directly (a multi-channel contraction derives the ONE fused product loop — see
    :attr:`Fold.loop`); a zero-axis ``Fold``
    is read off the top-level body — the annotated reduce loop is a top-level stmt (a
    single-flat-reduce cell); a nested / multi reduce stays un-annotated (``role=FREE`` — flat
    fallback) and is invisible here, so it materializes on the scalar tier.

    For the LOOP NEST itself — a caller that consumes a body. A node-level FACT (the role, the
    reduce axis, the operand edges) is a stored param on :func:`head`'s node: read it there, not
    off a nest synthesized to be thrown away."""
    if isinstance(op, Fold) and op.axis is not None:
        return op.loop
    if isinstance(op, Fold) and op.operands:
        return reduce_loop(op.operands[0])  # the zero-axis node projecting over its source
    for s in op.body:
        if isinstance(s, (Loop, StridedLoop)) and s.role.is_reduce:
            return s
    return None


def reduce_plan(tile):
    """The tile's reduce partition (:class:`~emmy.compiler.ir.schedule.ReducePlan`), read from
    ``TileOp.schedule`` for the PRIMARY :class:`~emmy.compiler.ir.tile.ir.Fold` — when ``tile.op``
    is a ``Fold`` (bare, or wrapped via ``operands``), else ``None`` (a pure pointwise / scalar
    per-cell zero-axis ``Fold`` has no partition). An unstamped fold reads the empty plan (the scalar serial
    fold), matching the node field's default. The single accessor the materializer /
    ``030_split_reduce`` read."""
    node = head(tile.op)
    if node is None:
        return None
    plan = sched_of(tile).get("REDUCE", node)
    return plan if plan is not None else ReducePlan()


def axis_role(op) -> AxisRole:
    """The reduce :class:`~emmy.compiler.ir.axis.AxisRole` of a kernel's outermost reduction: a
    ``CONTRACTION`` contraction, a ``TWISTED`` (online-softmax / flash) or ``PLANAR`` (plain
    ``sum`` / ``max`` / ``mean``) reduce, or ``FREE`` for a pure pointwise / flat-fallback
    zero-axis ``Fold``. This is what the schedule / materialize passes dispatch on.

    Read off the NODE (:func:`head`) — the role is that node's own derived property
    (``Fold.role``), so the dispatch costs a field access. The
    annotated-``Loop`` scan survives only for the raw-loop-IR escape (an un-recognized cell,
    ``030``'s finalize, the coop fused-tail sibling): no node to ask, the stamped ``Loop.role``
    IS the fact. Never synthesize a nest to answer this — :attr:`Fold.loop` splices every operand
    edge and flattens nested nodes, and hands back the same property it was given."""
    node = head(op)
    if node is not None:
        return node.role
    for s in op.body:  # the escape: a flat zero-axis fold whose recognition-stamped reduce Loop is a top-level stmt
        if isinstance(s, (Loop, StridedLoop)) and s.role.is_reduce:
            return s.role
    return AxisRole.FREE


# ``term_key`` (kernel identity) lives in its own module — it is not a compute read. Re-exported
# here because ``ops.term_key`` is the layer's ONE public identity name (``op_cache_key`` /
# ``Graph.structural_key`` reach it through this module). The structural dump is NOT re-exported:
# it has no consumer outside ``_dump`` itself, so a shim here would serve nothing.
from emmy.compiler.ir.tile._key import term_key  # noqa: E402

__all__ = [
    "Sched",
    "axis_names",
    "axis_role",
    "cone_seam",
    "head",
    "projection_tail",
    "reduce_loop",
    "reduce_plan",
    "sched_of",
    "seal_workers",
    "term_key",
]
