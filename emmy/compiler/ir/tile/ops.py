r"""The geometry-free compute layer — node lowering and the structural reads.

A kernel's compute is a stored :class:`~emmy.compiler.ir.tile.ir.Fold` (a bare reduce), a
:class:`~emmy.compiler.ir.tile.ir.Contraction` (a contraction cell — 1s), or a
:class:`~emmy.compiler.ir.tile.ir.Map` (a pure pointwise cell, or the projection wrapper over its
source node). :func:`head` is the ONE accessor reaching that node through the projection ``Map``,
and every structural fact a pass dispatches on — :func:`axis_role`, the reduce ``Axis``, the
operand edges — is a STORED param on it (a fold's role is derived from those params, never
stored). Reading a node fact off a SYNTHESIZED nest is the inversion this module exists to
prevent: :func:`reduce_loop` / :func:`lower` are for callers that consume a body.

This module is the thin lowering of any node back to its loop nest (:func:`lower` — a fold
flattens through :attr:`Fold.loop`, a wrapping projection appends) plus the shared
contraction-loop builder (:func:`contraction_loop`), the tree-path schedule accessor
(:class:`Sched`), kernel identity (:func:`term_key`) and the worker sealing
(:func:`seal_workers`). Stored trees are already resolved — a computed operand is an inline node
on its edge, so there is no name-resolution step ahead of a lowering walk."""

from __future__ import annotations

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.schedule import ReducePlan
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, StridedLoop
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.tile.ir import Fold, Map, deep_defines, deep_reads, effect_tail


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
    if not isinstance(cone, Fold) or cone.axis is not None or not cone.operands:
        return (), tuple(cone.body) if isinstance(cone, Fold) and cone.axis is None else (), ()
    pro = tuple(lower(cone.operands[0]))
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


def unplaced_slices(tile) -> list[tuple[str, object]]:
    """The kernel's schedule entries that NO stored node carries — the keys addressing DERIVED
    material (flash's synthesized PV, ``TILE@pj``), sorted by key.

    Every other slice reaches its reader by annotating the node it keys against. These cannot: the
    node they address is a consequence of the stored params, not one of them. They are a schedule
    fact either way, so the dump prints them in its schedule region rather than reconstructing the
    derived node inside the term to hang them on. Empty for a kernel whose sites are all stored —
    which, measured over the frontend's kernels, is all of them but causal flash."""
    if tile.op is None or not tile.schedule:
        return []
    from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415 — path imports ir; keep ops light

    sched = sched_of(tile)
    stored = [s.node for s in sites(tile.op) if not s.derived]
    claimed = {k for nd in stored for f in ("TILE", "REDUCE", "STAGE") if (k := sched.key(f, nd)) is not None}
    return sorted((k, v) for k, v in tile.schedule.items() if k not in claimed)


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
    """The kernel's EFFECTFUL projection stmt stream — the root ``Map``'s (pure) body with the
    kernel-boundary ``TileOp.stores`` reconstituted (:func:`~emmy.compiler.ir.tile.ir.effect_tail`).
    The ONE read every scheduler gate that inspects "the tail" goes through (1q), so a converted
    kernel (stores at the boundary) and a raw-loop-IR one (effects still in-body, empty ``stores``)
    answer identically — e.g. the ``b<n>t`` band's no-sweep-``Loop`` condition keeps excluding
    rms/softmax rows after their sweep moved to a ``Store`` decoration."""
    op = tile.op
    body = list(op.body) if isinstance(op, Fold) and op.axis is None else []
    return effect_tail(body, tile.stores)


def seal_workers(tile) -> None:
    """Derive and STAMP the kernel's ONE worker inventory (``TileOp.work`` + the ``WORK`` knob —
    the step-7 value-grammar family): the per-site ``w``/``n`` worker tokens factored out of the
    resolved ``TILE`` slices (1r), the cooperative width off the ``REDUCE`` slices (``b512`` →
    ``t512``), and the producer band off the resolved :class:`WarpSpec` (the ``WSPEC`` absorb —
    ``+p<n>``). FAILING LOUDLY on cross-site disagreement (one kernel, one inventory). A 1-thread
    inventory (a bare register strip) keeps ``None`` — the per-cell forms' launch geometry stays
    derived. Called by every option builder / split realizer after the schedule dict is
    assembled."""
    from dataclasses import replace  # noqa: PLC0415

    from emmy.compiler.ir.schedule import Workers, derive_workers  # noqa: PLC0415

    work = derive_workers(v for k, v in tile.schedule.items() if k.split("@", 1)[0] == "TILE")
    coop = max(
        (v.coop for k, v in tile.schedule.items() if k.split("@", 1)[0] == "REDUCE" and hasattr(v, "coop")),
        default=1,
    )
    if coop > 1:
        if work is not None and work.count != coop:
            raise ValueError(f"disagreeing worker geometry: TILE workers {work.spell()} vs coop width {coop} — one kernel, one inventory")
        if work is None:
            work = Workers(kind="thread", units=(coop, 1))
    ws = getattr(tile, "workers", None)
    if work is not None and ws is not None and getattr(ws, "aux_warps", 0):
        work = replace(work, producer=ws.aux_warps)
    tile.work = work
    tile.knobs["WORK"] = work.spell() if work is not None else ""


def head(op):
    """The kernel's compute NODE — a :class:`~emmy.compiler.ir.tile.ir.Fold` /
    :class:`~emmy.compiler.ir.tile.ir.Contraction`, bare or under its projection ``Map`` — or
    ``None`` for a pure pointwise cell / the raw-loop-IR escape (a ``Map`` with no sources).

    The ONE accessor for "which node is this kernel about", replacing the hand-spelled
    ``op.sources[0] if isinstance(op, Map) and op.sources else op`` ternary at every reader. Every
    node-level fact the scheduler dispatches on — the :class:`~emmy.compiler.ir.axis.AxisRole`, the
    reduce ``Axis``, the operand edges — is a STORED param on what this returns, so a scheduling
    read never needs :func:`reduce_loop` (which synthesizes a whole nest) to reach it."""
    node = op.operands[0] if isinstance(op, Fold) and op.axis is None and op.operands else op
    return node if isinstance(node, Fold) and node.axis is not None else None


def reduce_loop(op):
    """The kernel's outermost **annotated** reduce ``Loop`` (its ``role`` stamped by recognition),
    or ``None`` for a pure pointwise / flat-fallback ``Map`` (no annotated reduce). A
    :class:`~emmy.compiler.ir.tile.ir.Fold` / :class:`Contraction` synthesizes its loop
    directly (a multi-channel contraction derives the ONE fused product loop — see
    :attr:`Contraction.loop`); a ``Map``
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
    is a ``Fold`` (bare, or wrapped via ``Map.sources``), else ``None`` (a pure pointwise / scalar
    per-cell ``Map`` has no partition). An unstamped fold reads the empty plan (the scalar serial
    fold), matching the retired node field's default. The single accessor the materializer /
    ``030_split_reduce`` read."""
    node = head(tile.op)
    if node is None:
        return None
    plan = sched_of(tile).reduce_of(node)
    return plan if plan is not None else ReducePlan()


def nodify_reduce(op, like=None):
    """Nodify a kernel op into a :class:`~emmy.compiler.ir.tile.ir.Fold` node the reduce
    partition can key on (the plan itself lives in ``TileOp.schedule`` — the caller stores it
    against the returned fold). Returns ``(op2, fold)``. A stored contraction fold (the
    recognize-side per-cell contraction) is already the node — the coop / ILP K partition treats
    it as the degenerate algebra of its additive fold, any projection riding the wrapping ``Map``.
    A flat ``Map`` holding an annotated reduce ``Loop`` (a split partial) nodifies via
    :meth:`Fold.from_loop`, which reconstructs the identical annotated loop, so the lowering is
    byte-identical; a projection tail (a fused epilogue) rides a wrapping ``Map`` over the node, a
    bare reduce becomes the root node. ``like`` is the fold whose algebra a TWISTED loop extracts
    against (:meth:`Fold.from_loop`) — defaulted from ``op``'s own head when it is node-form
    (030 passes the pre-slice fold for its flat split partial).

    Used by the scheduler (the coop / ILP-K contraction) and ``030_split_reduce`` (the split partial) so
    EVERY partitioned reduce keys its plan off a node uniformly. For the ``Map`` form the reduce ``Loop``
    must be a top-level stmt of ``op.body`` with no prologue ahead of it (true for a split partial /
    bare sum: the reduce is the body's head); a projection tail after it becomes the wrapping
    ``Map`` body."""
    node = head(op)
    if node is not None and node.role is AxisRole.CONTRACTION:
        return op, node  # a bare node IS its own head — the contraction keys its own plan
    if like is None:
        like = node if isinstance(node, Fold) else None
    rloop = reduce_loop(op)
    red = Fold.from_loop(rloop, like)
    if red is None:  # not λ-representable (a raw-block slice) — the caller keeps the flat spelling
        return op, None
    body = list(op.body)
    idx = body.index(rloop)
    pre, tail = body[:idx], body[idx + 1 :]
    assert not pre, f"nodify_reduce: unexpected prologue ahead of the reduce loop: {pre}"
    return (Map(body=Body(tuple(tail)), sources=(red,)) if tail else red), red


def axis_role(op) -> AxisRole:
    """The reduce :class:`~emmy.compiler.ir.axis.AxisRole` of a kernel's outermost reduction: a
    ``CONTRACTION`` contraction, a ``TWISTED`` (online-softmax / flash) or ``PLANAR`` (plain
    ``sum`` / ``max`` / ``mean``) reduce, or ``FREE`` for a pure pointwise / flat-fallback
    ``Map``. This is what the schedule / materialize passes dispatch on.

    Read off the NODE (:func:`head`) — the role is that node's own derived property
    (``Fold.role`` / the ``Contraction`` kind), so the dispatch costs a field access. The
    annotated-``Loop`` scan survives only for the raw-loop-IR escape (an un-recognized cell,
    ``030``'s finalize, the coop fused-tail sibling): no node to ask, the stamped ``Loop.role``
    IS the fact. Never synthesize a nest to answer this — :attr:`Fold.loop` splices every operand
    edge and flattens nested nodes, and hands back the same property it was given."""
    node = head(op)
    if node is not None:
        return node.role
    for s in op.body:  # the escape: a flat Map whose recognition-stamped reduce Loop is a top-level stmt
        if isinstance(s, (Loop, StridedLoop)) and s.role.is_reduce:
            return s.role
    return AxisRole.FREE


def lower(op) -> list[Stmt]:
    """Lower the lift wrapper to loop-IR stmts — the ``Map``'s body verbatim. The carriers are
    already dissolved into loose fold ``Accum``\\ s (and the streaming ``merge`` for a twisted
    carrier) at recognition, and the reduce ``Loop``\\ s carry their role/carrier annotations, so
    one ``lower`` call emits the kernel's per-cell body with nothing left to expand. Stored trees
    are already resolved (computed operands are inline nodes), so there is no name-inlining step;
    a multi-channel contraction lowers through its own derived product loop
    (:attr:`Contraction.loop`)."""

    if isinstance(op, Fold):
        return op.lower()
    raise TypeError(f"lower: expected a Fold, got {type(op).__name__}")


def contraction_loop(lift, fold, operand_bodies, reduce_axis) -> Loop:
    """Build the contraction (matmul) reduce ``Loop`` in the recognizable ``Accum``-in-``Loop``
    form: expand each operand source's stmts (siblings), the ``⊗`` lift ``Assign``
    (``fold.value = lift(operands…)``), and the additive ``fold`` ⊕ (its identity init is the
    ``Loop``'s immediate-``Accum`` prelude — no explicit ``Init``). The loop is stamped
    ``CONTRACTION`` — its algebra IS the body's additive ``Accum`` — so the
    warp / cooperative tiers read the operands structurally off
    the body. Shared by the flash score producer and the scalar register-tile contraction."""
    body: list[Stmt] = []
    names: list[str] = []
    for ob in operand_bodies:
        stmts = list(ob)
        body += stmts
        names.append(stmts[-1].defines()[-1])
    body.append(Assign(name=fold.value, op=lift, args=tuple(names)))
    body.append(fold)
    return Loop(axis=reduce_axis, body=Body(tuple(body)), role=AxisRole.CONTRACTION)


def _term_names(root) -> tuple[list[str], list[str]]:
    """Every SSA name and buffer name in ``root``'s term, in DETERMINISTIC first-appearance
    order over the canonical walk (operand edges in tuple order, then lift params / body defs /
    results, then the combine results; a ``Map``'s fn params / body / results, then sources).
    The order is a function of the stored params only, so the renumber map — and with it
    :func:`term_key` — is α-invariant."""
    from emmy.compiler.ir.stmt import Load  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import Fold  # noqa: PLC0415

    names: list[str] = []
    bufs: list[str] = []
    seen: set[str] = set()
    bseen: set[str] = set()

    def note(n) -> None:
        if isinstance(n, str) and n not in seen:
            seen.add(n)
            names.append(n)

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            walk(s)
            return
        if isinstance(s, Load) and s.input not in bseen:
            bseen.add(s.input)
            bufs.append(s.input)
        for n in s.defines():
            note(n)
        for b in s.nested():
            for c in b:
                note_stmt(c)

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        if node.axis is None:
            # The zero-axis reading names its binder first, then walks the sources — the order
            # the projection wrapper always used, so canonical renumbering is unchanged.
            for p in node.lift.params:
                note(p)
            for s in node.lift.body:
                note_stmt(s)
            for r in node.lift.results:
                note(r)
            for src in node.operands:
                note_stmt(src)
            return
        for e in node.operands:
            note_stmt(e)
        for p in node.lift.params:
            note(p)
        for s in node.lift.body:
            note_stmt(s)
        for r in node.lift.results:
            note(r)
        for r in node.combine.results:
            note(r)

    walk(root)
    return names, bufs


def _canon_order(stmts: tuple) -> tuple:
    """A DETERMINISTIC dependency-respecting order for an ANF stmt sequence — the hash-time
    body-order canonicalization (step 7): Kahn's algorithm over the def/use edges, ready stmts
    picked by a NAME-INDEPENDENT token (stmt kind, op spelling, buffer, arity), ties keeping the
    original relative order. Two α-equivalent lift bodies that differ only in the interleaving of
    independent stmts canonicalize to one order; the stored term itself is never reordered — the
    lowered nest depends on storage order, identity does not."""
    stmts = tuple(stmts)
    if len(stmts) <= 1:
        return stmts

    def token(s) -> tuple:
        from emmy.compiler.ir.stmt import Load  # noqa: PLC0415

        op = getattr(s, "op", None)
        return (
            type(s).__name__,
            getattr(op, "name", "") if op is not None else "",
            s.input if isinstance(s, Load) else "",
            len(getattr(s, "args", ()) or ()),
        )

    def reads(s) -> set:
        out = set(s.deps())
        for b in s.nested():
            for c in b:
                out |= set(reads(c))
        return out

    defs_of = [set(s.defines()) | {d for b in s.nested() for c in b for d in c.defines()} for s in stmts]
    read_of = [reads(s) for s in stmts]
    placed: list = []
    done: set = set()
    remaining = list(range(len(stmts)))
    while remaining:
        ready = [i for i in remaining if not (read_of[i] & {n for j in remaining if j != i for n in defs_of[j]} - done)]
        if not ready:
            return stmts  # a cycle (state-reading merge material) — keep the stored order
        pick = min(ready, key=lambda i: (token(stmts[i]), i))
        placed.append(stmts[pick])
        done |= defs_of[pick]
        remaining.remove(pick)
    return tuple(placed)


def _canon_tree(node):
    """``node`` with every λ body in its tree re-ordered by :func:`_canon_order` — the hash-side
    normal form ``term_key`` renumbers. Never stored."""
    from dataclasses import replace  # noqa: PLC0415

    from emmy.compiler.ir.stmt import Lambda  # noqa: PLC0415
    from emmy.compiler.ir.stmt.body import Body as _B  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import Fold  # noqa: PLC0415

    def canon_stmt(s):
        return _canon_tree(s) if isinstance(s, Fold) else s

    if not isinstance(node, Fold):
        return node
    if node._contraction is not None:
        # The bilinear reading's lift is generated, so only the EDGES canonicalize (reordering a
        # contraction's multiply stmts would be meaningless and would move the term key).
        return replace(node, operands=tuple(canon_stmt(e) for e in node.operands))
    body = tuple(canon_stmt(s) for s in _canon_order(tuple(node.lift.body)))
    lift = node.lift
    if all(st.pure for st in body):
        lift = Lambda(params=lift.params, body=_B(body), results=lift.results)
    return replace(node, lift=lift, operands=tuple(canon_stmt(e) for e in node.operands))


def term_key(root) -> str:
    """The α-INVARIANT identity of a stored term (step 7 — kernel identity keys off the
    canonically renumbered TERM, no longer the lowered loop nest): every SSA name maps to
    ``s<i>`` and every buffer to ``b<i>`` in the deterministic first-appearance order of
    :func:`_term_names`, the rename applied through the ONE ``_rewrite`` registry (the Fold /
    Map handlers rename lift / combine in lockstep, regenerating the
    exp-family programs over the renamed state), and the renumbered term's ``repr`` is the key
    text. Two terms differing only in SSA / buffer spelling — trace naming, fusion suffixes —
    key identically; any structural difference (an op, an axis, an operand shape) keys apart."""
    from emmy.compiler.ir.sigma import Sigma  # noqa: PLC0415
    from emmy.compiler.ir.stmt.passes import rewrite as _stmt_rewrite  # noqa: PLC0415

    if root is None:
        return ""
    root = _canon_tree(root) if isinstance(root, Fold) else root
    names, bufs = _term_names(root)
    ren = {n: f"s{i}" for i, n in enumerate(names)}
    canon = _stmt_rewrite(root, lambda n: ren.get(n, n), Sigma.IDENTITY, lambda a: a)
    text = repr(canon)
    # Buffer names appear in the repr only as ``input='<buf>'`` (the term carries no ``Write``
    # since 1q) — canonicalize them positionally, longest first so no name prefixes another.
    order = {b: i for i, b in enumerate(bufs)}
    for b in sorted(bufs, key=len, reverse=True):
        text = text.replace(f"input='{b}'", f"input='B{order[b]}'")
    return text


# --------------------------------------------------------------------------- #
# The structural dump — the STORED term as a tree, and NOTHING derived.
#
# The tile term is a tree of three node kinds over operand EDGES, and every fact a pass
# dispatches on is a stored param on a node (this module's whole premise). The dump renders
# exactly that: each node's own header, its stored params as labelled branches, and each operand
# edge recursed into — so an inline COMPUTED edge is visibly a subtree and a MATERIALIZED one
# visibly a leaf ``Load``.
#
# It renders NO derived material. The structure is already complete in the stored tree — the
# operand edges and their nesting — and a derived evaluation (the per-cell step, the synthesized
# nodes inside it, the loop nest a node lowers to) is a CONSEQUENCE of the stored params, exactly
# as re-derivable as ``lower()``'s output. Printing it beside storage is the inversion this module
# exists to prevent, and it is bulk: measured over eight kernels the step branch restated
# ``lift`` + ``combine`` and contributed no schedule site on seven of them. ``--ir loop`` is where
# a reader goes for a body.
#
# Schedule slices are not on the term at all — they annotate a node from the owning
# ``TileOp.schedule`` when one is supplied. The one slice that can address DERIVED material
# (flash's synthesized PV, ``TILE@pj``) has no stored node to annotate, so it prints in the
# ``schedule`` region beside the term rather than dragging the derived node into it.
# --------------------------------------------------------------------------- #

_TEE, _ELBOW, _PIPE, _GAP = "├─ ", "└─ ", "│  ", "   "


class _Ctx:
    """The dump's read-only context — the owning ``TileOp``'s schedule view (so each STORED node
    can be annotated with the slices keyed against it) and the iteration space a λ's capture set
    is measured against. ``None`` everywhere when a bare term is printed without its op."""

    def __init__(self, tile, root=None) -> None:
        self.sched = sched_of(tile) if tile is not None and tile.op is not None else None
        # The ITERATION SPACE a capture set is measured against. Only the OWNING ``TileOp`` knows
        # it in full: the term's own axes (:func:`axis_names`), the placement's free/grid axes, and
        # a boundary store's sweep axis (off-term since 1q) — the same three the cut's closure
        # check unions. Without a tile the placement is unknown, so ``captures`` declines to
        # answer rather than report grid coordinates as captured values.
        self.axes = None if tile is None else axis_names(root) if root is not None else set()
        if tile is not None:
            self.axes |= {a.name for a in (*tile.place.free, *tile.place.grid)}
            self.axes |= {st.sweep.name for st in tile.stores if st.sweep is not None}

    def captures(self, lam) -> tuple[str, ...]:
        """The VALUE names ``lam``'s body reads but neither binds nor takes from the iteration
        space — the same reading the cut's closure predicate applies
        (``_cut._captured_values``). Non-empty means the λ is NOT closed, which is exactly what
        makes a subtree unhoistable to an operand edge: flash's ``P = exp(s − m)`` reads the
        online-softmax carrier's running max, so it can never be an edge and its seam is not
        cuttable. Empty when the iteration space is unknown (no owning ``TileOp``) — an unanswered
        question prints as no annotation, never as "closed"."""
        return () if self.axes is None else tuple(sorted(lam.free_names() - self.axes))

    def note(self, node) -> str:
        """The schedule annotation for ``node`` — every slice the kernel keys against it, spelled
        by the codec (``''`` = the family's decided-empty)."""
        if self.sched is None:
            return ""
        bits = []
        for family in ("TILE", "REDUCE", "STAGE"):
            slice_ = self.sched.get(family, node)
            if slice_ is not None:
                bits.append(f"{family}={slice_.spell() or '·'}")
        return f"   ⟨{' '.join(bits)}⟩" if bits else ""


def _lam_sig(lam, ctx: _Ctx | None = None) -> str:
    """A lambda's one-line signature. A float result is the ι literal injected in the lift
    (softmax's singleton ``(x, 1)``), which has no def to name.

    A non-empty CAPTURE set is spelled between the params and the results — without it a λ that
    reads an enclosing value would print as though it were closed, which is the one property the
    reader most needs (an unclosed subtree can never become an operand edge)."""
    rs = ", ".join(r if isinstance(r, str) else format(r, "g") for r in lam.results)
    cap = ctx.captures(lam) if ctx is not None else ()
    free = f" [captures {', '.join(cap)}]" if cap else ""
    return f"λ({', '.join(lam.params)}){free} -> ({rs})"


def _axis_span(axis) -> str:
    win = getattr(axis, "window", None)
    parent = f" ⊂ {win.parent.name}" if win is not None and win.parent is not None else ""
    return f"{axis.name} in 0..{axis.extent}{parent}"


def _kind(edge) -> str:
    """An operand edge's inhabitant — the two things an input can be."""
    return "materialized" if isinstance(edge, Load) else "computed"


def _head(node, ctx: _Ctx) -> str:
    """One node's header line — its kind and the stored params that fit on a line. A λ-valued
    field is NOT one of them: its signature belongs on its own branch, next to the body it binds
    (``lift:`` / ``combine:`` / ``fn:``), not one screenful above it."""
    if not isinstance(node, Fold):
        return str(node)
    if node.axis is None:
        text = "Fold  free" + ("" if node.operands else "  ‹pointwise›")
    else:
        text = f"Fold[{_axis_span(node.axis)}] {node.role.name.lower()}" + (" unroll" if node.unroll else "")
    return text + ctx.note(node)


def _stmts(stmts, ctx: _Ctx):
    """Render a λ body (a ``lift`` / ``combine`` / ``Map.fn``) — indented two under the signature
    line that binds it, so the program reads as the binder's body rather than as a sibling of the
    branch labels. A structural NODE may occupy a statement position here (a demoted cone's inline
    node); it expands in place, since a lift body is storage like any other."""

    def render(cont: str) -> list[str]:
        out: list[str] = []
        for s in stmts:
            if isinstance(s, Fold):
                out.append(f"{cont}  {_head(s, ctx)}")
                out.extend(_branch(_items(s, ctx), cont + "  "))
            else:
                out.extend(pretty_body(Body((s,)), cont + "  "))
        return out

    return render


def _subtree(node, ctx: _Ctx):
    return lambda cont: _branch(_items(node, ctx), cont)


def _edge(label: str, edge, ctx: _Ctx) -> tuple[str, object]:
    """One operand edge as a tree item — a ``Load`` is a leaf spelled inline, a computed edge
    recurses into the node stored on it."""
    if isinstance(edge, Load):
        return f"{label}: {edge.pretty()[0].strip()}   ‹materialized›", lambda cont: []
    return f"{label}: {_head(edge, ctx)}   ‹computed›", _subtree(edge, ctx)


def _items(node, ctx: _Ctx) -> list[tuple[str, object]]:
    """A node's STORED children, each a labelled branch. Nothing derived: the step, the
    synthesized nodes inside it and the lowered nest are all consequences of these params."""
    items: list[tuple[str, object]] = []
    if not isinstance(node, Fold):
        return items
    con = node._contraction
    if con is not None:
        # The bilinear reading labels its edges ``a`` / ``b`` — the same labels the path codec
        # spells, so a reader can match a dump line to a ``PLACE@a`` key by eye.
        a, chans = con
        items.append(_edge("operand[a]", a, ctx))
        one = len(chans) == 1
        items += [_edge("operand[b]" if one else f"operand[b{i}] -> {ch.acc}", ch.b, ctx) for i, ch in enumerate(chans)]
    else:
        items += [_edge(f"operand[{i}]", e, ctx) for i, e in enumerate(node.operands)]
    if node.axis is not None:
        init = ", ".join(x if isinstance(x, str) else format(x, "g") for x in node.init)
        items.append((f"init: ({init})", lambda cont: []))
    # Always emitted, even for an empty body: the branch carries the SIGNATURE, and a node's
    # binder is storage whether or not it computes anything (an identity projection binds too).
    items.append((f"lift: {_lam_sig(node.lift, ctx)}", _stmts(node.lift.body, ctx)))
    if node.combine is not None:
        items.append((f"combine: {_lam_sig(node.combine, ctx)}", _stmts(node.combine.body, ctx)))
    return items


def _branch(items: list[tuple[str, object]], cont: str) -> list[str]:
    out: list[str] = []
    for i, (head, sub) in enumerate(items):
        last = i == len(items) - 1
        out.append(f"{cont}{_ELBOW if last else _TEE}{head}")
        out.extend(sub(cont + (_GAP if last else _PIPE)))
    return out


def pretty(op, indent: str = "", *, tile=None) -> list[str]:
    """Structurally pretty-print a kernel op (for dumps) as the STORED tree and nothing else —
    each node's kind and params, its operand edges recursed into. No derived material: the
    per-cell step, the nodes synthesized inside it and the lowered nest all follow from these
    params (``--ir loop`` is where a body lives). Pass ``tile`` — the owning ``TileOp`` — to
    annotate each node with the schedule slices keyed against it. A bare stmt falls back to its
    own pretty."""
    ctx = _Ctx(tile, root=op)
    if isinstance(op, Fold):
        return [f"{indent}{_head(op, ctx)}", *_branch(_items(op, ctx), indent)]
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = [
    "Map",
    "axis_role",
    "Sched",
    "cone_seam",
    "unplaced_slices",
    "contraction_loop",
    "lower",
    "nodify_reduce",
    "pretty",
    "projection_tail",
    "reduce_loop",
    "reduce_plan",
    "sched_of",
    "seal_workers",
    "axis_names",
    "term_key",
]
