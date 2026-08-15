"""The recognition CORE — the pure ``LoopOp`` body → recognized ``TileOp`` read.

Factored out of the ``010_recognize`` pass shell so the strict golden decode can derive a
RECORD's kernel identity through the exact functions the live compile recognizes with: one
shared code path, so record-side and fork-side identity cannot drift. Everything here is pure
structure — no graph surgery, no placement, no pins; the pass (`010_recognize`) adds the
placement fork and the cut realization around :func:`recognized_tile`.

Recognition reads through the two shared algebra parsers and nothing else: the λ-fold reading
(:func:`~._fromloop.fold_from_loop`) and the ⊗-lift reading (``_atomize._bilinear_reads``); see
the pass docstring for the step order.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import (
    Channel,
    Fold,
    Placement,
    TileOp,
    split_effects,
)
from emmy.compiler.pipeline.passes.lowering._reduction import loop_state_head
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction, make_cone
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.pipeline.passes.lowering.tile._softmax import _fuse
from emmy.compiler.pipeline.pipeline import LoweringError

# --------------------------------------------------------------------------- #
# Lift — peel the free (parallel) axes off and lift the per-cell compute into ONE
# zero-axis ``Fold`` whose body holds the annotated reduce ``Loop`` + projection.
# --------------------------------------------------------------------------- #


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(free_axes, per_cell_stmts)``.

    The outer chain of **free** loops becomes the parallel axes. At every level of the
    chain a leading run of pure stmts (``Load`` / ``Assign`` / ``Init`` — loop-invariant
    loads hoisted above or between the free loops, e.g. a broadcast row scale ``rs[m]``
    read once per ``m``) is sunk into the per-cell body, re-evaluated per cell. The chain
    stops at the first reduce loop, branch, or non-pure stmt — everything from there down
    is the per-cell body (the fold and its epilogue / output sweep), run serially by one
    thread per cell."""
    axes: list = []
    prefix: list[Stmt] = []
    cur = list(body)
    while True:
        i = 0
        while i < len(cur) and isinstance(cur[i], (Load, Assign, Init)):
            i += 1
        head, rest = cur[:i], cur[i:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + cur
        prefix += head
        axes.append(rest[0].axis)
        cur = list(rest[0].body)


def _flat_cell(cell: list[Stmt]) -> tuple[Fold, tuple]:
    """A FLAT zero-axis ``Fold`` of the cell stmts — the pointwise / un-recognized spelling — with the
    trailing root ``Write``\\ s split to boundary :class:`Store`\\ s when the cell is otherwise
    pure (the pointwise / concat forms; 1q). A reduce-bearing or interleaved-effect cell keeps
    the raw-loop-IR spelling verbatim (the scalar-tier escape — split declines). Sweep stores are
    not taken on a flat cell: without a fold source the materializer's flat-root arm reattaches
    plain ``Write``\\ s only."""
    split = split_effects(tuple(cell))
    if split is not None and split[1] and all(st.sweep is None for st in split[1]):
        pure, stores = split
        return Fold.projection(body=Body(pure)), stores
    return Fold.projection(body=tuple(cell)), ()


def _reduce_in(stmts) -> bool:
    """Any reduce ``Loop`` reachable in ``stmts`` (deep)."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            return True
        if any(_reduce_in(b) for b in s.nested()):
            return True
    return False


def _reads(stmts) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep — through ``deps`` + nested bodies)."""
    out: set[str] = set()
    for s in stmts:
        out.update(s.deps())
        for b in s.nested():
            out |= _reads(list(b))
    return out


def _bilinear_candidate(body: list[Stmt], k_name: str) -> bool:
    """Whether a flat reduce MAY read as a bilinear contraction: ONE additive fold, a ⊗ lift that
    distributes over it with two arguments, and a K-indexed load reachable. Deliberately LIBERAL —
    this is candidacy, not a parser: the ONE binder (:func:`bind_contraction`) arbitrates every
    operand shape (direct loads, hoistable k-invariant factor chains, computed cones, the
    both-computed decode pair), and an unbindable candidate derives ``PLANAR`` at
    :func:`_nodify_contraction`. Recognition holds no second reading of the algebra — the old
    clean/computed/both-computed decision tree here was a parallel parser of exactly what the
    binder parses, kept in sync by hand."""
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return False
    fold = accs[0]
    lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
    if lift is None or not lift.op.distributes_over(fold.op) or len(lift.args) != 2:
        return False
    return any(isinstance(s, Load) and k_name in {v for e in s.index for v in e.free_vars()} for s in body)


def _annotate_reduce(rloop: Loop, pre_reduce: tuple[Stmt, ...]) -> Loop | None:
    """Annotate a single FLAT reduce ``Loop`` with its :class:`AxisRole`,
    moving any reduce-feeding ``pre_reduce`` prologue INTO the loop body (so the cooperative
    register fold replicates it per accumulator chain). The role is the ONLY annotation — the
    algebra is the body itself (its dissolved fold ``Accum``\\ s / streaming merge, read back by
    ``Fold.from_loop``). An already-annotated loop (online-softmax from ``_fuse``) keeps its
    role; a clean contraction stamps ``CONTRACTION``; a single-``Accum`` reduce stamps
    ``PLANAR``. Returns ``None`` (→ flat zero-axis fold fallback) when the shape
    can't be read (several ``Accum``\\ s, no fold)."""
    body = (*pre_reduce, *rloop.body)
    if rloop.role is not AxisRole.FREE:
        return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=rloop.role)
    if _bilinear_candidate(list(body), rloop.axis.name):
        return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.CONTRACTION)
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return None
    return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.PLANAR)


def _rewrite_deep(stmt: Stmt, rename) -> Stmt:
    """Apply an SSA rename through a stmt, recursing into ``Loop`` bodies."""
    if isinstance(stmt, Loop):
        from dataclasses import replace  # noqa: PLC0415

        return replace(stmt, body=Body(tuple(_rewrite_deep(s, rename) for s in stmt.body)))
    return stmt.rewrite(rename)


def _lift_cell(cell: list[Stmt], free: list, output: str) -> tuple[Fold, tuple]:
    """Lift the per-cell stmts into a zero-axis ``Fold`` whose body is the annotated loop nest, returning
    ``(node, stores)`` — the 1q boundary split: a projected reduce's root ``Write`` (and the
    rms/softmax output-sweep ``Loop`` around it) leaves the term for a :class:`Store` decoration
    when :func:`split_effects`' byte-identity gate accepts the shape; a declining projection (a
    contraction tail's composed loop) keeps the raw-loop-IR spelling with empty stores. A pure
    pointwise cell (no reduce) is a flat zero-axis fold of its stmts; a single flat reduce annotates that
    reduce ``Loop`` in place (``CONTRACTION`` / ``PLANAR`` / pre-annotated ``TWISTED``), its body
    holding the reduce loop followed by the projection — stripped to just the loop when the only
    epilogue is the grid-cell ``Write`` (materialize stores ``out`` as glue). A cell with no, or
    several, or a nested reduce stays a flat zero-axis fold (un-annotated → the scalar tier)."""
    reduces = [i for i, s in enumerate(cell) if isinstance(s, Loop) and s.is_reduce]
    if len(reduces) != 1:
        return _flat_cell(cell)
    idx = reduces[0]
    rloop = cell[idx]
    if _reduce_in(list(rloop.body)):
        return _flat_cell(cell)  # nested (non-flash) reduce — keep loop-IR form
    # Route the loop-invariant prologue (stmts above the reduce, sans the regenerated ``Init``
    # seeds) one dependency cone at a time: stmts feeding the reduce move INTO the loop
    # (``pre_reduce``), while independent stmts feeding only the epilogue stay after it. Treating
    # the whole preamble as one unit demoted contractions with both kinds of independent values —
    # e.g. DiT's GELU constants feed computed A while the linear bias feeds the epilogue. A single
    # stmt/cone feeding BOTH still can't be placed by reordering, so keep that cell as a flat
    # zero-axis ``Fold`` (its loop-IR order is preserved verbatim).
    before = [s for s in cell[:idx] if not isinstance(s, Init)]
    after = list(cell[idx + 1 :])
    reduce_need = _reads(list(rloop.body))
    epilogue_need = _reads(after)
    reduce_idx: set[int] = set()
    epilogue_idx: set[int] = set()
    dup_renames: dict[str, str] = {}
    for i in range(len(before) - 1, -1, -1):
        stmt = before[i]
        defs = set(stmt.defines())
        feeds_reduce = bool(defs & reduce_need)
        feeds_epilogue = bool(defs & epilogue_need)
        if feeds_reduce and feeds_epilogue:
            if isinstance(stmt, Load) and not stmt.deps() and not any(e.free_vars() for e in stmt.index or ()):
                # An axis-invariant scalar ``Load`` demanded by BOTH sides (a shared mask / eps
                # constant the splicer hoisted to cell scope): pure, so a renamed copy rides the
                # reduce side while the original stays with the epilogue — no reordering needed.
                for name in defs:
                    dup_renames[name] = f"{name}__stat"
                reduce_idx.add(i)
                epilogue_idx.add(i)
                continue
            return _flat_cell(cell)
        if feeds_reduce:
            reduce_idx.add(i)
            reduce_need.update(stmt.deps())
        else:
            # Keep unused pure preamble stmts on the epilogue side, preserving the old behavior
            # and original order. If a later epilogue stmt depends on this one, the reverse walk
            # has already added that dependency to ``epilogue_need``.
            epilogue_idx.add(i)
            if feeds_epilogue:
                epilogue_need.update(stmt.deps())
    dup_idx = reduce_idx & epilogue_idx
    if any(not dup_renames.keys() & set(before[i].defines()) for i in dup_idx):
        return _flat_cell(cell)
    pre_reduce = tuple(s for i, s in enumerate(before) if i in reduce_idx)
    pre_epilogue = tuple(s for i, s in enumerate(before) if i in epilogue_idx)
    if dup_renames:
        rename = lambda n: dup_renames.get(n, n)  # noqa: E731
        pre_reduce = tuple(_rewrite_deep(s, rename) for s in pre_reduce)
        rloop = _rewrite_deep(rloop, rename)
    annotated = _annotate_reduce(rloop, pre_reduce)
    if annotated is None:
        return _flat_cell(cell)
    grid_index = tuple(Var(ax.name) for ax in free)
    bare = (
        not before
        and len(after) == 1
        and isinstance(after[0], Write)
        and after[0].is_scalar
        and after[0].value == loop_state_head(annotated)
        and after[0].output == output
        and after[0].index == grid_index
    )
    # ``bare`` ⇒ materialize writes ``carrier.out`` at the grid cell (empty projection).
    projection = () if bare else (*pre_epilogue, *after)
    # A PLANAR / TWISTED reduce lifts to a typed ``Fold`` node (its ⊕ carrier + structure split
    # out, the fold loop synthesized on demand); a ``CONTRACTION`` is nodified to a
    # contraction right after the free axes are ordered (:func:`_nodify_contraction`).
    # ``lower`` flattens either back identically.
    if annotated.role in (AxisRole.PLANAR, AxisRole.TWISTED):
        reduction = fold_from_loop(annotated)
        if reduction is None:  # not λ-representable — the raw-loop-IR escape (the flat zero-axis fold)
            return Fold.projection(body=(annotated, *projection)), ()
        # A projected reduce (softmax / RMSNorm) is the ``source`` of a zero-axis ``Fold`` whose
        # body IS that projection — pure, its root store (and the output sweep around it) a
        # boundary ``Store`` when the split gate accepts; the composed-tail shapes (norm→linear's
        # column loop) decline and keep the raw-loop-IR spelling for
        # ``bind_prologue_contraction`` to parse. ``_project`` is that rule, and a bare reduce
        # (empty projection) passes straight through it.
        return _project(reduction, Body(projection))
    return Fold.projection(body=(annotated, *projection)), ()


def _nodify_contraction(node, free: tuple):
    """Nodify a freshly-lifted flat ``CONTRACTION`` zero-axis ``Fold`` into the contraction
    structural node with a **deferred** per-cell ``TilePlan()`` (the schedule fork re-tiles it),
    resolving the operand→role binding ONCE, recognize-side (:func:`bind_contraction` over the
    ordered ``free`` axes' trailing ``(m, n)``). An unbindable contraction — a 1-D output (a
    matvec-shaped cell) or no (m, n)-bearing K-loads — keeps its loads INLINE in the fold's step
    (no operand hoist), so the fold **derives PLANAR** (``Fold.role``) and takes the reduce tiers
    at schedule dispatch, gaining the cooperative / ILP partitions a per-cell serial fold never
    offered — the old recognition-time role rewrite is gone. After this step no flat zero-axis fold
    carries an annotated ``CONTRACTION`` loop — the scheduler and materializer read contraction
    structure only off the node. A computed-A cone is stored INLINE on the ``a`` edge
    (:func:`make_cone`)."""
    if not (isinstance(node, Fold) and node.axis is None) or node.operands or len(node.body) == 0:
        return node, ()
    rloop = node.body[0]
    if not isinstance(rloop, Loop) or rloop.role is not AxisRole.CONTRACTION:
        return node, ()
    projection = Body(tuple(node.body[1:]))
    if len(free) >= 2:
        try:
            a_load, b_load, acc, epi = bind_contraction(rloop, free[-2].name, free[-1].name, projection)
        except LoweringError:
            pass
        else:
            con = Fold.contraction(
                k_axis=rloop.axis,
                a=a_load if isinstance(a_load, Load) else make_cone(a_load, rloop.axis.name),
                # A computed B is a closed zero-axis operand node. Unlike A's norm/activation
                # cone it has no row-statistic seam to split; its whole generic MAP tree is
                # evaluated at each (k, n) slab cell by the smem compute fill.
                channels=(Channel(b=b_load if isinstance(b_load, Load) else Fold.projection(body=Body(tuple(b_load))), acc=acc),),
            )
            # ONE home for the projection: the wrapping zero-axis fold's lift body, never a node field. The
            # STORED form is the contraction itself (1s) — pure algebra; the
            # output axes / tile / stage are caller facts, stamped at the point of use.
            return _project(con, epi)
    red = fold_from_loop(rloop)  # loads stay inline in the lift — the fold derives PLANAR
    if red is None:  # not λ-representable — the raw-loop-IR escape (the flat zero-axis fold)
        return Fold.projection(body=(rloop, *projection)), ()
    return _project(red, projection)


def _project(fold: Fold, projection: Body) -> tuple[Fold, tuple]:
    """Wrap ``fold`` in its projecting zero-axis ``Fold``, the root store split to the boundary (1q):
    ``split_effects``' byte-identity gate accepts the epilogue's trailing ``Write`` (and an
    output sweep) or the raw-loop-IR spelling stands. A bare fold passes through (its grid-cell
    store stays materialize glue)."""
    if not len(projection):
        return fold, ()
    split = split_effects(tuple(projection))
    if split is not None:
        pure, stores = split
        return Fold.projection(body=Body(pure), operands=(fold,)), stores
    return Fold.projection(body=projection, operands=(fold,)), ()


def _lift(stmts: list[Stmt], output: str) -> tuple[Fold, tuple, tuple]:
    """Peel the free axes and lift the per-cell compute, returning ``(root node, free axes,
    boundary stores)``. The free axes are the schedule's (carried on the ``TileOp``, not the node);
    The schedule maps them onto the grid, and the ``stores`` ride
    ``TileOp.stores`` (1q). A ``CONTRACTION`` cell
    nodifies to a contraction once the free axes are output-ordered (the binding needs
    the final ``(m, n)``)."""
    free, cell = _peel(Body(tuple(stmts)))
    node, stores = _lift_cell(cell, free, output)
    free = _order_free_by_output(node, free, stores)
    node, con_stores = _nodify_contraction(node, free)
    # At most one of the two conversion points fires: ``_nodify_contraction`` only converts the
    # flat CONTRACTION shape, which ``_lift_cell`` always emits store-less.
    return node, free, stores or con_stores


def _order_free_by_output(node: Fold, free: list, stores: tuple = ()) -> tuple:
    """Order the free (grid) axes to match the **output Write's index order**, so the innermost
    grid axis is the output's *contiguous* dim. The contraction tier needs ``n_axis == grid[-1] ==``
    the contiguous output axis — the mma fragment store coalesces a ``float2`` along it, and the
    cuda lowering's ``ldm`` is the output's inner extent — but the peel / loop-naming order can
    diverge from the output layout (e.g. a batched ``Q@Kᵀ`` whose ``kv`` got named before ``m``).
    The root ``Write`` is read from the body (a raw-loop-IR spelling) or the boundary ``stores``
    (a converted projection — its top-level store only: a sweep store's ``Write`` was never a
    top-level stmt, so it never ordered the grid). A node with no explicit output ``Write`` (a
    bare contraction whose grid-cell store is synthesized
    at materialize, already in free order) is left as-is."""
    body = node.lower() if isinstance(node, Fold) else getattr(node, "body", ())
    write = next((s for s in body if isinstance(s, Write)), None)
    if write is None:
        write = next((st.write for st in stores if st.sweep is None), None)
    if write is None:
        return tuple(free)
    pos = {e.name: i for i, e in enumerate(write.index) if isinstance(e, Var)}
    if not all(ax.name in pos for ax in free):
        return tuple(free)  # a free axis absent from the output index — leave the peel order
    return tuple(sorted(free, key=lambda ax: pos[ax.name]))


def recognized_tile(op: LoopOp, output_name: str, name: str = "") -> TileOp:
    """The pure recognition read: ``op``'s body → the UNMAPPED recognized ``TileOp`` (online
    softmax fused, free axes peeled and output-ordered, the cell lifted / nodified). This is the
    schedule fork's ``root_op`` on the fused arm, and therefore the carrier of the kernel's
    structural identity — the strict golden decode derives a record's identity by calling exactly
    this on the record's own lowered target."""
    fused, _ = _fuse(op.body)
    node, free, stores = _lift(list(fused), output_name)
    return TileOp(op=node, name=name, place=Placement(free=free), inputs=dict(op.inputs), stores=stores)
