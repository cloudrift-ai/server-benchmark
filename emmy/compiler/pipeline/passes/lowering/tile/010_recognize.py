"""Recognize a ``LoopOp``'s algebraic structure and lift it to an UNMAPPED ``TileOp`` — the
STRUCTURAL half of the Loop-IR → Tile-IR boundary.

After this rule nothing downstream traffics in ``LoopOp``. Recognition reads the algebra off the
body and lifts the per-cell compute into a :class:`~emmy.compiler.ir.tile.ir.Fold` whose body is the
**annotated loop nest** (the reduce ``Loop`` stamped with its
:class:`~emmy.compiler.ir.axis.AxisRole` — the only loop annotation; the algebra is the body),
wrapped in a :class:`~emmy.compiler.ir.tile.ir.TileOp` whose ``place`` carries just the free axes.
That op IS the rewrite's result. The schedule picks it up on the next rule sweep, maps the free
axes onto the grid and offers the scheduling forks; materialization back to loop IR happens in
``lowering/kernel``.

Nothing here reads a knob or a pin — recognition is structure, and every choice it makes is
unconditional. (The one exception is PLACEMENT ROUTING, step 3.5: a routing golden / ``PLACE`` pin
cuts the recognized tree into a fragment of un-mapped ``LoopOp``\\ s, and it must resolve BEFORE any
schedule fork exists, so it cannot wait for ``020``.)

All recognition lives in THIS one rule (no separate flash / softmax pass), in order (each
step unconditional — no knobs):

1. **Flash attention** — a softmax-then-P@V kernel (+ its clean scaled-QK producer) is
   the online-softmax twisted reduce over a streaming KV axis; rewrite the pair to one fused
   flash ``TileOp`` (the ``(m, l, O)`` ``TWISTED`` kv loop over a nested ``CONTRACTION`` score
   loop), with its free ``(batch…, m, d)`` axes carried on the schedule. Graph rewrite —
   consumes the score producer. Recognition + construction live in the ``_flash`` helper
   (``try_flash``). Because the fusion reads the score producer's Q/K as plain ``Load``\\ s, a
   node that IS such a producer is *deferred* (left a ``LoopOp``, :func:`is_flash_score_producer`)
   so step 3 doesn't lift it out from under its consumer.
2. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same input fuses
   into one streaming online-softmax loop: a ``TWISTED`` reduce ``Loop`` carrying the ``(m, d)``
   exp-family merge dissolved in the body. The ``_softmax`` helper
   (``_fuse``).
3. **Lift** — peel the free (parallel) axes off the kernel and lift the per-cell compute into a
   zero-axis ``Fold`` whose body holds the annotated reduce ``Loop`` + projection: a pure pointwise body is a
   flat zero-axis fold; a single flat reduce is annotated in place — ``CONTRACTION`` (clean contraction)
   / ``PLANAR`` (plain ``sum`` / ``max`` / ``mean``) / pre-annotated ``TWISTED`` (online softmax) —
   with the projection after it. The free axes ride on
   the ``TileOp``'s schedule (the root's concern); ``_schedule`` maps them onto the grid. A cell
   the lift can't cleanly factor (no reduce, several reduces, or a nested non-flash reduce) stays a
   flat un-annotated zero-axis ``Fold`` (→ the scalar tier).
4. **The MONOID-producer composition** — a lifted ``Fold.projection(source=Fold)`` whose body is the
   statistic's scalar epilogue + a fresh free (column) ``Loop`` over one or more ⊗-folds of ONE
   shared A value reading the statistic (the fused norm→linear edge ``rmsnorm(x)·nw @ w``; its
   N-channel form the gate/up MLP edge ``swiglu(x̂@Wg, x̂@Wu)`` — a product-monoid fold) ALSO
   nodifies to ``Fold.projection(body=projection, operands=(Contraction,))``: ONE computed-A product
   :class:`Contraction` with a :class:`Channel` per ⊗-fold, the A cone stored inline on its edge
   (a real node tree — the per-row statistic its ``Fold`` source)
   (``_atomize.bind_prologue_contraction``, structure-only), its column axis joining the grid.
   Recognition only *builds* that reading (for the routing router's reference tree, below);
   The schedule re-derives it and merges both forms' candidates into ONE fork, because which
   of the two a row realizes is a decision about the SCHEDULE.

Flash must precede online-softmax which must precede the lift: each later step consumes the
``Accum``\\ s an earlier one matches. A **symbolic** axis (dynamic ``seq_len``) is left
un-lifted (the scalar ``Tile`` decode needs static extents) — the ``LoopOp`` stays put for
the dynamic-shape tier.

This is case-by-case recognition today (flash / online-softmax / contraction patterns);
the intent is to grow it toward ONE algorithmic algebra recognizer.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import (
    Channel,
    Contraction,
    Fold,
    Placement,
    TileOp,
    split_effects,
)
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering._reduction import loop_state_head
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction, bind_prologue_contraction, make_cone, map_cone
from emmy.compiler.pipeline.passes.lowering.tile._cut import realize_cut, route_cut
from emmy.compiler.pipeline.passes.lowering.tile._flash import is_flash_score_producer, try_flash
from emmy.compiler.pipeline.passes.lowering.tile._softmax import _fuse
from emmy.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", LoopOp)]


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


def _is_clean_contraction(body: list[Stmt], k_name: str) -> bool:
    """True iff ``body`` (a reduce loop's body, possibly with a moved-in prologue) is a clean
    contraction whose lift multiplies the operand loads **directly** — body is exactly the
    K-indexed operand loads + the ``⊗`` lift ``Assign`` (distributing over the fold) + the
    additive fold ``Accum``, contracting ≥ 2 operand loads, with NO loop-invariant
    load or per-operand preprocessing (a pre-scaled ``sum_k (x·s)·(y·s)`` is NOT clean — it
    becomes a degenerate ``PLANAR`` reduce so the scale survives in the loop body). The two
    operands may be different affine views of one packed buffer (for example load-time-concatenated
    QKV); operand identity is the load/index, not the backing allocation."""
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return False
    fold = accs[0]
    lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
    if lift is None or not lift.op.distributes_over(fold.op):
        return False
    k_loads = [ld for ld in body if isinstance(ld, Load) and k_name in {v for e in ld.index for v in e.free_vars()}]
    if len(k_loads) < 2:
        return False
    all_loads = [s for s in body if isinstance(s, Load)]
    if len(body) == len(all_loads) + 2 and len(all_loads) == len(k_loads) and set(lift.args) == {ld.names[0] for ld in k_loads}:
        return True
    # COMPUTED-A contraction: the lift multiplies ONE K-indexed operand load by a value this loop
    # computes (a fused operand cone). Still a contraction — the cone is the A tile — so mark it and
    # let ``bind_contraction`` bind the cone; the alternative is a PLANAR scalar fold, which is what
    # took gemma-4's GeGLU->down_proj edge off the mma tier entirely (66x).
    if len(lift.args) != 2:
        return False
    operand = next((ld for ld in k_loads if ld.names[0] in lift.args), None)
    if operand is None:
        return False
    cone = map_cone(list(body), next(a for a in lift.args if a != operand.names[0]))
    return bool(cone) and any(isinstance(st, Load) and k_name in {v for e in st.index for v in e.free_vars()} for st in cone)


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
    if _is_clean_contraction(list(body), rloop.axis.name):
        return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.CONTRACTION)
    accs = [s for s in body if isinstance(s, Accum)]
    if len(accs) != 1:
        return None
    return Loop(axis=rloop.axis, body=Body(body), unroll=rloop.unroll, role=AxisRole.PLANAR)


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
    for i in range(len(before) - 1, -1, -1):
        stmt = before[i]
        defs = set(stmt.defines())
        feeds_reduce = bool(defs & reduce_need)
        feeds_epilogue = bool(defs & epilogue_need)
        if feeds_reduce and feeds_epilogue:
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
    if reduce_idx & epilogue_idx:
        return _flat_cell(cell)
    pre_reduce = tuple(s for i, s in enumerate(before) if i in reduce_idx)
    pre_epilogue = tuple(s for i, s in enumerate(before) if i in epilogue_idx)
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
    # :class:`Contraction` right after the free axes are ordered (:func:`_nodify_contraction`).
    # ``lower`` flattens either back identically.
    if annotated.role in (AxisRole.PLANAR, AxisRole.TWISTED):
        reduction = Fold.from_loop(annotated)
        if reduction is None:  # not λ-representable — the raw-loop-IR escape (the flat zero-axis fold)
            return Fold.projection(body=(annotated, *projection)), ()
        if bare:
            return reduction, ()
        # A projected reduce (softmax / RMSNorm) is the ``source`` of a zero-axis ``Fold`` whose body IS that
        # projection — pure, its root store (and the output sweep around it) a boundary ``Store``
        # when the split gate accepts; the composed-tail shapes (norm→linear's column loop) decline
        # and keep the raw-loop-IR spelling for ``bind_prologue_contraction`` to parse.
        split = split_effects(projection)
        if split is not None:
            pure, stores = split
            return Fold.projection(body=Body(pure), operands=(reduction,)), stores
        return Fold.projection(body=Body(projection), operands=(reduction,)), ()
    return Fold.projection(body=(annotated, *projection)), ()


def _nodify_contraction(node, free: tuple):
    """Nodify a freshly-lifted flat ``CONTRACTION`` zero-axis ``Fold`` into the :class:`Contraction`
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
            con = Contraction(
                k_axis=rloop.axis,
                a=a_load if isinstance(a_load, Load) else make_cone(a_load, rloop.axis.name),
                channels=(Channel(b=b_load, acc=acc),),
            )
            # ONE home for the projection: the wrapping zero-axis fold's lift body, never a node field. The
            # STORED form is the :class:`Contraction` node itself (1s) — pure algebra; the
            # output axes / tile / stage are caller facts, stamped at the point of use.
            return _project(con, epi)
    red = Fold.from_loop(rloop)  # loads stay inline in the lift — the fold derives PLANAR
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
    nodifies to a :class:`Contraction` once the free axes are output-ordered (the binding needs
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


def rewrite(match: Match, root: Node, ctx=None) -> TileOp | Graph | None:
    # (1) Flash attention — a graph rewrite that fuses a softmax-then-P@V kernel with its
    # scaled-QK producer. Tried first on every node; flash precedes online-softmax precedes
    # normalize, each consuming the Accums the next would match. The fusion is unconditional:
    # a kernel flash recognition can certify is always fused (an uncertifiable one — RoPE'd QK —
    # falls through to the separate score producer + softmax-then-P@V kernels below).
    graph = match.graph
    flash = try_flash(graph, root)
    if flash is not None:
        return flash
    # (2) Defer a flash score producer: the general lift below would turn this scaled-QK
    # matmul into a ``TileOp`` before its softmax-then-P@V consumer fuses, and that fusion
    # reads the producer's Q/K as plain ``Load``s. Leave it a ``LoopOp`` until the consumer
    # has had its chance to consume it (a later scan re-visits this node, by then removed).
    if is_flash_score_producer(graph, root):
        raise RuleSkipped("flash score producer — defer to its consumer's fusion")
    loop: LoopOp = root.op
    # (3) Online softmax — the sibling-fold tupling: fuse the adjacent (rowmax, Σexp) reduce
    # pair into one streaming pass.
    fused, _ = _fuse(loop.body)
    node, free, stores = _lift(list(fused), root.output.name)
    # A symbolic FREE (parallel) axis rides a **symbolic grid**: the ``Tile`` decode sizes the
    # launch from the runtime extent (``_gid < ∏extents``, the ``Dim`` name threaded as an
    # ``int`` arg by the cuda lowering) — the dynamic-grid tier. A symbolic REDUCE /
    # output-sweep axis is likewise supported (the reduce loop strides to the runtime extent,
    # the ``< seq_len`` cap masking the tail). Register-tiled symbolic axes mask their tail
    # cell (clamp-read + guarded write) in ``lowering/kernel``.
    # Wrap the lifted node + its unmapped placement in an UNMAPPED ``TileOp`` — recognition's
    # OUTPUT. The schedule picks it up on the next rule sweep, maps the free axes onto the grid
    # and offers the per-axis scheduling forks (``REDUCE`` partition / ``TILE`` output tile).
    # ``inputs`` is seeded from the matched ``LoopOp`` (the matcher populated its real Tensors) so
    # the scheduler can read operand shapes (the shared-row stage detection); the matcher refreshes
    # it from the graph again when a later pass matches the scheduled op.
    map_tile = TileOp(op=node, name=loop.name, place=Placement(free=free), inputs=dict(loop.inputs), stores=stores)
    pro = bind_prologue_contraction(node, free)
    # (3.5) PLACEMENT ROUTING (phase 4) — resolved FIRST, before any schedule fork exists: a
    # ROUTING golden (PLACE-only knobs for this kernel's (kind, shape)) or an authoritative
    # PLACE pin cuts the recognized tree into a fragment of un-mapped LoopOps; each piece
    # re-recognizes as a fresh root on the pass-scan restart and resolves its OWN entry
    # (recursive — a piece's entry may itself cut). No entry / no pin = fuse = the recognized
    # form, by absence. The fused (computed-A) reading is the routing reference tree when it
    # binds — its seams (the `a` cone edge) are the ones a routing entry spells.
    route_tree, route_free, route_stores = (pro[0], (*free, pro[1]), pro[2]) if pro is not None else (node, free, stores)
    seam = route_cut(ctx, dict(loop.knobs or {}), route_tree, route_stores, route_free)
    if seam is not None:
        return realize_cut(match, root, route_tree, route_free, route_stores, seam)
    # Recognition ends here: the UNMAPPED tile is the rewrite's result. The MONOID-producer
    # composition (``pro``) is re-derived by the schedule — it is a decision about the SCHEDULE
    # (which of the two readings of this one loop each fork row realizes), not about the structure,
    # and it needs the schedule results to arbitrate (a warp ``TILE`` pin keeps the contraction
    # rows alone; a contraction form with no legal row demotes back to the PLANAR reduce).
    return map_tile
