"""The SPLIT realizer — a ``SPLIT`` decision partitions a reduce axis across kernels.

``SPLIT@<axis> = g<w>[a|k]`` on a reducing fold splits the kernel along that axis. Each partition
reduces its contiguous slice ``[s·B, (s+1)·B)`` (``B = extent/w``) and contributes its carrier
*state*; the ``finalize`` letter says how the partitions recombine:

- ``k`` — the partial writes its state to a ``ws[w, *free]`` **f32** workspace and a sibling
  **finalize kernel** folds the workspace over the split axis, then runs the original projection
  epilogue. **Two kernels.** The only legal arm for a twisted carrier (the streaming-softmax
  rescale is not an atomic).
- ``a`` — the partial ``atomicAdd``\\ s its (additive) state into the zero-init'd output, applying
  the projection per-partition first, which is legal exactly when that projection *distributes*
  over the add. **One kernel.** Additive single-component carriers only.

**The pieces are completely new kernels.** Nothing is inherited: each is an UNMAPPED ``TileOp``, so
``020_schedule`` picks it up and it resolves its own schedule fork, its own knobs and its own
``S_*`` identity, and records its own rows. The pre-split kernel keeps no row of its own — its
latency is the Σ of the pieces' best latencies, which is what the structural-fork pricing already
computes (``search/policy/greedy._priced_pick``).

**Everything happens in Tile IR.** The pieces are built by σ-reindexing the recognized tree's
operand edges and shrinking its axis — never by lowering to Loop IR and handing the result back to
recognition. A round-trip loses every structure recognition already derived: on the fused gate⊗up
cone it dropped the computed-A binding and the partial enumerated **11 rows, 0 of them mma**,
against the unsplit kernel's 3637 rows / 3626 mma.

Staying in Tile IR is also what makes a re-split gate unnecessary: ``010_recognize`` matches
``LoopOp``, so a piece is never handed back to routing and cannot cascade.

This is the same shape as ``_cut``: a routing decision resolved BEFORE any schedule fork exists,
offered as a structural fork when no pin decides, and recorded on the piece that owns the output as
the exact pin that replays it.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from emmy import config
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Lambda, Load, Write
from emmy.compiler.ir.stmt.algebra import M
from emmy.compiler.ir.stmt.passes import projection_distributes
from emmy.compiler.ir.tile import Fold, Placement, Store, TileOp
from emmy.compiler.ir.tile.ir import effect_tail
from emmy.compiler.ir.tile.path import Site, family_sites, resolve, sites, spell
from emmy.compiler.pipeline.knob import family_of, parse_knob_spec
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.pipeline import RuleSkipped

logger = logging.getLogger(__name__)

#: The partition axis introduced on the partial — a free axis of the partial kernel, indexing the
#: workspace's leading dimension.
_SPLIT_AXIS = "_ksplit"

#: The partition widths offered when nothing pins. Divisor legality (the width must divide a static
#: extent) is checked per site; everything finer is the piece's OWN schedule question, which is what
#: the routing hoist buys — the pre-split kernel no longer has to know the partial's warp K-step to
#: decide whether a width is legal.
WIDTHS: tuple[int, ...] = (2, 4, 8)

#: The finalize arms, by codec letter.
ARMS: tuple[str, ...] = ("k", "a")


def _split_pins() -> dict[str, str]:
    """The live ``SPLIT`` pins — authoritative over routing entries, same convention as
    ``_cut._place_pins``: ``SPLIT@…`` keys ride the ``EMMY_KNOBS`` aggregate (an ``@`` key is not a
    shell-variable name), a bare ``EMMY_SPLIT`` rides its own var. An EMPTY spelling is "no split",
    the decided-empty reading every codec knob gives ``""``."""
    pins = {k: v for k, v in parse_knob_spec(config.knobs_aggregate()).items() if family_of(k) == "SPLIT" and str(v).strip()}
    bare = config.knob_raw("SPLIT")
    if bare is not None and bare.strip() and "SPLIT" not in pins:
        pins["SPLIT"] = bare
    return pins


def parse_move(value: str) -> tuple[int, str]:
    """Decode a ``g<w>[a|k]`` codec value into ``(width, arm)``. Raises on anything else — a
    misspelled pin is a loud error, never a silent fallback to the unsplit form."""
    s = str(value).strip()
    if not s.startswith("g") or len(s) < 3 or s[-1] not in ARMS or not s[1:-1].isdigit():
        raise ValueError(f"SPLIT {value!r}: expected g<width>[a|k] (e.g. g8k)")
    return int(s[1:-1]), s[-1]


def spell_move(width: int, arm: str) -> str:
    return f"g{width}{arm}"


def splittable_sites(root, free: tuple = ()) -> tuple[Site, ...]:
    """Every reducing fold in ``root``'s tree whose axis a split may partition — shallowest first,
    so a bare ``SPLIT`` pin resolves to the primary reduce the way every other bare family key
    does."""
    del free  # the split partitions a reduce axis; the placement's free axes never gate it
    return tuple(sorted((s for s in family_sites("SPLIT", sites(root)) if s.axis != _SPLIT_AXIS), key=lambda s: s.depth))


def legal_moves(site: Site, tile_op=None, stores: tuple = ()) -> list[tuple[int, str]]:
    """The ``(width, arm)`` pairs legal on ``site``: a width dividing the static extent, and the
    atomic arm only where the carrier is single-component and the projection epilogue distributes
    over the add (else each partition's contribution is mis-scaled).

    Everything the pre-hoist enumeration also gated on — the warp K-step of the partial's tile, a
    computed-B cone's schedule slice, a producer band — is gone: those were consequences of
    deciding the split AFTER the schedule, and the partial now answers them at its own fork."""
    node = site.node
    extent = node.axis.extent
    if not extent.is_static:
        return []
    k = extent.as_static()
    alg = Reduction(node)
    tail = _projection(tile_op, stores) if tile_op is not None else ()
    atomic_ok = len(alg.names) == 1 and (not tail or projection_distributes(tail, alg.names))
    moves: list[tuple[int, str]] = []
    for w in WIDTHS:
        if k % w:
            continue
        moves.append((w, "k"))
        if atomic_ok:
            moves.append((w, "a"))
    return moves


def route_split(root, free: tuple = (), stores: tuple = ()) -> tuple[Site, int, str] | None:
    """The ``SPLIT`` pin resolution for a freshly-recognized kernel: the pinned ``(site, width,
    arm)``, or ``None`` when no pin decides (the structural FORK owns the choice). A key naming no
    reducing fold on THIS tree is skipped — one whole-model pin sweeps every kernel, and the shapes
    it does not name are simply not the shapes it targets. A key that DOES name a site here is
    handed on, so an illegal width or arm raises in the realizer rather than silently deploying the
    unsplit form."""
    del stores
    pins = _split_pins()
    if not pins:
        return None
    candidates = splittable_sites(root, free)
    if not candidates:
        return None
    all_sites = sites(root)
    for key, value in pins.items():
        width, arm = parse_move(value)
        if key == "SPLIT":
            return candidates[0], width, arm
        try:
            site = resolve(root, key, all_sites=all_sites)
        except ValueError:
            continue
        if site is not None and site in candidates:
            return site, width, arm
    return None


def _projection(tree: Fold, stores: tuple) -> tuple:
    """The kernel's projection epilogue — the root zero-axis fold's body with the kernel-boundary
    stores reconstituted. Same reading as ``ops.projection_tail``, off the recognized TREE (which
    is what routing holds; the ``TileOp`` does not exist yet at this point)."""
    body = list(tree.body) if isinstance(tree, Fold) and tree.axis is None else []
    return tuple(effect_tail(body, stores))


def _sliced_edge(edge, sigma: Sigma):
    """An operand edge σ-reindexed to absolute k for one partition — the SAME rule on either edge.
    A MATERIALIZED edge rewrites its gmem index; a COMPUTED cone rewrites its per-cell BODY only —
    the REDUNDANT-STATISTIC split: the cone's row-invariant prologue (the per-row statistic) spans
    the whole row and stays FULL-ROW in every partition, each recomputing it. That redundancy is
    what the split trades for parallelism; whether it pays on a shape is evidence's decision."""
    if isinstance(edge, Load):
        return replace(edge, index=tuple(sigma.apply(e) for e in edge.index))
    return edge.with_bodies((Body(tuple(s.rewrite(lambda nm: nm, sigma) for s in edge.body)),))


def _sliced(node: Fold, b: int) -> Fold:
    """``node`` restricted to one partition: the reduce axis shrunk to ``B`` and every read of it
    offset by ``_ksplit·B``, so the fold walks ``[0, B)`` while reading ``[s·B, (s+1)·B)``.

    ONE rule for every reducing fold. A contraction's reads sit on its operand edges and a plain
    reduce's inside its lift, so both are rewritten — no dispatch on the node's role, and the
    algebra (``init`` / ``combine``) rides through untouched because a partition folds the same
    monoid over fewer elements."""
    ident = lambda n: n  # noqa: E731
    sigma = Sigma({node.axis.name: BinaryExpr("+", Var(node.axis.name), BinaryExpr("*", Var(_SPLIT_AXIS), Literal(b, "int")))})
    return replace(
        node,
        axis=replace(node.axis, extent=Dim(b)),
        operands=tuple(_sliced_edge(e, sigma) for e in node.operands),
        lift=replace(node.lift, body=Body(tuple(s.rewrite(ident, sigma) for s in node.lift.body))),
    )


def _folded(node: Fold, alg: Reduction, ws: str, index, states: tuple[str, ...], split_axis: Axis) -> Fold:
    """The cross-partition fold: a FRESH fold over the workspace, carrying the same monoid the
    original fold carried.

    Built, not mutated. Deriving it by ``replace``-ing the original node keeps a contraction's
    bilinear structure, so the finalize reads back as a matmul over the workspace and computes
    garbage (92.8% of elements wrong). What carries over is only the ALGEBRA: a degenerate carrier
    rebuilds its componentwise ⊕ through the one monoid constructor, and a twisted carrier reuses
    its stored exp-family merge verbatim — which is why the twisted case needs no special arm."""
    edges = tuple(Load(name=states[i], input=ws, index=index(i)) for i in range(len(states)))
    lift = Lambda(params=(split_axis.name, *states), body=Body(()), results=states)
    if alg.ops is not None:
        init, combine = M(*alg.ops, names=states)
    else:
        init, combine = node.init, node.combine
    return Fold(axis=split_axis, operands=edges, lift=lift, init=init, combine=combine)


def _piece(op: Fold, free: tuple, stores: tuple = ()) -> TileOp:
    """One piece as an UNMAPPED ``TileOp``: free axes, no grid. That is what ``020_schedule`` picks
    up, so the piece resolves its OWN schedule fork. (The pre-hoist realizer built a *mapped* op
    here, which is exactly why the old partial had no fork and carried its parent's row.)"""
    return TileOp(op=op, place=Placement(free=tuple(free)), stores=tuple(stores))


def _stamp(piece: TileOp, graph) -> None:
    """Re-stamp a piece's ``S_*`` structural identity from its own body — the one stamper
    (``structure_features``), fed the piece's Tile IR lowered for the read alone."""
    from emmy.compiler.pipeline.passes.loop.stamp._stamp import structure_features  # noqa: PLC0415

    body = Body(tuple(effect_tail(piece.op.lower(), piece.stores)))
    piece.knobs = {k: v for k, v in (piece.knobs or {}).items() if not str(k).startswith("S_")}
    piece.knobs.update(structure_features(body, graph))


def _inputs(piece: TileOp, root: Node) -> list[str]:
    """The graph inputs ``piece`` reads, in the root's order — the finalize included: its
    projection epilogue can read plain graph inputs of its own (``mean``'s divisor, SDPA's scale)."""
    reads = {ld.input for ld in Body(tuple(effect_tail(piece.op.lower(), piece.stores))).loads}
    return [i for i in root.inputs if i in reads]


def realize_split(match, root: Node, tile_op, free: tuple, stores: tuple, site: Site, width: int, arm: str) -> Graph:
    """Partition ``site``'s reduce axis ``width`` ways, returning the fragment of new kernels."""
    out = root.output
    node = site.node
    extent = node.axis.extent.as_static()
    if extent % width:
        raise NotImplementedError(f"split width {width} does not divide {node.axis.name}={extent}")
    ws = f"{out.name}__partial"
    if ws in match.graph.nodes:
        raise RuleSkipped(f"already split — {ws} exists")

    alg = Reduction(node)
    states = alg.names
    n_comp = len(states)
    split_axis = Axis(name=_SPLIT_AXIS, extent=Dim(width))
    projection = _projection(tile_op, stores)
    spelled = spell(tile_op, "SPLIT", node, all_sites=sites(tile_op))
    logger.info("split: %s = %s on %s", spelled, spell_move(width, arm), root.id)

    sliced = _sliced(node, extent // width)
    cell = tuple(Var(a.name) for a in free)
    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)

    # --- the ATOMIC arm: ONE kernel, each partition accumulating into the zero-init'd output ----
    if arm == "a":
        if n_comp != 1:
            raise NotImplementedError("atomic finalize needs an additive (1-component) carrier; pin the g<w>k arm")
        if projection and not projection_distributes(projection, states):
            raise NotImplementedError(
                "atomic finalize cannot carry a non-distributive projection epilogue (e.g. l2's sqrt); "
                "pin the deferred-kernel finalize instead (SPLIT=g<w>k)"
            )
        epilogue = (
            tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in projection)
            if projection
            else (Write(output=out.name, index=cell, value=states[0], atomic=True),)
        )
        piece = _piece(Fold.projection(body=Body(epilogue), operands=(sliced,)), (split_axis, *free))
        frag.add_node(op=piece, inputs=_inputs(piece, root), output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
        frag.outputs = [out.name]
        _stamp(piece, frag)
        piece.knobs = {**(piece.knobs or {}), spelled: spell_move(width, arm)}
        return frag

    # --- the KERNEL arm: partial → f32 workspace, finalize folds it over the partition axis -----
    # The workspace is **f32**: it holds raw pre-projection carrier state, which must not round-trip
    # through the output dtype. It is sized by the FREE extents — a rank mismatch against the index
    # would flatten without strides and collide the partitions.
    ws_shape = (Dim(width), *(a.extent for a in free)) if n_comp == 1 else (Dim(n_comp), Dim(width), *(a.extent for a in free))

    def ws_index(i: int) -> tuple:
        lead = (Var(_SPLIT_AXIS),) if n_comp == 1 else (Literal(i, "int"), Var(_SPLIT_AXIS))
        return (*lead, *cell)

    accs = tuple(sliced.defines())
    partial = _piece(
        Fold.projection(operands=(sliced,)),
        (split_axis, *free),
        tuple(Store(write=Write(output=ws, index=ws_index(i), value=accs[i])) for i in range(n_comp)),
    )
    folded = _folded(node, alg, ws, ws_index, states, split_axis)
    finalize = _piece(Fold.projection(body=Body(projection), operands=(folded,)) if projection else folded, free)

    frag.add_node(op=partial, inputs=_inputs(partial, root), output=Tensor(ws, ws_shape, F32), node_id=ws)
    frag.add_node(op=finalize, inputs=[ws, *_inputs(finalize, root)], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    for nid in (ws, out.name):
        _stamp(frag.nodes[nid].op, frag)
    # The decision rides the piece that owns the output, spelled exactly as the pin that replays it
    # — the same no-side-channel rule ``_cut`` follows.
    finalize.knobs = {**(finalize.knobs or {}), spelled: spell_move(width, arm)}
    return frag


__all__ = ["ARMS", "WIDTHS", "legal_moves", "parse_move", "realize_split", "route_split", "spell_move", "splittable_sites"]
