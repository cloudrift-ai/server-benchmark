"""Materialize a stored Fold edge as a kernel boundary.

The cut is structural: the child Fold keeps its algebra and becomes its own kernel, the parent
reads what it produced through ordinary ``Load`` edges. Both pieces are fresh unmapped ``TileOp``
objects. Pinned cuts consume placement on both pieces before the cut pass continues with cross-CTA
reduction splitting; unpinned cuts may expose smaller seams.

A seam has three realizations, and the seam itself decides which — one site stays ONE decision,
because at each seam one of them dominates the others outright and there is no trade for the
evidence to weigh:

- the WORKSPACE cut, the general case: the piece writes every state component to a fresh buffer.
- the STORAGE-FRONTIER cut (contraction-operand seams whose cone passes through a decode, see
  :func:`storage_frontier`): the buffer holds the raw storage bits, exact and narrower than the
  re-rounded result, and the consumer keeps the decode-plus-factors residue.
- the OUTPUT-OWNING cut (:func:`_output_owners`): where the seam's cone solely produces some of the
  kernel's OWN outputs, the piece writes those outputs and the sibling piece keeps the rest. The
  workspace would have held the output's exact bytes at the output's exact dtype, leaving the
  sibling nothing to do for them but copy, so this deletes a buffer and a copy. It also leaves both
  pieces single-output, which is what lets the shared-sweep promotion
  (:func:`~emmy.compiler.ir.tile.ir.promoted_sweep`) bind a sweep the fused kernel had to serialize.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import (
    Fold,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.schedule.packing import match_packed_pair_node
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ir import promoted_sweep
from emmy.compiler.ir.tile.ops import UnbindableProjection, carries_partition, edge_dtypes, output_regions
from emmy.compiler.ir.tile.path import family_sites, sites, spell
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.lowering.tile._split import add_output_piece, output_root
from emmy.compiler.structural import digest
from emmy.compiler.tensor import Tensor


@dataclass(frozen=True)
class CutSite:
    """All stored occurrences of one canonically shared child Fold. ``dtypes`` is the workspace's
    per-component materialization, decided at offer time so the realization stores exactly what
    was offered. ``frontier`` (contraction-operand seams only) moves the cut to the cone's storage
    waypoint — see :class:`Frontier`."""

    node: Fold
    spelling: str
    axes: tuple
    dtypes: tuple
    frontier: Frontier | None = None
    #: Duplicate cones this seam ALSO stands for — alpha-equivalent up to their captured axis
    #: names (contraction-operand seams only): each sibling is
    #: ``(node, ((rep axis name, sibling axis name), …))`` — the positional capture correspondence
    #: the clustering proved. One placement decision materializes the value once; the realization
    #: replaces every sibling with workspace loads spelled through its own axes. Object sharing is
    #: the degenerate case (identity, with the identity correspondence).
    siblings: tuple = ()
    #: ``(tail, stores)`` when this seam's cone solely produces some of the kernel's OWN output
    #: specifications — those stores and the projection statements only they read — else ``None``.
    #: A seam with ``owned`` realizes as the output-owning cut (module docstring) and writes no
    #: workspace, so its ``dtypes`` are empty.
    owned: tuple | None = None


@dataclass(frozen=True)
class Frontier:
    """A contraction-operand cone's STORAGE waypoint: a decode (the ``ElementwiseImpl.decodes``
    trait) of a value the cone itself computes. The seam materializes there instead of at the
    cone's result — the workspace holds the raw storage bits (exact, the element the graph's own
    quantize produced), the producer piece computes ``producer`` (the encode prefix), and the
    consumer keeps ``residue`` (the decode plus the factor chain), which the normalize-time
    decode hoist then absorbs into a raw storage-dtype load with the factors on the accumulator
    epilogue — the same ``sum_k a*(s*w) = s*sum_k a*w`` reassociation as the materialized case."""

    name: str  # the encoded value the workspace holds
    producer: tuple  # the prefix stmts computing ``name`` (spliceable operand bodies inlined)
    residue: tuple  # the decode + factor stmts the consumer keeps
    dtype: object  # the storage DataType the decode op names


def _spliceable(edge) -> tuple | None:
    """A zero-axis operand's flat stmt list, or ``None`` when it cannot splice inline (an
    iterating fold, nested operands, or non-scalar members)."""
    if not isinstance(edge, Fold) or edge.axis is not None or edge.operands:
        return None
    members = tuple(edge.lift.body)
    return members if all(isinstance(stmt, (Load, Assign)) for stmt in members) else None


def storage_frontier(node: Fold) -> Frontier | None:
    """``node``'s storage frontier, or ``None`` when it has none the cut can separate.

    The shape is semantic, not an op list: exactly one decode of a value DEFINED by the cone's own
    body (a decode of a materialized load was already absorbed by normalization), whose backward
    cone separates cleanly — only the decode reads a prefix-computed name, so the residue's value
    is a pure function of the stored bits and its own leaves. Every operand must splice inline
    (each side takes the operand bodies it reads), keeping both pieces free of nested edges."""
    if not isinstance(node, Fold) or node.axis is not None or len(node.lift.results) != 1:
        return None
    lift = node.applied  # the operands' spelling: what the spliced sides read
    body = lift.body
    if any(not isinstance(stmt, (Load, Assign)) for stmt in body):
        return None
    computed = {name for stmt in body if isinstance(stmt, Assign) for name in stmt.defines()}
    decodes = [
        stmt
        for stmt in body
        if isinstance(stmt, Assign) and stmt.op.decodes is not None and len(stmt.args) == 1 and stmt.args[0] in computed
    ]
    if len(decodes) != 1:
        return None
    decode = decodes[0]
    frontier = decode.args[0]
    spliced = [_spliceable(edge) for edge in node.operands]
    if any(members is None for members in spliced):
        return None
    prefix = tuple(body.backward_cone((frontier,)).members)
    prefix_ids = {id(stmt) for stmt in prefix}
    prefix_defs = {name for stmt in prefix for name in stmt.defines()}
    residue = tuple(stmt for stmt in body if id(stmt) not in prefix_ids)
    for stmt in residue:
        crossing = Body((stmt,)).ssa_uses & prefix_defs
        if crossing and (stmt is not decode or crossing != {frontier}):
            return None  # a residue stmt reads past the frontier — the waypoint does not separate
    if lift.results[0] in prefix_defs:
        return None
    result = get_dtype(decode.op.decodes)

    def side(stmts: tuple) -> tuple:
        reads = Body(stmts).ssa_uses
        inlined: list = []
        for edge, members in zip(node.operands, spliced, strict=True):
            needed = set(edge.exposes) & reads
            if needed:  # only the cone the side reads — a dead spliced def would decline the decode hoist
                inlined.extend(Body(members).backward_cone(tuple(sorted(needed))).members)
        return (*inlined, *stmts)

    return Frontier(name=frontier, producer=side(prefix), residue=side(residue), dtype=result)


def _external_reads(node: Fold) -> frozenset[str]:
    """Everything ``node`` needs supplied from outside — read off its DECLARATION.

    Was: lower the term and walk the result for free names. That asked a term to re-derive what it
    already states, re-lowered on every call, and returned a superset (names the term binds
    internally, which every caller here discards). :attr:`Fold.free_axes` is the declaration —
    the term's own axes unioned with its operands', asked of the term rather than derived here. A
    term is closed: its values arrive through its operand edges, so its coordinates are all it
    takes from outside."""
    return node.free_axes


def _closed_at(node: Fold, axes: tuple) -> bool:
    """Whether ``node`` has no capture other than the axes (by name) in scope at its incoming edge."""
    return _external_reads(node) <= set(axes)


def _fed_store_dtype(tile: TileOp, consumer: Fold):
    """The dtype ``consumer`` stores its result at: the output its accumulators transitively feed
    (a forward closure over the root's lowered stmts covers any epilogue between the two), or
    ``None`` when the fed dtypes are not a singleton. A multi-output kernel can store siblings at
    other dtypes (w8a8's fp8 encode beside the f16 linear), so only the contraction's own stores
    speak for its slabs — and when it feeds outputs at SEVERAL dtypes no one of them does, so the
    seam stays undetermined and unoffered rather than resolved by list order."""
    if not tile.output_specs:  # the default store: the root's result to the kernel's one output
        tensor = next(iter(tile.outputs.values()), None)
        return None if tensor is None else tensor.dtype
    dependent = set(consumer.exposes)
    stmts = tile.op.lower(axes=tile.axes)
    for _ in stmts:
        grown = False
        for stmt in stmts:
            defines = Body((stmt,)).ssa_defs
            if not defines <= dependent and Body((stmt,)).ssa_uses & dependent:
                dependent |= defines
                grown = True
        if not grown:
            break
    fed = {
        tensor.dtype
        for store in tile.output_specs
        if store.write.value in dependent
        if (tensor := tile.outputs.get(store.write.output)) is not None
    }
    return fed.pop() if len(fed) == 1 else None


def _workspace_dtypes(node: Fold, tile: TileOp, consumer: Fold | None, table: dict[int, tuple]) -> tuple | None:
    """The cut workspace's per-component dtypes, or ``None`` when they cannot be determined.
    Reduction carrier precision is a Kernel IR policy — every Fold state is f32 until lowering
    stamps the concrete Accum/Init pair; a zero-axis value has no carrier and is inferred from its
    typed pure program instead. A seam standing in for a contraction OPERAND (``consumer`` is the
    consuming contraction) is the exception: it materializes explicitly at the dtype that
    contraction's output is stored at — the element the fused slab would have stored — never the
    carrier its cone computed in (only the ``a`` edge has a converting fill, so an f32 workspace on
    a ``b`` edge could feed no warp atom). A seam whose dtypes stay undetermined is not offered:
    the offer and the realization must agree, and a raise past the offer would kill the compile."""
    names = node.exposes
    if consumer is not None:
        dtype = _fed_store_dtype(tile, consumer)
        return None if dtype is None else (dtype,) * len(names)
    dtypes = (F32,) * len(names) if node.axis is not None else table.get(id(node), ())
    if len(dtypes) != len(names) or any(dtype is None for dtype in dtypes):
        return None
    return dtypes


def _dtype_table(tile: TileOp) -> dict[int, tuple]:
    """Each stored edge's inferred result dtypes, keyed by edge identity.

    One edge, one answer: a term is CLOSED, so its values arrive through its own operand edges and
    it types the same wherever it occurs. The occurrence-agreement filter this used to apply — one
    shared cone under two scopes having two answers — cannot arise once no term captures."""
    cache: dict[int, tuple] = {}
    edge_dtypes(tile.op, tile.inputs, cache)
    return dict(cache)


def _output_owners(tile: TileOp) -> dict[int, tuple]:
    """The root operands that solely produce some of this kernel's outputs AND would bind a grid
    axis the fused kernel cannot — keyed by operand identity, each mapped to ``(tail, stores)``.

    Two conditions, both structural, both asked of rules that already exist.

    OWNERSHIP is :func:`~emmy.compiler.ir.tile.ops.output_regions`: every output specification must
    read exactly one operand, the operands' cones over the root body must be disjoint, and together
    they must cover it. Without it the pieces would not be a partition of the kernel — one of them
    would have to recompute what it no longer owns.

    RANK is :func:`~emmy.compiler.ir.tile.ir.promoted_sweep`, asked twice: once of the fused kernel,
    once of the candidate piece. A piece promotes a sweep the whole kernel could not exactly when
    its stores agree on an axis the sibling's stores do not, so the intersection over ALL stores
    came up empty — the NVFP4 encode, whose packed codes ride the feature axis and whose block
    scales ride one sixteenth of it. Where the piece promotes nothing more than the kernel already
    does, splitting buys a second launch and no grid, so the seam keeps its workspace reading.
    """
    op = tile.op
    if len(tile.output_specs) < 2 or not isinstance(op, Fold) or op.axis is not None or len(op.operands) < 2:
        return {}
    try:
        regions = output_regions(op, tile.output_specs)
    except UnbindableProjection:
        return {}
    fused = promoted_sweep(op, tile.output_specs)
    return {id(region): (tail, stores) for region, tail, stores in regions if stores and not promoted_sweep(region, stores) <= fused}


def cuttable_seams(tile: TileOp) -> tuple[CutSite, ...]:
    """Every semantically closed stored Fold edge a cut can hand its own kernel, grouped only by
    object sharing. A contraction's operand edges are seams too — cutting one materializes the cone
    feeding the operand into its own kernel and the contraction reads it back as an ordinary load —
    and they take the explicit contraction-operand dtype rule (`_workspace_dtypes`), except on a
    block-scaled packed pair, whose operand cones are not seams at all. A seam is offered only where
    the cone is closed at the axes of every occurrence; a term is closed by construction, so that
    check names a malformed tree rather than a capture to resolve.

    Workspace dtypes must be determined for the seam to be offered — the offer and the realization
    must agree, and a raise past the offer would kill the compile. An OUTPUT-OWNING seam
    (:func:`_output_owners`) writes no workspace, so it carries none and that condition does not
    apply to it."""
    all_sites = sites(tile.op)
    owners = _output_owners(tile)
    store_dtype_consumers = {
        id(edge): site.node
        for site in all_sites
        if site.node.as_contraction() is not None
        for edge in site.node.operands
        if isinstance(edge, Fold) and edge.as_slab() is None
    }
    outer = tuple(axis.name for axis in (*tile.place.free, *(axis for store in tile.output_specs for axis in store.sweep)))
    occurrence_axes: dict[int, list[tuple]] = {}
    for site in all_sites[1:]:
        occurrence_axes.setdefault(id(site.node), []).append((*outer, *site.scope))
    dtype_table: dict[int, tuple] = {}
    if isinstance(tile.op, Fold):
        dtype_table = _dtype_table(tile)
    out: list[CutSite] = []
    seen: set[int] = set()
    for site in family_sites("PLACE", all_sites):
        node = site.node
        scopes = occurrence_axes.get(id(node), ())
        if not isinstance(node, Fold) or node.as_slab() is not None or id(node) in seen or not scopes:
            continue
        if not all(_closed_at(node, scope) for scope in scopes):
            continue
        if node.observe is not None:
            # An observed fold's per-step results exist only inside its stream — a cut would
            # separate the scan from its streamed boundary store, which no piece can then spell.
            continue
        consumer = store_dtype_consumers.get(id(node))
        if consumer is not None and match_packed_pair_node(consumer, tile.inputs) is not None:
            # An operand cone of a BLOCK-SCALED packed pair reaches gmem already: its codes and
            # its block scale are loads, and the cone only decodes them. Materializing it stores
            # the decoded values instead, so the consumer holds neither the codes nor the
            # per-block scale the cell multiplies — the reading is gone for every occurrence the
            # seam covers, and no piece can put it back. Nothing is hoisted either way, so this
            # is not a placement trade. The contraction's OWN seam stays offered, and that is the
            # cut that gives the piece the output-axis pair a fragment needs.
            continue
        # A frontier REPLACES the fed-store realization at this seam rather than joining the
        # offer: the raw bits dominate the fed-store workspace on both precision (exact vs
        # re-rounded) and footprint (storage width vs store width), so there is no trade for the
        # evidence to decide — one site stays one decision.
        owned = owners.get(id(node))
        frontier = storage_frontier(node) if consumer is not None and owned is None else None
        if owned is not None:
            dtypes = ()  # the piece writes the kernel's own outputs; there is no workspace to type
        else:
            dtypes = (frontier.dtype,) if frontier is not None else _workspace_dtypes(node, tile, consumer, dtype_table)
            if dtypes is None:
                continue
        # A seam is evaluated over the coordinates its term READS, not the whole ambient scope: an
        # output sweep the grid carries for a sibling's sake is not one of this workspace's axes.
        axes = tuple(tile.axis_of(name) for name in dict.fromkeys(name for scope in scopes for name in scope) if name in node.free_axes)
        if any(axis.window is not None and axis.window.block for axis in axes):
            # A BLOCK is a working set INSIDE one kernel — the pivot's pass reaches its end and
            # the channels read what it left behind. A seam evaluated over a block coordinate
            # would write that working set to gmem once per block, which is the cost blocking
            # exists to avoid, and leaves a piece whose whole output is one block.
            continue
        seen.add(id(node))
        out.append(
            CutSite(
                node=node,
                spelling=spell(tile.op, "PLACE", node, all_sites=all_sites),
                axes=axes,
                dtypes=dtypes,
                frontier=frontier,
                owned=owned,
            )
        )
    return _cluster_value_seams(
        out,
        {id(seam.node): seam.frontier is None and seam.owned is None and store_dtype_consumers.get(id(seam.node)) for seam in out},
    )


def _cluster_value_seams(seams: list[CutSite], operand_of: dict[int, object]) -> tuple[CutSite, ...]:
    """Fold duplicate operand cones — alpha-equivalent up to captured axis names — into ONE seam per value.

    Object sharing groups occurrences of one stored node; a traced graph can also hold several
    ALPHA-EQUIVALENT copies of the same computation captured under different axis names —
    attention's normalized K cone appears once per score contraction. Those copies are one VALUE:
    the cluster's first seam becomes the decision for all of them, carrying each duplicate as a
    sibling with its positional capture correspondence (:class:`CutSite`). Membership reuses the
    closure alpha-equivalence the semiring canonicalization already trusts
    (:meth:`~emmy.compiler.ir.pure.fold.Fold.canonical`); a member joins only when its
    paired axes agree on extent and window, its workspace dtypes match, and every workspace axis
    is a mapped capture — otherwise it stays its own seam."""
    eligible = [index for index, seam in enumerate(seams) if operand_of.get(id(seam.node))]
    if len(eligible) < 2:
        return tuple(seams)
    captured = {index: tuple(axis.name for axis in seams[index].axes) for index in eligible}
    # The seam's own capture correspondence, and the alpha-quotient taken under it: an operand cone
    # is a TERM, so it quotients as one (``Fold.canonical``) rather than as a scoped lambda.
    scoped = {index: tuple(axis for axis in captured[index] if axis in seams[index].node.free_axes) for index in eligible}
    clusters: dict[object, list[int]] = {}
    for index in eligible:
        clusters.setdefault(seams[index].node.canonical(), []).append(index)
    drop: set[int] = set()
    merged: dict[int, CutSite] = {}
    for rep_index, *rest in clusters.values():
        if not rest:
            continue
        rep = seams[rep_index]
        rep_params = scoped[rep_index]
        rep_axes = {axis.name: axis for axis in rep.axes}
        if {axis.name for axis in _workspace_axes(rep, rep.node)} - set(rep_params):
            continue  # a workspace axis with no capture to map has no sibling spelling
        siblings = []
        for member_index in rest:
            member = seams[member_index]
            member_params = scoped[member_index]
            member_axes = {axis.name: axis for axis in member.axes}
            aligned = member.dtypes == rep.dtypes and all(
                (a := rep_axes[rn]).extent == (b := member_axes[mn]).extent and a.window == b.window
                for rn, mn in zip(rep_params, member_params, strict=True)
            )
            if not aligned:
                continue
            siblings.append((member.node, tuple(zip(rep_params, member_params, strict=True))))
            drop.add(member_index)
        if siblings:
            merged[rep_index] = replace(rep, siblings=tuple(siblings))
    return tuple(merged.get(index, seam) for index, seam in enumerate(seams) if index not in drop)


def _unchanged(pieces: tuple, members) -> bool:
    return len(pieces) == len(members) and all(piece is member for piece, member in zip(pieces, members, strict=True))


def _replace_member(member, targets: dict[int, tuple]):
    if id(member) in targets:
        return targets[id(member)]
    if isinstance(member, Fold):
        return (_replace_fold(member, targets),)
    nested = member.nested()
    if not nested:
        return (member,)
    bodies = []
    changed = False
    for body in nested:
        replaced = tuple(piece for child in body for piece in _replace_member(child, targets))
        changed = changed or not _unchanged(replaced, body)
        bodies.append(Body(replaced))
    return (member.with_bodies(tuple(bodies)) if changed else member,)


def _replace_fold(node: Fold, targets: dict[int, tuple]) -> Fold:
    """Replace every stored occurrence of the target Folds in ONE walk — ``targets`` maps
    ``id(node)`` to its replacement stmts. One walk, because the rebuild copies every node on the
    way down: a second walk's target objects no longer exist in the first walk's output, so
    sequential replacement silently loses every decision after the first. IDENTITY-PRESERVING off
    the replacement spine: a subtree holding no target returns the SAME object, so untouched
    Lambdas are not reconstructed (construction normalization over a large fused body is where a
    copying walk turns quadratic) and shared-node grouping keeps its identities."""
    operands = tuple(piece for edge in node.operands for piece in _replace_member(edge, targets))
    body = tuple(piece for stmt in node.lift.body for piece in _replace_member(stmt, targets))
    if _unchanged(operands, node.operands) and _unchanged(body, node.lift.body):
        return node
    return replace(node, operands=operands, lift=replace(node.lift, body=Body(body)))


def _workspace_axes(seam: CutSite, produced: Fold) -> tuple:
    """The seam axes the PRODUCED piece actually sweeps — its workspace dimensions. ``produced``
    is the seam node, or the frontier prefix when the seam materializes at a storage waypoint.

    Static unit axes consume no additional storage, but retain the producer's schedule geometry.
    Dropping one lets a later split axis take its place as a contraction fragment axis even though
    operand indices still read it as the outer partition coordinate."""
    read = _external_reads(produced)
    return tuple(axis for axis in seam.axes if axis.name in read or (axis.extent.is_static and axis.extent.as_static() == 1))


def _buffer_reads(node: Fold) -> set[str]:
    """The gmem buffers ``node``'s STORED tree reads — every lift body's loads, through the operand
    edges (a slab's body is its one load). Read off the tree, never by lowering it."""
    out = {load.input for load in node.lift.body.loads}
    for edge in node.operands:
        out |= _buffer_reads(edge)
    return out


def _piece_inputs(root: Node, fold: Fold, first: tuple[str, ...] = ()) -> list[str]:
    """A piece's graph inputs: its workspaces, then every buffer of ``root``'s the piece reads.

    The read set is the STORED tree's (:func:`_buffer_reads`), walked through the operand edges a
    body cannot reach. A walk that stopped at the stored body named fewer inputs than the kernel
    went on to read; the workspace producers then had no consumer edge, were pruned as orphans,
    and the launch asked for a buffer nothing had allocated."""
    reads = _buffer_reads(fold)
    return [*first, *(name for name in root.inputs if name in reads)]


def _input_fragment(match: Match, root: Node) -> Graph:
    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(name), node_id=name)
    return fragment


#: The temporary every piece of a placement cut travels under while the fragment is spliced in.
#: One suffix per rewrite (``_split`` mints ``__split``), so a fragment's names say which decision
#: minted them.
_PLACED = "__placed"


def output_map(root: Node) -> dict[str, str]:
    """Stable temporary output names used by every cut sibling of ``root``."""
    return {name: f"{name}{_PLACED}" for name in root.buffer_names()}


def _in_source_order(stores: tuple, order: list[str]) -> tuple:
    """``stores`` in the kernel's own output-specification order. The ownership partition groups by
    region; ``apply_output_specs`` reads consecutive same-path stores as one sweep nest, so the
    pieces must keep the order the kernel spelled rather than the order the partition returned."""
    return tuple(sorted(stores, key=lambda store: order.index(store.write.output)))


def _region_term(regions: tuple, body, results: tuple) -> Fold:
    """A zero-axis term over ``regions`` and the projection statements they own, exposing
    ``results`` — the values the stores that stay with it are written from. The results are named
    rather than derived from the body's last definition: a piece may own several stores, and one
    that owns a store read straight off a region has no body at all."""
    bound = tuple(name for region in regions for name in region.exposes)
    return Fold(operands=regions, lift=Lambda.closing(bound, Body.coerce(body), results))


def _region_piece(tile: TileOp, regions: tuple, tail, stores: tuple, placement_decided: bool, split_consumed: bool, spelling: str):
    """One output-owning piece: the regions' term, the outputs they produce, and the PARENT's free
    axes. The placement is deliberately the parent's and not the seam's own axes — the piece is a
    kernel writing the kernel's own outputs, so its grid is settled by the same shared-sweep
    promotion that settles any single-output kernel's, applied now that the sibling's stores are no
    longer in the intersection."""
    piece = TileOp(
        op=_region_term(regions, tail, tuple(dict.fromkeys(value for store in stores for value in store.write.values))),
        # The seam token keeps recursive pieces' kernel names distinct, as it does for a workspace
        # producer — two same-named pieces from different cut levels would launch one kernel twice.
        name=f"{tile.name}__place_{digest(tile.identity_key(structural=False) or '', spelling)[:10]}",
        place=Placement(free=tuple(tile.place.free)),
        axes=tile.axes,
        output_specs=stores,
        placement_decided=placement_decided,
        split_consumed=split_consumed,
    )
    return replace(piece, knobs=consume_kernel_row(piece.knobs))


def _read_name(name: str, token: str, ordinal: int | None = None) -> str:
    """The SSA name a workspace read binds: the cone's own result name, tagged with the seam whose
    workspace it now comes from (and, for a clustered duplicate, its ordinal in the cluster).

    A lowered body reads PRODUCER names throughout — a consumer's params are spelled as the result
    names of the edge they bind (:attr:`~emmy.compiler.ir.pure.fold.Fold.applied`) — so an edge's
    result name is what the emitted kernel declares, and the cone's own name is NOT unique among
    what one kernel binds. The value a cut materializes can still be computed in place beside the
    read: a structurally equal cone the replacement did not reach (replacement follows object
    sharing), or a second seam exposing that same value. Under one name those are two declarations
    at two different addresses, which is an SSA fault and which nvcc rejects (*already declared in
    the current scope*). Reads of ONE workspace at one address keep one name, so the emitted body
    still binds each value once.
    """
    return f"{name}__ws{token}" if ordinal is None else f"{name}__ws{token}s{ordinal}"


def _producer_order(pieces) -> list:
    """Topologically order cut producers by the workspaces their stored Fold reads.

    Containment can make one produced piece read another even when the corresponding seams have
    no provider requirement. Dependency COUNT is not an order: two pieces may each read one
    workspace while one of those workspaces is produced by the other piece.
    """
    workspaces = {buffer for *_, buffers in pieces for buffer in buffers}
    remaining = list(pieces)
    ordered = []
    available: set[str] = set()
    while remaining:
        ready = [piece for piece in remaining if _buffer_reads(piece[1]) & workspaces <= available]
        assert ready, "strict Fold containment makes the cut-workspace dependency graph acyclic"
        ordered.extend(ready)
        available.update(buffer for *_, buffers in ready for buffer in buffers)
        remaining = [piece for piece in remaining if not any(piece is chosen for chosen in ready)]
    return ordered


def realize(
    match: Match,
    root: Node,
    seams,
    *,
    placement_decided: bool = False,
) -> Graph:
    """Build the cut fragment for ``seams`` — one piece per seam plus the ONE sibling piece that
    reads what they produced. A single seam is the two-kernel cut; several seams are one COMPOSED
    placement decision (a pinned compile consumes every scoped PLACE pin that resolves on this
    kernel at once, so the pieces stay decided and the knob row records every spelling).

    A frontier seam cuts at the cone's storage waypoint: the piece computes the encode prefix, the
    workspace holds the raw bits, and the sibling keeps the decode + factor residue as its operand
    cone (which normalization then binds as a raw storage-dtype load with the factors hoisted onto
    the accumulator epilogue). An OUTPUT-OWNING seam writes the kernel's own outputs instead of a
    workspace, and the sibling keeps the rest; the two kinds compose, because a workspace cut nested
    under an output-owning region is applied to that region's term like any other consumer's.

    ``placement_decided`` consumes an authoritative pinned PLACE restriction on every piece.
    Unpinned cuts leave it false so fresh pieces can expose and decide smaller seams. A placement
    cut never erases an earlier cross-CTA decision: every piece inherits the parent's explicit or
    sliced-axis split receipt."""
    tile: TileOp = root.op
    split_consumed = tile.split_consumed or carries_partition(tile)
    owning = tuple(seam for seam in seams if seam.owned is not None)
    seams = tuple(seam for seam in seams if seam.owned is None)
    pieces = []
    # What each replaced cone's result is called once the consumer reads it back — the ONE
    # rename this pass mints, collected as it is minted. A term's readers follow it for free (a
    # consumer's params are spelled as the result names of the edge they bind), but the kernel's
    # boundary stores are NOT part of the term: ``TileOp.output_specs`` names the stored value as
    # a plain string, so a store of a cut cone's own result has to be re-spelled here or it names
    # a value the consumer no longer defines.
    read_names: dict[str, str] = {}
    for seam in seams:
        child = seam.node
        front = seam.frontier
        if front is not None:
            names = (front.name,)
            produced = Fold(operands=(), lift=Lambda.closing((), Body.coerce(Body(front.producer)), names))
        else:
            names = child.exposes
            produced = child
        axes = _workspace_axes(seam, produced)
        index = tuple(Var(axis.name) for axis in axes)
        token = digest(tile.identity_key(structural=False) or "", seam.spelling)[:10]
        buffers = tuple(f"{root.id}__place_{token}_{i}" for i in range(len(names)))

        # SLABS, not bare Loads: these replace an operand edge, and an operand is a term. The
        # workspace read declares the seam axes it indexes, exactly as any other gmem read does.
        loads = tuple(
            Fold.slab(Load(name=_read_name(name, token), input=buffer, index=index)) for name, buffer in zip(names, buffers, strict=True)
        )
        if front is not None:
            # The raw storage read at the frontier's dtype stays INLINE under its decode residue —
            # the storage-decode cone the operand readers recognize (a raw ``b8`` fill), not a
            # projection over a slab. The residue is a lambda over that read, so tagging what it
            # EXPOSES renames its defining statements in lockstep and leaves the read's own
            # internal spelling alone.
            raw = Load(name=names[0], input=buffers[0], index=index, dtype=front.dtype)
            residue = Lambda.closing((), Body.coerce(Body((raw, *front.residue))), child.lift.results)
            loads = (Fold(operands=(), lift=residue.rename({name: _read_name(name, token) for name in residue.results})),)
        # The names the consumer reads this workspace back under. A frontier seam's workspace holds
        # the raw storage waypoint, so its piece is named after the FRONTIER while the consumer
        # still exposes the cone's decoded results — the rename is over those.
        read_names.update({name: _read_name(name, token) for name in (names if front is None else child.lift.results)})
        replacements = {id(child): loads}
        for ordinal, (sibling, pairs) in enumerate(seam.siblings):
            # A clustered duplicate reads the SAME workspace, spelled through its own captured
            # axes via the correspondence the clustering proved — and under its own read names,
            # since it reads that workspace at a DIFFERENT address than the representative.
            mapping = dict(pairs)
            sibling_index = tuple(Var(mapping[axis.name]) for axis in axes)
            replacements[id(sibling)] = tuple(
                Fold.slab(Load(name=_read_name(name, token, ordinal), input=buffer, index=sibling_index))
                for name, buffer in zip(sibling.exposes, buffers, strict=True)
            )
            # The representative wins a shared name: a boundary store of a value both occurrences
            # expose reads the term's own, and only the representative sits on the term's path.
            for name in sibling.exposes:
                read_names.setdefault(name, _read_name(name, token, ordinal))
        pieces.append((seam, produced, axes, index, token, names, buffers, replacements))

    # Every replacement applies to the consumer AND to every OTHER seam's produced piece: a
    # composed decision may cut a cone nested inside another seam's value (attention's statistics
    # cone contains the score dots whose operand cones are cut beside it), and that producer must
    # read the workspace like any other consumer. Containment is strict, so order is free.
    everything = {target: loads for *_, replacements in pieces for target, loads in replacements.items()}
    parent_fold = _replace_fold(tile.op, everything)
    produced_pieces = []
    for seam, produced, axes, index, token, names, buffers, replacements in pieces:
        others = {target: loads for target, loads in everything.items() if target not in replacements}
        produced_pieces.append((seam, _replace_fold(produced, others) if others else produced, axes, index, token, names, buffers))

    fragment = _input_fragment(match, root)
    all_buffers = [buffer for *_, buffers in produced_pieces for buffer in buffers]
    # A producer reading another seam's workspace must follow the node that writes it. Strict
    # containment makes this dependency graph acyclic, including chains whose members have the
    # same number of direct workspace reads.
    for seam, produced, axes, index, token, names, buffers in _producer_order(produced_pieces):
        producer = TileOp(
            op=produced,
            # The seam token keeps recursive pieces' kernel names distinct — the one-name-one-source
            # launch rule stated beside ``nvcc.load_cubin_function``: two same-named producers from
            # different cut levels would launch one kernel twice.
            name=f"{tile.name}__place_{token}",
            place=Placement(free=axes),
            axes=tile.axes,
            output_specs=tuple(
                OutputSpec(Write(output=buffer, index=index, value=name)) for name, buffer in zip(names, buffers, strict=True)
            ),
            placement_decided=placement_decided,
            split_consumed=split_consumed,
        )
        producer = replace(producer, knobs=consume_kernel_row(producer.knobs))
        shape = tuple(axis.extent for axis in axes)
        workspace_tensors = tuple(Tensor(name=buffer, shape=shape, dtype=dtype) for buffer, dtype in zip(buffers, seam.dtypes, strict=True))
        reads = _buffer_reads(produced)
        fragment.add_node(
            op=producer,
            inputs=_piece_inputs(root, produced, tuple(buffer for buffer in all_buffers if buffer in reads)),
            outputs=workspace_tensors,
            node_id=buffers[0],
        )

    # A bare reduction carries NO output specification — its grid-cell store is materializer glue —
    # so the sibling's ports are read off the graph node, not off the stores.
    consumer_fold, consumer_stores = parent_fold, tile.output_specs
    consumer_outputs = set(root.buffer_names())
    if owning:
        chosen = {index: seam for index, edge in enumerate(tile.op.operands) for seam in owning if seam.node is edge}
        # The ownership partition, re-derived over the REPLACED root: a workspace cut composed with
        # these swapped a cone for a slab under one of the regions, and the piece has to carry that
        # swap. With no such seam the replacement is the identity and this is the same call the
        # offer made, so the two agree by construction; a composed one replaces operand edges and
        # leaves the root body alone, which is what the partition reads. Either way it is positional
        # over the operands, so the regions line up with the seams that chose them.
        regions = output_regions(parent_fold, tile.output_specs)
        order = [store.write.output for store in tile.output_specs]
        for index, seam in chosen.items():
            region, tail, stores = regions[index]
            stores = _in_source_order(stores, order)
            piece = _region_piece(tile, (region,), tail, stores, placement_decided, split_consumed, seam.spelling)
            reads = _buffer_reads(piece.op)
            add_output_piece(
                match,
                fragment,
                output_root(root, {store.write.output for store in stores}),
                piece,
                _piece_inputs(root, piece.op, tuple(buffer for buffer in all_buffers if buffer in reads)),
                suffix=_PLACED,
            )
        kept = [regions[index] for index in range(len(regions)) if index not in chosen]
        consumer_stores = _in_source_order(tuple(store for _, _, stores in kept for store in stores), order)
        # A composed decision may hand every output away; then there is no sibling piece to emit.
        if not consumer_stores:
            return fragment
        consumer_outputs = {store.write.output for store in consumer_stores}
        consumer_fold = _region_term(
            tuple(region for region, _, _ in kept),
            Body(tuple(stmt for _, tail, _ in kept for stmt in tail)),
            tuple(dict.fromkeys(value for store in consumer_stores for value in store.write.values)),
        )

    consumer = TileOp(
        op=consumer_fold,
        name=tile.name,
        place=tile.place,
        axes=tile.axes,
        # The stored value each boundary write reads is re-spelled where a workspace cut renamed it;
        # ``add_output_piece`` re-spells the BUFFER each write targets.
        output_specs=tuple(
            replace(store, write=replace(store.write, values=tuple(read_names.get(value, value) for value in store.write.values)))
            for store in consumer_stores
        ),
        placement_decided=placement_decided,
        split_consumed=split_consumed,
    )
    consumer = replace(consumer, knobs=consume_kernel_row(consumer.knobs))
    add_output_piece(
        match,
        fragment,
        output_root(root, consumer_outputs),
        consumer,
        _piece_inputs(root, consumer_fold, tuple(all_buffers)),
        suffix=_PLACED,
    )
    return fragment


__all__ = ["CutSite", "Frontier", "cuttable_seams", "output_map", "realize", "storage_frontier"]
