"""Materialize a stored Fold edge as a workspace kernel boundary.

The cut is structural: the child Fold keeps its algebra, writes every state component to a
workspace, and the parent reads those components through ordinary ``Load`` edges. Both pieces
are fresh unmapped ``TileOp`` objects. Pinned cuts consume placement on both pieces before the
cut pass continues with cross-CTA reduction splitting; unpinned cuts may expose smaller seams.
A contraction-operand seam whose cone passes through a storage waypoint cuts THERE instead
(:func:`storage_frontier`): the workspace holds the raw storage bits and the consumer keeps the
decode-plus-factors residue.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import (
    Fold,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.pure.tree import walk
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import carries_partition, edge_dtypes
from emmy.compiler.ir.tile.path import family_sites, sites, spell
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row
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
    #: Sibling stmts (``Load``/``Assign``, fold-free) the produced piece prepends to close the
    #: cone — a body-member fold may capture host-provided scalars (attention's mask/scale
    #: constants) that a plain-closure seam carries inside its cone. Identical across every
    #: hosting body, or the seam is not offered.
    providers: tuple = ()
    #: Captured names this cone can only take from ANOTHER seam's workspace: ``(name, producer
    #: Fold)`` pairs where the producer is itself an offered seam (attention's second statistics
    #: pass reading the first pass's accumulator). Cutting this seam is only realizable as a
    #: composed decision that also cuts every producer; the produced piece reads the name back
    #: through the producer's workspace loads.
    requires: tuple = ()


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
    body = node.lift.body
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
    if node.lift.results[0] in prefix_defs:
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
    """Everything ``node`` needs supplied from outside — read off its DECLARATIONS.

    Was: lower the term and walk the result for free names. That asked a term to re-derive what it
    already states, re-lowered on every call, and returned a superset (names the term binds
    internally, which every caller here discards). :attr:`Fold.index_space` is the declaration —
    the term's own axes unioned with its operands', asked of the term rather than derived here."""
    return node.index_space


def _closed_at(node: Fold, axes: tuple) -> bool:
    """Whether ``node`` has no capture other than axes available at its incoming edge."""
    return _external_reads(node) <= {axis.name for axis in axes}


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
    stmts = tuple(tile.op.lower())
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


def _environments(root: Fold) -> dict[int, list[tuple[Fold, ...]]]:
    """Each stored Fold occurrence's lexical environment, innermost scope first.

    Operand edges and body members have the same lexical scoping. The walk follows every tree
    occurrence, including shared nodes, because provider closure must prove that all occurrences
    resolve to the same sources before one placement seam may represent them.
    """
    environments: dict[int, list[tuple[Fold, ...]]] = {}

    def visit(node: Fold, outer: tuple[Fold, ...]) -> None:
        inner = (node, *outer)
        for child in (*node.operands, *node.lift.body):
            for stored in _stored_folds(child):
                environments.setdefault(id(stored), []).append(inner)
                visit(stored, inner)

    visit(root, ())
    return environments


def _stored_folds(member):
    """``member`` itself when it is a Fold, else every Fold stored in its nested bodies.

    A plain statement binds axes, not SSA definitions, so it is not a resolution scope — but it can
    HOLD a stored fold (a ``ProjectionRegion`` keeps its cones), and one reached only that way had
    no environment at all, which drops its seam. ``ir.fold_tree.walk`` alternates node-wise and
    statement-wise for the same reason."""
    if isinstance(member, Fold):
        yield member
        return
    for body in member.nested():
        for stmt in body:
            yield from _stored_folds(stmt)


def _closed_in(node: Fold, environment: tuple[Fold, ...], axis_names: set[str]) -> tuple[tuple, tuple] | None:
    """Resolve one occurrence's captures outward through its lexical environment."""
    needed = set(_external_reads(node)) - axis_names
    levels: list[tuple] = []
    requires: list = []
    for scope in environment:
        if not needed:
            break
        defined: dict[str, object] = {}
        for stmt in scope.lift.body:
            for name in stmt.defines():
                defined[name] = stmt
        for edge in scope.operands:
            for name in edge.exposes:
                defined.setdefault(name, edge)
        order = {id(stmt): position for position, stmt in enumerate(scope.lift.body)}
        providers: list = []
        outward: set[str] = set()
        resolved: set[str] = set()
        queue = sorted(needed)
        while queue:
            name = queue.pop()
            if name in resolved:
                continue
            resolved.add(name)
            producer = defined.get(name)
            if producer is None:
                outward.add(name)
                continue
            if producer is node:
                return None
            if isinstance(producer, Fold):
                requires.append((name, producer))
                continue
            if not isinstance(producer, (Load, Assign)):
                return None
            if not any(chosen is producer for chosen in providers):
                providers.append(producer)
            queue.extend(sorted(Body((producer,)).ssa_uses - axis_names - resolved))
        providers.sort(key=lambda stmt: order.get(id(stmt), -1))
        levels.append(tuple(providers))
        needed = outward
    if needed:
        return None
    providers = tuple(stmt for level in reversed(levels) for stmt in level)
    return providers, tuple(sorted(requires, key=lambda pair: pair[0]))


def _provider_closure(node: Fold, scopes, environments: list[tuple[Fold, ...]]) -> tuple[tuple, tuple] | None:
    """Close a capturing seam identically at every stored occurrence, or return ``None``."""
    if not environments:
        return None
    axis_names = {axis.name for scope in scopes for axis in scope}
    closures = []
    for environment in environments:
        closure = _closed_in(node, environment, axis_names)
        if closure is None:
            return None
        closures.append(closure)
    first_providers, first_requires = closures[0]
    for providers, requires in closures[1:]:
        aligned = providers == first_providers and len(requires) == len(first_requires)
        if not aligned or any(
            a != b or producer_a is not producer_b for (a, producer_a), (b, producer_b) in zip(requires, first_requires, strict=True)
        ):
            return None  # occurrences resolving to different sources cannot share one seam
    provided = {name for stmt in first_providers for name in stmt.defines()}
    provided.update(name for name, _ in first_requires)
    outside = (set(_external_reads(node)) | {name for stmt in first_providers for name in Body((stmt,)).ssa_uses}) - provided
    if any(outside - {axis.name for axis in scope} for scope in scopes):
        return None
    return first_providers, first_requires


def _dtype_table(tile: TileOp) -> dict[int, tuple]:
    """Each stored edge's inferred result dtypes, keyed by edge identity.

    One edge, one answer: a term is CLOSED, so its values arrive through its own operand edges and
    it types the same wherever it occurs. The occurrence-agreement filter this used to apply — one
    shared cone under two scopes having two answers — cannot arise once no term captures."""
    cache: dict[int, tuple] = {}
    edge_dtypes(tile.op, tile.inputs, cache)
    return dict(cache)


def cuttable_seams(tile: TileOp) -> tuple[CutSite, ...]:
    """Every semantically closed stored Fold edge whose workspace dtypes are determined, grouped
    only by object sharing. A contraction's operand edges are seams too — cutting one materializes
    the cone feeding the operand into its own kernel and the contraction reads it back as an
    ordinary load — and they take the explicit contraction-operand dtype rule (`_workspace_dtypes`).
    A body-member fold capturing host-provided names is offered through provider closure
    (:func:`_provider_closure`); one whose closure needs another seam's workspace is offered as a
    dependent seam and drops out unless that producer is offered too."""
    all_sites = sites(tile.op)
    store_dtype_consumers = {
        id(edge): site.node
        for site in all_sites
        if site.node.as_contraction() is not None
        for edge in site.node.operands
        if isinstance(edge, Fold) and not edge.is_slab
    }
    outer = (*tile.place.free, *(store.sweep for store in tile.output_specs if store.sweep is not None))
    occurrence_axes: dict[int, list[tuple]] = {}
    for visit in islice(walk(tile.op, outer), 1, None):
        occurrence_axes.setdefault(id(visit.node), []).append(visit.axes)
    environments: dict[int, list[tuple[Fold, ...]]] = {}
    dtype_table: dict[int, tuple] = {}
    if isinstance(tile.op, Fold):
        environments = _environments(tile.op)
        dtype_table = _dtype_table(tile)
    out: list[CutSite] = []
    seen: set[int] = set()
    for site in family_sites("PLACE", all_sites):
        node = site.node
        scopes = occurrence_axes.get(id(node), ())
        if not isinstance(node, Fold) or node.is_slab or id(node) in seen or not scopes:
            continue
        providers: tuple = ()
        requires: tuple = ()
        if not all(_closed_at(node, scope) for scope in scopes):
            closure = _provider_closure(node, scopes, environments.get(id(node), []))
            if closure is None:
                continue
            providers, requires = closure
        if node.observed:
            # An observed fold's per-step results exist only inside its stream — a cut would
            # separate the scan from its streamed boundary store, which no piece can then spell.
            continue
        consumer = store_dtype_consumers.get(id(node))
        # A frontier REPLACES the fed-store realization at this seam rather than joining the
        # offer: the raw bits dominate the fed-store workspace on both precision (exact vs
        # re-rounded) and footprint (storage width vs store width), so there is no trade for the
        # evidence to decide — one site stays one decision.
        frontier = storage_frontier(node) if consumer is not None else None
        dtypes = (frontier.dtype,) if frontier is not None else _workspace_dtypes(node, tile, consumer, dtype_table)
        if dtypes is None:
            continue
        seen.add(id(node))
        axes = tuple({axis.name: axis for scope in scopes for axis in scope}.values())
        out.append(
            CutSite(
                node=node,
                spelling=spell(tile.op, "PLACE", node, all_sites=all_sites),
                axes=axes,
                dtypes=dtypes,
                frontier=frontier,
                providers=providers,
                requires=requires,
            )
        )
    # A dependent seam stands only while every required producer is itself offered; dropping one
    # can orphan the next link of a chain, so filter to the fixpoint.
    while True:
        offered = {id(seam.node) for seam in out}
        kept = [seam for seam in out if all(id(producer) in offered for _, producer in seam.requires)]
        if len(kept) == len(out):
            break
        out = kept
    return _cluster_value_seams(out, {id(seam.node): seam.frontier is None and store_dtype_consumers.get(id(seam.node)) for seam in out})


def _closure_key(seam: CutSite) -> tuple:
    """What a seam closes over, as a comparable value: its provider stmts and the SOURCES its
    requirements name. Two cones can be alpha-equivalent and still read different sources — a
    capture is a free name, and one host may define ``x = sum(a)`` where the other defines
    ``x = sum(b)`` — so the closure is part of the value, and clustering compares it."""
    return (seam.providers, tuple((name, id(producer)) for name, producer in seam.requires))


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
    # A seam ANOTHER seam requires never joins a cluster, as representative or as member. A
    # dependent's piece reads its producer's workspace by the name that producer binds, and
    # clustering re-points a value at the representative — whose result names are its own. Leaving
    # a required producer alone keeps the requirement pointing at a seam that still exists and
    # still binds the name the dependent reads.
    required = {id(producer) for seam in seams for _, producer in seam.requires}
    eligible = [index for index, seam in enumerate(seams) if operand_of.get(id(seam.node)) and id(seam.node) not in required]
    if len(eligible) < 2:
        return tuple(seams)
    captured = {index: tuple(axis.name for axis in seams[index].axes) for index in eligible}
    # The seam's own capture correspondence, and the alpha-quotient taken under it: an operand cone
    # is a TERM, so it quotients as one (``Fold.canonical``) rather than as a scoped lambda.
    scoped = {index: tuple(axis for axis in captured[index] if axis in seams[index].node.index_space) for index in eligible}
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
            aligned = (
                member.dtypes == rep.dtypes
                and _closure_key(member) == _closure_key(rep)
                and all(
                    (a := rep_axes[rn]).extent == (b := member_axes[mn]).extent and a.window == b.window
                    for rn, mn in zip(rep_params, member_params, strict=True)
                )
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
    captures = _external_reads(produced)
    return tuple(axis for axis in seam.axes if axis.name in captures or (axis.extent.is_static and axis.extent.as_static() == 1))


def _piece_inputs(root: Node, fold: Fold, first: tuple[str, ...] = ()) -> list[str]:
    """A piece's graph inputs: its workspaces, then every buffer of ``root``'s the piece reads.

    The read set is :attr:`Fold.loads` — the STORED tree's, walked through the operand edges a
    body cannot reach. A walk that stopped at the stored body named fewer inputs than the kernel
    went on to read; the workspace producers then had no consumer edge, were pruned as orphans,
    and the launch asked for a buffer nothing had allocated."""
    reads = {load.input for load in fold.loads}
    return [*first, *(name for name in root.inputs if name in reads)]


def _input_fragment(match: Match, root: Node) -> Graph:
    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(name), node_id=name)
    return fragment


def output_map(root: Node) -> dict[str, str]:
    """Stable temporary output names used by every cut sibling of ``root``."""
    return {name: f"{name}__placed" for name in root.buffer_names()}


def _requires_order(seams) -> tuple:
    """``seams`` with every required producer ahead of its dependents; a producer absent from the
    composed decision is left for the realization's per-name error. Requirement chains follow SSA
    definition order, so the graph is acyclic by construction."""
    present = {id(seam.node) for seam in seams}
    remaining = list(seams)
    ordered: list = []
    placed: set[int] = set()
    while remaining:
        step = [seam for seam in remaining if all(id(producer) in placed or id(producer) not in present for _, producer in seam.requires)]
        assert step, "cyclic seam requirements cannot arise from SSA definition order"
        ordered.extend(step)
        placed.update(id(seam.node) for seam in step)
        remaining = [seam for seam in remaining if not any(seam is chosen for chosen in step)]
    return tuple(ordered)


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
        ready = [piece for piece in remaining if {load.input for load in piece[1].loads} & workspaces <= available]
        assert ready, "strict Fold containment makes the cut-workspace dependency graph acyclic"
        ordered.extend(ready)
        available.update(buffer for *_, buffers in ready for buffer in buffers)
        remaining = [piece for piece in remaining if not any(piece is chosen for chosen in ready)]
    return ordered


def realize(
    match: Match,
    root: Node,
    seams,
    renamed_outputs: dict[str, str],
    *,
    placement_decided: bool = False,
) -> Graph:
    """Build the cut fragment for ``seams`` — one producer kernel per seam plus the ONE consumer
    reading every workspace back. A single seam is the two-kernel cut; several seams are one
    COMPOSED placement decision (a pinned compile consumes every scoped PLACE pin that resolves
    on this kernel at once, so the pieces stay decided and the knob row records every spelling).
    A frontier seam cuts at the cone's storage waypoint: the producer computes the encode prefix,
    the workspace holds the raw bits, and the consumer keeps the decode + factor residue as its
    operand cone (which normalization then binds as a raw storage-dtype load with the factors
    hoisted onto the accumulator epilogue).

    ``placement_decided`` consumes an authoritative pinned PLACE restriction on every piece.
    Unpinned cuts leave it false so fresh pieces can expose and decide smaller seams. A placement
    cut never erases an earlier cross-CTA decision: every piece inherits the parent's explicit or
    sliced-axis split receipt."""
    tile: TileOp = root.op
    split_consumed = tile.split_consumed or carries_partition(tile.op)
    pieces = []
    workspace_loads: dict[int, tuple] = {}
    for seam in _requires_order(seams):
        child = seam.node
        front = seam.frontier
        if front is not None:
            names = (front.name,)
            produced = Fold(
                operands=(),
                lift=Lambda.closing(tuple(name for edge in () for name in edge.exposes), Body.coerce(Body(front.producer)), names),
            )
        else:
            names = child.exposes
            produced = child
        if seam.providers or seam.requires:
            # Provider closure: the piece carries the host-provided scalar chain, and every
            # required name arrives through its producer seam's workspace load — the same Load
            # objects the replacement spells, so the two spellings cannot drift apart.
            prefix = list(seam.providers)
            for name, producer in seam.requires:
                producer_loads = workspace_loads.get(id(producer))
                if producer_loads is None:
                    raise ValueError(f"seam {seam.spelling!r} requires {name!r} from a producer seam absent from this composed cut")
                # Whatever binds the name, taken by the name it BINDS: a plain workspace ``Load``,
                # or — when the producer cut at a storage frontier — the projection wrapping the
                # raw load and its decode residue. Matching on ``Load`` alone skipped the frontier
                # form and left the requirement unbound in the piece.
                supplied = [entry for entry in producer_loads if name in entry.exposes]
                if not supplied:
                    raise ValueError(f"seam {seam.spelling!r} requires {name!r}, which its producer seam does not bind")
                prefix.extend(supplied)
            produced = Fold(
                operands=(),
                lift=Lambda.closing(tuple(name for edge in () for name in edge.exposes), Body.coerce(Body((*prefix, produced))), names),
            )
        axes = _workspace_axes(seam, produced)
        index = tuple(Var(axis.name) for axis in axes)
        token = digest(tile.identity_key(structural=False) or "", seam.spelling)[:10]
        buffers = tuple(f"{root.id}__place_{token}_{i}" for i in range(len(names)))

        # SLABS, not bare Loads: these replace an operand edge, and an operand is a term. The
        # workspace read declares the seam axes it indexes, exactly as any other gmem read does.
        loads = tuple(
            Fold.slab(Load(name=name, input=buffer, index=index), axes) for name, buffer in zip(names, buffers, strict=True)
        )
        if front is not None:
            raw = replace(loads[0], dtype=front.dtype)
            loads = (
                Fold(
                    operands=(),
                    lift=Lambda.closing(
                        tuple(name for edge in () for name in edge.exposes), Body.coerce(Body((raw, *front.residue))), child.lift.results
                    ),
                ),
            )
        replacements = {id(child): loads}
        workspace_loads[id(child)] = loads
        for sibling, pairs in seam.siblings:
            # A clustered duplicate reads the SAME workspace, spelled through its own captured
            # axes via the correspondence the clustering proved.
            mapping = dict(pairs)
            sibling_index = tuple(Var(mapping[axis.name]) for axis in axes)
            replacements[id(sibling)] = tuple(
                Load(name=name, input=buffer, index=sibling_index) for name, buffer in zip(sibling.exposes, buffers, strict=True)
            )
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
            output_specs=tuple(
                OutputSpec(Write(output=buffer, index=index, value=name)) for name, buffer in zip(names, buffers, strict=True)
            ),
            placement_decided=placement_decided,
            split_consumed=split_consumed,
        )
        producer = replace(producer, knobs=consume_kernel_row(producer.knobs))
        shape = tuple(axis.extent for axis in axes)
        workspace_tensors = tuple(Tensor(name=buffer, shape=shape, dtype=dtype) for buffer, dtype in zip(buffers, seam.dtypes, strict=True))
        reads = {load.input for load in produced.loads}
        fragment.add_node(
            op=producer,
            inputs=_piece_inputs(root, produced, tuple(buffer for buffer in all_buffers if buffer in reads)),
            outputs=workspace_tensors,
            node_id=buffers[0],
        )

    parent_stores = tuple(
        replace(store, write=replace(store.write, output=renamed_outputs.get(store.write.output, store.write.output)))
        for store in tile.output_specs
    )
    consumer = TileOp(
        op=parent_fold,
        name=tile.name,
        place=tile.place,
        output_specs=parent_stores,
        placement_decided=placement_decided,
        split_consumed=split_consumed,
    )
    consumer = replace(consumer, knobs=consume_kernel_row(consumer.knobs))
    output_tensors = (
        replace(root.outputs[0], name=root.outputs[0].name),
        *(replace(tensor, name=renamed_outputs[name]) for name, tensor in zip(root.buffer_names()[1:], root.outputs[1:], strict=True)),
    )
    all_buffers = [buffer for *_, buffers in produced_pieces for buffer in buffers]
    fragment.add_node(
        op=consumer,
        inputs=_piece_inputs(root, parent_fold, all_buffers),
        outputs=output_tensors,
        node_id=renamed_outputs[root.id],
    )
    fragment.outputs.extend(renamed_outputs.values())
    return fragment


__all__ = ["CutSite", "Frontier", "cuttable_seams", "output_map", "realize", "storage_frontier"]
