r"""The geometry-free compute layer — node lowering and the structural reads.

A kernel's compute is one stored :class:`~emmy.compiler.ir.pure.fold.Fold`: a bare reduction, a
pure pointwise cell, or the zero-axis projection over another Fold. :func:`head` reaches the
iterating node through that projection, and every structural fact a pass dispatches on — its
derived role, reduce ``Axis``, and operand edges — comes directly from the tree. Reading those
facts off a synthesized nest is the inversion this module exists to prevent; :meth:`Fold.lower`
is for callers that consume a body.

This module holds the structural reads over a node tree — the cone seam (:func:`~emmy.compiler.ir.schedule.views.cone_seam`), the
iteration-space names (:func:`axis_names`) — plus the typed schedule accessor (:class:`Sched`). Lowering itself
has ONE spelling and it lives on the node: :meth:`Fold.lower` (a fold flattens through
:attr:`Fold.loop`, a wrapping projection appends its operand nests). Stored trees are already
resolved — a computed operand is an inline node on its edge, so there is no name-resolution step
ahead of a lowering walk."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.ir.pure.fold import (
    Fold,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.schedule import PlacedTile, Reduce
from emmy.compiler.ir.schedule.classic import (
    ReductionSchedule,
    classic_node_key,
    classic_stage_key,
)
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Select, refs_axis, stmt_axis_names
from emmy.compiler.ir.stmt.base import Stmt, dtype_promote
from emmy.compiler.ir.tile.ir import TileOp, apply_output_specs
from emmy.compiler.ir.tile.path import UnknownSiteError, sites


def cone_stat_dtypes(pro: tuple, stats: tuple[str, ...], inputs) -> dict[str, object]:
    """The dtype each value bridged across the cone's K seam carries (:func:`cone_seam`).

    The prologue runs once per tile row and publishes its results through smem rows the cell reads
    back, so those rows are the only place a bridged value's dtype is declared. Declaring them all
    float would decide the CELL's arithmetic too: a value that crosses as an integer — a shift
    amount, a nibble mask — comes back f32, and the bit operations reading it have no f32 spelling.

    Typed the same way :func:`edge_dtypes` types an edge's results, over the prologue's flat stmt
    list. A name whose stmt kind carries no dtype is absent from the result; its row keeps the
    float default."""
    env: dict[str, object] = {}
    for stmt in pro:
        if isinstance(stmt, Load):
            tensor = inputs.get(stmt.input) if inputs else None
            env.update((name, tensor.dtype if tensor is not None else None) for name in stmt.names)
        elif isinstance(stmt, Assign):
            args = [env.get(name) for name in stmt.args]
            env[stmt.name] = stmt.dtype or (get_dtype(dtype_promote(stmt.op.name, [d.name for d in args])) if all(args) else None)
        elif isinstance(stmt, Init):
            env[stmt.name] = stmt.dtype
        else:
            env.update((name, None) for name in stmt.defines())
    return {nm: dt for nm in stats if (dt := env.get(nm)) is not None}


def edge_dtypes(edge, inputs, cache: dict[int, tuple] | None = None, scope: dict | None = None) -> tuple:
    """Infer an edge's result dtypes in the lexical scope where the edge occurs.

    A term is closed, so an edge types the same wherever it occurs: its values arrive positionally
    through its own operand edges, never from the scope around it. The ``cache`` is therefore keyed
    by object identity alone, and a shared node is typed once.
    """
    cache = {} if cache is None else cache
    # Identity alone: a term is CLOSED, so it captures nothing and its dtypes cannot depend on the
    # occurrence. Keying is valid while the tree is alive, which it is for every caller — each
    # holds the root across the walk.
    key = id(edge)
    if key in cache:
        return cache[key]
    if edge.as_slab() is not None:
        load = edge.as_slab().load
        tensor = inputs.get(load.input) if inputs else None
        result = (tensor.dtype if tensor is not None else None,) * len(load.names)
        cache[key] = result
        return result
    if not isinstance(edge, Fold):
        result = (None,) * len(edge.exposes)
        cache[key] = result
        return result

    env = dict(scope or {})
    for param, operand, index in edge.bindings:
        env[param] = edge_dtypes(operand, inputs, cache, env)[index]
    for stmt in edge.lift.body:
        if isinstance(stmt, Load):
            tensor = inputs.get(stmt.input) if inputs else None
            env.update((name, tensor.dtype if tensor is not None else None) for name in stmt.names)
        elif isinstance(stmt, Fold):
            env.update(zip(stmt.exposes, edge_dtypes(stmt, inputs, cache, env), strict=True))
        elif isinstance(stmt, Assign):
            args = [env.get(name) for name in stmt.args]
            env[stmt.name] = stmt.dtype or (get_dtype(dtype_promote(stmt.op.name, [dtype.name for dtype in args])) if all(args) else None)
        elif isinstance(stmt, Init):
            env[stmt.name] = stmt.dtype
        elif isinstance(stmt, Select):
            branch_dtypes = [env.get(branch.value) for branch in stmt.branches]
            env[stmt.name] = branch_dtypes[0] if branch_dtypes and all(dtype == branch_dtypes[0] for dtype in branch_dtypes) else None
        else:
            env.update((name, None) for name in stmt.defines())

    lifted = tuple(env.get(result) for result in edge.lift.results)
    if edge.axis is None:
        result = lifted
    else:
        carried = lifted[: len(edge.combine.results)]
        result = carried + tuple(env.get(name) for name in edge.exposes[len(carried) :])
    cache[key] = result
    return result


def make_cone(cell: list, k_name: str, stat=None, sweep=()) -> Fold:
    """Build a computed-A **cone** as a real node tree — the inline node its consuming operand edge
    stores (there is no let table: sharing is the product contraction's channel arity). The
    inverse of :func:`cone_seam`, kept beside it so the K-seam layout has one home.

    The K seam is decided HERE, once, and lives ON the node: the maximal leading run of cone stmts
    that never index the contraction axis is **row-invariant**, so it joins the per-row statistic
    (``stat``, the :class:`Fold`, plus its scalar ``sweep``) in the cone's SOURCE node — one
    projected reduce, exactly the RMSNorm shape; the k-varying remainder is the cone's ``body``, the
    per-cell normalize. Everything downstream then READS that boundary (:func:`cone_seam`) instead of
    re-scanning stmts: the scheduler sizes the stat smem rows off it, the materializer runs the
    prologue once per tile row and the body per cell.

    A producer NODE in ``cell`` is not a stmt of either side: it hangs off the cone's OPERANDS,
    where ``cone_seam`` splits it by the same K seam — k-varying, so it is the per-cell producer
    spliced ahead of its first use.

    The cell's λ is CLOSED: every prologue value it reads — a statistic component the prologue
    binds (softmax's ``m``, read by the per-cell ``exp(s − m)``) or a row-invariant def — passes
    through as a further prologue RESULT, so the cell binds it positionally like any operand.
    Nothing in a stored term captures; the seam between statistic and normalize is then a
    positional edge like every other, and ``cone_seam``'s bridge is that edge's extra results."""
    nodes = tuple(s for s in cell if isinstance(s, Fold))
    cell = [s for s in cell if not isinstance(s, Fold)]
    # A stmt READING a k-varying producer varies with k as surely as one that indexes it: the K
    # seam is a dependency question, not only an index question. Without this, attention's
    # ``exp(s − m)`` chain — which names the score rather than the KV axis — hoists into the
    # row-invariant prologue, where the per-cell score it reads is not yet defined.
    varying = {nm for n in nodes if k_name in n.free_axes for nm in n.exposes}
    pro: list = []
    rest = list(cell)
    while rest and not refs_axis(rest[0], k_name) and not (set(rest[0].deps()) & varying):
        pro.append(rest.pop(0))
    pro_body = Body((*sweep, *pro))
    pro_ops = () if stat is None else (stat,)
    # The projection's result is its body's last definition — the value a consumer reads back.
    pro_bound = tuple(name for edge in pro_ops for name in edge.exposes)
    pro_results = next((stmt.defines()[-1:] for stmt in reversed(tuple(pro_body)) if stmt.defines()), ())
    prologue = Fold(operands=pro_ops, lift=Lambda.closing(pro_bound, pro_body, pro_results))
    cell_reads = Body(rest).ssa_uses
    bound = (*(n for e in pro_ops for n in e.exposes), *(n for s in pro_body for n in s.defines()))
    bridged = tuple(n for n in dict.fromkeys(bound) if n in cell_reads and n not in prologue.lift.results)
    if bridged:
        prologue = Fold(operands=pro_ops, lift=Lambda.closing(pro_bound, pro_body, (*prologue.lift.results, *bridged)))
    src = (prologue,) if (pro or sweep or stat is not None) else ()
    cell_body = Body(tuple(rest))
    cell_ops = (*src, *nodes)
    cell_bound = tuple(name for edge in cell_ops for name in edge.exposes)
    cell_results = next((stmt.defines()[-1:] for stmt in reversed(tuple(cell_body)) if stmt.defines()), ())
    return Fold(operands=cell_ops, lift=Lambda.closing(cell_bound, cell_body, cell_results))


class Sched:
    """Read-only view of one kernel's schedule choices and materialization facts.

    Node families use stable integer identities; transport is keyed by consumer and operand.
    ``PLACE`` is structural and remains outside this view. A node outside the problem, or a family
    outside that node's classified schedule sum, fails loudly.
    """

    def __init__(self, tile, place=None, schedule=None, materialization=None) -> None:
        if not isinstance(getattr(tile, "op", None), Fold):
            raise TypeError("a schedule view reads a TileOp — it is the term's site index")
        self.tile = tile
        self.root = tile.op
        self.axis_of = tile.axis_of  # the kernel's axis table: a term names its axes, the tile holds their extents
        self.schedule = schedule
        self.materialization = materialization
        #: The kernel's free→grid :class:`~emmy.compiler.ir.schedule.Placement`. A ``TILE`` slice
        #: is geometry over an ``(m, n)`` output pair, and WHICH pair is a function of the site's
        #: position in the tree — so the binding belongs here, on the scheduling structure, and
        #: not at each reader (:meth:`tile_of`).
        self.place = place
        self._sites = None
        self._site_by_id = None
        self._mn_by_id = {}

    def _all_sites(self):
        if self._sites is None:
            self._sites = sites(self.root)
        return self._sites

    def site_of(self, node):
        """The :class:`~emmy.compiler.ir.tile.path.Site` of ``node`` on this tree — how a consumer
        holding a node learns its own address. A shared subtree keeps its FIRST site (one node,
        one schedule, however many paths reach it). A node that is not a site RAISES
        :class:`~emmy.compiler.ir.tile.path.UnknownSiteError` — an identity miss (a copied or
        rebuilt node) must be loud, never a silent fall-through to the untiled path."""
        if self._site_by_id is None:
            by_id = {}
            for s in self._all_sites():
                by_id.setdefault(id(s.node), s)
            self._site_by_id = by_id
        site = self._site_by_id.get(id(node))
        if site is None:
            raise UnknownSiteError(
                f"{type(node).__name__} is not a site of this tree — the caller holds a copied or "
                f"rebuilt node, not the stored object the site walk enumerated"
            )
        return site

    def key(self, family: str, node) -> str | None:
        """The canonical node-family key, or ``None`` when the family does not apply."""
        try:
            site = self.tile.node_id(node)
        except KeyError as error:
            raise UnknownSiteError(str(error)) from None
        if family == "STAGE":
            family_sites = tuple(dict.fromkeys(edge[0] for edge in self.tile.stage_edges))
        elif family in self.tile.family_sites:
            family_sites = self.tile.family_sites[family]
        else:
            raise ValueError(f"unknown classic schedule family {family!r}")
        if site not in family_sites:
            return None
        if family == "STAGE":
            return classic_stage_key(self.tile, next(edge for edge in self.tile.stage_edges if edge[0] == site))
        return classic_node_key(self.tile, family, site)

    def get(self, family: str, node):
        if self.schedule is None:
            return None
        site = self.tile.node_id(node)
        assignment = self.schedule.nodes[site]
        if family == "TILE":
            return assignment.tile if assignment.tile.is_tiled else None
        if family == "REDUCE":
            if not isinstance(assignment, ReductionSchedule) or not assignment.reduce.stages:
                return None
            return assignment.reduce
        if family == "STAGE":
            if self.materialization is None:
                return None
            stages = {stage for edge, stage in self.materialization.stages.items() if edge[0] == site}
            if len(stages) > 1:
                raise ValueError("current kernel lowering requires one resolved transport across a node's operand edges")
            return next(iter(stages), None)
        raise ValueError(f"unknown classic schedule family {family!r}")

    def tile_of(self, node):
        """The node's ``TILE`` slice, ALREADY PLACED — its ``(m, n)`` output axes bound
        (:attr:`Tile.axes`), so a reader gets the geometry off the slice and never re-derives
        placement of its own. ``axes`` stays ``compare=False`` / ``repr=False``, so binding it here
        cannot reach ``spell()``, a stamped row, a golden or a prior key.

        The pair is a function of the SITE (:meth:`_mn_for`), which is what makes one rule
        possible where there used to be three hand-written ``.at(...)`` calls."""
        if self.schedule is None or self.materialization is None:
            return None
        return self.materialization.tiles.get(self.tile.node_id(node))

    def placed(self, node, plan):
        """``plan`` bound to the ``(m, n)`` output axes a ``TILE`` slice at ``node``'s site tiles —
        the same one rule :meth:`tile_of` reads through, offered to a caller holding a CANDIDATE
        plan the table does not carry yet (the enumeration, whose legality predicates read the
        placed geometry). Already-placed and unplaceable plans pass through."""
        if plan is None or self.place is None or isinstance(plan, PlacedTile):
            return plan
        mn = self._mn_for(node)
        return plan.at(*mn) if mn is not None else plan

    def _mn_for(self, node):
        """The cached ``(m, n)`` output axes for ``node``. Placement is a site fact: candidate
        plans change tile sizes, never which output axes they tile."""
        key = id(node)
        if key not in self._mn_by_id:
            self._mn_by_id[key] = self._derive_mn(node)
        return self._mn_by_id[key]

    def _derive_mn(self, node):
        """The ``(m, n)`` output axes a ``TILE`` slice at ``node``'s site tiles, or ``None`` when
        the placement cannot supply them (an unmapped / rank-<2 grid — the caller's untiled path).

        THREE site shapes, one rule each — the geometry the retired ``Contraction`` node used to
        carry as stamped fields, now read off the tree instead:

        - a ROOT contraction, including one directly under a zero-axis projection that groups
          several kernel outputs, tiles the kernel grid's trailing pair (``Placement.root_mn``,
          the same reading the scheduler binds through at option assembly);
        - a derived unit-axis contraction inherits its parent Fold's reduction domain, so its
          result tiles the placement's trailing free pair;
        - any other nested contraction takes the free m axis and its nearest ENCLOSING fold's axis as n —
          read through a slice partial's window PARENT, so the view carries the pre-slice geometry
          the fragment clamps were built against.
        """
        free = tuple(self.place.free)
        site = self.site_of(node)
        ancestors = tuple(candidate for candidate in self._all_sites() if site.under(candidate))

        def orient(mn):
            # The pair is each side's own free axis, the first operand's leading — the placement
            # binds both, and a sibling output's sweep promoted beside them (the fused q/k/v
            # projections, N 64 beside N 32) never stands in for either. With several own axes a
            # side the trailing one is the role and the rest ride the grid; a side without one
            # (the unit-row matvec) leaves the trailing pair to the placement.
            view = node.as_contraction()
            if mn is None or view is None:
                return mn
            order = {axis.name: (position, axis) for position, axis in enumerate((*self.place.free, *self.place.grid))}
            left = max((order[name] for name in view.left_axes if name in order), default=None)
            right = max((order[name] for name in view.right_axes if name in order), default=None)
            if left is not None and right is not None:
                return (left[1], right[1])
            first, second = mn
            return (second, first) if second.name == view.left and first.name != view.left else mn

        if all(getattr(candidate.node, "axis", None) is None for candidate in ancestors):
            return orient(self.place.root_mn)
        if len(free) < 2:
            return None
        # The nearest ENCLOSING fold, through any zero-axis projection between them — a projection
        # binds no coordinate, so it cannot be the one the result is evaluated over. A weight cone
        # reaches the score it is built over through exactly one such level.
        parent = next(
            (
                found
                for depth in range(len(site.hops) - 1, -1, -1)
                if (found := next((s for s in self._all_sites() if s.hops == site.hops[:depth]), None)) is not None
                and getattr(found.node, "axis", None) is not None
            ),
            None,
        )
        ax = self.axis_of(parent.node.axis) if parent is not None else None
        if ax is None:
            return None
        return orient((free[-2], ax.window.parent if ax.window is not None else ax))


def sched_of(tile) -> Sched:
    """Return the typed schedule view of a ``TileOp``."""
    return Sched(
        tile,
        place=tile.place,
        schedule=tile.schedule,
        materialization=tile.materialization,
    )


def scheduled(
    op,
    *,
    name: str,
    place,
    knobs: dict,
    output_specs: tuple = (),
    schedule=None,
    materialization=None,
    workers=None,
    axes: tuple = (),
):
    """Build a scheduled ``TileOp`` from one accepted semantic assignment.

    The one constructor every row materializer shares (a split piece is not built here — it leaves
    ``030_cut`` unscheduled and reaches this through its own row). The accepted assignment
    is the sole worker-inventory source; the encoded row must agree with it."""
    if schedule is None:
        raise ValueError("cannot construct a scheduled TileOp without a Schedule")
    work = schedule.kernel.work
    producer = workers.producer_warps if workers is not None else 0
    if work.producer != producer:
        raise ValueError(f"WORK producer band {work.producer} disagrees with WarpSpec producer band {producer}")
    if knobs.get("WORK") != work.spell():
        raise ValueError("encoded WORK does not agree with the accepted classic assignment")
    return TileOp(
        op=op,
        name=name,
        place=place,
        workers=workers,
        knobs=knobs,
        output_specs=tuple(output_specs),
        axes=axes,
        schedule=schedule,
        materialization=materialization,
    )


def axis_names(root) -> set[str]:
    """Every ITERATION-SPACE name in ``root``'s tree — the structural nodes' axes plus every loop
    induction variable in their bodies, over the ONE node walk (``path.sites``). An induction
    variable is bound by the enclosing loop nest, not by any value tree, so a subtree reading one
    is never capturing a value.

    The ONE reading that separates the two kinds of free name a λ body can carry: an iteration var
    (bound by the nest, free by construction) and a captured VALUE. The structural dump shows
    what remains as the λ's capture set."""
    out: set[str] = set()
    for site in sites(root):
        node = site.node
        if not isinstance(node, Fold):
            continue
        if node.axis is None:
            out |= stmt_axis_names(node.lift.body)
        else:
            out.add(node.axis)
            out |= stmt_axis_names(tuple(node.lift.body))
    return out


def projection_tail(tile) -> list[Stmt]:
    """The kernel's EFFECTFUL projection stmt stream — the root zero-axis fold's (pure) body with the
    kernel-boundary ``TileOp.output_specs`` reconstituted (:func:`~emmy.compiler.ir.tile.ir.apply_output_specs`).
    The ONE read every scheduler gate that inspects "the tail" goes through, so the
    ``coop-t`` band's no-sweep-``Loop`` condition keeps excluding rms/softmax rows after their
    sweep moved to an ``OutputSpec`` decoration."""
    op = tile.op
    body = list(op.lift.body) if isinstance(op, Fold) and op.axis is None else []
    return apply_output_specs(body, tile.output_specs)


class UnbindableProjection(ValueError):
    """The projection's outputs do not partition by producing root, so the multi-root binding has
    no realization for this row — a legality fact about the OFFERED row, not a malformed tree.
    Typed so the materializer can decline the row (``RuleSkipped`` → the greedy blocklist retry
    moves to the next one) without masking genuinely malformed input, which stays a plain
    ``ValueError``."""


def projection_root(edge: Fold) -> Fold | None:
    """The reducing term a root projection's operand is ABOUT: the operand itself, or the one
    reducing operand of a zero-axis projection over it — an output sweep's epilogue, evaluated
    over the sweep coordinate. ``None`` when the projection reads several reduces or none."""
    if edge.axis is not None:
        return edge
    reducing = [operand for operand in edge.operands if operand.axis is not None]
    return reducing[0] if len(reducing) == 1 else None


def projection_regions(op: Fold, output_specs: tuple) -> tuple[tuple[Fold, Fold, Body, tuple], ...]:
    """Partition an independent projection by producing root — ``(root, region, tail, stores)`` per
    operand of ``op``: the reducing root (:func:`projection_root`), the operand term that carries
    it (the root itself, or its epilogue projection), the root-body statements only that operand's
    outputs read, and those output specifications.

    Each output specification must read exactly one region, the regions' backward cones over the
    root body must be disjoint, and together those cones must cover it. This is the structural
    ownership rule shared by kernel binding and rewrites that turn one MIMO TileOp into fresh
    pieces. Refusals raise :class:`UnbindableProjection`.
    """
    # Rendered: the tail and the stores in the operands' spelling, the form every consumer emits.
    lift = op.applied
    spelled = dict(zip(op.lift.params, lift.params, strict=True))
    output_specs = tuple(replace(spec, write=spec.write.rewrite(lambda name: spelled.get(name, name))) for spec in output_specs)
    regions = op.operands
    by_name = {name: region for region in regions for name in region.exposes}
    members: dict[int, set] = {id(region): set() for region in regions}
    grouped: dict[int, list] = {id(region): [] for region in regions}
    for spec in output_specs:
        cone = lift.body.backward_cone(tuple(spec.write.values))
        used = {id(by_name[name]) for name in (*cone.external_reads, *spec.write.values) if name in by_name}
        if len(used) != 1:
            raise UnbindableProjection("an output-tiled root must own each output specification independently")
        owner = used.pop()
        members[owner].update(cone.members)
        grouped[owner].append(spec)

    claimed: set = set().union(*members.values()) if members else set()
    if claimed != set(lift.body) or any(not grouped[id(region)] for region in regions):
        raise UnbindableProjection("an output-tiled root forest must cover the complete projection")
    if any(members[id(left)] & members[id(right)] for i, left in enumerate(regions) for right in regions[i + 1 :]):
        raise UnbindableProjection("output-tiled root projections may not share tail statements")
    out = []
    for region in regions:
        root = projection_root(region)
        if root is None:
            raise UnbindableProjection("an output-tiled root must own each output specification independently")
        out.append((root, region, Body(stmt for stmt in lift.body if stmt in members[id(region)]), tuple(grouped[id(region)])))
    return tuple(out)


def head(op):
    """The kernel's compute NODE — a :class:`~emmy.compiler.ir.pure.fold.Fold` at any role, bare or
    under its projection (zero-axis) fold — or ``None`` for a pure pointwise cell.

    The ONE accessor for "which node is this kernel about", replacing the hand-spelled
    ``op.operands[0] if op.axis is None and op.operands else op`` ternary at every reader. Every
    node-level fact the scheduler dispatches on — the views, the
    reduce ``Axis``, the operand edges — is a STORED param on what this returns."""
    node = op
    # A term composes through operands, so a projection's node is its first edge — through every
    # zero-axis wrapper on the way (an output sweep's projection over its reduce).
    while isinstance(node, Fold) and node.axis is None and node.operands:
        node = node.operands[0]
    return node if isinstance(node, Fold) and node.axis is not None else None


def kernel_roots(op) -> tuple[Fold, ...]:
    """The reduce nodes the kernel binder builds the kernel AROUND — the ones whose ``REDUCE``
    partition it realizes. The binder peels each zero-axis projection to one operand: the
    contraction root of a tiled edge (every such root at once for a multi-output kernel), else the
    first operand; every other reduce in the tree lowers serially inside its reader, so a partition
    offered on it would price a kernel the binder never builds. This is that peel, read off the
    term alone, so the schedule projection offers the partition catalog only where it is realized."""
    node = op
    while isinstance(node, Fold) and node.axis is None and node.operands:
        tiled = [root for edge in node.operands if (root := projection_root(edge)) is not None and root.as_contraction() is not None]
        if len(tiled) > 1:
            return tuple(tiled)
        node = tiled[0] if tiled else node.operands[0]
    return (node,) if isinstance(node, Fold) else ()


def chain_members(root: Fold) -> tuple[Fold, ...]:
    """The reduce folds a kernel root's cones close over — reached from ``root`` through zero-axis
    operand edges and the axis-invariant (hoisted) reduce operands of members, deepest first, so a
    member another member's cone reads comes ahead of it. This is the CHAIN the binder emits in
    body order around one shared lane axis. A reduce read per step of another (the score inside
    the twist) lowers inside that reduce's loop and is no member, and a contraction root has no
    chain: its cone's statistic is the tiled fill's business, not a fold beside the root's."""
    out: list[Fold] = []
    if not isinstance(root, Fold) or root.axis is None or root.as_contraction() is not None:
        return ()

    def visit(node: Fold, hoisted_from: str | None) -> None:
        for edge in node.operands:
            if edge.as_slab() is not None:
                continue
            if edge.axis is None:
                visit(edge, hoisted_from)
            elif hoisted_from is None or hoisted_from not in edge.free_axes:
                visit(edge, edge.axis)
                if all(edge is not member for member in out):
                    out.append(edge)

    visit(root, root.axis)
    return tuple(out)


def chain_form(root: Fold) -> bool:
    """Whether a reduce root binds as a CHAIN — its members, or a computed provider cone hoisted
    ahead of its loop (a workspace row and its rsqrt), sit beside its own fold. The transposed
    band's σ-substitution and guarded close assume the fold stands alone at the kernel root, so a
    chain root takes no transposed band."""
    if not isinstance(root, Fold) or root.axis is None:
        return False
    if chain_members(root):
        return True
    return any(
        edge.axis is None
        and root.axis not in edge.free_axes
        and any(not isinstance(stmt, Load) for stmt in edge.lift.body)  # a computed provider, not a slab or a constant read
        for edge in root.operands
    )


def cone_stat(cone, axes: tuple) -> Fold | None:
    """The per-row STATISTIC fold of a computed-A cone — the reduce its prologue (the cone's first
    operand, the row-invariant edge) materializes first: the fold whose carried state the first
    reduce ``Loop`` of the prologue's lowering folds. ``None`` for a cone without one — the
    caller's serial fallback."""
    prologue = cone.operands[0] if isinstance(cone, Fold) and cone.axis is None and cone.operands else None
    if prologue is None:
        return None
    first = next((stmt for stmt in prologue.lower(axes=axes) if isinstance(stmt, Loop) and stmt.is_reduce), None)
    if first is None:
        return None
    carried = {stmt.name for stmt in first.body if isinstance(stmt, Accum)}
    pending = [prologue]
    while pending:
        term = pending.pop()
        if term.axis is not None and set(term.combine.results) <= carried:
            return term
        pending.extend(reversed(term.operands))
    return None


def carries_partition(tile) -> bool:
    """Whether this kernel's IR already records a realized cross-CTA split — the ``Window``
    receipt the split offer and the schedule walk's pin path read, KERNEL-scoped because that is
    the scope of the decision consumed: ``REDUCE`` is one pin and a bare one fans out to every
    kernel, so reading the receipt per axis would let the same pin split a piece again on a
    DIFFERENT reduce axis (a fused cone's per-row statistic fold — three kernels from one pinned
    split), and per-axis alone never terminates on its own axis (a pinned split re-applies to its
    own partial, halving K every sweep).

    The receipt sits on the sliced axis, and that axis is not always a NODE: a computed-A cone can
    keep its sliced contraction inside the lift as a plain ``Loop``, so a ``sites``-only scan
    misses it. Scan the loop bodies too; the receipt is in the IR either way."""

    def loops(stmts):
        for s in stmts:
            if isinstance(s, Loop):
                yield s
                yield from loops(s.body)

    for site in sites(tile.op):
        node = site.node
        ax = tile.axis_of(node.axis) if getattr(node, "axis", None) is not None else None
        if ax is not None and ax.window is not None and ax.window.partition:
            return True
        bodies = [node.lift.body, *([node.lift.body] if getattr(node, "lift", None) is not None else [])]
        if any(lp.axis.window is not None and lp.axis.window.partition for b in bodies for lp in loops(b)):
            return True
    return False


def reduce_plan(tile):
    """The tile's reduce partition (:class:`~emmy.compiler.ir.schedule.Reduce`), read from
    ``TileOp.schedule`` for the primary :class:`~emmy.compiler.ir.pure.fold.Fold` — when ``tile.op``
    is a ``Fold`` (bare, or wrapped via ``operands``), else ``None`` (a pure pointwise / scalar
    per-cell zero-axis ``Fold`` has no partition). An unscheduled fold reads the direct plan. The
    materializer's single accessor."""
    node = head(tile.op)
    if node is None:
        return None
    plan = sched_of(tile).get("REDUCE", node)
    return plan if plan is not None else Reduce()


# Kernel identity is the Loop IR a term lowers to (``TileOp.identity_key``); a term has no key of
# its own. The structural dump is NOT re-exported: it has no consumer outside ``_dump`` itself, so
# a shim here would serve nothing.

__all__ = [
    "Sched",
    "axis_names",
    "carries_partition",
    "cone_stat_dtypes",
    "edge_dtypes",
    "head",
    "chain_form",
    "chain_members",
    "kernel_roots",
    "make_cone",
    "projection_regions",
    "projection_tail",
    "reduce_plan",
    "sched_of",
]
