r"""The geometry-free compute layer — node lowering and the structural reads.

A kernel's compute is one stored :class:`~emmy.compiler.ir.pure.fold.Fold`: a bare reduction, a
pure pointwise cell, or the zero-axis projection over another Fold. :func:`head` reaches the
iterating node through that projection, and every structural fact a pass dispatches on — its
derived role, reduce ``Axis``, and operand edges — comes directly from the tree. Reading those
facts off a synthesized nest is the inversion this module exists to prevent; :meth:`Fold.lower`
is for callers that consume a body.

This module holds the structural reads over a node tree — the cone seam (:func:`cone_seam`), the
iteration-space names (:func:`axis_names`) — plus the tree-path schedule accessor (:class:`Sched`),
kernel identity (:func:`structural_key`) and the worker sealing (:func:`seal_workers`). Lowering itself
has ONE spelling and it lives on the node: :meth:`Fold.lower` (a fold flattens through
:attr:`Fold.loop`, a wrapping projection appends its operand nests). Stored trees are already
resolved — a computed operand is an inline node on its edge, so there is no name-resolution step
ahead of a lowering walk."""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.ir.pure.fold import (
    Fold,
    _operand_result_names,
    deep_reads,
    edge_refs_axis,
    is_contraction,
    refs_axis,
    splice_operands,
    stmt_axis_names,
)
from emmy.compiler.ir.schedule import ReducePlan, derive_inventory
from emmy.compiler.ir.stmt import Assign, Body, Init, Load, Select
from emmy.compiler.ir.stmt.base import Stmt, dtype_promote
from emmy.compiler.ir.tile.ir import TileOp, effect_tail
from emmy.compiler.ir.tile.path import resolve, sites, spell


def cone_seam(cone, k_name: str) -> tuple[tuple, tuple, tuple[str, ...]]:
    """The computed-A cone's ``(prologue, cell, stats)`` — read off the NODE BOUNDARY, not by
    scanning stmts: the cone is ``Fold.projection(body=<the per-cell normalize>, operands=(<the row-invariant
    prologue>, <any per-cell producer>…))``, and the prologue node IS the per-row statistic (its
    own zero-axis ``Fold`` over the stat ``Fold``) plus any row-invariant cone prefix, placed there
    when the cone was built (:func:`make_cone` splits at the K seam once, structurally).

    The split is the K SEAM, on the edges as on the stmts: an edge that never indexes the
    contraction axis ``k_name`` is row-invariant and belongs to the prologue; a k-VARYING producer
    edge (the attention score contraction the cone's ``exp(s − m)`` reads) is per-cell and splices
    into the cell ahead of its first use, like any operand edge. Every fused norm→linear cone
    carries the single row-invariant edge, so its seam reads exactly as it always did.

    ``stats`` are the prologue results the cell reads — the values bridged through the stat smem
    rows. Internal definitions are excluded: the prologue and cell may independently use the same
    local SSA name. A prologue whose results go unread is dropped (nothing to bridge). The ONE seam
    both sides read: the scheduler sizes the stat rows into the sync stage's smem budget, the
    materializer fills them (``sync_stat_fill``)."""
    if not isinstance(cone, Fold) or cone.axis is not None or not cone.operands:
        return (), tuple(cone.body) if isinstance(cone, Fold) and cone.axis is None else (), ()
    varying = [edge_refs_axis(e, k_name) for e in cone.operands]
    pro = tuple(s for e, k in zip(cone.operands, varying, strict=True) if not k for s in e.lower())
    cell = splice_operands(tuple(e for e, k in zip(cone.operands, varying, strict=True) if k), tuple(cone.body))
    pro_results = {nm for edge, varies in zip(cone.operands, varying, strict=True) if not varies for nm in _operand_result_names(edge)}
    stats = tuple(sorted(pro_results & deep_reads(list(cell))))
    return (pro, cell, stats) if stats else ((), cell, ())


def edge_dtypes(edge, inputs, cache: dict[int, tuple] | None = None) -> tuple:
    """Infer a pure operand edge's result dtypes from its typed leaves and SSA program."""
    cache = {} if cache is None else cache
    if id(edge) in cache:
        return cache[id(edge)]
    if isinstance(edge, Load):
        tensor = inputs.get(edge.input) if inputs else None
        result = (tensor.dtype if tensor is not None else None,) * len(edge.names)
        cache[id(edge)] = result
        return result
    if not isinstance(edge, Fold):
        result = (None,) * len(_operand_result_names(edge))
        cache[id(edge)] = result
        return result

    env = {}
    for operand in edge.operands:
        env.update(zip(_operand_result_names(operand), edge_dtypes(operand, inputs, cache), strict=True))
    for stmt in edge.lift.body:
        if isinstance(stmt, Load):
            tensor = inputs.get(stmt.input) if inputs else None
            env.update((name, tensor.dtype if tensor is not None else None) for name in stmt.names)
        elif isinstance(stmt, Fold):
            env.update(zip(_operand_result_names(stmt), edge_dtypes(stmt, inputs, cache), strict=True))
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

    lifted = tuple(env.get(result) if isinstance(result, str) else F32 for result in edge.lift.results)
    result = lifted if edge.axis is None else lifted[: len(edge.combine.results)]
    cache[id(edge)] = result
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
    varying = {nm for n in nodes if edge_refs_axis(n, k_name) for nm in _operand_result_names(n)}
    pro: list = []
    rest = list(cell)
    while rest and not refs_axis(rest[0], k_name) and not (set(rest[0].deps()) & varying):
        pro.append(rest.pop(0))
    pro_body = Body((*sweep, *pro))
    pro_ops = () if stat is None else (stat,)
    prologue = Fold.projection(body=pro_body, operands=pro_ops)
    cell_reads = deep_reads(rest)
    bound = (*(n for e in pro_ops for n in _operand_result_names(e)), *(n for s in pro_body for n in s.defines()))
    bridged = tuple(n for n in dict.fromkeys(bound) if n in cell_reads and n not in prologue.lift.results)
    if bridged:
        prologue = Fold.projection(body=pro_body, operands=pro_ops, results=(*prologue.lift.results, *bridged))
    src = (prologue,) if (pro or sweep or stat is not None) else ()
    return Fold.projection(body=Body(tuple(rest)), operands=(*src, *nodes))


def split_invariant_factors(body: list, value: str, axis_name: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The general additive-fold factor split ``Σₖ c·xₖ = c·Σₖ xₖ``: flatten the two-arg
    ``multiply`` spine defining ``value`` over a reduce-loop body and split the leaf factor
    names into ``(c — the loop-invariant factors, names defined outside the body; the
    loop-varying leaves)``, left-to-right. The loop axis itself counts as loop-varying. The
    spine must be private to the product — a spine temp read by any other body stmt (or a
    non-binary multiply) returns ``None``, and the caller keeps the loop's current reading.
    A bare leaf is the degenerate product: ``((), (value,))``.

    The ALGEBRAIC LICENSE is a semiring fact: a factor constant along the fold axis commutes out
    of the fold because ⊗ is associative + commutative and distributes over the fold's ⊕ — the
    same reassociation category as split-K and the mul-hoist. The one registered ⊗ today is
    ``multiply`` (``ElementwiseImpl._SEMIRING``), which this helper spells directly."""
    defs: dict[str, object] = {n: s for s in body for n in s.defines()}
    spine: list[str] = []
    leaves: list[str] = []

    def flatten(n: str) -> bool:
        d = defs.get(n)
        if isinstance(d, Assign) and d.op.name == "multiply":
            if len(d.args) != 2:
                return False
            spine.append(n)
            return flatten(d.args[0]) and flatten(d.args[1])
        leaves.append(n)
        return True

    if not flatten(value):
        return None
    spine_reads = {n for n in spine if n != value}
    for s in body:
        if not (isinstance(s, Assign) and s.name in spine) and set(s.deps()) & spine_reads:
            return None
    inv = tuple(n for n in leaves if n not in defs and n != axis_name)
    return inv, tuple(n for n in leaves if n in defs or n == axis_name)


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
        if self._sites is None:
            self._sites = sites(self.root)
        return self._sites

    def key(self, family: str, node) -> str | None:
        """The canonical codec key addressing ``node`` under ``family`` — ``None`` when the node
        is not a site of that family on this tree (nothing to key)."""
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
        return self.placed(node, self.get("TILE", node))

    def placed(self, node, plan):
        """``plan`` bound to the ``(m, n)`` output axes a ``TILE`` slice at ``node``'s site tiles —
        the same one rule :meth:`tile_of` reads through, offered to a caller holding a CANDIDATE
        plan the table does not carry yet (the enumeration, whose legality predicates read the
        placed geometry). Already-placed and unplaceable plans pass through."""
        if plan is None or self.place is None or plan.axes is not None:
            return plan
        mn = self._mn_for(node)
        return plan.at(*mn) if mn is not None else plan

    def _mn_for(self, node):
        """The ``(m, n)`` output axes a ``TILE`` slice at ``node``'s site tiles, or ``None`` when
        the placement cannot supply them (an unmapped / rank-<2 grid — the caller's untiled path).

        THREE site shapes, one rule each — the geometry the retired ``Contraction`` node used to
        carry as stamped fields, now read off the tree instead:

        - a ROOT contraction, including one directly under a zero-axis projection that groups
          several kernel outputs, tiles the kernel grid's trailing pair (``Placement.root_mn``,
          the same reading the scheduler binds through at option assembly);
        - a derived unit-axis contraction inherits its parent Fold's reduction domain, so its
          result tiles the placement's trailing free pair;
        - any other nested contraction takes the free m axis and its PARENT fold's axis as n —
          read through a slice partial's window PARENT, so the view carries the pre-slice geometry
          the fragment clamps were built against.
        """
        free = tuple(self.place.free)
        site = next((s for s in self._all_sites() if s.node is node), None)
        if site is None:
            return None
        ancestors = tuple(
            candidate
            for candidate in self._all_sites()
            if len(candidate.segments) < len(site.segments) and site.segments[: len(candidate.segments)] == candidate.segments
        )

        def orient(mn):
            if mn is None or not is_contraction(node):
                return mn
            first, second = mn
            first_refs = edge_refs_axis(node.a, first.name)
            second_refs = edge_refs_axis(node.a, second.name)
            return (second, first) if second_refs and not first_refs else mn

        if site.depth == 1 or all(getattr(candidate.node, "axis", None) is None for candidate in ancestors):
            return orient(self.place.root_mn)
        if len(free) < 2:
            return None
        if site.derived and node.axis.extent.is_static and node.axis.extent.as_static() == 1:
            return orient((free[-2], free[-1]))
        parent = next((s for s in self._all_sites() if s.segments == site.segments[:-1]), None)
        ax = getattr(parent.node, "axis", None) if parent is not None else None
        if ax is None:
            return None
        return orient((free[-2], ax.window.parent if ax.window is not None else ax))


def sched_of(tile) -> Sched:
    """The :class:`Sched` view of a ``TileOp`` (binds its ``schedule`` dict to its op tree)."""
    return Sched(tile.op, tile.schedule, place=tile.place)


def scheduled(op, *, name: str, place, knobs: dict, stores: tuple = (), slices=(), schedule: dict | None = None, workers=None):
    """Build a SCHEDULED ``TileOp``: the term + placement, its schedule slices written through
    :class:`Sched` (the canonical key spelling), and the ``WORK`` inventory sealed.

    The one constructor every option builder and split realizer shares. Sealing is what makes a
    ``TileOp`` scheduled — an unsealed one carries no ``work`` and stamps no ``WORK`` knob — so
    pairing it with construction here is what stops a new builder forgetting it.

    ``slices`` are ``(family, node, value)`` triples keyed on the way in; ``schedule`` is an
    ALREADY-KEYED dict (``030_split_reduce`` re-keys against the partial's own tree before it gets
    here). ``None`` slice values are skipped, so a resolver that declined needs no guard."""
    source = Sched(op, {}, place=place)
    out = TileOp(op=op, name=name, place=place, workers=workers, knobs=knobs, schedule=dict(schedule or {}), stores=tuple(stores))
    sched = sched_of(out)
    for family, node, value in slices:
        if value is not None:
            key = source.key(family, node)
            if key is None:
                raise ValueError(f"{family} does not apply to this {type(node).__name__} — no site to key the slice on")
            sched.put(family, resolve(out.op, key).node, value)
    seal_workers(out)
    return out


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
            out |= stmt_axis_names(node.body)
        else:
            out.add(node.axis.name)
            out |= stmt_axis_names(node.step_stmts())
    return out


def projection_tail(tile) -> list[Stmt]:
    """The kernel's EFFECTFUL projection stmt stream — the root zero-axis fold's (pure) body with the
    kernel-boundary ``TileOp.stores`` reconstituted (:func:`~emmy.compiler.ir.tile.ir.effect_tail`).
    The ONE read every scheduler gate that inspects "the tail" goes through, so the
    ``b<n>t`` band's no-sweep-``Loop`` condition keeps excluding rms/softmax rows after their
    sweep moved to a ``Store`` decoration."""
    op = tile.op
    body = list(op.body) if isinstance(op, Fold) and op.axis is None else []
    return effect_tail(body, tile.stores)


def projection_regions(op: Fold, stores: tuple) -> tuple[tuple[Fold, Body, tuple], ...]:
    """Partition an independent projection's pure body and stores by producing Fold.

    Each boundary store must read exactly one root, the roots' backward cones must be disjoint,
    and together those cones must cover the projection body. This is the structural ownership
    rule shared by kernel binding and rewrites that turn one MIMO TileOp into fresh pieces.
    """
    roots = tuple(edge for edge in op.operands if isinstance(edge, Fold))
    by_name = {name: root for root in roots for name in root.defines()}
    members: dict[int, set] = {id(root): set() for root in roots}
    grouped: dict[int, list] = {id(root): [] for root in roots}
    for store in stores:
        cone = op.body.backward_cone((store.write.value,))
        used = {id(by_name[name]) for name in cone.external_reads if name in by_name}
        if len(used) != 1:
            raise ValueError("an output-tiled root must own each boundary store independently")
        owner = used.pop()
        members[owner].update(cone.members)
        grouped[owner].append(store)

    claimed: set = set().union(*members.values()) if members else set()
    if claimed != set(op.body) or any(not grouped[id(root)] for root in roots):
        raise ValueError("an output-tiled root forest must cover the complete projection")
    if any(members[id(left)] & members[id(right)] for i, left in enumerate(roots) for right in roots[i + 1 :]):
        raise ValueError("output-tiled root projections may not share tail statements")
    return tuple((root, Body(stmt for stmt in op.body if stmt in members[id(root)]), tuple(grouped[id(root)])) for root in roots)


def seal_workers(tile) -> None:
    """Derive and STAMP the kernel's ONE worker inventory (``TileOp.work`` + the ``WORK`` knob —
    the step-7 value-grammar family): the per-site ``w``/``n`` worker tokens factored out of the
    resolved ``TILE`` slices, the cooperative width off the ``REDUCE`` slices (``b512`` →
    ``t512``), and the producer band off the resolved :class:`WarpSpec` (the ``WSPEC`` absorb —
    ``+p<n>``). FAILING LOUDLY on cross-site disagreement (one kernel, one inventory). A 1-thread
    inventory (a bare register strip) keeps ``None`` — the per-cell forms' launch geometry stays
    derived. Called by every option builder / split realizer after the schedule dict is
    assembled."""
    coop = max(
        (v.coop for k, v in tile.schedule.items() if k.split("@", 1)[0] == "REDUCE"),
        default=1,
    )
    work = derive_inventory(
        (v for k, v in tile.schedule.items() if k.split("@", 1)[0] == "TILE"),
        coop=coop,
        producer=tile.workers.producer_warps if tile.workers is not None else 0,
    )
    tile.work = work
    tile.knobs["WORK"] = work.spell() if work is not None else ""


def head(op):
    """The kernel's compute NODE — a :class:`~emmy.compiler.ir.pure.fold.Fold` at any role, bare or
    under its projection (zero-axis) fold — or ``None`` for a pure pointwise cell.

    The ONE accessor for "which node is this kernel about", replacing the hand-spelled
    ``op.operands[0] if op.axis is None and op.operands else op`` ternary at every reader. Every
    node-level fact the scheduler dispatches on — the :class:`~emmy.compiler.ir.axis.AxisRole`, the
    reduce ``Axis``, the operand edges — is a STORED param on what this returns."""
    node = op
    if isinstance(op, Fold) and op.axis is None:
        if op.operands:
            node = op.operands[0]
        else:
            # The chain form's sweep case: the column fold reads the boundary store's sweep axis,
            # so root formation keeps it as the projection's one fold BODY member.
            members = [s for s in op.body if isinstance(s, Fold)]
            node = members[0] if len(members) == 1 else op
    return node if isinstance(node, Fold) and node.axis is not None else None


def reduce_plan(tile):
    """The tile's reduce partition (:class:`~emmy.compiler.ir.schedule.ReducePlan`), read from
    ``TileOp.schedule`` for the PRIMARY :class:`~emmy.compiler.ir.pure.fold.Fold` — when ``tile.op``
    is a ``Fold`` (bare, or wrapped via ``operands``), else ``None`` (a pure pointwise / scalar
    per-cell zero-axis ``Fold`` has no partition). An unstamped fold reads the empty plan (the scalar serial
    fold), matching the node field's default. The single accessor the materializer /
    ``030_split_reduce`` read."""
    node = head(tile.op)
    if node is None:
        return None
    plan = sched_of(tile).get("REDUCE", node)
    return plan if plan is not None else ReducePlan()


# Kernel identity lives in its own module (``tile/_key.py``) — it is not a compute read — and its
# ONE public name is the ``Structural`` method, ``Fold.structural_key()`` / ``TileOp.structural_key()``
# (``Op.cache_key`` / ``Graph.structural_key`` reach it there). The structural dump is NOT re-exported:
# it has no consumer outside ``_dump`` itself, so a shim here would serve nothing.

__all__ = [
    "Sched",
    "axis_names",
    "cone_seam",
    "edge_dtypes",
    "head",
    "make_cone",
    "projection_regions",
    "projection_tail",
    "reduce_plan",
    "sched_of",
    "seal_workers",
    "split_invariant_factors",
]
