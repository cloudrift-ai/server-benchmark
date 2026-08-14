"""The post-recognition placement fork and cut realizer.

``PLACE@<child-path> = cut`` on an in-tree parent↔child seam splits the kernel: the child
subtree becomes its own graph node (a plain un-mapped ``LoopOp``, re-entering recognition as a
fresh tree), the seam value materializes to a workspace buffer, and the parent consumes a plain
``Load`` where the child was. With no authoritative pin or routing entry, every structurally legal
seam is offered beside the maximal form as a structural ``PLACE`` fork. Outer search measures those choices;
a cold deploy keeps the first, fused option, while trusted evidence can price the fragment as the sum
of its independently scheduled kernels. This happens before ``020_schedule``. Resolution is recursive:
every cut piece re-enters recognition
and may offer its own placement fork.

The realizer is seam-agnostic by design: the two seam shapes (a zero-axis ``Fold`` projection seam, a fold
operand edge) fall out of the node kinds — the child's index space is DERIVED (the enclosing
iteration axes its lowered body reads: parent free axes + ancestor fold axes), the workspace
dtype from the seam kind (a fold child's carrier state is **f32**, mirroring the split-reduce
workspace rule; a value seam keeps its leaf operand dtype — the same bytes the fused form's A
slab stored), and the piece bodies from ``Fold.lower`` with loop-invariant stmts placed at the
shallowest level that defines their reads. Legality is structural (edge-iff-closed holds by
construction); an open seam cannot be spelled because ``PLACE`` sites are tree children.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from emmy import config
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp, splice_loop_ops
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write, lexical_free_values
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.tile.ir import (
    Fold,
    TileOp,
    _operand_result_names,
    deep_defines,
    deep_reads,
    effect_tail,
    is_contraction,
    operand_name,
)
from emmy.compiler.ir.tile.ops import axis_names
from emmy.compiler.ir.tile.path import Site, family_sites, resolve, sites, spell
from emmy.compiler.pipeline.fork import OptionFork
from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, family_of, parse_knob_spec
from emmy.compiler.pipeline.passes.loop.stamp._stamp import restamp_structural_features
from emmy.compiler.pipeline.pipeline import RuleSkipped

logger = logging.getLogger(__name__)

#: The one value that routes a rewrite; ``fuse`` (or absence) is the recognized form.
_CUT = "cut"
_FUSE = "fuse"
_PRODUCT_KEY = "PLACE@product"
_BROADCAST_KEY = "PLACE@broadcast"
_NESTED_KEY = "PLACE@nested"


@dataclass(frozen=True)
class _CompactBroadcastCut:
    """A root-store sweep that is virtual layout, not a workspace dimension.

    ``axis_positions`` maps the recognized kernel's compact free axes to their direct
    positions in the expanded boundary write.  The realizer uses that inverse map to
    rewrite every Loop-IR consumer load onto the compact workspace. ``body`` is the
    complete compact producer cell before that virtual sweep and ``value`` is the SSA
    value it stores. Keeping both on the alternative makes an extracted ``Store`` and a
    raw effect tail (the multi-reduction escape) realize through the same path.
    """

    axis_positions: tuple[int, ...]
    body: tuple[Stmt, ...]
    value: str
    output: str


@dataclass(frozen=True)
class _NestedReductionCut:
    """A closed contraction cone embedded in a raw nested-reduction cell.

    The raw cell is recognition's semantics-preserving escape for reductions that no current
    atom composes. ``members`` is the immediate-body cone to lift, ``container`` its one owner,
    and ``axes`` the compact materialized domain after the inner reduction axis disappears.
    """

    container: Body
    prologue: tuple[Stmt, ...]
    members: tuple[Stmt, ...]
    result: str
    axes: tuple[Axis, ...]


@dataclass(frozen=True)
class _ProductReductionCollapse:
    """A placement residue whose expanded product has one additive-reduction consumer."""

    producer: str


type _PlacementSite = Site | _CompactBroadcastCut | _NestedReductionCut | _ProductReductionCollapse


@dataclass(frozen=True)
class _PlacementAlternative:
    """One replayable structural alternative and the value that realizes it."""

    key: str
    site: _PlacementSite
    realized_value: str = _CUT


@dataclass(frozen=True)
class _PlacementDecision:
    """An authoritative placement result, or the undecided alternative set to enumerate."""

    decided: bool
    selected: _PlacementAlternative | None
    alternatives: tuple[_PlacementAlternative, ...]
    knobs: dict


def _place_pins() -> dict[str, str]:
    """The live ``PLACE`` pins — authoritative over routing entries. ``PLACE@…`` keys ride the
    ``EMMY_KNOBS`` aggregate (an ``@`` key is not a shell-var name); a bare ``EMMY_PLACE`` pin
    rides its own var and resolves like any bare family key (the primary seam)."""
    pins = {k: v for k, v in parse_knob_spec(config.knobs_aggregate()).items() if family_of(k) == "PLACE"}
    bare = config.knob_raw("PLACE")
    if bare is not None and "PLACE" not in pins:
        pins["PLACE"] = bare
    return pins


#: The schedule knob families a golden / ``--ab`` row pins. A live pin from any of them marks a
#: pinned re-record compile, where the pin — not a recorded routing entry — must decide the form.
#: ONE list (``knob.SCHEDULE_FAMILIES``): the retired ``WSPEC`` alias is gone from it because
#: nothing reads that pin any more, so treating it as live suppressed routing for no decision.


def _schedule_pins_live() -> bool:
    """Whether any schedule-family knob pin is live (a bare ``EMMY_<KNOB>`` var or an
    ``EMMY_KNOBS`` aggregate key). Pins are authoritative over every golden tier — a recorded
    ROUTING entry must not reroute a pinned compile: the pinned fused row would silently compile
    the cut's pieces and gate as ``realized (off)`` (the 2026-07-31 fused re-record dead end,
    where every fused golden replay failed against its own recorded spelling as soon as a
    same-shape ``.cut`` routing row landed). Bare schedule pins apply compile-wide, so this
    suppression is compile-wide too — matching ``Knob.narrow``'s bare-pin scope."""
    if any(config.knob_raw(f) is not None for f in SCHEDULE_FAMILIES):
        return True
    return any(family_of(k) in SCHEDULE_FAMILIES for k in parse_knob_spec(config.knobs_aggregate()))


def _has_computed_a(node) -> bool:
    """Whether the tree carries a computed-A contraction — an ``a`` edge stored INLINE (a cone
    node) rather than materialized (a gmem ``Load``). The structural twin of the offer signal
    ``greedy._fork_shape_key`` keys the fused convention on (only computed-A resolvers enumerate
    the ``sync`` compute-fill): at the routing consult no offer exists yet, but the routing
    reference tree does, and the edge inhabitant is the same fact.

    Walked through ``path.sites`` — the ONE node walk in the layer, already imported here — rather
    than a private recursion over ``operands``."""
    return any(is_contraction(s.node) and not isinstance(s.node.a, Load) for s in sites(node))


def _compact_broadcast_cut(root, stores: tuple, free: tuple) -> _CompactBroadcastCut | None:
    """Return the compact placement for a pure broadcast boundary, if one exists.

    A detached output ``Store.sweep`` is semantically virtual when the stored value is
    independent of that sweep.  Materializing the store's full domain multiplies a compact
    producer by the broadcast extent before its consumer reads one compact value per reduction
    coordinate.  Select the producer's actual free-axis domain instead.  The direct-coordinate
    gate makes the inverse rewrite exact and generic; affine/non-bijective layouts keep the
    ordinary fused representation.
    """
    if not free:
        return None

    # The common projected-reduction spelling has already split the sweep into a
    # boundary Store. A cell with several reductions deliberately remains raw Loop IR;
    # in that escape spelling the exact same boundary is the terminal non-reduce loop.
    # Recognize only a write-only terminal sweep: any computation under the loop may
    # depend on its axis and is therefore not a broadcast.
    body = tuple(root.lower())
    if len(stores) == 1 and stores[0].sweep is not None:
        sweep = stores[0].sweep
        write = stores[0].write
    elif not stores and body and isinstance(body[-1], Loop) and not body[-1].is_reduce:
        tail = body[-1]
        if len(tail.body) != 1 or not isinstance(tail.body[0], Write):
            return None
        sweep = tail.axis
        write = tail.body[0]
        body = body[:-1]
    else:
        return None

    if sweep.name in deep_reads(body):
        return None
    try:
        if sweep.extent.as_static() == 1:
            return None
    except TypeError:
        pass  # symbolic tensor extents are positive and may expand at runtime
    positions: list[int] = []
    for axis in free:
        hits = [i for i, expr in enumerate(write.index) if isinstance(expr, Var) and expr.name == axis.name]
        if len(hits) != 1:
            return None
        positions.append(hits[0])
    sweep_hits = [i for i, expr in enumerate(write.index) if isinstance(expr, Var) and expr.name == sweep.name]
    if len(sweep_hits) != 1 or sweep_hits[0] in positions:
        return None
    return _CompactBroadcastCut(tuple(positions), body, write.value, write.output)


def _is_contraction_loop(loop: Loop) -> bool:
    """Whether ``loop`` structurally carries an additive product contraction.

    Placement alternatives are algebraic and therefore identical on every target. A square/sum
    statistic has one reduction-indexed source and declines; a contraction has a multiply in an
    additive accumulator's backward cone and at least two reduction-indexed source loads.
    Hardware capability only filters schedules after a placement option has been selected.
    """
    accums = tuple(s for s in loop.body if isinstance(s, Accum) and s.op.reduce_canon == "add")
    if not accums:
        return False
    for accum in accums:
        cone = loop.body.backward_cone((accum.value,))
        cone_body = Body(cone.members)
        if not any(isinstance(s, Assign) and s.op.name == "multiply" for s in cone_body.iter()):
            continue
        loads = tuple(load for load in cone_body.loads if loop.axis.name in {name for expr in load.index for name in expr.free_vars()})
        if len(loads) < 2:
            continue
        if len(loads) >= 2:
            return True
    return False


def _nested_reduction_cut(root, free: tuple) -> _NestedReductionCut | None:
    """Find a closed inner contraction embedded in a raw nested-reduction cell.

    Recognition intentionally preserves a cell with nested non-flash reductions as raw Loop IR.
    That representation is always correct, but it has no contraction site and therefore offers
    only scalar/planar schedules. Placement may lift an inner additive product fold's closed result
    cone to a compact workspace. Pointwise projection after the fold joins the cone
    only while it introduces no new iteration axis; thus a gated activation materializes at its
    natural ``M×K`` boundary, while multiplying it by a downstream ``K×N`` weight stays in the
    residue rather than expanding the workspace to ``M×K×N``.
    """
    if not isinstance(root, Fold) or root.axis is not None:
        return None
    source_body = Body(stmt for operand in root.operands for stmt in operand.lower())
    axis_by_name = {axis.name: axis for axis in free}
    for stmt in root.body.iter():
        if isinstance(stmt, Loop):
            axis_by_name.setdefault(stmt.axis.name, stmt.axis)
    all_axis_names = frozenset(axis_by_name)

    def scan(body: Body, scopes: tuple[Body, ...]) -> _NestedReductionCut | None:
        for stmt in body:
            if not isinstance(stmt, Loop):
                continue
            nested = scan(stmt.body, (*scopes, body))
            if nested is not None:
                return nested
            if not stmt.is_reduce or not _is_contraction_loop(stmt):
                continue
            accums = tuple(s for s in stmt.body if isinstance(s, Accum) and s.op.reduce_canon == "add")
            base_axes = set().union(*(body.deps_closure.get(acc.name, frozenset()) for acc in accums)) & all_axis_names
            forward = body.forward_cone((stmt,)).members
            result = accums[0].name if len(accums) == 1 else None
            after = False
            for candidate in forward:
                if candidate is stmt:
                    after = True
                    continue
                if not after:
                    continue
                if not isinstance(candidate, Assign):
                    break
                candidate_axes = set(body.deps_closure.get(candidate.name, frozenset())) & all_axis_names
                if not candidate_axes <= base_axes:
                    break
                result = candidate.name
            if result is None:
                continue  # a multi-component fold needs a scalar projection seam
            result_deps = body.deps_closure.get(result, frozenset())
            if len(accums) > 1 and not {acc.name for acc in accums} <= result_deps:
                continue
            cone = body.backward_cone((result,))
            if stmt not in cone.members:
                continue
            member_ids = {id(member) for member in cone.members}
            consumers = tuple(
                candidate for candidate in body.iter() if id(candidate) not in member_ids and result in _member_reads(candidate)
            )
            if not any(candidate is not stmt and isinstance(candidate, (Loop, Accum)) for candidate in forward) or not body.defs_die_at(
                cone.members, roots=(result,), allowed=consumers
            ):
                continue
            axes = tuple(axis_by_name[name] for name in axis_by_name if name in (set(result_deps) & all_axis_names))
            captured = set(cone.external_reads - all_axis_names)
            groups: list[tuple[Stmt, ...]] = []
            # Close over invariant definitions in enclosing scopes.  A fused producer such as
            # RMSNorm defines its row statistic outside the projected contraction's free-N loop;
            # that is a legal, compact dependency, not a reason to retain an M×K×N scalar cell.
            # Walk scopes inside-out to resolve the names, then emit their cones outside-in.
            for scope in reversed(scopes):
                if not captured:
                    break
                outer = scope.backward_cone(captured)
                if not outer.members:
                    continue
                groups.append(outer.members)
                captured = set(outer.external_reads - all_axis_names)
            if captured:
                continue  # some value would still be read from a scope the child does not own
            prologue = tuple(member for group in reversed(groups) for member in group)
            return _NestedReductionCut(
                container=body,
                prologue=prologue,
                members=cone.members,
                result=result,
                axes=axes,
            )
        return None

    return scan(root.body, (source_body,) if source_body else ())


def _routing_entry(ctx, knobs: dict, root=None):
    """The live card's ROUTING golden for this kernel's ``(kind, shape)`` — fastest-first, or
    ``None``. Gated like the schedule golden tier: goldens are -O3 truth, so a correctness-lane
    (-O1) compile never consults them; off-GPU / unseeded cards read an empty set.

    The consult key follows the FUSED-KIND CONVENTION ``greedy._fork_shape_key`` documents: a
    computed-A cone's stamped histogram cannot always fire ``kind="fused"``: a stat-free cone
    like the geglu→down edge has no statistic or second reduce axis. The fork-side rebuild reads
    the offer's sync-STAGE signal; here no offer exists yet, so the TREE supplies the structural
    computed-A fact (``_has_computed_a``). The golden builder reads ``PLACE@a`` as the same fact,
    keeping this a single-key lookup without letting the plain key recursively match a cut piece.
    Found live: the Laguna activation→down entry measured correctly under a pin but could not
    deploy from its golden."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.base import _O3_OPT  # noqa: PLC0415

    gpu_name = getattr(ctx, "gpu_name", None)
    if not gpu_name:
        return None
    try:
        h_opt = float(ctx.features().get("H_opt", _O3_OPT))
    except Exception:  # noqa: BLE001
        h_opt = _O3_OPT
    if h_opt != _O3_OPT:
        return None  # goldens are -O3 truth — the correctness lane never routes off an entry
    try:
        key = ShapeKey.from_s_features(knobs)
        if key.kind == "" and root is not None and _has_computed_a(root):
            from dataclasses import replace  # noqa: PLC0415

            key = replace(key, kind="fused", is_warp=True)
        cap = tuple(ctx.compute_capability)
        entries = [
            g for g in GOLDEN_RECORDS if g.is_routing and g.gpu_name == gpu_name and tuple(g.compute_cap) == cap and key.joins(g.shape_key)
        ]
    except Exception:  # noqa: BLE001 — a routing consult failure must never break compile
        return None
    if not entries:
        return None
    return min(entries, key=lambda g: g.emmy_us or float("inf"))


def _captured_values(root, axes: set[str]) -> tuple[str, ...]:
    """The VALUE names ``root``'s subtree reads but does not itself define — its capture set, with
    iteration-space names (``axes``) excluded.

    DEMOTED to a validation check at 1q: operands bind POSITIONALLY to lift params, so an edge
    cannot see the fold's state or its siblings — **edge iff closed holds by construction** — and
    the cut is the one decision that still consults the scan. It is the executable statement of
    that invariant, and the honest reading of the one legal capture: flash's ``P = exp(s − m)``
    genuinely reads the online-softmax carrier's running max, updated by the merge stmts of the
    very loop step that consumes it (legal: an inline operand's one home is always inside the
    scope it captures from) — which is why that seam is not cuttable.

    Returned sorted, so callers can put it straight into an error message."""
    return tuple(sorted(lexical_free_values(root.lower(), bound=axes)))


def _cuttable(root, site: Site, stores: tuple, free: tuple) -> bool:
    """Structural cut legality — the plan's edge-iff-closed reading plus two v1 shape gates:

    - the child produces ONE component (a product fold's per-component separation is
      deliberately forfeited at tile level — the sanctioned cut for the fused edge is the
      shared ``a`` operand, never a channel);
    - the child is CLOSED over values (:func:`_captured_values` — its demoted validation role: a
      state-capturing composition like flash's ``P`` sits in the step at its semantic position
      and is simply not cuttable);
    - the seam is not the pure-copy degenerate: cutting a root zero-axis fold's only source when the
      projection body is empty and the store is a plain write leaves a parent that merely
      copies the workspace back out — the child IS the kernel, the tree does not shrink, and
      the recursion never terminates."""
    child = site.node
    if len(_operand_result_names(child)) != 1:
        return False
    if _captured_values(child, axis_names(root) | {a.name for a in free}):
        return False
    trivial_body = (isinstance(root, Fold) and root.axis is None) and not len(root.body) and all(st.sweep is None for st in stores)
    if trivial_body and any(s is child for s in root.operands):
        return False
    # The PARENT must be closed once the seam materializes: replace the child with its workspace
    # ``Load`` and require no residual free value reads. A seam whose subtree feeds the parent
    # through a SECOND dataflow path — the geglu map form, where the projection reads the up
    # channel's accumulator while the seam value is only the gate's — is not a cut: only the
    # seam value crosses the kernel boundary, so the second path's def would vanish with the
    # subtree (found live: ``Assign v9: arg 'acc1' not defined`` on the m4096 geglu cut).
    probe = Load(name=operand_name(child), input="__seam_probe", index=())
    parent_tree = _replace_edge(root, child, probe)
    sweep_axes = {st.sweep.name for st in stores if st.sweep is not None}  # boundary-store sweeps live off-term (1q)
    if _captured_values(parent_tree, axis_names(parent_tree) | sweep_axes | {a.name for a in free}):
        return False
    return True


def _placement_alternatives(root, stores: tuple, free: tuple, graph: Graph | None, graph_root: Node | None):
    """Enumerate the target-independent placement space for this recognized root.

    A product workspace left by an earlier selected cut is already a split reading. Its local
    maximal alternative is recomposition, so it forms a standalone two-sided fork and will
    reconsider any ordinary seams after the recomposed cell re-enters recognition.
    """
    product = _product_reduction_producer(graph, graph_root) if graph is not None and graph_root is not None else None
    collapse = _PlacementAlternative(_PRODUCT_KEY, _ProductReductionCollapse(product.id), _FUSE) if product is not None else None
    if collapse is not None and not any(isinstance(stmt, Accum) for stmt in product.op.body.iter()):
        return (collapse,)

    all_sites = sites(root)
    out = [
        _PlacementAlternative(spell(root, "PLACE", site.node, all_sites=all_sites), site)
        for site in family_sites("PLACE", all_sites)
        if _cuttable(root, site, stores, free)
    ]
    if (compact := _compact_broadcast_cut(root, stores, free)) is not None:
        out.append(_PlacementAlternative(_BROADCAST_KEY, compact))
    if (nested := _nested_reduction_cut(root, free)) is not None:
        out.append(_PlacementAlternative(_NESTED_KEY, nested))
    if collapse is not None and _wider_than_every_edge(graph, product):
        # A producer carrying its own nested reductions recomposes only when its workspace is
        # STATICALLY wider than every edge it bridges — the trace-native pre-reduction product
        # (a 512-token Llama MLP block materializes 120 GB there). Symbolic shapes decline
        # conservatively and keep the ordinary fork.
        out.append(collapse)
    return tuple(out)


def _wider_than_every_edge(graph: Graph, producer: Node) -> bool:
    """Whether the pair's workspace is statically larger than each edge it bridges."""

    def elems(tensor) -> int | None:
        count = 1
        for dim in tensor.shape:
            if not getattr(dim, "is_static", False):
                return None
            count *= dim.as_static()
        return count

    out = elems(producer.output)
    if out is None:
        return False
    consumer = graph.nodes[next(iter(graph.users(producer.id)))]
    edges = [node.output for inp in producer.inputs if (node := graph.nodes.get(inp)) is not None and node.output is not None]
    edges.append(consumer.output)
    return all((size := elems(tensor)) is not None and out > size for tensor in edges)


def _match_alternative(key: str, root, alternatives: tuple[_PlacementAlternative, ...]):
    """Resolve a canonical specialized key or any accepted ordinary path alias."""
    exact = next((alt for alt in alternatives if alt.key == key), None)
    if exact is not None:
        return exact
    ordinary = tuple(alt for alt in alternatives if isinstance(alt.site, Site))
    try:
        site = resolve(root, key, all_sites=sites(root))
    except ValueError:
        return None
    return next((alt for alt in ordinary if alt.site == site), None)


def _decide_values(values: dict, root, alternatives: tuple[_PlacementAlternative, ...], *, source: str):
    """Map a pin/golden row to its one active alternative and canonical decision row."""
    bare = values.get("PLACE")
    # ``PLACE`` is both the legacy bare command and a valid canonical key for the
    # unique shallowest ordinary seam.  When that exact alternative exists, read it
    # alongside every scoped key so a full row such as
    # ``PLACE=fuse,PLACE@broadcast=cut`` remains replayable.  Only trees without a
    # canonical bare seam use the legacy "primary/maximal" command semantics.
    if bare is not None and not any(alt.key == "PLACE" for alt in alternatives):
        active = next((alt for alt in alternatives if alt.realized_value == str(bare)), None)
        return active, {"PLACE": bare}

    matched: dict[str, str] = {}
    active: list[_PlacementAlternative] = []
    for key, value in values.items():
        if family_of(key) != "PLACE":
            continue
        alt = _match_alternative(key, root, alternatives)
        if alt is None:
            continue
        matched[alt.key] = str(value)
        if str(value) == alt.realized_value:
            active.append(alt)
    if len(active) > 1:
        raise NotImplementedError(f"{source}: exactly one placement alternative may be active, got {[alt.key for alt in active]}")
    return (active[0] if active else None), matched


def _placement_decision(
    ctx,
    knobs: dict,
    root,
    stores: tuple = (),
    free: tuple = (),
    graph: Graph | None = None,
    graph_root: Node | None = None,
) -> _PlacementDecision:
    """Resolve authoritative evidence, otherwise expose every algebraic alternative."""
    alternatives = _placement_alternatives(root, stores, free, graph, graph_root)
    if not alternatives:
        return _PlacementDecision(True, None, (), {})
    if pins := _place_pins():
        selected, row = _decide_values(pins, root, alternatives, source="placement pin")
        if not row:
            selected = next((alt for alt in alternatives if alt.realized_value == _FUSE), None)
        return _PlacementDecision(True, selected, alternatives, row)
    if _schedule_pins_live():
        return _PlacementDecision(True, None, alternatives, {})
    entry = _routing_entry(ctx, knobs, root)
    if entry is None:
        return _PlacementDecision(False, None, alternatives, {})
    selected, row = _decide_values(dict(entry.knobs), root, alternatives, source=f"routing golden {entry.name!r}")
    if not row:
        logger.warning("routing golden %r no longer names a placement alternative; using the maximal form", entry.name)
        selected = next((alt for alt in alternatives if alt.realized_value == _FUSE), None)
    return _PlacementDecision(True, selected, alternatives, row)


def _stamp_decision(graph: Graph, row: dict) -> None:
    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op.decision_knobs.update(row)


def placement_options(ctx, knobs: dict, match, root: Node, tree, free: tuple, stores: tuple, fused):
    """Return authoritative placement or maximal-first replayable structural alternatives."""
    decision = _placement_decision(ctx, knobs, tree, stores, free, match.graph, root)
    if decision.decided:
        if decision.selected is not None:
            placed = realize_cut(match, root, tree, free, stores, decision.selected.site)
            if placed is not None:
                _stamp_decision(placed, decision.knobs)
                return placed
        return replace(fused, decision_knobs={**fused.decision_knobs, **decision.knobs}) if decision.knobs else fused

    original_consumed, original_output = set(match.consumed), match.output
    realized = []
    for alternative in decision.alternatives:
        match.consumed, match.output = set(original_consumed), original_output
        try:
            placed = realize_cut(match, root, tree, free, stores, alternative.site)
        except (RuleSkipped, ValueError) as exc:
            logger.debug("placement: alternative %s is not realizable: %s", alternative.key, exc)
            continue
        if placed is not None:
            realized.append((alternative, placed, frozenset(match.consumed), match.output))
    match.consumed, match.output = original_consumed, original_output
    if not realized:
        return fused

    alternatives = tuple(alt for alt, _graph, _consumed, _output in realized)
    base_row = {alt.key: (_CUT if alt.realized_value == _FUSE else _FUSE) for alt in alternatives}
    structural = {key: value for key, value in knobs.items() if key.startswith("S_")}
    fused = replace(fused, decision_knobs={**fused.decision_knobs, **base_row})
    base = OptionFork(option=fused, knobs={**structural, **base_row})
    realized_options = []
    for alternative, placed, consumed, output in realized:
        row = {**base_row, alternative.key: alternative.realized_value}
        _stamp_decision(placed, row)
        realized_options.append(
            OptionFork(
                option=placed,
                knobs={**structural, **row},
                consumed=consumed,
                output=output,
            )
        )
    # Only a repair may lead the cold option 0: a re-fusing recomposition (spelled ``fuse``)
    # strictly shrinks memory, and the nested-reduction lift strictly shrinks work (the raw
    # fused reading replays the inner fold per outer cell — the reading recognition keeps only
    # as a functional fallback). An ordinary value-seam ``cut`` materializes its seam value,
    # which for an accumulator seam is the whole pre-reduction product; letting it lead cold
    # turned a 512-token MLP block into a 120 GB workspace. Ordinary cuts stay behind the
    # maximal fused base and win only by evidence.
    def leads_cold(alt: _PlacementAlternative) -> bool:
        return alt.realized_value == _FUSE or alt.key == _NESTED_KEY

    repairs = [option for alt, option in zip(alternatives, realized_options) if leads_cold(alt)]
    cuts = [option for alt, option in zip(alternatives, realized_options) if not leads_cold(alt)]
    return [*repairs, base, *cuts]


def _child_axes(child, free: tuple, ancestors: tuple) -> list[Axis]:
    """The seam value's index space — the enclosing iteration axes the child's lowered body
    reads (its own bound axes excluded), in enclosing order: the parent placement's free axes,
    then the ancestor fold axes along the path down to the seam."""
    reads: set[str] = set()
    for s in child.lower():
        reads |= _member_reads(s)
    return [a for a in (*free, *ancestors) if a.name in reads]


def _ancestor_axes(root, child) -> tuple[Axis, ...]:
    """The fold axes on the path from ``root`` down to (excluding) ``child``, outer→inner."""

    def walk(node, acc: tuple[Axis, ...]) -> tuple[Axis, ...] | None:
        if node is child:
            return acc
        edges = node.operands if isinstance(node, Fold) else ()
        # A ZERO-AXIS node contributes no fold axis to the path (it does not iterate).
        below = (*acc, node.axis) if isinstance(node, Fold) and node.axis is not None else acc
        for e in edges:
            if isinstance(e, Fold):
                got = walk(e, below)
                if got is not None:
                    return got
        return None

    got = walk(root, ())
    assert got is not None, "cut seam is not a child of this tree"
    return got


def _stmt_level(stmts: list[Stmt], axes: list[Axis]) -> list[int]:
    """The nesting level of each stmt — the deepest axis it (transitively) reads: stmts join the
    shallowest loop that defines everything they read, so a row-invariant statistic runs once per
    row while the per-cell remainder rides the inner sweep (loop-invariant placement, derived)."""
    pos = {a.name: i + 1 for i, a in enumerate(axes)}
    level_of: dict[str, int] = {}
    out: list[int] = []
    for s in stmts:
        reads = _member_reads(s)
        lvl = max([pos.get(n, 0) for n in reads] + [level_of.get(n, 0) for n in reads] + [0])
        for d in s.defines():
            level_of[d] = lvl
        for b in s.nested():
            for c in b.iter():
                for d in c.defines():
                    level_of[d] = lvl
        out.append(lvl)
    return out


def _nest(stmts: list[Stmt], axes: list[Axis]) -> list[Stmt]:
    """Nest ``stmts`` under loops over ``axes`` (outer→inner), each stmt at its derived level."""
    if not axes:
        return list(stmts)
    levels = _stmt_level(stmts, axes)
    body: list[Stmt] = [s for s, lvl in zip(stmts, levels, strict=True) if lvl >= len(axes)]
    for depth in range(len(axes) - 1, -1, -1):
        body = [*(s for s, lvl in zip(stmts, levels, strict=True) if lvl == depth), Loop(axis=axes[depth], body=Body(tuple(body)))]
    return body


def _ws_dtype(child, inputs: dict):
    """The seam workspace dtype: a fold child bridges raw carrier STATE — **f32**, the
    split-reduce workspace rule (a reduced statistic must not round-trip through the output
    dtype) — while a value seam (a zero-axis ``Fold`` child — the cone's per-cell normalize) keeps its leaf
    operand's dtype: the same bytes the fused form's A slab stored, so numerics match. In the
    one-kind IR "fold child" means the child FOLDS AN AXIS: a zero-axis projection is the value
    seam, whatever reduces ride inside its operands. (``isinstance(child, Fold)`` alone matches
    every node since the ``Map``/``Contraction`` merge — that spelling made every seam f32, and
    an f32 A cannot ride the warp atoms bare, so every cut consumer re-wrapped into a demoting
    cone, keyed ``fused``, and deployed the parent's fused golden instead of its own matmul row.)"""
    if isinstance(child, Fold) and child.axis is not None:
        return F32
    for s in child.lower():
        for ld in Body(tuple([s])).loads:
            t = (inputs or {}).get(ld.input)
            if t is not None:
                return t.dtype
    return F32


def _replace_edge(node, child, load: Load):
    """The parent tree with the seam child replaced by ``load`` — the cut terminal (every edge
    admits ``Load``). Positional bindings hold: the load defines the child's bound name."""
    from dataclasses import replace as _dc_replace  # noqa: PLC0415

    if not isinstance(node, Fold):
        return node
    # ONE arm for the one stored kind: every reading's edges are ``operands``, so replacing the
    # seam is the same rewrite whether the node reads as a map, a reduce or a contraction.
    if any(e is child for e in node.operands):
        return _dc_replace(node, operands=tuple(load if e is child else e for e in node.operands))
    return _dc_replace(node, operands=tuple(_replace_edge(e, child, load) if isinstance(e, Fold) else e for e in node.operands))


def _replace_nested_region(body: Body, cut: _NestedReductionCut, load: Load) -> Body:
    """Replace ``cut``'s one immediate-body cone with its workspace load.

    The lifted child owns a copy of ``cut.prologue``, but those definitions are not
    necessarily child-exclusive.  A later sibling in the parent can still read one of
    them (for example, Q/K normalization constants and the normalized Q value shared by
    consecutive attention reductions).  Remove only the transitive prologue slice that
    is dead after replacing the cone; retaining every prologue statement would preserve
    correctness but can leave the very nested reduction that placement is meant to split.
    """
    prologue_ids = {id(member) for member in cut.prologue}
    member_ids = {id(member) for member in cut.members}
    first = id(cut.members[0])

    def rewrite(region: Body, drop: set[int]) -> Body:
        out: list[Stmt] = []
        for stmt in region:
            stmt_id = id(stmt)
            if stmt_id in drop:
                continue
            if region is cut.container and stmt_id == first:
                out.append(load)
            if region is cut.container and stmt_id in member_ids:
                continue
            nested = stmt.nested()
            out.append(stmt if not nested else stmt.with_bodies(tuple(rewrite(child, drop) for child in nested)))
        return Body(out)

    # Compute residue liveness with the entire duplicated prologue absent.  A name that
    # resurfaces as an external read is a real parent dependency.  Walk the original
    # prologue backwards (SSA/topological order) to retain its complete dependency slice.
    stripped = rewrite(body, prologue_ids)
    needed = set().union(*(_member_reads(stmt) for stmt in stripped)) if stripped else set()
    keep: set[int] = set()
    for stmt in reversed(cut.prologue):
        defines = deep_defines(stmt)
        if not defines & needed:
            continue
        keep.add(id(stmt))
        needed.difference_update(defines)
        needed.update(_member_reads(stmt))
    return rewrite(body, prologue_ids - keep)


def _realize_nested_reduction(match, root: Node, tile_op, free: tuple, stores: tuple, cut: _NestedReductionCut) -> Graph:
    """Lift one raw nested contraction cone to a compact materialized workspace."""
    out = root.output
    ws = f"{out.name}__cut_{cut.result}"
    if ws in match.graph.nodes:
        raise RuleSkipped(f"nested-reduction seam already cut — {ws} exists")
    ws_index = tuple(Var(axis.name) for axis in cut.axes)
    child_cell = [*cut.prologue, *cut.members, Write(output=ws, index=ws_index, value=cut.result)]
    child_op = LoopOp(body=Body(tuple(_nest(child_cell, list(cut.axes)))))

    load = Load(name=cut.result, input=ws, index=ws_index)
    rewritten = _replace_nested_region(tile_op.body, cut, load)
    # ``tile_op.lift`` is the impure raw-cell lambda built by ``Fold.projection``'s
    # Loop-IR adapter.  Rebuild through that adapter: dataclass-replacing the Lambda
    # invokes its public purity validator and rejects the deliberately impure body.
    rewritten_reads = deep_reads(rewritten)
    live_operands = tuple(operand for operand in tile_op.operands if set(_operand_result_names(operand)) & rewritten_reads)
    parent_tree = Fold.projection(body=rewritten, operands=live_operands)
    parent_cell = Body(effect_tail(parent_tree.lower(), stores))
    parent_op = LoopOp(body=Body(tuple(_nest(list(parent_cell), list(free)))))

    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    child_reads = {ld.input for ld in child_op.body.loads if ld.input != ws}
    parent_reads = {ld.input for ld in parent_op.body.loads if ld.input != ws}
    frag.add_node(
        op=child_op,
        inputs=[inp for inp in root.inputs if inp in child_reads],
        # The lifted value replaces a tensor boundary that fusion had inlined.  Preserve that
        # boundary dtype (including its rounding semantics); an arbitrary source leaf may be a
        # wider statistic even when the projected activation is f16.
        output=Tensor(ws, tuple(axis.extent for axis in cut.axes), out.dtype),
        node_id=ws,
    )
    frag.add_node(
        op=parent_op,
        inputs=[*(inp for inp in root.inputs if inp in parent_reads), ws],
        output=Tensor(out.name, out.shape, out.dtype),
        node_id=out.name,
    )
    frag.outputs = [out.name]
    for node_id in (ws, out.name):
        restamp_structural_features(frag.nodes[node_id].op, frag)
    logger.info(
        "placement: materializing nested contraction result %s on compact axes %s before scheduling residue %s",
        cut.result,
        tuple(axis.name for axis in cut.axes),
        out.name,
    )
    return frag


def _compact_consumer(op: LoopOp, old: str, ws: str, positions: tuple[int, ...], old_output: str, new_output: str) -> LoopOp | None:
    """Redirect one expanded-buffer consumer onto the compact coordinate projection."""
    found = False

    def rewrite_stmt(stmt):
        nonlocal found
        if isinstance(stmt, Load) and stmt.input == old:
            if any(pos >= len(stmt.index) for pos in positions):
                raise ValueError("expanded consumer index is shorter than the compact placement map")
            found = True
            return Load(names=stmt.names, input=ws, index=tuple(stmt.index[pos] for pos in positions), dtype=stmt.dtype)
        if isinstance(stmt, Write) and stmt.output == old:
            raise ValueError("expanded buffer is also written by a consumer")
        if isinstance(stmt, Write) and stmt.output != old_output:
            return stmt
        if isinstance(stmt, Write):
            return replace(stmt, output=new_output)
        return stmt

    body = op.body.map(rewrite_stmt)
    return LoopOp(body=body) if found else None


def _realize_compact_broadcast(match, root: Node, tile_op, free: tuple, stores: tuple, cut: _CompactBroadcastCut) -> Graph | None:
    """Materialize the value domain before a virtual boundary broadcast.

    The expanded buffer is internal-only and every consumer must still be Loop IR so its loads
    can be inverted through the direct boundary-write map.  If either condition is false the
    recognized fused form is retained; placement never changes a graph ABI or guesses at an
    opaque consumer layout.
    """
    graph = match.graph
    old = root.output.name
    if old in graph.outputs or cut.output != old:
        return None
    users = sorted(graph.buffer_users(old))
    consumers: list[tuple[Node, LoopOp, str]] = []
    for consumer_id in users:
        consumer = graph.nodes[consumer_id]
        if not isinstance(consumer.op, LoopOp) or len(consumer.outputs) != 1:
            return None
        compact_id = f"{consumer_id}__compact_consumer"
        try:
            compact_op = _compact_consumer(
                consumer.op,
                old,
                f"{old}__cut_compact",
                cut.axis_positions,
                consumer.output.name,
                compact_id,
            )
        except ValueError:
            return None
        if compact_op is None:
            return None
        compact_op.source = consumer.op
        consumers.append((consumer, compact_op, compact_id))
    if not consumers:
        return None

    ws = f"{old}__cut_compact"
    if ws in graph.nodes:
        raise RuleSkipped(f"compact placement already exists — {ws}")
    axes = list(free)
    ws_index = tuple(Var(axis.name) for axis in axes)
    child_stmts = [*cut.body, Write(output=ws, index=ws_index, value=cut.value)]
    child_op = LoopOp(body=Body(tuple(_nest(child_stmts, axes))))

    frag = Graph()
    external = dict.fromkeys(
        inp
        for op in (child_op, *(compact_op for _consumer, compact_op, _cid in consumers))
        for inp in (load.input for load in op.body.loads)
        if inp != ws
    )
    for inp in external:
        tensor = graph.buffer(inp)
        if tensor is None:
            return None
        frag.add_node(op=InputOp(), inputs=[], output=tensor, node_id=inp)
    frag.add_node(
        op=child_op,
        inputs=list(dict.fromkeys(load.input for load in child_op.body.loads if load.input != ws)),
        output=Tensor(ws, tuple(axis.extent for axis in axes), root.output.dtype),
        node_id=ws,
    )
    output_map: dict[str, str] = {}
    for consumer, compact_op, compact_id in consumers:
        inputs = list(dict.fromkeys(load.input for load in compact_op.body.loads))
        frag.add_node(
            op=compact_op,
            inputs=inputs,
            output=Tensor(consumer.output.name, consumer.output.shape, consumer.output.dtype),
            node_id=compact_id,
        )
        output_map[consumer.id] = compact_id
    frag.outputs = list(output_map.values())
    match.consumed = {root.id, *output_map}
    match.output = output_map
    for nid in (ws, *output_map.values()):
        restamp_structural_features(frag.nodes[nid].op, frag)
    logger.info(
        "placement: compacting broadcast workspace %s from %s to %s before %d consumer(s)",
        old,
        root.output.shape,
        tuple(axis.extent for axis in axes),
        len(consumers),
    )
    return frag


def _product_reduction_pair(graph: Graph, producer: Node) -> Node | None:
    """The sole additive-reduction consumer of a virtual wide product, if exact.

    The product may be exposed directly by maximal loop fusion (an internal f32 carrier) or by
    an earlier placement boundary (a ``__cut_`` input). A real narrow tensor product is not a
    reassociable carrier and therefore remains materialized. A producer carrying its own nested
    reductions (a fused projection chain feeding the product) recomposes too: the collapsed cell
    re-enters recognition, where the nested-reduction lift leads the cold fork, so the inner
    folds never replicate per output cell.
    """
    virtual_product = producer.output.dtype is F32 or any("__cut_" in inp for inp in producer.inputs)
    if not virtual_product or len(graph.users(producer.id)) != 1:
        return None
    if "__cut_" in producer.output.name:
        # A workspace a placement decision just materialized (a lifted nested fold) must not
        # re-merge into its consumer: the lift and the recomposition are inverse rewrites, and
        # letting each lead the next fork alternates them forever.
        return None
    consumer_id = next(iter(graph.users(producer.id)))
    consumer = graph.nodes[consumer_id]
    consumer_loop = _placed_loop(consumer)
    if not isinstance(producer.op, LoopOp) or consumer_loop is None:
        return None
    if not any(isinstance(stmt, Assign) and stmt.op.name == "multiply" for stmt in producer.op.body.iter()):
        return None
    direct = {load.name for load in consumer_loop.body.loads if load.input == producer.id}
    if not any(accum.op.reduce_canon == "add" and accum.value in direct for accum in consumer_loop.body.accums):
        return None
    return consumer


def _product_reduction_producer(graph: Graph, node: Node) -> Node | None:
    """The placement-product endpoint of ``node``'s exact product→sum pair.

    Recognition is independently queued for both graph endpoints.  The consumer can reach
    placement before a rewritten producer is revisited; requiring the producer match to win that
    race lets the consumer freeze an avoidable scalar grid split first.  Resolve the same proven
    pair from either endpoint so recomposition is scheduler-order invariant.
    """
    if _product_reduction_pair(graph, node) is not None:
        return node
    for inp in node.inputs:
        producer = graph.producer(inp)
        if producer is not None and _product_reduction_pair(graph, producer) is node:
            return producer
    return None


def _placed_loop(node: Node) -> LoopOp | None:
    """Return the semantic Loop IR for an unscheduled or already-placed graph node.

    Recognition scans nodes independently.  By the time a placement residue re-enters the
    pass, its downstream reduction may already be a ``TileOp``.  Reconstituting that tile's
    structural term, boundary stores, and free-axis loops is lossless: schedule fields do not
    change the algebra and are intentionally discarded when the merged loop re-enters
    recognition.  A store-less tile declines because the pairwise splicer needs a concrete
    boundary write to solve the producer-to-consumer coordinates.
    """
    if isinstance(node.op, LoopOp):
        return node.op
    if not isinstance(node.op, TileOp) or node.op.op is None:
        return None
    cell = effect_tail(node.op.op.lower(), node.op.stores)
    if not any(write.output == node.output.name for write in Body(cell).writes):
        return None
    loop = LoopOp(body=Body(tuple(_nest(list(cell), list(node.op.place.free)))))
    return loop


def _collapse_materialized_product_reduction(match, producer: Node) -> Graph | None:
    """Reconstitute a contraction split by a virtual product workspace.

    A compact activation cut can leave the residue as ``A[K] * B[K,N] -> P[K,N]`` followed by
    one additive reduction of ``P``.  The workspace is correct but defeats contraction
    recognition and launches a bandwidth-bound product plus reduction.  This is a placement
    repair, not fusion policy: only a virtual wide carrier with exactly one Loop consumer is
    eligible, and the generic Loop splicer must prove the coordinate substitution. A real narrow
    product tensor remains a semantic boundary.
    The merged LoopOp re-enters recognition as one product-reduction cell (and binds a
    contraction atom whenever its output geometry admits one).
    """
    consumer = _product_reduction_pair(match.graph, producer)
    if consumer is None:
        return None
    consumer_loop = _placed_loop(consumer)
    if consumer_loop is None:
        return None
    merged = splice_loop_ops(producer.op, consumer_loop, producer.id)
    if merged is None:
        return None

    frag = Graph()
    reads = list(dict.fromkeys(load.input for load in merged.body.loads))
    for inp in reads:
        tensor = match.graph.buffer(inp)
        if tensor is None:
            return None
        frag.add_node(op=InputOp(), inputs=[], output=tensor, node_id=inp)
    frag.add_node(
        op=merged,
        inputs=reads,
        output=Tensor(consumer.output.name, consumer.output.shape, consumer.output.dtype),
        node_id=consumer.id,
    )
    frag.outputs = [consumer.id]
    match.consumed = {producer.id, consumer.id}
    match.output = consumer.id
    restamp_structural_features(merged, frag)
    logger.info(
        "placement: collapsing product workspace %s into its sole additive reduction %s",
        producer.id,
        consumer.id,
    )
    return frag


def realize_cut(match, root: Node, tile_op, free: tuple, stores: tuple, site: _PlacementSite) -> Graph | None:
    """Split the recognized tree at ``site``'s seam into a two-kernel fragment: the CHILD piece
    computes the seam value into a workspace over its derived index space; the PARENT piece is
    the same tree with the seam edge replaced by a plain workspace ``Load``. Both pieces are
    plain un-mapped ``LoopOp``\\ s — the pass-scan restart hands them back to ``010_recognize``,
    so each re-recognizes as a fresh root, resolves its OWN routing/schedule entries, and the
    recursion terminates because trees strictly shrink. Structural features are re-stamped per
    piece (a cut consumer is a plain matmul that must join the matmul evidence, not the fused
    kind)."""
    if isinstance(site, _ProductReductionCollapse):
        producer = match.graph.nodes.get(site.producer)
        return _collapse_materialized_product_reduction(match, producer) if producer is not None else None
    if isinstance(site, _CompactBroadcastCut):
        return _realize_compact_broadcast(match, root, tile_op, free, stores, site)
    if isinstance(site, _NestedReductionCut):
        return _realize_nested_reduction(match, root, tile_op, free, stores, site)

    out = root.output
    child = site.node
    child_name = operand_name(child)
    ws = f"{out.name}__cut_{child_name}"
    if ws in match.graph.nodes:
        raise RuleSkipped(f"seam already cut — {ws} exists")
    anc = _ancestor_axes(tile_op, child)
    axes = _child_axes(child, tuple(free), anc)
    ws_index = tuple(Var(a.name) for a in axes)
    ws_dtype = _ws_dtype(child, getattr(root.op, "inputs", None) or {})
    spelled = spell(tile_op, "PLACE", child, all_sites=sites(tile_op))
    logger.info("placement: cutting %s (%s → %s + residue) on %s", spelled, root.id, ws, out.name)

    # --- the CHILD piece: the seam subtree, its value stored to the workspace ------------------
    child_stmts = [*child.lower(), Write(output=ws, index=ws_index, value=child_name)]
    child_body = _nest(child_stmts, axes)
    child_op = LoopOp(body=Body(tuple(child_body)))

    # --- the PARENT piece: the tree with the seam edge → a workspace Load ----------------------
    load = Load(name=child_name, input=ws, index=ws_index)
    parent_tree = _replace_edge(tile_op, child, load)
    parent_cell = effect_tail(parent_tree.lower(), stores)
    parent_body = _nest(list(parent_cell), list(free))
    parent_op = LoopOp(body=Body(tuple(parent_body)))

    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    child_reads = {ld.input for ld in child_op.body.loads if ld.input != ws}
    parent_reads = {ld.input for ld in parent_op.body.loads if ld.input != ws}
    frag.add_node(
        op=child_op,
        inputs=[i for i in root.inputs if i in child_reads],
        output=Tensor(ws, tuple(a.extent for a in axes), ws_dtype),
        node_id=ws,
    )
    frag.add_node(
        op=parent_op,
        inputs=[*(i for i in root.inputs if i in parent_reads), ws],
        output=Tensor(out.name, out.shape, out.dtype),
        node_id=out.name,
    )
    frag.outputs = [out.name]
    for nid in (ws, out.name):
        restamp_structural_features(frag.nodes[nid].op, frag)
    return frag


__all__ = ["placement_options", "realize_cut"]
