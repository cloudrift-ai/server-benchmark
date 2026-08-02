"""The placement CUT realizer (phase 4) — routing entries partition the recognized tree.

``PLACE@<child-path> = cut`` on an in-tree parent↔child seam splits the kernel: the child
subtree becomes its own graph node (a plain un-mapped ``LoopOp``, re-entering recognition as a
fresh tree), the seam value materializes to a workspace buffer, and the parent consumes a plain
``Load`` where the child was. Resolution is TWO-LEVEL and RECURSIVE: the ROUTING entry (a golden
whose knobs are ``PLACE`` keys only — cuts, never schedules) or an authoritative ``PLACE`` pin
decides the cut BEFORE any schedule fork is built; every resulting piece then re-recognizes on
the pass-scan restart and resolves its OWN ``(kind, shape)`` entry through the full deploy
hierarchy — a piece's entry may itself cut (the cone piece re-recognizes as the rms_norm shape
and its routing entry cuts the statistic out). NO routing entry = fuse = the recognized form —
the deployment-safety default, spelled as absence.

The realizer is seam-agnostic by design: the two seam shapes (a zero-axis ``Fold`` projection seam, a fold
operand edge) fall out of the node kinds — the child's index space is DERIVED (the enclosing
iteration axes its lowered body reads: parent free axes + ancestor fold axes), the workspace
dtype from the seam kind (a fold child's carrier state is **f32**, mirroring the split-reduce
workspace rule; a value seam keeps its leaf operand dtype — the same bytes the fused form's A
slab stored), and the piece bodies from ``ops.lower`` with loop-invariant stmts placed at the
shallowest level that defines their reads. Legality is structural (edge-iff-closed holds by
construction); an open seam cannot be spelled because ``PLACE`` sites are tree children.
"""

from __future__ import annotations

import logging

from emmy import config
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.tile.ir import (
    Fold,
    _operand_name,
    _operand_result_names,
    deep_defines,
    deep_reads,
    effect_tail,
)
from emmy.compiler.ir.tile.ops import axis_names, lower
from emmy.compiler.ir.tile.path import Site, family_sites, resolve, sites, spell
from emmy.compiler.pipeline.knob import family_of, parse_knob_spec
from emmy.compiler.pipeline.passes.loop.stamp._stamp import restamp_structural_features
from emmy.compiler.pipeline.pipeline import RuleSkipped

logger = logging.getLogger(__name__)

#: The one value that routes a rewrite; ``fuse`` (or absence) is the recognized form.
_CUT = "cut"


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
_SCHEDULE_FAMILIES = ("TILE", "STAGE", "REDUCE", "WORK", "RASTER", "WSPEC")


def _schedule_pins_live() -> bool:
    """Whether any schedule-family knob pin is live (a bare ``EMMY_<KNOB>`` var or an
    ``EMMY_KNOBS`` aggregate key). Pins are authoritative over every golden tier — a recorded
    ROUTING entry must not reroute a pinned compile: the pinned fused row would silently compile
    the cut's pieces and gate as ``realized (off)`` (the 2026-07-31 fused re-record dead end,
    where every fused golden replay failed against its own recorded spelling as soon as a
    same-shape ``.cut`` routing row landed). Bare schedule pins apply compile-wide, so this
    suppression is compile-wide too — matching ``Knob.narrow``'s bare-pin scope."""
    if any(config.knob_raw(f) is not None for f in _SCHEDULE_FAMILIES):
        return True
    return any(family_of(k) in _SCHEDULE_FAMILIES for k in parse_knob_spec(config.knobs_aggregate()))


def _card_has_routing(gpu_name, cap) -> bool:
    """Whether ANY routing golden exists for this card — the cheap gate that keeps the per-kernel
    seam scan off the common compile (no pins, no routing entries → recognition is untouched)."""
    if not gpu_name:
        return False
    try:
        from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS  # noqa: PLC0415

        return any(g.is_routing and g.gpu_name == gpu_name and tuple(g.compute_cap) == cap for g in GOLDEN_CONFIGS)
    except Exception:  # noqa: BLE001
        return False


def _has_computed_a(node) -> bool:
    """Whether the tree carries a computed-A :class:`Contraction` — an ``a`` edge stored INLINE
    (a cone node) rather than materialized (a gmem ``Load``). The structural twin of the offer
    signal ``greedy._fork_shape_key`` keys the fused convention on (only computed-A resolvers
    enumerate the ``sync`` compute-fill): at the routing consult no offer exists yet, but the
    routing reference tree does, and the edge inhabitant is the same fact."""
    if isinstance(node, Contraction) and not isinstance(node.a, Load):
        return True
    if isinstance(node, Map):
        children = node.sources
    elif isinstance(node, Fold):
        children = node.operands
    elif isinstance(node, Contraction):
        children = (node.a, *(ch.b for ch in node.channels))
    else:
        return False
    return any(_has_computed_a(c) for c in children if isinstance(c, (Map, Fold, Contraction)))


def _routing_entry(ctx, knobs: dict, root=None):
    """The live card's ROUTING golden for this kernel's ``(kind, shape)`` — fastest-first, or
    ``None``. Gated like the schedule golden tier: goldens are -O3 truth, so a correctness-lane
    (-O1) compile never consults them; off-GPU / unseeded cards read an empty set.

    The consult key follows the FUSED-KIND CONVENTION ``greedy._fork_shape_key`` documents: a
    computed-A cone's stamped histogram cannot fire ``kind="fused"`` (the statistic — when there
    is one at all — lives in the nested A-cone sub-body; a stat-free cone like the geglu→down
    edge has nothing to fire on), so a raw ``from_s_features`` key reads ``kind=""`` and misses
    every ``fused``-keyed routing entry. The fork-side rebuild reads the offer's sync-STAGE
    signal; here no offer exists yet, so the rebuild reads the TREE instead — a computed-A
    contraction is structural (``_has_computed_a``). Found live: the gemma-4 geglu→down cut
    entries never fired in-model, so every ``mlp_down`` deployed the fused computed-A form —
    the m4096 chunk-prefill TTFT regression of the 2026-07-31 article-repro session."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS  # noqa: PLC0415
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
            g
            for g in GOLDEN_CONFIGS
            if g.is_routing and g.gpu_name == gpu_name and tuple(g.compute_cap) == cap and key.joins(g.shape_key())
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
    stmts = list(lower(root))
    defs: set[str] = set()
    for s in stmts:
        defs |= deep_defines(s)
    return tuple(sorted(deep_reads(stmts) - defs - axes))


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
    probe = Load(name=_operand_name(child), input="__seam_probe", index=())
    parent_tree = _replace_edge(root, child, probe)
    sweep_axes = {st.sweep.name for st in stores if st.sweep is not None}  # boundary-store sweeps live off-term (1q)
    if _captured_values(parent_tree, axis_names(parent_tree) | sweep_axes | {a.name for a in free}):
        return False
    return True


def route_cut(ctx, knobs: dict, root, stores: tuple = (), free: tuple = ()) -> Site | None:
    """The routing resolution for a freshly-recognized kernel: the cut seam to realize, or
    ``None`` (= fuse, the default — spelled as the ABSENCE of a routing entry). ``PLACE`` pins
    are authoritative over the recorded routing entry (a ``fuse`` pin suppresses a recorded
    cut), and so is any live schedule-family pin (a pinned re-record / ``--ab`` compile keeps
    the recognized form so the pinned row can realize); a key that names no seam (or an uncuttable one) on this tree is skipped for a pin (a
    whole-model pin targets one kernel shape) and falls through with a warning for an entry
    (the drift case — deploy keeps the recognized form). A bare ``PLACE=cut`` pin takes the
    shallowest CUTTABLE seam."""
    pins = _place_pins()
    if not pins and not _card_has_routing(getattr(ctx, "gpu_name", None), tuple(getattr(ctx, "compute_capability", ()) or ())):
        return None  # nothing could ever route — skip the seam scan (the common compile)
    all_sites = sites(root)
    seams = [s for s in family_sites("PLACE", all_sites) if _cuttable(root, s, stores, free)]
    if not seams:
        return None  # a bare fold / flat cell has no in-tree seam (or none is legal)
    for key, value in pins.items():
        if key == "PLACE":
            if value == _CUT:
                return min(seams, key=lambda s: s.depth)
            return None  # bare fuse pin — authoritative
        try:
            site = resolve(root, key, all_sites=all_sites)
        except ValueError:
            continue  # the pin names no seam on THIS tree (a whole-model pin targets one kernel)
        if site is None or site not in seams:
            continue
        return site if value == _CUT else None  # an explicit fuse pin suppresses any routing entry
    if _schedule_pins_live():
        return None  # a pinned compile: the pin decides the form — recorded routing entries do not fire
    entry = _routing_entry(ctx, knobs, root)
    if entry is None:
        return None
    cuts = [k for k, v in entry.knobs.items() if str(v) == _CUT]
    if len(cuts) != 1:
        raise NotImplementedError(f"routing golden {entry.name!r}: exactly ONE cut per entry for now, got {sorted(cuts)}")
    if cuts[0] == "PLACE":
        # A bare routing cut takes the SHALLOWEST CUTTABLE seam — the same rule as the bare pin.
        # The consult tree is the PRE-fork recognized form (the fused kinds' map form), whose
        # canonical primary seam can be uncuttable (the fold) or spell differently than the
        # warp-form tree a suffixed key was recorded against; the recursion reaches the same
        # cascade from whichever legal seam goes first, so the shallowest-cuttable rule is the
        # tree-robust reading of a bare entry (measured: the cone-cut A/Bs ran exactly this).
        return min(seams, key=lambda s: s.depth)
    try:
        site = resolve(root, cuts[0], all_sites=all_sites)
    except ValueError as e:
        logger.warning("routing golden %r: %s — the recorded cut no longer names a seam; deploying the recognized form", entry.name, e)
        return None
    if site not in seams:
        logger.warning("routing golden %r: %s names an uncuttable seam — deploying the recognized form", entry.name, cuts[0])
        return None
    return site


def _child_axes(child, free: tuple, ancestors: tuple) -> list[Axis]:
    """The seam value's index space — the enclosing iteration axes the child's lowered body
    reads (its own bound axes excluded), in enclosing order: the parent placement's free axes,
    then the ancestor fold axes along the path down to the seam."""
    reads: set[str] = set()
    for s in lower(child):
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
    operand's dtype: the same bytes the fused form's A slab stored, so numerics match."""
    if isinstance(child, Fold):
        return F32
    for s in lower(child):
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


def realize_cut(match, root: Node, tile_op, free: tuple, stores: tuple, site: Site) -> Graph:
    """Split the recognized tree at ``site``'s seam into a two-kernel fragment: the CHILD piece
    computes the seam value into a workspace over its derived index space; the PARENT piece is
    the same tree with the seam edge replaced by a plain workspace ``Load``. Both pieces are
    plain un-mapped ``LoopOp``\\ s — the pass-scan restart hands them back to ``010_recognize``,
    so each re-recognizes as a fresh root, resolves its OWN routing/schedule entries, and the
    recursion terminates because trees strictly shrink. Structural features are re-stamped per
    piece (a cut consumer is a plain matmul that must join the matmul evidence, not the fused
    kind)."""
    out = root.output
    child = site.node
    child_name = _operand_name(child)
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
    child_stmts = [*lower(child), Write(output=ws, index=ws_index, value=child_name)]
    child_body = _nest(child_stmts, axes)
    child_op = LoopOp(body=Body(tuple(child_body)))

    # --- the PARENT piece: the tree with the seam edge → a workspace Load ----------------------
    load = Load(name=child_name, input=ws, index=ws_index)
    parent_tree = _replace_edge(tile_op, child, load)
    parent_cell = effect_tail(lower(parent_tree), stores)
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


__all__ = ["realize_cut", "route_cut"]
