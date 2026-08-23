"""The placement CUT realizer — a ``PLACE`` pin partitions the recognized tree.

``PLACE@<child-path> = cut`` on an in-tree parent↔child seam splits the kernel: the child
subtree becomes its own graph node (a plain un-mapped ``LoopOp``, re-entering recognition as a
fresh tree), the seam value materializes to a workspace buffer, and the parent consumes a plain
``Load`` where the child was. Resolution is RECURSIVE: a pin decides the cut BEFORE any schedule
fork is built; every resulting piece then re-recognizes on the pass-scan restart — a piece may
itself be cut by a deeper pin key. NO pin = fuse = the recognized form, spelled as absence. Pins
are the codec's exploration mechanism (``--ab`` and tune trajectories); a pass consults no
deploy evidence.

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

import numpy as np

from emmy import config
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, deep_defines, deep_reads, operand_name
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.ir.tile.ir import effect_tail
from emmy.compiler.ir.tile.ops import axis_names
from emmy.compiler.ir.tile.path import Site, family_sites, resolve, sites, spell
from emmy.compiler.pipeline.knob import consume_kernel_row, family_of, parse_knob_spec
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


def _captured_values(root, axes: set[str]) -> tuple[str, ...]:
    """The VALUE names ``root``'s subtree reads but does not itself define — its capture set, with
    iteration-space names (``axes``) excluded.

    DEMOTED to a validation check at 1q: operands bind POSITIONALLY to lift params, so an edge
    cannot see the fold's state or its siblings — **edge iff closed holds by construction** — and
    the cut is the one decision that still consults the scan. It is the executable statement of
    that invariant. No stored term captures: the computed-A cone passes every statistic value its
    per-cell normalize reads through the prologue's results (``ops.make_cone``), so softmax's
    ``exp(s − m)`` binds ``m`` positionally; only a hand-built tree can fail this check.

    Returned sorted, so callers can put it straight into an error message."""
    stmts = list(root.lower())
    defs: set[str] = set()
    for s in stmts:
        defs |= deep_defines(s)
    return tuple(sorted(deep_reads(stmts) - defs - axes))


def _cuttable(root, site: Site, stores: tuple, free: tuple) -> bool:
    """Structural cut legality — the plan's edge-iff-closed reading plus two v1 shape gates:

    - the child produces ONE component (a product fold's per-component separation is
      deliberately forfeited at tile level — the sanctioned cut for the fused edge is the
      shared ``a`` operand, never a channel);
    - the child is CLOSED over values (:func:`_captured_values` — its demoted validation role);
    - the seam is not the pure-copy degenerate: cutting a root zero-axis fold's only source when the
      projection body is empty and the store is a plain write leaves a parent that merely
      copies the workspace back out — the child IS the kernel, the tree does not shrink, and
      the recursion never terminates."""
    child = site.node
    if len(_operand_result_names(child)) != 1:
        return False
    # A boundary store's SWEEP axis is an iteration axis of the kernel (it lives off-term on the
    # ``Store``): a child under the sweep reads it as a coordinate, never as a captured value —
    # the cut piece iterates it as one of its own free axes (:func:`realize_cut`).
    sweep_axes = {st.sweep.name for st in stores if st.sweep is not None}
    if _captured_values(child, axis_names(root) | {a.name for a in free} | sweep_axes):
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
    if _captured_values(parent_tree, axis_names(parent_tree) | sweep_axes | {a.name for a in free}):
        return False
    return True


def cuttable_seams(root, stores: tuple = (), free: tuple = ()) -> list[Site]:
    """Every legal ``PLACE`` seam on this tree, shallowest first — the placement fork's option
    list and the pin resolver's candidate set, one derivation."""
    all_sites = sites(root)
    return sorted((s for s in family_sites("PLACE", all_sites) if _cuttable(root, s, stores, free)), key=lambda s: s.depth)


def route_cut(ctx, knobs: dict, root, stores: tuple = (), free: tuple = ()) -> tuple[str | None, Site | None]:  # noqa: ARG001 — ctx/knobs kept for the rewrite-rule call signature
    """The ``PLACE`` pin resolution for a freshly-recognized kernel: ``("cut", seam)`` when a pin
    cuts, ``("fuse", None)`` when a pin names this tree and keeps it fused (authoritative — no
    placement fork is offered), ``(None, None)`` when no pin decides (the placement FORK owns the
    choice). A key that names no seam (or an uncuttable one) on this tree is skipped (a
    whole-model pin targets one kernel shape). A bare ``PLACE=cut`` pin takes the shallowest
    CUTTABLE seam."""
    pins = _place_pins()
    if not pins:
        return None, None
    all_sites = sites(root)
    seams = [s for s in family_sites("PLACE", all_sites) if _cuttable(root, s, stores, free)]
    if not seams:
        return None, None  # a bare fold / flat cell has no in-tree seam (or none is legal)
    for key, value in pins.items():
        if key == "PLACE":
            if value == _CUT:
                return _CUT, min(seams, key=lambda s: s.depth)
            return "fuse", None  # bare fuse pin — authoritative
        try:
            site = resolve(root, key, all_sites=all_sites)
        except ValueError:
            continue  # the pin names no seam on THIS tree (a whole-model pin targets one kernel)
        if site is None or site not in seams:
            continue
        return (_CUT, site) if value == _CUT else ("fuse", None)
    return None, None


def _is_contraction_operand(root, child) -> bool:
    """Whether ``child`` is the ``a`` edge or a channel's ``b`` edge of some contraction in the tree."""
    from emmy.compiler.ir.pure.fold import is_contraction  # noqa: PLC0415

    def walk(node) -> bool:
        if not isinstance(node, Fold):
            return False
        if is_contraction(node) and (node.a is child or any(ch.b is child for ch in node.channels)):
            return True
        members = (*node.operands, *(m for m in node.body if isinstance(m, Fold))) if node.axis is None else node.operands
        return any(walk(e) for e in members if isinstance(e, Fold))

    return walk(root)


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
        if not isinstance(node, Fold):
            return None
        # A ZERO-AXIS node contributes no fold axis to the path (it does not iterate). Its body
        # members that are nodes (a chained column fold reading the store's sweep) are children
        # like its operand edges.
        below = (*acc, node.axis) if node.axis is not None else acc
        members = (*node.operands, *(m for m in node.body if isinstance(m, Fold))) if node.axis is None else node.operands
        for e in members:
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
    dtype) — while a value seam (a zero-axis ``Fold`` child — the cone's per-cell normalize) holds
    the seam VALUE, so it takes that value's dtype: the same bytes the fused form's A slab stored,
    so numerics match. In the one-kind IR "fold child" means the child FOLDS AN AXIS: a zero-axis
    projection is the value seam, whatever reduces ride inside its operands. (``isinstance(child,
    Fold)`` alone matches every node since the ``Map``/``Contraction`` merge — that spelling made
    every seam f32, and an f32 A cannot ride the warp atoms bare, so every cut consumer re-wrapped
    into a demoting cone, keyed ``fused``, and deployed the parent's fused golden instead of its
    own matmul row.)

    The value's own dtype is the defining statement's when it CONVERTS (a coded-weight decode
    cone reads integer tables and produces f16 — reading the leaf dtype there stores the fp16
    value through an ``int`` workspace and rounds every element to an integer). Only a
    conversion carries an explicit dtype this early; everything else passes its leaf operand
    through unconverted, which is what the leaf lookup below answers.
    """
    if isinstance(child, Fold) and child.axis is not None:
        return F32
    lowered = child.lower()
    seam = Body(tuple(lowered)).definitions.get(operand_name(child))
    if (converted := getattr(seam, "dtype", None)) is not None:
        return converted
    # An INTEGER leaf never types the seam. Types are stamped later (``030_stamp_types`` runs
    # ``dtype_promote`` over the body), so all this can see is what the cone READS — and a decode
    # cone reads nothing but codes, shifts and masks while computing a real value from them. Taking
    # the first leaf typed the reconstruction seam ``i32``, and the producer's ``__half`` was then
    # stored through an ``int*`` workspace: every decoded weight rounded to an integer, a
    # whole-model ``max_diff`` of 0.27 against a 0.005 gate. Only an explicit declared dtype (the
    # branch above) may make a seam integer.
    # The seam holds what the fused form's operand slab would have stored: the TENSOR the cone
    # reads (an indexed load), not a scalar constant it also reads — a normalize cone's eps /
    # count constants are f32 scalars beside f16 activations, and typing the seam off them
    # makes an f32 B operand no warp atom can copy into its slab.
    loads = [ld for st in lowered for ld in Body((st,)).loads]
    for indexed in (True, False):
        for ld in loads:
            if bool(any(e.free_vars() for e in ld.index or ())) != indexed:
                continue
            t = (inputs or {}).get(ld.input)
            if t is not None and np.issubdtype(t.dtype.np, np.floating):
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
    out = _dc_replace(node, operands=tuple(_replace_edge(e, child, load) if isinstance(e, Fold) else e for e in node.operands))
    if node.axis is None and any(isinstance(m, Fold) for m in node.body):
        # A zero-axis node's body members that are nodes (the chained column fold in a sweep)
        # carry seams too; the seam child itself becomes the load in place.
        body = tuple(load if m is child else (_replace_edge(m, child, load) if isinstance(m, Fold) else m) for m in node.body)
        out = out.with_bodies((Body(body),))
    return out


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
    child_name = operand_name(child)
    ws = f"{out.name}__cut_{child_name}"
    if ws in match.graph.nodes:
        raise RuleSkipped(f"seam already cut — {ws} exists")
    anc = _ancestor_axes(tile_op, child)
    sweeps = tuple(st.sweep for st in stores if st.sweep is not None)
    axes = _child_axes(child, (*free, *sweeps), anc)
    ws_index = tuple(Var(a.name) for a in axes)
    ws_dtype = _ws_dtype(child, getattr(root.op, "inputs", None) or {})
    if _is_contraction_operand(tile_op, child) and out.dtype.nbytes == 2:
        # A seam standing in for a CONTRACTION OPERAND holds what the fused form's operand slab
        # stored: the fill rounds a computed operand to the atom's 16-bit element on the slab
        # store, so the materialized operand takes the contraction's 16-bit output dtype — the
        # same numerics, and a B slab the warp atoms can copy (only ``a`` has a converting fill).
        ws_dtype = out.dtype
    spelled = spell(tile_op, "PLACE", child, all_sites=sites(tile_op))
    logger.info("placement: cutting %s (%s → %s + residue) on %s", spelled, root.id, ws, out.name)

    # --- the CHILD piece: the seam subtree, its value stored to the workspace ------------------
    child_stmts = [*child.lower(), Write(output=ws, index=ws_index, value=child_name)]
    child_body = _nest(child_stmts, axes)
    child_op = LoopOp(body=rename_ssa_sequential(Body(tuple(child_body))))

    # --- the PARENT piece: the tree with the seam edge → a workspace Load ----------------------
    load = Load(name=child_name, input=ws, index=ws_index)
    parent_tree = _replace_edge(tile_op, child, load)
    parent_cell = effect_tail(parent_tree.lower(), stores)
    parent_body = _nest(list(parent_cell), list(free))
    # A piece is a RAW loop body: the tree's λ-local SSA names (each lift names its own loads
    # ``in0``…) flatten into one scope, where a name two λs share collides. Canonical sequential
    # names make the flattened body a valid single-scope program; the re-recognition lifts and
    # renames the piece again, so nothing downstream reads these names.
    parent_op = LoopOp(body=rename_ssa_sequential(Body(tuple(parent_body))))

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
    # A cut CONSUMES the kernel it replaces: both pieces drop its schedule row and its structural
    # identity, so each arrives at the identity stamp / ``020_schedule`` as a
    # brand-new kernel stamped and scheduled from its OWN body. They are built fresh here, so this
    # is the contract stated rather than work done — and it stays stated, because a rule that mints
    # a kernel is the thing that has to say the kernel is new.
    for nid in (ws, out.name):
        op = frag.nodes[nid].op
        op.knobs = consume_kernel_row(op.knobs)
    # The decision is spelled onto the OPTION, exactly as the pin that replays it, so a recorded
    # routing golden can match this fragment against the seam it names. The splice then CONSUMES it
    # along with everything else the pieces came with (the pieces' own ``consume_kernel_row``): once the cut
    # has happened the graph holds two kernels where it held one, and that is the record.
    parent = frag.nodes[out.name].op
    parent.knobs = {**(parent.knobs or {}), spelled: _CUT}
    return frag


__all__ = ["cuttable_seams", "realize_cut", "route_cut"]
