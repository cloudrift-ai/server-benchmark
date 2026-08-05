r"""Schedule a recognized (UNMAPPED) ``TileOp`` — the generic row enumerator.

**Every role emits rows through ONE recursive walk of the site tree; no role builds ``TileOp``\ s
directly, and no term shape gets its own path.** A row is a joint assignment across every scheduling
SITE of a term, and the tree that generates it is the term's own:

.. code-block:: text

    for work in _inventories(readings)        # the kernel's ONE inventory, CHOSEN at the root
      for term in _readings(tile)             # the tree REWRITES — the one union above the product
        for raster in _raster_values(term)    # kernel-global, like work
          for row in _rows_at(root, work)     # the product over the site tree

    _rows_at(site, work) = for value in _site_values(site, work)      # RESOLVED against work
                             for combo in product(_rows_at(child, work) for child in children)
                               _merge(site, value, combo)             # spells each slice ONCE

``WORK`` leads because the codec says so: :meth:`TilePlan.parse` and :meth:`ReducePlan.parse` both
take the inventory as an INPUT — a ``TILE`` value's unit widths and a ``REDUCE`` value's coop width
are READ OFF it — so the dependency runs work → slice, and a candidate that cannot spell against the
chosen inventory is simply not in ``_site_values(site, work)``. :func:`derive_inventory` stays, as
the VALIDATION that a row's own slices imply the inventory it claims.

Three layers, each with one job:

- the candidate DOMAIN is generated from its bounds in ``search/space.py`` (the tile spaces) or
  listed in its catalog there (the families with no multiplicative coupling — stages, split widths,
  the coop partitions, the raster orders);
- per-node LEGALITY — what a domain cannot know because it depends on this term's K, N, dtype and
  smem cap — is :mod:`._legality`, one predicate per rule, raise-vs-drop chosen by ``pinned``;
- THIS module chooses: which families a SITE offers, the conservative option-0 each leads with, and
  how a row becomes a ``TileOp``.

**Dispatch is two stored-param predicates on the node, never the** :class:`AxisRole`: ``axis is
None`` selects the register-strip values, :func:`is_contraction` the tile × stage × reduce product,
and everything else falls through to the reduce partition. The role is a LOOP annotation and a
materializer read; it never selects a catalog here.

What the walk covers:

- the pointwise cell: the register-strip ladder (``TILE=f<r>``, a TERM VARIANT applied at
  materialization);
- the reduce partition (``REDUCE``): the conservative heuristic pick, then the coop / ILP catalog;
- the contraction: the ``TILE × STAGE × REDUCE × RASTER`` legal product over the scalar and warp
  (mma) tiers, split-K rows routing through the structural ``Fold ⊃ Fold`` composition that
  ``030_split_reduce`` consumes;
- a COMPUTED ``a`` edge (the fused norm→linear / gate⊗up cone): the warp tier over the mandatory
  ``sync`` compute-fill, with the cone's own statistic site under the same inventory — a
  ``_site_values`` entry plus legality, not an emitter of its own;
- the STREAMING PAIR (flash): the hoisted score edge and the derived P@V each enumerate their half
  of the twisted geometry, the stream reconciles the pair and sizes its K/V transport against it,
  and the chain is the same P@V site under the ``""`` inventory — again values plus legality, with
  no emitter and no form dispatch.

A term this walk cannot schedule yields NO rows, and ``020_schedule`` leaves it unmapped rather than
guessing. That is the guardrail contract: empty enumeration returns ``[]``, never raises.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import (
    Raster,
    ReducePlan,
    Stage,
    TilePlan,
    WarpSpec,
    Workers,
    derive_inventory,
    plan_workers,
    resolve_site_tile,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Lambda, Load, Stmt, Write
from emmy.compiler.ir.stmt.algebra import M
from emmy.compiler.ir.stmt.passes import has_contraction_tail, projection_distributes
from emmy.compiler.ir.tile import Fold, Placement, Store, TileOp
from emmy.compiler.ir.tile.ir import is_contraction, operand_body
from emmy.compiler.ir.tile.ops import Sched, head, projection_tail, scheduled
from emmy.compiler.ir.tile.path import Site, sites
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.knob import canonical_row_key, family_of, values_equal
from emmy.compiler.pipeline.passes.lowering._addr import gmem_row_stride
from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction, make_cone
from emmy.compiler.pipeline.search.space import (
    RASTER,
    REDUCE,
    STAGE,
    TILE,
    WORK,
    coop_reduce_moves,
    map_tile_moves,
    raster_moves,
    scalar_tile_moves,
    splitk_moves,
    stage_moves,
    twisted_warp_moves,
    warp_tile_moves,
)

logger = logging.getLogger(__name__)

#: The per-site schedule families this enumeration decides, in the order their keys lead the fork
#: levels. ``WORK`` and ``RASTER`` are kernel-global and bracket them; ``PLACE`` is the seam
#: family — resolved from routing goldens / pins, never enumerated here.
FAMILIES = ("TILE", "STAGE", "REDUCE")

#: The ``Knob`` each family pins through.
_KNOBS = {"TILE": TILE, "STAGE": STAGE, "REDUCE": REDUCE}

#: The most rows one kernel's enumeration may produce. The product across sites is GENERATED, so a
#: term that widens it silently would hand the search a space it cannot walk and the prior a
#: feature space it cannot cover. Exceeding it is a LOUD failure, never a truncation — a truncated
#: enumeration reads as "covered everything" while dropping whichever rows the walk reached last.
#: Measured headroom: the widest live term (a static f16 square matmul, both tiers, every stage /
#: split / raster) enumerates ~133k rows.
MAX_ROWS = 400_000


# ---- structural reads over the stored term ----------------------------------------------------- #


def _node_loads(node) -> list[Load]:
    """Every gmem ``Load`` the term reads, as a deep walk over the STORED structure: an operand
    edge's MATERIALIZED inhabitant plus the loads sitting inline in a lift body, recursing through
    a COMPUTED edge's own node.

    Deliberately hand-written rather than a ``Body`` walk: an operand edge is not a body
    dependency, and ``Fold.nested()`` withholds a contraction's edges precisely so generic walkers
    do not read its multiply arguments as statements. Crossing the edges is the whole point here,
    so the walk alternates node-wise and statement-wise and visits each node exactly once."""
    out: list[Load] = []

    def walk_stmts(stmts) -> None:
        for s in stmts:
            if isinstance(s, Fold):
                walk(s)  # an inline COMPUTED edge is a node, not a statement — one visit, over there
                continue
            if isinstance(s, Load):
                out.append(s)
            for b in s.nested():
                walk_stmts(list(b))

    def walk(n) -> None:
        if isinstance(n, Load):
            out.append(n)
        elif isinstance(n, Fold):
            for e in n.operands:
                walk(e)
            # A contraction's lift IS the synthesized multiply — its reads are the edges above.
            if not is_contraction(n):
                walk_stmts(list(n.lift.body))

    walk(node)
    return out


def _projection(op) -> Body:
    """The kernel's per-cell projection — the wrapping zero-axis fold's body, or empty when the
    term is a bare node. A projection has ONE home (the wrapper's lift), never a node field."""
    return op.lift.body if isinstance(op, Fold) and op.axis is None and op.operands else Body(())


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _free_cells(place: Placement) -> int:
    """How many output cells the kernel's free grid covers (hint-resolved)."""
    return prod(_hint_extent(a) for a in place.free) if place.free else 1


def _inner_free(place: Placement) -> Axis | None:
    """The innermost NON-UNIT free axis — the m1 recognizer's synthesized unit axis can sit
    innermost, and it is not the axis the transposed emitter sweeps."""
    if not place.free:
        return None
    return next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)


def _matvec_b_kstride(term: _Term, carrier) -> int | None:
    """B's gmem stride along the reduce axis at the per-cell MATVEC tier, or ``None`` when no
    layout gate applies. A contraction demoted to PLANAR carries BOTH a vector operand (a load
    along the reduce axis touching no non-unit free axis — A) and a matrix operand indexed by the
    reduce axis AND a non-unit free axis (B); only that two-operand shape is gated. ``1`` means the
    reduce axis is B's fastest-varying dimension (the serving ``F.linear`` N×K layout); ``>1`` is
    k-major (canonical ``B[k, n]``)."""
    nonunit = {a.name for a in term.place.free if not (a.extent.is_static and a.extent.as_static() == 1)}
    k_name = carrier.axis.name
    a_seen = False
    strides = set()
    for ld in _node_loads(term.tile.op):
        used = set().union(*(e.free_vars() for e in ld.index)) if ld.index else set()
        if k_name not in used:
            continue
        if used & nonunit:
            strides.add(gmem_row_stride(ld, k_name, term.tile.inputs))
        else:
            a_seen = True
    return strides.pop() if a_seen and len(strides) == 1 else None


def _k_contiguous(term: _Term, node) -> bool | None:
    """Whether B stores its reduce axis FASTEST — one question, two ways to answer it because the
    two node shapes carry different evidence. A contraction has a B EDGE, so ``b_trans`` reads its
    index directly; a matvec DEMOTED to a plain fold keeps its loads inline in the lift (the
    formation fact), so there is no edge and the classifier walks the loads instead. ``None`` when
    no layout gate applies. Both answers feed the ONE predicate (``_legality.coop_band_layout``) —
    which is what stops the rule being stated twice with opposite polarity."""
    if is_contraction(node):
        return node.b_trans
    stride = _matvec_b_kstride(term, node)
    return None if stride is None else stride == 1


def _shared_row_buf(carrier_loads, tail, grid_vars, raxis: Axis, inputs) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — read in
    the carrier at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``, its trailing dim the
    (static) reduce extent. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input for s in carrier_loads if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in (ld for ld in Body(tail).loads if ld.is_scalar):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _row_stage(term: _Term, node) -> Stage | None:
    """The shared-row :class:`Stage` a COOPERATIVE reduce over ``node`` can drive, or ``None``. It
    is a RESOLVER, not a choice: the row spells ``d1/sync`` exactly when the shape carries an
    operand the CTA can hold as a shared row across the reduce and its contraction tail, and the
    materializer re-resolves the same buffer off the same term."""
    tail = projection_tail(term.tile)
    if not has_contraction_tail(tail):
        return None
    grid_vars = tuple(Var(a.name) for a in term.place.grid)
    carrier_loads = [ld for ld in _node_loads(node) if ld.is_scalar]
    buf = _shared_row_buf(carrier_loads, tail, grid_vars, node.axis, term.tile.inputs)
    return Stage(transport="sync", smem=(buf,)) if buf is not None else None


def _strippable(term: _Term) -> bool:
    """Whether the pointwise cell admits the register strip: a pure zero-axis fold with no operands
    whose body is FLAT elementwise (per-cell ``Load`` / ``Assign`` + boundary root stores, no nested
    ``Loop`` / carried state), over a static innermost free axis."""
    op, place = term.tile.op, term.place
    if not (isinstance(op, Fold) and op.axis is None and not op.operands) or not place.free:
        return False
    if not place.free[-1].extent.is_static:
        return False
    return all(isinstance(s, (Load, Assign, Write)) for s in op.body) and all(st.sweep is None for st in term.tile.stores)


# ---- the site tree ------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class _Node:
    """One node of the SITE TREE the enumeration walks: a schedule site, the canonical key each
    family spells it with, and the sites nested under it."""

    site: Site
    keys: dict[str, str]
    children: tuple[_Node, ...]


def _site_tree(op, key) -> tuple[_Node, ...]:
    """``op``'s scheduling sites as a TREE — the topmost ones first, each carrying the sites nested
    under it. The walker is ``path.sites``; this only groups its output by containment, so a term
    shape never gets a site list of its own.

    A node IS a site exactly when some family can key it, and ``key`` (``_Term.key`` over
    ``Sched.key``) already answers that — it spells a family site and returns ``None`` otherwise —
    so the key dict each node carries doubles as the membership test, and there is no second site
    predicate to keep in step with the codec's."""
    keyed = [(s, {f: k for f in FAMILIES if (k := key(f, s.node)) is not None}) for s in sites(op)]
    keyed = [(s, keys) for s, keys in keyed if keys]

    def under(parent: Site, child: Site) -> bool:
        return len(child.segments) > len(parent.segments) and child.segments[: len(parent.segments)] == parent.segments

    def build(parent: Site | None) -> tuple[_Node, ...]:
        kids = [(s, keys) for s, keys in keyed if parent is None or under(parent, s)]
        tops = [(s, keys) for s, keys in kids if not any(t is not s and under(t, s) for t, _ in kids)]
        return tuple(_Node(site=s, keys=keys, children=build(s)) for s, keys in tops)

    return build(None)


def _streams(node) -> bool:
    """Whether ``node``'s DERIVED evaluation carries schedule sites of its OWN — the STREAMING
    PAIR: an iterating fold that is not itself a contraction but whose blocked evaluation contracts
    (flash's hoisted score edge at the head of the step, its synthesized PV below the seam). Those
    are real sites (``TILE@dd`` / ``TILE@pj``), which is exactly what makes the stream a two-site
    term and the enumeration recursive.

    Structural, and deliberately NOT the ``TWISTED`` role: the role is derived by matching the
    combine's operation family, so a scheduling decision keyed on it would be an operation match
    wearing an algebraic name. What this asks is how the term is BUILT."""
    return node.axis is not None and not is_contraction(node) and any(is_contraction(s) for s in node.step_stmts())


def _keeps_children(site: Site) -> bool:
    """Whether the site's nested sites stay sites under the values this term offers. Two shapes
    keep them: a CONTRACTION, whose COMPUTED operand edges carry their own families (a MATERIALIZED
    operand is not a site — its transport is the parent's ``STAGE``), and a STREAMING fold, whose
    derived evaluation schedules the pair it contracts through. Under every other tier the fold's
    edges lower INLINE in its body, so they are not separately scheduled."""
    return is_contraction(site.node) or _streams(site.node)


def _kids(node: _Node) -> tuple[_Node, ...]:
    """The children the site tree descends into — :func:`_keeps_children` applied, in ONE place.
    Four walks need it (the row product, the inventory scan, the key spelling and the
    materializer's slice stamping) and they must agree exactly: what materializes is what was
    enumerated, so a walk that pruned differently would stamp a key no row decided."""
    return node.children if _keeps_children(node.site) else ()


# ---- the candidate values, per site ------------------------------------------------------------- #


class _Term:
    """Everything the enumeration reads about ONE READING of a term — the op, its grid placement,
    the target and the key speller — plus the per-site catalogs, built once and grouped by the
    inventory each candidate implies (the enumeration visits every site once per inventory).

    ``ref`` is the union's ONE key namespace (:func:`_readings`): the reference reading's tree,
    consulted before this reading's own so a site the two SHARE spells the same key in both rows.
    That is what keeps ``REDUCE`` bare on the contraction's K fold and the cone's statistic at
    ``REDUCE@<axis>`` on BOTH readings of a fused term."""

    def __init__(self, tile: TileOp, place: Placement, ctx, *, ref: Sched | None = None) -> None:
        self.tile = tile
        self.place = place
        self.ctx = ctx
        self.sched = Sched(tile.op, {}, place=place)
        self.ref = ref if ref is not None else self.sched
        self.proj = _projection(tile.op)
        self.tree = _site_tree(tile.op, self.key)
        self._tiles: dict[int, dict[str, list[TilePlan]]] = {}
        self._streams: dict[int, _Stream | None] = {}
        self._k_contig: dict[int, bool | None] = {}
        #: The refusal a schedule PIN drew, kept until the walk is done. One inventory declining
        #: a pin is ordinary (the widths are read OFF the inventory, so the pin names a different
        #: plan under each); a pin NO inventory could spell is malformed, and that is loud.
        self.pin_error: ValueError | None = None
        self.pin_spelled = False
        #: Set when any site offers a tensor-core row — a structural fact about the KERNEL, stamped
        #: on EVERY row so the priors can price "a scalar tile where tensor cores were on offer".
        self.warp_eligible = False

    def key(self, family: str, node) -> str | None:
        """The canonical key ``family`` spells ``node`` with in the UNION's namespace — the
        reference reading's tree first, this reading's own as the fallback. A node the reference
        tree does not carry is one this reading REWROTE (the collapse's spliced fold, the mixed-A
        promotion's coned contraction), and a rewrite keeps the site's tree POSITION, so the two
        spellings coincide there by construction. ``None`` when the family has no site to key."""
        return self.ref.key(family, node) or self.sched.key(family, node)

    def pin(self, family: str, node) -> str | None:
        """The live env pin for this site's ``family`` key — ``EMMY_KNOBS``'s ``FAMILY@<element>``
        entry, falling back to the bare ``EMMY_<FAMILY>``. Unset reads ``None``, which is the
        distinction the enumeration keys on: an unset ``TILE`` offers the catalog, a set one is
        authoritative."""
        key = self.key(family, node)
        if key is None:
            return None
        knob, element = _KNOBS[family], key.partition("@")[2]
        return knob.narrow_at(element) if element else knob.raw()

    def keyed_pin(self, family: str, node) -> str | None:
        """The EXPLICIT ``FAMILY@<element>`` pin for this site — no bare fallback. The two reads
        are different questions where a family has SEVERAL sites: an explicit key names one site
        and is authoritative there, while a bare pin fans out to every eligible site and cannot say
        which it meant (``knob.pin_key_matches``), so it narrows by MATCHING each site's own
        catalog and leaves a site it names nothing at alone."""
        key = self.key(family, node)
        element = key.partition("@")[2] if key is not None else ""
        return _KNOBS[family].pin_at(element) if element else None

    def stream(self, node) -> _Stream | None:
        """The mma facts a STREAMING fold's two sites resolve against, or ``None`` when the tier
        does not apply (a non-16-bit operand, an underivable gmem row stride, a score prologue no
        fragment can realize). Built once per node — every site of every inventory asks."""
        if id(node) not in self._streams:
            self._streams[id(node)] = _stream_of(self, node)
        return self._streams[id(node)]

    def k_contiguous(self, node) -> bool | None:
        """Whether B stores the reduce axis FASTEST (:func:`_k_contiguous`), memoized per node.

        Asked once per TILE CANDIDATE by ``_contraction_reduces``, and the demoted-matvec answer
        walks every load in the term to get there — so on a wide contraction the enumeration paid
        for one whole-term scan per candidate. The term is immutable during the walk, so the answer
        cannot move."""
        if id(node) not in self._k_contig:
            self._k_contig[id(node)] = _k_contiguous(self, node)
        return self._k_contig[id(node)]

    def tiles(self, node) -> dict[str, list[TilePlan]]:
        """The contraction node's ``TILE`` catalog, grouped by the ``WORK`` spelling each candidate
        implies (``""`` for the untiled per-cell tile, which composes with any inventory a
        cooperative ``REDUCE`` claims)."""
        if id(node) not in self._tiles:
            self._tiles[id(node)] = self._build_tiles(node)
        return self._tiles[id(node)]

    def _build_tiles(self, node) -> dict[str, list[TilePlan]]:
        atoms = _warp_atoms(self, node)
        warp = [p for p in warp_tile_moves(atoms) if _tile_ok(self, node, p)] if atoms else []
        self.warp_eligible = self.warp_eligible or bool(warp)
        # A COMPUTED ``a`` edge is warp-ONLY: the fill that evaluates a producer cone is the mma
        # tier's compute fill, and a per-cell / scalar expansion would re-run the cone on every K
        # step. The reduce tiers stay reachable — through the COLLAPSE reading, whose spliced fold
        # computes the whole body per cell (:func:`_readings`).
        scalar = scalar_tile_moves() if isinstance(node.a, Load) else []
        grouped: dict[str, list[TilePlan]] = {}
        for plan in scalar + warp:
            w = plan_workers(plan)
            grouped.setdefault(w.spell() if w is not None else "", []).append(plan)
        # A ``TILE`` pin is authoritative over the VALUES but not over the inventories: its unit
        # widths are read OFF ``WORK``, so the pin names a different plan under each one and is
        # re-resolved per inventory in :func:`_contraction_values`. The catalog still answers
        # "which inventories can this site spell against".
        return grouped


# ---- the streaming pair: what its two sites resolve against -------------------------------------- #


@dataclass(frozen=True)
class _Stream:
    """The mma facts a STREAMING fold's two schedule sites share — read ONCE off the term, because
    both sites and the transport resolver ask the same questions of it.

    The score atom is always the f32-ACCUMULATE one: the QK product feeds the online-softmax
    statistics, so it has no precision to trade. Only the PV gains the f16-accumulate sibling
    (``pv_atoms``), whose partials promote per streaming block in the realizer."""

    qk: Fold  # the hoisted score edge — the head of the derived step
    pv: Fold  # the synthesized P@V contraction — combine material, below the seam
    atom: object  # the score's mma atom
    pv_atoms: tuple  # the P@V atom, plus its f16-accumulate sibling when the gate offers it
    d_v: int  # the value dim the P@V fragment tiles exactly
    tma: bool  # whether a TMA box can encode this trace's K/V operands
    stageable: bool  # the K/V slabs byte-copy at the atom's operand width
    q_stageable: bool  # ... and so does Q, which the ``split`` groups additionally stage


def _kv_penultimate(load: Load, kv_name: str) -> bool:
    """Whether a K/V operand's gmem index puts the STREAM axis where TMA's box origin expects it.
    The box coords are derived POSITIONALLY — batch dims lead, the kv row sits at ``[-2]``, the
    contiguous head dim last — so an un-transposed ``(B, S, H, D)`` trace (kv at position 1) would
    leak the raw axis var into the emitted coords. cp.async is unaffected: its fill substitutes by
    axis NAME."""
    idx = load.index
    if len(idx) < 2 or not isinstance(idx[-2], Var) or idx[-2].name != kv_name:
        return False
    return not any(kv_name in e.free_vars() for e in (*idx[:-2], idx[-1]))


def _stream_of(term: _Term, node) -> _Stream | None:
    """Read the streaming pair off ``node``, or ``None`` when the fragment tier does not apply.

    The refusals here are TERM readings, not candidate legality: a non-16-bit operand dtype, a
    score prologue whose bias is indexed beyond ``(m, kv)`` (no fragment realization for it), an
    underivable gmem row stride (the fragment loaders step operand rows at the buffer's REAL
    stride, derived from the index + shape) and a fragment-unrealizable projection. Each is a
    property of the TERM, so it belongs in the choice layer — what a CANDIDATE must satisfy is
    :mod:`._legality`."""
    steps = list(node.step_stmts())
    qk = steps[0] if steps and is_contraction(steps[0]) else None
    pv = next((s for s in steps[1:] if is_contraction(s)), None)
    inputs = term.tile.inputs
    if qk is None or pv is None or not inputs or legal.fragment_epilogue(term.proj) is not None:
        return None
    if not (isinstance(qk.a, Load) and isinstance(qk.b, Load) and isinstance(pv.b, Load)):
        return None
    if not qk.axis.extent.is_static:
        return None  # the score's fragment K-steps are unrolled — a symbolic head dim has no tier
    atoms = atoms_for(_a_dtype(qk, inputs))
    if not atoms:
        return None
    atom = ATOM_REGISTRY[atoms[0]]
    # The two sites' PLACED readings — the ``(m, n)`` pair is a function of the SITE, so it is
    # taken off the one rule (``Sched._mn_for``, through a bare probe plan) rather than restated.
    qk_mn, pv_mn = term.sched.placed(qk, TilePlan()).axes, term.sched.placed(pv, TilePlan()).axes
    if qk_mn is None or pv_mn is None or not pv_mn[1].extent.is_static:
        return None
    d_v, kv_name, m_name = pv_mn[1].extent.as_static(), node.axis.name, qk_mn[0].name
    for s in node.lift.body:
        if isinstance(s, Load) and s.index and not {v for e in s.index for v in e.free_vars()} <= {m_name, kv_name}:
            return None  # a score bias indexed beyond (m, kv) — the fragment realizer cannot load it
    strides = (
        gmem_row_stride(qk.a, m_name, inputs),
        gmem_row_stride(qk.b, (qk_mn[1] if qk.b_trans else qk.axis).name, inputs),
        gmem_row_stride(pv.b, (pv_mn[1] if pv.b_trans else node.axis).name, inputs),
    )
    if any(s is None for s in strides):
        return None

    def dtype_of(load: Load):
        t = inputs.get(load.input)
        return getattr(t, "dtype", None)

    b_dt, a_dt = atom.operand_dtype("b"), atom.operand_dtype("a")
    # Which atoms EXIST for this operand dtype is a term fact; whether the precision-trading sibling
    # is OFFERED is the choice layer's gate (:func:`_twisted_values`), which a pin bypasses.
    sibling = atoms_for(_a_dtype(qk, inputs), acc=_a_dtype(qk, inputs))
    return _Stream(
        qk=qk,
        pv=pv,
        atom=atom,
        pv_atoms=(atom, *(ATOM_REGISTRY[n] for n in sibling)),
        d_v=d_v,
        tma=term.ctx.has_tma and _kv_penultimate(qk.b, kv_name) and _kv_penultimate(pv.b, kv_name),
        # Staging byte-COPIES the operands into slabs typed at the atom's operand width, so an
        # operand traced at another dtype would deposit wrong-sized elements; gmem-direct fragment
        # loads convert per element, which is why a mismatch keeps the tier and drops its stage rows.
        stageable=dtype_of(qk.b) == b_dt and dtype_of(pv.b) == b_dt and qk.axis.extent.as_static() % atom.atom_k == 0,
        q_stageable=dtype_of(qk.a) == a_dt,
    )


# ---- the term READINGS: the one mechanism above the product ------------------------------------- #


def _reading(tile: TileOp, op, ctx, *, free=None, stores=None, ref: Sched | None = None) -> _Term:
    """One reading as a ``_Term`` — the rewritten ``op`` over its own placement / boundary stores,
    on the grid. A reading is never a mutation: each is its own term, with its own ``term_key`` and
    ``op_cache_key``."""
    place = Placement(free=tuple(tile.place.free if free is None else free))
    alt = TileOp(op=op, name=tile.name, place=place, inputs=dict(tile.inputs), stores=tile.stores if stores is None else stores)
    return _Term(alt, place.on_grid(), ctx, ref=ref)


def _fused_op(tile: TileOp):
    """The MONOID-producer composition of this term — the fused norm→linear / gate⊗up reading, whose
    contraction reads its normalized row off a COMPUTED ``a`` edge (``bind_prologue_contraction``);
    ``None`` when the term is not that shape. It ADDS the contraction and the cone's statistic sites
    to the map form's single reduce site, so it is a READING, not a value."""
    return bind_prologue_contraction(tile.op, tuple(tile.place.free))


def _promoted(node, inputs):
    """A mixed-dtype contraction — a plain **f32** ``a`` ``Load`` against 16-bit channels — with its
    ``a`` edge re-expressed as a one-``Load`` COMPUTED cone, so it rides the mandatory ``sync``
    compute-fill whose slab store demotes the value to the atom dtype. The copy transports move raw
    bytes and cannot convert, which is why the warp tier is reachable only through the cone form.
    ``None`` when the term is not that shape.

    The signature can only enter a traced graph through an ERASED dtype cast (torch cannot execute a
    mixed matmul, so the model itself rounded A — Gemma's ``self._norm(x.float()).type_as(x)`` ahead
    of every f16 projection; the tracer maps ``to`` / ``type_as`` to pass-throughs). B's values carry
    16 bits, the accumulate stays f32, and this is a fork SIBLING, so the demotion is searchable and
    costs ~2⁻¹¹ relative noise on A — the rounding the model performed anyway."""
    if node is None or not isinstance(node.a, Load) or not inputs:
        return None
    t = inputs.get(node.a.input)
    if getattr(getattr(t, "dtype", None), "name", None) != "f32" or not _channel_atoms(node, inputs):
        return None
    # ``a`` is a DERIVED reading, so the rewrite REBUILDS the bilinear fold over the new edge — the
    # one-``Load`` cone keeps the edge's bound name, so the regenerated lift is the same program.
    return Fold.contraction(k_axis=node.axis, a=make_cone([node.a], node.axis.name), channels=node.channels)


def _readings(tile: TileOp, ctx) -> list[_Term]:
    """The term READINGS whose rows the fork unions — at most two, and the ONE mechanism that sits
    ABOVE the site product (a reading REWRITES the tree, so it changes the site SET, which is
    exactly what the product cannot absorb).

    The base reading is the stored term. Its sibling is whichever rewrite applies — they are
    mutually exclusive by shape:

    - the MONOID-producer composition (:func:`_fused_op`) — the map form's reduce site plus the
      fused contraction's own tree. Its tree is the REFERENCE namespace: bare ``REDUCE`` must mean
      the contraction's K fold, so the map reading spells its statistic at ``REDUCE@<axis>`` too;
    - the COLLAPSE (:meth:`Fold.demoted`) — a computed ``a`` edge spliced back inline, REMOVING its
      site. It is what carries a stat-free cone (``f(x) @ w``) on the reduce tiers, and what a
      computed-A term with no legal warp row falls back to;
    - the mixed-A PROMOTION (:func:`_promoted`) — a materialized f32 ``a`` turned into a cone,
      ADDING one site.

    No reading's rows depend on whether its sibling produced any: each gate is a local predicate on
    its own term (a 16-bit atom, a resolvable fill, an inventory a value can spell against)."""
    base = _Term(tile, tile.place.on_grid(), ctx)
    pro = _fused_op(tile)
    if pro is not None:
        fused = _reading(tile, pro[0], ctx, free=(*tile.place.free, pro[1]), stores=pro[2])
        return [_Term(tile, tile.place.on_grid(), ctx, ref=fused.sched), fused]
    node = head(tile.op)
    if node is None or not is_contraction(node):
        return [base]
    if not isinstance(node.a, Load):
        return [base, _reading(tile, _rewrap(tile.op, node.demoted()), ctx, ref=base.sched)]
    promoted = _promoted(node, tile.inputs)
    if promoted is None:
        return [base]
    return [base, _reading(tile, _rewrap(tile.op, promoted), ctx, ref=base.sched)]


def _rewrap(op, node):
    """``op`` with its compute node replaced — the projection wrapper preserved when the term has
    one (a projection has ONE home, and a reading never moves it)."""
    return replace(op, operands=(node,)) if op is not head(op) else node


def _tile_ok(term: _Term, node, plan: TilePlan) -> bool:
    """Whether a warp tile candidate is realizable on ``node`` — the K-step divisibility every warp
    row needs, plus the exact-cover geometry a COMPUTED ``a`` edge's compute fill adds. Both are
    ``_legality`` predicates, dropped here and RAISED on a pin (:func:`_contraction_values`)."""
    if not legal.enforce(legal.warp_k_step(node, plan), pinned=False):
        return False
    if isinstance(node.a, Load):
        return True
    placed = plan.placed_on(term.place)
    if placed.axes is None:
        return False  # no (m, n) pair on the grid — nothing to place a compute-filled tile on
    return legal.enforce(legal.computed_a_cover(node, placed), pinned=False)


def _f16acc_allowed(ctx) -> bool:
    """Whether the f16-accumulate atom forks may be OFFERED — a precision-trading gate, off by
    default: the precise ``F16_MMA_F32_ACC`` pin is authoritative on every target; unset, the
    ``FAST_MATH`` umbrella offers it on the consumer dies (``Context.f16acc_is_faster``) where the
    f32-accumulate half-rate nerf makes it profitable. A ``TILE`` pin naming the atom bypasses this
    gate entirely (pins are authoritative)."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, precision_pin  # noqa: PLC0415

    raw = F16_MMA_F32_ACC.raw()
    if raw is not None:
        return F16_MMA_F32_ACC.parse(raw)
    return precision_pin(F16_MMA_F32_ACC) and ctx.f16acc_is_faster


def _a_dtype(node, inputs):
    """The ``a`` edge's element dtype — the value the mma fragment reads. A MATERIALIZED edge reads
    its gmem tensor's; a COMPUTED cone reads its K-indexed leaf ``Load``'s, which is the value the
    sync compute-fill stores to the A slab."""
    ld = node.a
    if not isinstance(ld, Load):
        k = node.axis.name
        ld = next((s for s in operand_body(node.a) if isinstance(s, Load) and k in {v for e in s.index for v in e.free_vars()}), None)
    t = inputs.get(ld.input) if ld is not None else None
    return t.dtype if t is not None else None


def _channel_dtype(node, inputs):
    """The one element dtype every channel's B agrees on, or ``None`` — the dtype an f32 ``a`` still
    rides when the sync compute-fill DEMOTES it on the slab store."""
    bs = [ch.b for ch in node.channels]
    if not bs or not all(isinstance(b, Load) for b in bs):
        return None
    dts = {getattr(inputs.get(b.input), "dtype", None) for b in bs}
    return next(iter(dts)) if len(dts) == 1 else None


def _channel_atoms(node, inputs) -> tuple[str, ...]:
    """The atoms every channel's B agrees on — what the mixed-A promotion would BUY, and ``()`` when
    it would buy nothing (so the promotion is not offered)."""
    return atoms_for(_channel_dtype(node, inputs))


def _warp_atoms(term: _Term, node) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (a non-16-bit operand dtype, a fragment-unrealizable gather epilogue), extended
    with the f16-accumulate siblings when :func:`_f16acc_allowed`. Reads pure algebra off the STORED
    node — the placement / tile would be unread.

    A COMPUTED ``a`` edge whose leaf is f32 still rides the CHANNELS' 16-bit atom: the compute fill
    converts on the slab store, which is the whole reason the mixed-A promotion routes through the
    cone form. A materialized f32 ``a`` stays ineligible here (a copy transport moves raw bytes) and
    reaches the warp tier through :func:`_promoted` instead.

    This is the CHOICE half of the dtype rule; a ``TILE`` pin bypasses the choice layer by design, so
    it re-asks the same question as a CHECK (``_legality.warp_operand_dtype``)."""
    inputs = term.tile.inputs
    if not inputs or legal.fragment_epilogue(term.proj) is not None:
        return ()
    ab = _a_dtype(node, inputs)
    if not atoms_for(ab) and not isinstance(node.a, Load):
        ab = _channel_dtype(node, inputs)  # the demoting compute fill — an f32 cone on 16-bit B
    atoms = atoms_for(ab)
    if not atoms or not _f16acc_allowed(term.ctx):
        return atoms
    return atoms + atoms_for(ab, acc=ab)  # the f16-accumulate siblings, registry order preserved


# --- the pointwise cell: the register strip ---


def _strip_width(plan: TilePlan) -> int:
    """The strip ratio ``r`` a strip row's ``TILE`` names — the inner register width. A warp codec
    names none (there is no fragment on a pointwise cell), so it reads ``0`` and is dropped."""
    return 0 if plan.is_warp else plan.reg_n


def _strip_values(term: _Term, node) -> list[dict]:
    """The register-strip values: the flat per-cell tile (option-0), then the catalog's ladder.
    ``r`` IS the spelled ``TILE=f<r>`` — the strip is a TERM VARIANT applied at materialization, a
    function of the ROW, not a member of a pre-enumerated variant set."""
    pin = term.pin("TILE", node)
    ext = term.place.free[-1].extent.as_static() if _strippable(term) else 0
    try:
        plans = [resolve_site_tile(pin, None)] if pin is not None else [TilePlan(), *map_tile_moves()]
    except ValueError as e:
        # A pin the strip site cannot SPELL — a warp atom, which needs an inventory a pointwise
        # cell never has. Same rule as everywhere: the candidate is simply not in
        # ``values(site, work)``, so the cell degrades to option-0. This is PIN BLEED (one env pin,
        # several kernels in the graph, and this is not the one it was written for), which is why
        # it degrades rather than emptying the fork; ``_enumerate`` still raises the recorded error
        # if NOTHING in the term could spell it.
        term.pin_error = e
        plans = []
    out = []
    for plan in plans:
        # A strip WIDTH the cell cannot carry (a stateful / sweep body, a symbolic or indivisible
        # inner extent, a warp codec on a pointwise cell) drops the row; the flat per-cell base
        # below is always offered, so a narrowing pin degrades to option-0.
        if legal.enforce(legal.strip_width(ext, _strip_width(plan)), pinned=False):
            out.append({"TILE": plan})
    return out or [{"TILE": TilePlan()}]


# --- the reduce partition ---

# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells


def _prevpow2(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_coop(extent: int, free: int, *, has_tail: bool = False) -> int:
    """The conservative whole-CTA cooperative-thread count for a reduce of static ``extent`` over
    ``free`` output cells, or ``1`` (stay serial). Cooperate only on a wide reduce feeding a small
    grid — otherwise the scalar tier already saturates the GPU; ``has_tail`` lifts the grid cap (a
    fused contraction tail multiplies each cell's work by its column extent)."""
    if extent < _COOP_MIN_EXTENT or (free > _FREE_CAP and not has_tail):
        return 1
    coop = min(_prevpow2(extent // _SERIAL_TARGET), _MAX_COOP)
    return coop if coop >= 2 else 1


def _reduce_specs(term: _Term, node) -> list[ReducePlan]:
    """The reduce-partition candidates for a non-contraction fold — option-0 is the conservative
    heuristic pick (:func:`_pick_coop`, so a cold greedy compile keeps its historical deploy), then
    the legal :func:`coop_reduce_moves` catalog + serial as fork siblings. The catalog rows are what
    keep the 16- / 32-wide reduce goldens reachable. An env pin is authoritative."""
    pin = term.pin("REDUCE", node)
    if pin is not None:
        return [ReducePlan.parse(pin, Workers.parse(WORK.raw()))]
    extent = _hint_extent(node.axis)
    free = _free_cells(term.place)
    tail = projection_tail(term.tile)
    coop = _pick_coop(extent, free, has_tail=has_contraction_tail(tail))
    # The layout gate (WS5, the cold-poison hardening): at the matvec tier the coop bands are only
    # coalesced on ONE B orientation — the plain band interleaves lanes along K, the transposed
    # band sweeps lanes along the output axis. Enumeration is the single choke point every tier
    # resolves through, so the gate lives here; an env pin stays authoritative and un-gated.
    k_contig = term.k_contiguous(node)
    if k_contig and extent >= _COOP_MIN_EXTENT and free >= _FREE_CAP:
        # The wide-K COALESCED matvec: a 32-wide band is the measured deploy, and option-0 is what
        # the prior-free paths take. It is a RANKING decision, so it leads the list rather than
        # replacing it — the row SET stays a function of the term alone, never of a context flag.
        coop = 32
    elif coop > 1 and k_contig is False:
        coop = 1  # the heuristic option-0 is a plain band too — uncoalesced on a k-major B
    cands = [ReducePlan.of(coop=coop)]
    inner = _inner_free(term.place)
    k_static = node.axis.extent.as_static() if node.axis.extent.is_static else None
    epilogue = legal.coop_band_epilogue(tail)  # term-wide, so it is asked ONCE, not per candidate
    for p in coop_reduce_moves():
        if not legal.enforce(legal.coop_band_layout(p, k_contig), pinned=False):
            continue
        if p.needs_split and not p.coop_transposed:
            # A cross-CTA split is offered on this tier only in COMPOSITE with the transposed band
            # — every split candidate in the catalog is one, so this states the catalog's shape
            # rather than adding a rule.
            continue
        if p.coop_transposed:
            # The band's own requirements: the geometry (shared with the contraction tier) plus
            # this tier's epilogue condition.
            if not legal.enforce(legal.coop_band_geometry(p, k_static, inner), pinned=False):
                continue
            if not legal.enforce(epilogue, pinned=False):
                continue
        if p.coop <= extent and p.reg <= extent and p not in cands:
            cands.append(p)
    if ReducePlan() not in cands:
        cands.append(ReducePlan())
    return cands


def _reduce_values(term: _Term, node) -> list[dict]:
    """The reduce-partition values a non-contraction fold offers: the partition itself plus the
    shared-row ``STAGE`` a cooperative band can drive (a resolver, not a choice — see
    :func:`_row_stage`). Which of them SPELL against the kernel's chosen inventory is the row's
    question, not this site's (:func:`_work_holds`) — a serial fold claims no workers at all, so at
    a NESTED site it composes with any parent inventory."""
    return [{"REDUCE": plan, "STAGE": _row_stage(term, node) if plan.coop > 1 else None} for plan in _reduce_specs(term, node)]


def _fill_realized(parent: _Node | None, site: Site) -> bool:
    """Whether the PARENT form realizes this nested fold's partition ITSELF, leaving the site's own
    value the decided empty. One form does today: the sync compute-fill's per-row statistic
    prologue stripes a cone's statistic ONE ROW PER WARP, the warp's 32 lanes striding the fold and
    closing it on the shuffle butterfly (``lowering/kernel/_stage.sync_stat_fill``) — a single
    hardwired partition, so any value here would stamp a knob no kernel realizes."""
    if parent is None or not is_contraction(parent.site.node) or isinstance(parent.site.node.a, Load):
        return False
    depth = len(parent.site.segments)
    return len(site.segments) > depth and site.segments[depth] == "a"


def _band_of(plan: ReducePlan) -> Workers | None:
    """The inventory a reduce partition implies — the 1-D cooperative band, or ``None`` (a serial /
    register-ILP fold keeps the derived per-cell launch geometry)."""
    return Workers(kind="thread", units=(plan.coop, 1)) if plan.coop > 1 else None


# --- the contraction: tile x stage x reduce ---

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied.
_SPLITK_MAX_CTAS = 1024


def _tile_area(plan: TilePlan) -> int:
    """The output cells one CTA covers under ``plan`` — the occupancy denominator."""
    am, an = (plan.atom.atom_m, plan.atom.atom_n) if plan.is_warp else (1, 1)
    return max(plan.units_m * plan.reg_m * am * plan.units_n * plan.reg_n * an, 1)


def _resolve_stage(term: _Term, node, tile: TilePlan, want: Stage | None) -> Stage | None:
    """The ONE transport-resolver dispatch — which resolver a node's ``a`` edge and tier select.

    A COMPUTED ``a`` edge takes the sync compute-fill, which is MANDATORY (no copy transport can
    evaluate a cone), so ``want=None`` still resolves and only the DEPTH is ever free. A
    MATERIALIZED edge takes the mma resolver on a warp tile and the scalar one otherwise, with
    ``want=None`` the gmem-direct baseline; TMA declines below sm_90 rather than failing to
    compile. ``tile`` is the PLACED slice.

    Enumeration, the split-K composition and re-materialization all reach the resolvers through
    here, so a row's resolved spelling is reproducible BY CONSTRUCTION rather than by three copies
    of the dispatch staying in step."""
    budget = term.ctx.max_dynamic_smem
    if not isinstance(node.a, Load):
        return legal.resolve_sync_stage(node, tile, budget, want.depth if want is not None else 1)
    if want is None or (want.transport == "tma" and not term.ctx.has_tma):
        return None
    if tile.is_warp:
        return legal.resolve_warp_stage(node, tile, want, budget)
    return legal.resolve_scalar_stage(node, tile, want, term.tile.inputs, budget)


def _resolved(moves, resolve, *, gmem_direct: bool = True) -> list[Stage | None]:
    """``moves`` resolved against the term and deduped on the RESOLVED spelling — the shape every
    stage-value site shares. Dedupe is on the resolved spelling, never the catalog move: a depth
    that clamps under the smem budget spells identically to its shallower sibling and must yield
    ONE row, or the fork carries two leaves naming one kernel.

    ``gmem_direct`` seeds the conservative option-0 (``None``, no slab). The compute-fill tier has
    no gmem-direct sibling — a computed ``a`` edge must land somewhere — so it seeds nothing, and a
    caller that declines every move returns the empty list rather than a silent fallback."""
    out: list[Stage | None] = [None] if gmem_direct else []
    spelled = {""} if gmem_direct else set()
    for move in moves:
        r = resolve(move)
        if r is not None and r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


def _sync_values(term: _Term, node, tile: TilePlan) -> list[Stage | None]:
    """The RESOLVED compute-fill stages a COMPUTED ``a`` edge offers — its depths, and nothing else:
    the fill is MANDATORY (there is no gmem-direct ``None`` sibling and no copy transport can
    evaluate a cone), so a ``STAGE`` pin can only choose the depth. ``d1`` and the asymmetric B-only
    prefetch ring ``d2`` are fork siblings — the ring is measured per shape (see
    :func:`_legality.resolve_sync_stage`) — and a ``d2`` that clamps back to ``d1`` under the smem
    budget spells identically, so it dedupes to one row."""
    pin = term.pin("STAGE", node)
    depths = [Stage.parse(pin).depth] if pin else [1, 2]

    def resolve(st: Stage) -> Stage | None:
        r = _resolve_stage(term, node, tile, st)
        if r is None:  # per DECLINED depth, so a pin that fits no depth names its own budget
            legal.enforce(f"the sync compute-fill slabs exceed the {term.ctx.max_dynamic_smem} B smem budget", pinned=pin is not None)
        return r

    return _resolved((Stage(depth=d) for d in depths), resolve, gmem_direct=False)


def _stage_values(term: _Term, node, plan: TilePlan) -> list[Stage | None]:
    """The RESOLVED operand stages for one tile candidate — gmem-direct ``None`` first, then every
    catalog move that RESOLVES against the node with this ``plan``, so the leaf identity, the
    stamped knobs and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin alone,
    or gmem-direct when it declines."""
    if not plan.is_tiled:
        return [None]  # per-cell / unbindable — no operand slab to stage
    tile = plan.placed_on(term.place)
    if not isinstance(node.a, Load):
        return _sync_values(term, node, tile)

    def resolve(st: Stage) -> Stage | None:
        return _resolve_stage(term, node, tile, st)

    pinned = term.pin("STAGE", node)
    if pinned is not None:
        # A malformed pin RAISES through ``Stage.parse`` — this used to be swallowed into
        # gmem-direct, which made it the only silently-ignored pin in the family.
        return [resolve(Stage.parse(pinned)) if pinned else None]
    return _resolved(stage_moves(warp=plan.is_warp), resolve)


def _contraction_reduces(term: _Term, node, plan: TilePlan) -> list[ReducePlan]:
    """The contraction's ``REDUCE`` candidates — serial first (option-0), then the legal coop / ILP
    moves (per-cell tier only — the non-output-tiled contract) and the divisor- and occupancy-
    guarded split-K moves. An ATOMIC split is offered only on a single-channel node whose FULL
    projection tail distributes over the add; the deferred kernel finalize stays legal for any
    epilogue."""
    pin = term.pin("REDUCE", node)
    if pin is not None:
        pinned = ReducePlan.parse(pin, Workers.parse(WORK.raw()))
        if pinned.needs_split:
            return [pinned]
        if pinned.coop > 1 or pinned.reg > 1:
            # A tiled candidate contracts K serially per register cell — the coop / ILP partition is
            # the NON-output-tiled tier's, so a tiled tile has nothing to honor the pin with.
            return [] if plan.is_tiled else [pinned]
        return [ReducePlan()]
    out = [ReducePlan()]
    ext = node.axis.extent
    k = ext.as_static() if ext.is_static else None
    # The σ-reindex a cross-CTA split performs needs a gmem index on every channel's B, so a
    # COMPUTED B admits no split — in the bare ``g<n>`` moves OR in the transposed band's ``g<n>k``
    # composites. Asked HERE and not only at materialization: ``_splitk_option`` enforces it with
    # ``pinned=True``, so an unpinned candidate the enumeration offered became a raise instead of a
    # dropped row — the raise-vs-drop split this module exists to keep on one side.
    splittable = k is not None and legal.enforce(legal.splitk_materialized_b(node), pinned=False)
    if k is not None and not plan.is_tiled:
        inner = _inner_free(term.place)
        k_contig = term.k_contiguous(node)
        for p in coop_reduce_moves():
            if not (p.coop <= k and p.reg <= k):
                continue
            if p.needs_split and not splittable:
                continue
            if not legal.enforce(legal.coop_band_layout(p, k_contig), pinned=False):
                continue
            # The transposed lane swap also needs the structure its emitter assumes — the SAME
            # geometry the reduce tier requires, stated once in ``_legality``.
            if not legal.enforce(legal.coop_band_geometry(p, k, inner), pinned=False):
                continue
            out.append(p)
    if splittable and _free_cells(term.place) // _tile_area(plan) <= _SPLITK_MAX_CTAS:
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        tail = tuple(projection_tail(term.tile))
        atomic_ok = len(node.channels) == 1 and (len(tail) == 0 or projection_distributes(tail, (node.acc,)))
        for sp in splitk_moves():
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # a non-distributive projection would raise at 030_split_reduce
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(sp)
    return out


def _contraction_values(term: _Term, node, work: Workers | None) -> list[dict]:
    """The contraction's values at ``work``: the tile × stage × reduce legal product, over EITHER
    inhabitant of the ``a`` edge — a materialized ``Load`` (both tiers, every transport) or a
    COMPUTED cone (the warp tier alone, over the mandatory compute fill)."""
    pin = term.pin("TILE", node)
    if pin is not None:
        try:
            plans = [resolve_site_tile(pin, work, term.pin("REDUCE", node) or "")]
        except ValueError as e:
            # The pin cannot SPELL against this inventory (a warp atom needs a warp ``WORK``), so
            # the candidate is simply not in ``values(site, work)`` — the same rule every other
            # value follows. A pin that spells against NO inventory is a different failure and
            # :func:`_enumerate` raises it rather than quietly emptying the fork.
            term.pin_error = e
            return []
        term.pin_spelled = True
        for plan in plans:
            if plan.is_warp:
                # A PIN with an indivisible K-step or a gather epilogue RAISES — the same predicates
                # the unpinned catalog above drops on, one home each.
                legal.enforce(legal.warp_k_step(node, plan), pinned=True)
                legal.enforce(legal.fragment_epilogue(term.proj), pinned=True)
                if not isinstance(node.a, Load):
                    legal.enforce(legal.computed_a_cover(node, plan.placed_on(term.place)), pinned=True)
                # The operand-dtype rule DROPS even under a pin: an unconvertible A means the pinned
                # tier is realizable through the mixed-A PROMOTION reading's converting fill, not
                # here, so this is choosing the reading rather than ignoring the pin.
                elif not legal.enforce(legal.warp_operand_dtype(node, plan, _a_dtype(node, term.tile.inputs)), pinned=False):
                    return []
            elif not isinstance(node.a, Load):
                return []  # a scalar / per-cell pin asks for a tier the compute fill has no fill for
            else:
                # The CTA thread budget, raised HERE rather than left to materialization: a pinned
                # tile the hardware cannot launch is a user error, and `Pipeline.run`'s validity
                # retry would otherwise catch the materializer's raise and quietly deploy the next
                # leaf — the pin says yes, the deploy says something else.
                legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    else:
        base = replace(work, producer=0) if work is not None else None
        grouped = term.tiles(node)
        plans = grouped.get(base.spell() if base is not None else "", []) + (grouped.get("", []) if base is not None else [])
    out = []
    for plan in plans:
        for red in _contraction_reduces(term, node, plan):
            for stage in _stage_values(term, node, plan):
                if work is not None and work.producer and not legal.enforce(legal.producer_transport(stage, red), pinned=False):
                    continue
                out.append({"TILE": plan, "STAGE": stage, "REDUCE": red})
    return out


# --- the streaming pair: the two sites, then the stream that must agree with them ---

#: The chain's register-vector budget — one thread holds the WHOLE output row.
_CHAIN_MAX_D = 64


def _narrowed(term: _Term, node, plans: list[TilePlan]) -> tuple[list[TilePlan], bool]:
    """``plans`` narrowed by the site's live ``TILE`` pin, and whether the pin SELECTED here.

    An EXPLICIT ``TILE@<axis>`` pin is authoritative — it names this site, so a spelling the site
    does not offer empties it. A BARE pin fans out to every eligible site and cannot say which it
    meant, so it narrows by MATCHING and a site it names nothing at keeps its catalog
    (``Knob.narrow``'s no-match-keeps-full-list, the same degrade the pin layer applies everywhere).
    That is what lets the masked-flash golden form — one bare ``TILE`` spelling the f16-accumulate
    PV plan — select the PV variant while the score keeps its own f32-accumulate catalog, the pair
    then reconciling at the stream. The flag is what keeps that degrade from ALSO reading as
    authority: a pin that named nothing here decides nothing here, precision gates included."""
    pin = term.pin("TILE", node)
    if pin is None:
        return plans, False
    kept = [p for p in plans if values_equal(TILE.name, pin, p.spell())]
    if kept or term.keyed_pin("TILE", node) is not None:
        return kept, True
    return plans, False


def _chain_values(term: _Term, node, work: Workers | None) -> list[TilePlan]:
    """The CHAIN candidate at the PV site — the FA-2 shared-score form: the value axis leaves the
    grid and rides a per-thread REGISTER VECTOR, so the score is computed once per streamed key and
    shared across the columns (against the per-cell tier's redundant recompute per column). Offered
    on the ``""`` inventory (one thread per query row) when the value axis is the innermost grid
    axis and small enough to hold — the register budget, a hand-measured ladder stop."""
    if work is not None:
        return []
    mn = term.sched.placed(node, TilePlan()).axes
    grid = term.place.grid
    if mn is None or not grid or grid[-1].name != mn[1].name or not mn[1].extent.is_static:
        return []
    d = mn[1].extent.as_static()
    return [TilePlan(regs=(1, d))] if 1 < d <= _CHAIN_MAX_D else []


def _twisted_values(term: _Term, site: Site, work: Workers | None, parent: _Node) -> list[dict]:
    """The values a site of the STREAM's derived evaluation offers — the hoisted score edge
    (``TILE@dd``) and the synthesized P@V (``TILE@pj``, ``Site.derived``).

    Both are contractions, and neither is a root matmul: the geometry they carry is the STREAM's,
    so the dispatch is the site's POSITION (the same question :func:`_fill_realized` asks of a
    cone's statistic), not its node kind. Each site enumerates its own half of the twisted grid and
    the pair is reconciled at the stream (:func:`_stream_values`) — two sites that must agree, which
    is what the recursion is for. The DECIDED EMPTY leads: it is the reduce tiers' reading of the
    same term, where the edge lowers inline in the stream's body."""
    node, stream = site.node, parent.site.node
    out = [TilePlan()]
    ctx = term.stream(stream)
    if site.derived:
        out.extend(_chain_values(term, node, work))
    if ctx is not None and work is not None and work.kind == "warp" and legal.enforce(legal.twisted_warp_columns(work), pinned=False):
        atom = ctx.atom
        if legal.enforce(legal.twisted_atom(atom, ctx.d_v), pinned=False):
            um = work.units[0]
            bk = -(-ctx.qk.axis.extent.as_static() // atom.atom_k)  # ceil: an overhanging final atom gmem-zero-fills
            # The grid's ``warps_m`` is the INVENTORY, already fixed above, so what a site
            # enumerates is the free half — the key-atom / query-tile pair, once each.
            for nt, fm in dict.fromkeys((nt, fm) for _, nt, fm in twisted_warp_moves()):
                # A tensor-core row on offer is a fact about the KERNEL, wherever the site sits:
                # every row then carries ``S_warp_eligible`` so the priors can price "a scalar form
                # where tensor cores were on offer" (``features.D_scalar_on_warp_eligible``).
                term.warp_eligible = True
                if site.derived:
                    out.extend(
                        TilePlan(atom=pv, units=(um, 1), regs=(fm, ctx.d_v // pv.atom_n), bk=max(1, nt * atom.atom_n // pv.atom_k))
                        for pv in ctx.pv_atoms
                    )
                else:
                    out.append(TilePlan(atom=atom, units=(um, 1), regs=(fm, nt), bk=bk))
    plans, selected = _narrowed(term, node, out)
    if not selected and not _f16acc_allowed(term.ctx):
        # The f16-accumulate P@V sibling rides the precision gate. A pin that SELECTED here is
        # authoritative and bypasses it (a recorded golden spelling IS the pin); one that named
        # nothing here decided nothing here, so it must not unlock a precision trade by bleeding.
        plans = [p for p in plans if not p.is_warp or p.atom is ctx.atom]
    # Only the SCORE site carries the block-covers-the-extent gate: it is the site whose widths ARE
    # the streaming key block and the query-row block (the P@V tile covers the value dim by
    # construction, and its rows are the score's — reconciled at the stream).
    if not site.derived:
        plans = [p for p in plans if not p.is_warp or legal.enforce(legal.twisted_block(stream, term.sched.placed(node, p)), pinned=False)]
    return [{"TILE": p} for p in plans]


def _stream_tiles(kids: tuple) -> tuple[TilePlan | None, TilePlan | None]:
    """The ``(score, P@V)`` slices a stream's child rows decided — ``None`` where the site took the
    decided empty. The pair travels by SITE, never by position in a flattened list."""
    out: dict[bool, TilePlan | None] = {False: None, True: None}
    for child, row in kids:
        plan = row.plans.get(child.keys.get("TILE"))
        out[child.site.derived] = plan if plan is not None and plan.is_tiled else None
    return out[False], out[True]


def _stream_stages(term: _Term, node, ctx: _Stream, qk: TilePlan, pv: TilePlan) -> list[Stage | None]:
    """The K/V stream's RESOLVED transports for one geometry — gmem-direct ``None`` first (the
    conservative option-0), then every catalog move that resolves against the stream, deduped on the
    resolved spelling (a depth that clamps under the smem budget spells identically to its shallower
    sibling). A pinned ``STAGE`` is authoritative: the resolved pin alone, and NO gmem-direct
    fallback — only this tier stages, so a staging pin that fell through to the chain / reduce
    siblings would let the prior bury the (necessarily lower-occupancy) staged row under a
    higher-occupancy scalar one."""
    budget = term.ctx.max_dynamic_smem

    def resolve(st: Stage) -> Stage | None:
        if st.transport == "tma" and not ctx.tma:
            return None
        if st.split and not (ctx.q_stageable and legal.enforce(legal.stage_split_groups(node), pinned=False)):
            return None
        return legal.resolve_twisted_stage(node, ctx.qk, qk, pv, st, budget)

    pin = term.pin("STAGE", node)
    if pin is not None:
        return [resolve(Stage.parse(pin))] if pin else [None]
    if not ctx.stageable:
        return [None]  # a dtype-mismatched or overhanging operand keeps the tier and drops its stages
    return _resolved(stage_moves(warp=True), resolve)


def _stream_values(term: _Term, node, work: Workers | None, kids: tuple) -> list[dict]:
    """The STREAM fold's own values, given what its derived evaluation decided.

    A stream whose pair took the decided empty realizes per cell, so it offers the ordinary reduce
    partition (:func:`_reduce_values`) — the cooperative / ILP / serial tiers, unchanged. A stream
    whose pair is SCHEDULED realizes the stream itself (fragment residence, or the chain's register
    vector), and those tiers fold the axis their own way: the partition families would stamp a knob
    no kernel realizes, so the two are alternatives rather than a product. What the fragment tier
    does compose with is the cross-CTA SPLIT-KV — ``030_split_reduce`` consumes it, each partial
    keeping its fragment residence.

    The QK / PV SIBLING EQUALITY is enforced here, because this is where the pair is visible: a
    mismatched pair yields NO values, so the combination is never built."""
    qk, pv = _stream_tiles(kids)
    if qk is None and pv is None:
        return _reduce_values(term, node)
    if not legal.enforce(legal.twisted_sites_agree(qk or TilePlan(), pv or TilePlan()), pinned=False):
        return []
    ctx = term.stream(node)
    # A pin the SCHEDULED form cannot honor drops these rows rather than ignoring the pin: the
    # reduce tiers do realize a partition, so the term still maps — through them.
    red_pin = term.pin("REDUCE", node)
    if not (qk is not None and qk.is_warp):
        if red_pin is not None and ReducePlan.parse(red_pin, work) != ReducePlan():
            return []
        if ctx is not None and term.pin("STAGE", node):
            return []  # only the fragment tier stages; a staging pin must not fall through to the chain
        return [{"REDUCE": ReducePlan(), "STAGE": None}]  # the chain: one thread per query row, kv serial
    if ctx is None:
        return []
    placed_qk, placed_pv = term.sched.placed(ctx.qk, qk), term.sched.placed(ctx.pv, pv)
    if red_pin is not None:
        want = ReducePlan.parse(red_pin, work)
        # The pin names the PARTITION; which streaming geometry can carry it is the GEOMETRY's
        # legality, so an incompatible pair drops this row rather than raising — other geometries
        # still carry the pin, and a shape where none can falls to the reduce tiers, which honor it.
        if want != ReducePlan() and not (want.needs_split and legal.enforce(legal.splitkv_slice(node, placed_qk, want), pinned=False)):
            return []
        reduces = [want]
    else:
        reduces = [ReducePlan()]
        # Split-KV siblings on an under-occupied grid: the recorded entries need the rows OFFERED,
        # and a grid that already fills the card has nothing to win from a split. The occupancy read
        # is the WARP row's OWN launch grid — the shrunk query axis, the value axis gone — never the
        # pre-tiled per-cell one, whose element grid would gate the splits off always.
        if _stream_ctas(term, placed_qk, placed_pv) <= _SPLITK_MAX_CTAS:
            reduces += [p for p in splitk_moves() if legal.enforce(legal.splitkv_slice(node, placed_qk, p), pinned=False)]
    stages = _stream_stages(term, node, ctx, placed_qk, placed_pv)
    return [
        {"REDUCE": red, "STAGE": stage}
        for red in reduces
        for stage in stages
        if not (work is not None and work.producer and not legal.enforce(legal.producer_transport(stage, red), pinned=False))
    ]


def _stream_ctas(term: _Term, qk: TilePlan, pv: TilePlan) -> int:
    """How many CTAs a warp-streaming row launches — its own grid: every free axis but the value
    dim (which folds into the P@V fragment), the query axis in CTA blocks."""
    n = 1
    for ax in term.place.free:
        if ax.name == pv.n.axis.name:
            continue
        ext = _hint_extent(ax)
        n *= -(-ext // qk.tile_m) if ax.name == qk.m.axis.name else ext
    return n


def _raster_values(term: _Term) -> list[str]:
    """The ``RASTER`` candidates — kernel-global, and CONTRACTION-scoped: only a 2-D-tiled
    contraction grid decodes the swizzle. A symbolic-axis (masked-tile) grid renders through the
    dynamic decode path, which does not carry it, so offering ``gm8`` there would stamp a launch
    order the kernel doesn't realize."""
    if not any(is_contraction(n.site.node) for n in term.tree):
        return [""]
    if any(not ax.extent.is_static for ax in term.place.free):
        return [""]
    return list(RASTER.narrow(raster_moves()))


def _site_values(term: _Term, site: Site, work: Workers | None, parent: _Node | None = None, kids: tuple = ()) -> list[dict]:
    """The values ``site`` offers under the chosen inventory — TYPED schedule slices, keyed by
    family. Dispatch is the two stored-param predicates on the node, never the ``AxisRole``.

    Two questions a site cannot answer alone travel with it, and both are about the SITE TREE rather
    than the node: ``parent``, because a parent FORM can realize a nested decision itself (the
    cone's statistic, the streaming pair's geometry), and ``kids``, because a value can depend on
    what the subtree decided — the stream's transport is sized by the score tile it feeds, and the
    sibling equality between the two flash sites is checked where the pair is visible."""
    node = site.node
    if parent is not None and _streams(parent.site.node):
        return _twisted_values(term, site, work, parent)
    if node.axis is None:
        return _strip_values(term, node)
    if is_contraction(node):
        return _contraction_values(term, node, work)
    if _streams(node):
        return _stream_values(term, node, work, kids)
    if _fill_realized(parent, site):
        # The one site that offers nothing but the decided empty: its PARENT form realizes the
        # partition itself, so there is no choice left here to spell.
        return [{"REDUCE": ReducePlan(), "STAGE": None}]
    return _reduce_values(term, node)


# ---- the recursion: one row is a joint assignment across the site tree --------------------------- #


def _spell(value) -> str:
    """A slice's stored spelling — ``""`` is the DECIDED empty (the per-cell tile, the serial fold,
    gmem-direct), never an absent key."""
    return value.spell() if value is not None else ""


@dataclass(frozen=True)
class _Row:
    """One enumerated row — the SPELLED knob dict, plus the two facts the kernel's ONE worker
    inventory derives from: the resolved ``TILE`` slices the row carries and the cooperative
    ``REDUCE`` band it claims. ``derive_inventory`` over exactly those is what ``ops.seal_workers``
    computes at materialization, so :func:`_work_holds` and the seal cannot answer differently.

    The slices are kept BY KEY, not as a flat tuple: a parent whose value depends on its subtree
    (the stream's transport, sized by the score tile) must read the child's slice back, and reading
    it out of a flattened list by position is how a two-site term silently swaps its sites."""

    knobs: dict
    plans: dict = field(default_factory=dict)
    coop: int = 1

    @property
    def tiles(self) -> tuple:
        """The row's resolved ``TILE`` slices — what the inventory folds out of."""
        return tuple(self.plans.values())

    @classmethod
    def union(cls, parts: Iterable[_Row]) -> _Row | None:
        """Several rows as ONE — knobs and slices unioned, the cooperative claim RECONCILED.
        ``None`` when the parts cannot share one inventory.

        The claim is a CONSISTENCY, not a maximum. Since step 7 a ``REDUCE`` value spells no coop
        width — the width lives once in ``WORK`` — so two sites claiming DIFFERENT cooperative
        bands spell identically while naming kernels the wire format cannot tell apart. Folding
        them with ``max`` admitted all of them: on the two-site fixture four child widths rode one
        ``t32`` parent as four byte-identical rows. A serial part claims nothing (``coop == 1``)
        and still composes with any other, which is what lets a nested serial fold sit under a warp
        inventory at all.

        Used at BOTH levels a row is assembled — a site with its children (:func:`_merge`) and the
        forest of a term's root sites (:func:`_term_rows`). They stated the rule differently until
        this existed, and the looser of the two was the one that ran on multi-root terms."""
        knobs: dict = {}
        plans: dict = {}
        coop = 1
        for part in parts:
            knobs.update(part.knobs)
            plans.update(part.plans)
            if part.coop > 1:
                if coop > 1 and part.coop != coop:
                    return None  # two sites, two widths, one WORK entry to spell them in
                coop = part.coop
        return cls(knobs=knobs, plans=plans, coop=coop)


def _merge(node: _Node, value: dict, combo: tuple[_Row, ...]) -> _Row | None:
    """One site's row: each family's slice spelled at ITS canonical path key (``Sched.key`` spells
    ANY site, so there are no new keys and no new codec), unioned with the child rows — and with
    them the inventory claim, which is a fact about the whole row, never one site's. ``None`` when
    the sites cannot share ONE inventory (:meth:`_Row.union` owns that rule)."""
    red = value.get("REDUCE")
    tile = value.get("TILE")
    own = _Row(
        knobs={key: _spell(value.get(family)) for family, key in node.keys.items()},
        plans={node.keys["TILE"]: tile} if tile is not None and "TILE" in node.keys else {},
        coop=red.coop if red is not None else 1,
    )
    return _Row.union((own, *combo))


def _rows_at(term: _Term, node: _Node, work: Workers | None, parent: _Node | None = None) -> list[_Row]:
    """Every row the subtree rooted at ``node`` offers under ``work`` — this site's values crossed
    with each child's own rows. The children are enumerated ONCE per inventory, not once per parent
    value: under a fixed ``work`` a child's candidates do not depend on what the parent chose (that
    is what choosing the inventory at the root buys). The dependency that DOES exist runs the other
    way — a site's values may read what its subtree decided (:func:`_site_values`) — so the child
    rows lead and this site's values are asked per combination."""
    children = _kids(node)
    child_rows = [_rows_at(term, c, work, node) for c in children]
    out: list[_Row] = []
    for combo in product(*child_rows):
        kids = tuple(zip(children, combo, strict=True))
        for value in _site_values(term, node.site, work, parent, kids):
            row = _merge(node, value, combo)
            if row is not None:
                out.append(row)
    return out


def _work_holds(row: _Row, work: Workers | None) -> bool:
    """Whether the row's own slices really imply the inventory it claims — :func:`derive_inventory`
    as the VALIDATION the work-first order turns it into, stated ONCE over the whole row. A serial
    fold or an untiled cell claims nothing, so it composes with any parent inventory; a genuine
    conflict (tiled ``TILE`` workers beside a differing coop width, a producer band with no warp
    inventory) is not co-representable and the row is never built."""
    try:
        return derive_inventory(row.tiles, coop=row.coop, producer=work.producer if work is not None else 0) == work
    except ValueError:
        return False  # the enumerator DROPS what ``seal_workers`` raises on — same rule, one home


def _site_inventories(term: _Term, node: _Node, parent: _Node | None = None) -> list[Workers | None]:
    """The inventories the subtree rooted at ``node`` can spell a value against, **its own option-0
    first** — which is what makes the enumeration's leading row the conservative one every
    prior-free path deploys."""
    site = node.site
    out: list[Workers | None] = []
    if parent is not None and _streams(parent.site.node):
        return out  # the streaming pair's warp map is named ONCE, at the stream (below)
    if site.node.axis is None:
        out.append(None)
    elif is_contraction(site.node):
        out.append(None)  # the per-cell tile and the serial fold — the contraction's option-0
        out.extend(Workers.parse(spell) for spell in term.tiles(site.node) if spell)
        # The non-output-tiled tier folds K across a cooperative band, so a contraction claims
        # those inventories too — at the per-cell tile, where the coop moves are offered.
        out.extend(_band_of(p) for p in _contraction_reduces(term, site.node, TilePlan()))
    else:
        out.extend(_band_of(p) for p in _reduce_specs(term, site.node))
        if _streams(site.node) and term.stream(site.node) is not None:
            # The stream's own fragment tier: one warp map, shared by both sites it schedules
            # through (which is why the widths live in the ONE ``WORK`` entry and neither ``TILE``
            # value carries them). m-only — the twisted carrier has no cross-warp merge.
            out.extend(Workers(kind="warp", units=(um, 1)) for um in dict.fromkeys(m for m, _, _ in twisted_warp_moves()))
    for child in _kids(node):
        out.extend(_site_inventories(term, child, node))
    return out


def _inventories(terms: list[_Term]) -> list[Workers | None]:
    """The kernel's ``WORK`` candidates — every inventory any READING's catalogs imply, the OPTION-0
    one leading (the reduce tier's conservative cooperative band, or ``None`` for the per-cell /
    chain / pure-reduce tiers — a first-class inventory), then the ``+p`` producer bands a warp
    inventory can carry. CHOSEN at the root: every site of every reading resolves against it, so
    three of the parent/child couplings stop being rules at all.

    Kernel-global means kernel-global: the list spans the READINGS (a fork has ONE ``WORK`` level),
    which is also what makes the pin fallback below a single decision instead of one per reading."""
    out: list[Workers | None] = []
    seen: set[str] = set()
    for term in terms:
        for node in term.tree:
            for w in _site_inventories(term, node):
                if (spell := w.spell() if w is not None else "") not in seen:
                    seen.add(spell)
                    out.append(w)
    if "" not in seen:
        out.append(None)  # a term with no site still maps its placement — one all-empty row
    for w in list(out):
        if w is None or w.kind != "warp":
            continue
        for band in (1, 2):
            spec = WarpSpec(band)
            if legal.enforce(legal.producer_band(spec, w.count * 32), pinned=False):
                out.append(replace(w, producer=band))
    # The live ``WORK`` pin is AUTHORITATIVE, so the pinned inventory is offered whether or not a
    # catalog implies it — the unit widths a ``TILE`` pin reads off it are exactly what no catalog
    # can predict.
    raw = WORK.raw()
    if raw is None:
        return out
    kept = [w for w in out if values_equal(WORK.name, raw, w.spell() if w is not None else "")]
    if kept:
        return kept
    # THE ONE PLACE A PIN DOES NOT NARROW, and it is the PIN-BLEED rule: one env pin, several
    # kernels in the graph, and this is not the one it was written for (a recognition fork's reduce
    # sibling seeing a matmul's warp pin). The catalog's own inventories stay as siblings so the
    # term still maps — emptying the fork would leave a term unmapped over a pin that was never
    # about it, which is the same degrade the strip site applies to a warp ``TILE`` pin it cannot
    # spell. The reading that used to share this branch — a COVERAGE GAP, where narrowing is right
    # — is gone: the twisted streaming site enumerates its own warp geometry now, so a
    # ``w<M>x<N>`` pin narrows there like anywhere else.
    # ``test_work_pin_widens_only_where_the_site_offers_no_warp_inventory`` pins both halves.
    logger.warning(
        "WORK pin %r matches no candidate's worker inventory (%s offered); offering it beside the full fork",
        raw,
        ", ".join(repr(w.spell() if w is not None else "") for w in out) or "none",
    )
    return [Workers.parse(raw), *out]


def _level_keys(term: _Term) -> list[str]:
    """The site keys ``term``'s own tree decides, family by family, in fork-level order."""
    decided: dict[str, list[str]] = {f: [] for f in FAMILIES}

    def walk(node: _Node) -> None:
        for family, key in node.keys.items():
            decided[family].append(key)
        for child in _kids(node):
            walk(child)

    for node in term.tree:
        walk(node)
    return [k for family in FAMILIES for k in decided[family]]


def _union_keys(terms: list[_Term]) -> list[str]:
    """The fork's site keys between ``WORK`` and ``RASTER`` — the UNION over the readings, in
    fork-level order. A family no reading decided keys the BARE name, and a key one reading lacks is
    stamped there as a DECIDED empty: every leaf of one fork must spell the same family keys, or a
    prefix-consistent evidence pick lets a gmem-direct leaf inherit a staged row's measurement."""
    seen: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for term in terms:
        for key in _level_keys(term):
            fam = family_of(key)
            if key not in seen[fam]:
                seen[fam].append(key)
    return [k for family in FAMILIES for k in (seen[family] or [family])]


def _term_rows(term: _Term, work: Workers | None, rasters: list[str], spelled: str) -> list[dict]:
    """One reading's rows at one inventory — the site product over the term's ROOT sites, filtered
    by the row-level inventory validation, closed by the kernel-global ``RASTER``. The roots
    reconcile through the same :meth:`_Row.union` a site uses for its children: one rule, whichever
    level of the tree assembles the row."""
    out: list[dict] = []
    for combo in product(*(_rows_at(term, node, work) for node in term.tree)):
        row = _Row.union(combo)
        if row is None or not _work_holds(row, work):
            continue
        out.extend({**row.knobs, WORK.name: spelled, RASTER.name: raster} for raster in rasters)
    return out


def _enumerate(terms: list[_Term]) -> tuple[list[dict], list[str], dict]:
    """Every legal schedule row across the term READINGS, in the site value grammar, plus the fork's
    site keys and the row → reading map materialization dispatches on. An empty result is the
    guardrail contract, never a raise: the caller leaves the term unmapped.

    Reading identity must survive into the prior's key space — ``build_fork_tree`` keys leaves on the
    knob dict ALONE, so two readings whose rows spell identically would average two structurally
    different kernels under one feature row. The map below is keyed on exactly that content
    (``canonical_row_key``) and a collision RAISES: the fix is an ``S_*`` stamp, never a new knob."""
    keys = _union_keys(terms)
    #: Every key the union spells, decided-empty — a reading lacking a site stamps the empty there.
    empty = {k: "" for k in keys}
    rows: list[dict] = []
    origin: list[_Term] = []
    for work in _inventories(terms):
        spelled = work.spell() if work is not None else ""
        for term in terms:
            for row in _term_rows(term, work, _raster_values(term), spelled):
                rows.append({**empty, **row})
                origin.append(term)
        if len(rows) > MAX_ROWS:
            raise ValueError(
                f"schedule enumeration for {terms[0].tile.name!r} exceeds the {MAX_ROWS}-row budget "
                f"({len(terms)} readings, {len(keys)} site keys) — the product across sites widened; "
                f"narrow a catalog or add the legality predicate that bounds it, never truncate."
            )
    keys, rows = _decided(keys, rows)
    owner: dict[tuple, _Term] = {}
    for row, term in zip(rows, origin, strict=True):
        ident = canonical_row_key(row)
        if owner.setdefault(ident, term) is not term:
            raise ValueError(
                f"two readings of {term.tile.name!r} spell the SAME row {dict(ident)} — reading identity must "
                f"survive into the prior's key space; distinguish them with an S_* stamp, never a new knob key."
            )
    for term in terms:
        if not rows and term.pin_error is not None and not term.pin_spelled:
            raise term.pin_error  # NO inventory could spell the pin — a pin names a specific kernel
    if any(term.warp_eligible for term in terms):
        # ``S_``-prefixed — not a schedule family, so tile identity and prefix-consistency are
        # untouched; it prices "a scalar tile where tensor cores were on offer".
        for row in rows:
            row["S_warp_eligible"] = 1.0
    return rows, keys, owner


def _decided(keys: list[str], rows: list[dict]) -> tuple[list[str], list[dict]]:
    """The fork's keys and rows with the addressed ``REDUCE`` / ``STAGE`` keys NO row decides
    removed.

    The uniform-key obligation is that every leaf of one fork spells the SAME family keys — not that
    every SITE gets one per family. A site whose partition and transport are not its own to decide
    (the streaming pair's two contractions: their K-step rides ``TILE``, their operands ride the
    stream's ``STAGE``) decides nothing there, and dropping those keys keeps the FEATURIZER honest.
    It reads one node GROUP per distinct ``@<axis>`` element and gives each group the reduce
    geometry when the slice carries a ``REDUCE`` key at all — so a decided-empty ``REDUCE@dd``
    fabricates a partitioned reduce at a site that has none and sum-pools its occupancy into the
    row. Measured on a flash term: ``D_threads`` and ``D_splitk`` tripled and ``D_log2_ctas`` read
    18 instead of 6, which cost the chain and warp forms their cold deploy.

    ``TILE`` keys stay whatever the rows decide: that family is what NAMES the node group, and a
    site offering no tile on this shape is still the site a golden joins against. Bare family keys
    stay too — a bare key is the row's "this family declined" stamp, which the featurizer,
    ``stamp_schedule_families`` and the golden matcher all expect on every row."""
    dead = {k for k in keys if "@" in k and family_of(k) != "TILE" and not any(row.get(k) for row in rows)}
    if not dead:
        return keys, rows
    return [k for k in keys if k not in dead], [{k: v for k, v in row.items() if k not in dead} for row in rows]


# ---- materialization: one builder per form, all fed by the same row ------------------------------ #


def _stamp(term: _Term, op, name, knobs: dict, slices, workers=None, place: Placement | None = None) -> TileOp:
    """Build the scheduled ``TileOp`` — :func:`ops.scheduled` over this term's placement and root
    stores. The term stays pure algebra; no slice is ever a node field.

    ``place`` overrides the per-cell grid for a form that re-places it (the stream's warp shrink,
    the chain's truncation): every placement construction is a closed-form function of (row, term),
    built HERE rather than carried in a row, and ``free`` is never touched — the placed reading each
    site derives (``Sched._mn_for``) is over the free axes, not the grid."""
    return scheduled(
        op,
        name=name,
        place=term.place if place is None else place,
        knobs=knobs,
        stores=term.tile.stores,
        slices=slices,
        workers=workers,
    )


def _strip_variant(term: _Term, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """The pointwise register-STRIP term variant: hand each thread ``r`` CONTIGUOUS inner-axis
    elements. The inner free axis shrinks to ``extent/r`` (the grid walks it) and the cell body is
    unrolled ``r`` times — copy ``i`` reads/writes ``inner·r + i`` with its SSA names suffixed —
    then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride runs feed
    ``050_vectorize_loads`` / ``080_vectorize_stores``. A different term, hence a different
    ``term_key`` and ``op_cache_key`` — which is why it is applied HERE and not at recognition."""
    inner = term.place.free[-1]
    r = plan.reg_n
    op = term.tile.op
    ssa: set[str] = set()
    for s in op.body:
        ssa.update(s.defines())
    loads: list[Stmt] = []
    computes: list[Stmt] = []
    stores: list[Store] = []
    for i in range(r):

        def rename(n: str, i: int = i) -> str:  # suffix only the body's SSA names; axis vars stay
            return f"{n}__u{i}" if n in ssa else n

        sigma = Sigma({inner.name: BinaryExpr("+", BinaryExpr("*", Var(inner.name), Literal(r, "int")), Literal(i, "int"))})
        for s in op.body:
            s2 = s.rewrite(rename, sigma)
            (loads if isinstance(s2, Load) else computes).append(s2)
        stores.extend(Store(write=st.write.rewrite(rename, sigma)) for st in term.tile.stores)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*term.place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    return scheduled(Fold.projection(body=Body((*loads, *computes))), name=name, place=new_place, knobs=knobs, stores=tuple(stores))


def _free_option(term: _Term, plan: TilePlan, name: str, knobs: dict, nested: Sequence[tuple] = ()) -> TileOp:
    """One zero-axis row: the flat per-cell map (also the raw-loop-IR escape's one row), or the
    strip variant when the row's ``TILE`` names a register width. A zero-axis fold with no operands
    has no nested sites, so the strip arm takes none."""
    if _strip_width(plan) > 1:
        return _strip_variant(term, plan, name, knobs)
    return _stamp(term, term.tile.op, name, knobs, nested)


def _node_option(
    term: _Term, node, plan: TilePlan, rplan: ReducePlan, stage: Stage | None, work, name: str, knobs: dict, nested: Sequence[tuple] = ()
) -> TileOp:
    """One un-split row whose compute is a single fold — EITHER reading, and one rule for both.
    What it stores is a property of the resolved plan, not of a role:

    - an UNTILED output stores its K partition on the node. That is a plain reduce's cooperative /
      ILP band, and equally a non-output-tiled contraction's — the contraction is the degenerate
      carrier of its own additive fold, so ``_factor._tile_reduce_axis`` folds it identically;
    - a TILED output stores its tile + transport instead, and contracts K serially per register
      cell. The two tiers differ only in which budget the tile must fit.

    ``scheduled`` skips a ``None`` slice, so a declined resolver and a serial fold need no guard."""
    if not plan.is_tiled:
        own = [("REDUCE", node, rplan if rplan.stages else None), ("STAGE", node, stage)]
        return _stamp(term, term.tile.op, name, knobs, [*own, *nested])
    legal.enforce(legal.warp_k_step(node, plan) if plan.is_warp else legal.scalar_block_threads(plan), pinned=True)
    # The producer band is INVENTORY, and the enumeration only offered this inventory to rows whose
    # stage can drive it (``_legality.producer_transport``) — so there is nothing left to re-check.
    workers = WarpSpec(work.producer) if work is not None and work.producer else None
    own = [("TILE", node, plan.placed_on(term.place)), ("STAGE", node, stage)]
    return _stamp(term, term.tile.op, name, knobs, [*own, *nested], workers=workers)


def _row_stream_tiles(node: _Node, row: dict, work: Workers | None) -> tuple[TilePlan | None, TilePlan | None]:
    """The ``(score, P@V)`` slices a materializing row carries — ``None`` where the site spelled the
    decided empty. Read by SITE (``Site.derived`` tells the synthesized P@V from the hoisted score
    edge), never by position. The materialization twin of :func:`_stream_tiles`, and it ends at the
    same post-condition: each half is ``None`` or a TILED plan.

    The two read "decided" from different places, and that is not drift — it is what each one has.
    The enumeration reads a decided SLICE out of ``_Row.plans``, where an absent key means the site
    chose nothing; here there are only spelled values, and the empty spelling IS the decided empty.
    Resolving ``""`` instead of guarding on it would answer a different question: an empty ``TILE``
    beside a thread inventory legally resolves to a unit-register tile (``resolve_site_tile``'s one
    ambiguity), so every per-cell flash row would come back claiming a stream geometry it never
    chose — which is what the digest and the explicit-mask attention cases say when this guard is
    dropped."""
    out: dict[bool, TilePlan | None] = {False: None, True: None}
    for child in _kids(node):
        spec = row.get(child.keys.get("TILE"), "") or ""
        plan = resolve_site_tile(spec, work) if spec else None
        out[child.site.derived] = plan if plan is not None and plan.is_tiled else None
    return out[False], out[True]


def _warp_stream_place(term: _Term, qk: TilePlan, pv: TilePlan) -> Placement:
    """The grid a WARP-streaming row launches on: the query axis shrinks to its CTA-block count
    (``um`` warps × ``fm`` register query tiles × ``atom_m`` rows each, all read off the placed
    score slice) and the value axis leaves the grid entirely — it folds into the P@V fragment. The
    stream axis never maps: it IS the stream, walked serially per CTA."""
    rows, m_name, d_name = qk.tile_m, qk.m.axis.name, pv.n.axis.name
    grid = tuple(
        Axis(name=ax.name, extent=ax.extent.ceil_div(rows), window=Window(parent=ax.source_axis or ax)) if ax.name == m_name else ax
        for ax in term.place.free
        if ax.name != d_name
    )
    return Placement(free=term.place.free, grid=grid)


def _stream_option(
    term: _Term, node, root: _Node, row: dict, rplan: ReducePlan, work, name: str, knobs: dict, nested: Sequence[tuple] = ()
) -> TileOp:
    """One row of a stream whose DERIVED EVALUATION is scheduled — the fragment-resident warp form
    or the chain's register vector. Both re-place the grid (the reduce tiers do not, which is why
    they stay ``_node_option``'s), and both stamp the pair's slices through ``nested`` exactly as
    the enumeration keyed them; the stream's own ``REDUCE`` is the cross-CTA split-KV or nothing,
    and its ``STAGE`` is the K/V transport re-resolved against the same geometry."""
    qk, pv = _row_stream_tiles(root, row, work)
    ctx = term.stream(node)
    spec = row.get(root.keys.get("STAGE"), "") or ""
    stage = None
    if qk is not None and qk.is_warp and ctx is not None:
        placed_qk, placed_pv = term.sched.placed(ctx.qk, qk), term.sched.placed(ctx.pv, pv)
        place = _warp_stream_place(term, placed_qk, placed_pv)
        if spec:
            stage = legal.resolve_twisted_stage(node, ctx.qk, placed_qk, placed_pv, Stage.parse(spec), term.ctx.max_dynamic_smem)
    else:
        # The chain: one thread per query row, the value axis off the grid and into the register
        # vector the P@V slice names.
        place = Placement(free=term.place.free, grid=tuple(term.place.grid[:-1]))
    workers = WarpSpec(work.producer) if work is not None and work.producer else None
    own = [("REDUCE", node, rplan if rplan.stages else None), ("STAGE", node, stage)]
    return _stamp(term, term.tile.op, name, knobs, [*own, *nested], workers=workers, place=place)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``<k>_ks``) becomes the outer :class:`Fold`'s reduce axis, parallelized across CTAs and summed
    in the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) stays the inner contraction's.
    The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce."""
    legal.enforce(legal.splitk_width(k_axis, w), pinned=True)
    b = k_axis.extent.as_static() // w
    ksplit = Axis(name=f"{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _sliced_a(a, sigma: Sigma):
    """The ``a`` edge σ-reindexed to absolute k for a split partition. A MATERIALIZED edge rewrites
    its gmem index; a COMPUTED cone rewrites its per-cell BODY only — the REDUNDANT-STATISTIC split:
    the cone's row-invariant prologue (the per-row statistic, the K seam ``ops.cone_seam`` reads off
    the node boundary) spans the whole row and stays FULL-ROW in every partition, each recomputing
    it. That redundancy is what the split trades for parallelism, and it is cheap exactly where the
    split pays — the small-free decode shapes the offer is occupancy-gated to."""
    if isinstance(a, Load):
        return replace(a, index=tuple(sigma.apply(e) for e in a.index))
    return a.with_bodies((Body(tuple(s.rewrite(lambda nm: nm, sigma) for s in a.body)),))


def _splitk_option(
    term: _Term, plan: TilePlan, node, rplan: ReducePlan, name: str, knobs: dict, stage_spec: str, nested: Sequence[tuple] = ()
) -> TileOp:
    """One SPLIT-K contraction row — the structural ``Fold(axis=ksplit) ⊃ Fold(axis=kslice)``
    composition ``030_split_reduce`` consumes into the cross-CTA partial + finalize. The inner node
    is the SAME contraction a non-split matmul builds, over ``kslice`` with operands σ-reindexed to
    absolute k; the outer reduce is the IDENTITY-lift composition over it (``Fold.composed``).

    Knob keying stamps against the PRE-SPLIT tree, keeping the kernel single-eligible-axis so the
    golden bare-collapse and the prior featurizer stay invariant."""
    if not plan.is_warp:
        legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    w = rplan.cta
    legal.enforce(legal.splitk_slice_k_step(node, plan, w), pinned=True)
    legal.enforce(legal.splitk_materialized_b(node), pinned=True)
    ksplit, kslice, sigma = _factor_k(node.axis, w)
    inner = Fold.contraction(
        k_axis=kslice,
        a=_sliced_a(node.a, sigma),
        channels=tuple(replace(ch, b=replace(ch.b, index=tuple(sigma.apply(e) for e in ch.b.index))) for ch in node.channels),
    )
    placed = plan.placed_on(term.place)
    # Resolved against the SLICED node, whose K is K/w. A computed-A partial's compute fill is
    # MANDATORY, so it resolves whether or not the row spelled a depth — :func:`_resolve_stage`
    # states that, and the enforce is what turns a declining fill into a loud row rather than a
    # silently gmem-direct one.
    stage = _resolve_stage(term, inner, placed, Stage.parse(stage_spec) if stage_spec else None)
    if not isinstance(node.a, Load):
        budget = term.ctx.max_dynamic_smem
        legal.enforce(None if stage is not None else f"split-K: the sync slabs exceed the {budget} B smem budget", pinned=True)
    # ONE composition rule: the outer reduce is the IDENTITY lift over the sliced contraction
    # operand, its combine the componentwise additive ⊕ over the same accumulator names — the
    # reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``.
    accs = tuple(inner.defines())
    outer = Fold(
        axis=ksplit,
        operands=(inner,),
        lift=Lambda(params=(ksplit.name, *accs), body=Body(()), results=accs),
        **dict(zip(("init", "combine"), M(*(["add"] * len(accs)), names=accs), strict=True)),
    )
    op = Fold.projection(body=term.proj, operands=(outer,)) if len(term.proj) else outer
    # The nested triples key against the PRE-SPLIT nodes the enumeration walked, which the split
    # rewrite leaves untouched below the sliced contraction.
    own = [("REDUCE", outer, rplan), ("TILE", inner, placed), ("STAGE", inner, stage)]
    return _stamp(term, op, name, knobs, [*own, *nested])


def _materialize(term: _Term, row: dict, name: str, knobs: dict) -> TileOp:
    """One row → its ``TileOp``. Each family resolves ONCE against the row's ``WORK`` inventory,
    per key, through the same :class:`Sched` spelling the enumeration used — and the FORM is the
    two node predicates again, never a role."""
    work = Workers.parse(row.get(WORK.name) or None)
    # Structural stamps (``S_warp_eligible``) ride onto the op: fork rows carry them for branch
    # identity, but the MATERIALIZED op is what ``realized_knobs`` reads, and dropping them here
    # left leaf/evidence rows unstamped while fork rows were stamped — fracturing the ``S_*``
    # evidence signature (the 2026-07-07 5090 gate's 330× fp16 miss).
    op_knobs = {**knobs, **{k: v for k, v in row.items() if k.startswith("S_")}}
    raster_spec = row.get(RASTER.name, "")
    Raster.parse(raster_spec)  # loud pin contract — a malformed spelling fails the row here
    op_knobs = {**op_knobs, RASTER.name: raster_spec, **{k: v for k, v in row.items() if family_of(k) in FAMILIES}}

    # The row's own keys — spelled ONCE, when the site tree was built. A family the site does not
    # carry keys the BARE name, which is the decided empty every row spells.
    #
    # ONE root, and it is checked rather than assumed: ``_term_rows`` products over EVERY root of
    # ``term.tree``, so a second root would contribute knobs to the row and then be dropped here —
    # its nested slices never stamped, form dispatch reading the wrong node, and both silently. No
    # live term has one; if a reading ever produces one, this says so instead of mis-materializing.
    if len(term.tree) > 1:
        raise ValueError(
            f"{term.tile.name!r}: {len(term.tree)} root site trees — materialization stamps ONE. "
            "Walk the forest here (as _term_rows does) before a reading may produce this shape."
        )
    root = term.tree[0] if term.tree else None
    keys = root.keys if root is not None else {}

    def value(family: str) -> str:
        return row.get(keys.get(family, family), "") or ""

    site = root.site if root is not None else None
    nested = _nested_slices(term, root, row, work) if root is not None else []
    if site is None or site.node.axis is None:
        return _free_option(term, resolve_site_tile(value("TILE"), work), name, op_knobs, nested)
    node = site.node
    rplan = ReducePlan.parse(value("REDUCE"), work)
    if is_contraction(node) and rplan.needs_split:
        split_plan = resolve_site_tile(value("TILE"), work, rplan.spell())
        return _splitk_option(term, split_plan, node, rplan, name, op_knobs, value("STAGE"), nested)
    if _streams(node) and any(t is not None for t in _row_stream_tiles(root, row, work)):
        return _stream_option(term, node, root, row, rplan, work, name, op_knobs, nested)
    plan = resolve_site_tile(value("TILE"), work, rplan.spell())
    return _node_option(term, node, plan, rplan, _stage_of(term, node, plan, value("STAGE")), work, name, op_knobs, nested)


def _nested_slices(term: _Term, node: _Node, row: dict, work: Workers | None) -> list[tuple]:
    """Every NESTED site's resolved slices, as the ``(family, node, value)`` triples ``scheduled``
    keys — materialization's half of the recursion :func:`_rows_at` already does.

    The enumeration walks the whole site tree, so a row DECIDES every site; stamping the root alone
    left a nested key as a knob no kernel realized — the row said ``REDUCE@j=r2`` and the op's
    schedule came back empty. The walk descends through :func:`_kids`, the same accessor
    ``_rows_at`` uses, so what materializes is what was enumerated. A site whose value is the decided empty resolves
    to ``None`` and ``scheduled`` skips it, which is why the corpus terms — whose one nested site is
    the cone statistic the parent fill realizes — stamp nothing new."""
    out: list[tuple] = []
    for child in _kids(node):
        cnode, keys = child.site.node, child.keys

        def spec(family: str, keys: dict = keys) -> str:
            return row.get(keys.get(family), "") or ""

        rplan = ReducePlan.parse(spec("REDUCE"), work)
        tile_spec = spec("TILE")
        plan = resolve_site_tile(tile_spec, work, rplan.spell())
        out.append(("REDUCE", cnode, rplan if rplan.stages else None))
        out.append(("STAGE", cnode, _stage_of(term, cnode, plan, spec("STAGE"))))
        if "TILE" in keys and tile_spec:
            # Stored UNPLACED: which ``(m, n)`` pair a nested site tiles is a function of its
            # POSITION (``Sched._mn_for`` — the parent fold's axis for a hoisted edge, the trailing
            # free pair for a derived one), so binding it here with the ROOT's rule would name the
            # wrong axes. ``Sched.tile_of`` binds at read, through the one home.
            out.append(("TILE", cnode, plan))
        out.extend(_nested_slices(term, child, row, work))
    return out


def _stage_of(term: _Term, node, plan: TilePlan, spec: str) -> Stage | None:
    """The row's ``STAGE`` re-resolved against the node — the operand pipeline on a tiled
    contraction, the shared ROW buffer on any other fold, dispatched by the same predicate the
    enumeration used. The row carries what the enumeration RESOLVED, so this reproduces the slice
    the leaf identity was built from, through :func:`_resolve_stage`'s one dispatch."""
    if not spec:
        return None
    if not is_contraction(node):
        return _row_stage(term, node)
    if not plan.is_tiled:
        return None
    return _resolve_stage(term, node, plan.placed_on(term.place), Stage.parse(spec))


# ---- the entry point ----------------------------------------------------------------------------- #


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> Fork | list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling fork.

    Returns the lazy fork tree over the enumerated rows (levels ``[WORK, *site keys, RASTER]`` — the
    kernel-global worker inventory leads, so every deeper prefix row is self-decoding; the
    launch-order codec closes), a single ``TileOp`` when the space collapses to one row, or ``[]``
    when nothing is enumerable (the guardrail contract — the caller leaves the term unmapped)."""
    rows, keys, owner = _enumerate(_readings(tile, ctx))
    if not rows:
        return []

    def materialize(row: dict) -> TileOp:
        return _materialize(owner[canonical_row_key(row)], row, name, knobs)

    if len(rows) == 1:
        return materialize(rows[0])

    def _level(key: str) -> Level:
        return Level((key,), key=lambda r: (r.get(key, ""),))

    levels = [_level(WORK.name), *(_level(k) for k in keys), _level(RASTER.name)]
    return build_fork_tree(params=rows, levels=levels, materialize=materialize)


__all__ = ["FAMILIES", "MAX_ROWS", "schedule"]
