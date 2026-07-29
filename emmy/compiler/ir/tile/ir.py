"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine* — the :class:`Map` / :class:`Fold` / :class:`ContractionView`
nodes defined **in this module**, alongside ``ir/stmt/algebra``) directly,
plus a thin set of **root-global schedule fields** — the
free-axis → grid :class:`~.schedule.Placement` (``place``) and the warp split
(``workers``). The
per-node schedule slices ride the structural nodes themselves (a
:class:`ContractionView`'s ``tile``, a :class:`Fold`'s
``reduce``, a :class:`ContractionView`'s / :class:`Fold`'s ``stage`` — EVERY schedule slice rides
the node it decorates, none on the ``TileOp`` (flash is now a
``Map(sources=(Fold(step=[ContractionView(QK), …, ContractionView(PV)]),))`` node tree, so its
partition rides the node). There is no per-kind kernel/schedule type: the algebra is read
structurally off the axes' :class:`~emmy.compiler.ir.axis.AxisRole`
(``ops.axis_role``), so MAP / MONOID / SEMIRING all ride the same ``TileOp``.

**Operand sharing is arity.** "These two matmuls read the same A" is ONE
:class:`ContractionView` whose output is a tuple: one ``a`` edge plus N product
:class:`Channel`\\ s ``(b_i, acc_i)``, folding the componentwise ``(+, ×)`` —
the N-component product monoid the derived :attr:`ContractionView.loop` carries.
There is no let table and no name-reference mechanism: the one in-tree relation
that had several consumers is a single edge, so a shared subtree has exactly one
home by construction.

**An operand is an edge with two inhabitants** — the two things an input can
be: MATERIALIZED (a gmem ``Load``) or COMPUTED (the node itself, stored inline
on the edge). Tree ownership gives an inline node exactly one consumer — its
parent — so sharing needs no reference arm. The node boundary a reader wants
(``ops.cone_seam``'s prologue / per-cell split) is read straight off the edge;
``lower`` flattens it once, at the point of use. A subtree that reads no value
name from its enclosing body is **closed** (:func:`captured_values`); closure is
the precondition for lifting any subtree into its own kernel (a placement cut),
and nothing else requires it — flash's ``P`` legitimately captures the running
max its own loop step updates, and that seam is simply not cuttable.

**Every structural node is a ``Stmt``.** A composed step occupies a statement
position in another node's body — flash's ``Σ_dd Q·K`` and ``Σ_j P·V`` in a
reduce ``step``, split-K's sliced contraction, the fused sibling group inside
a split reduce — and its POSITION there is semantic: flash's PV reads the softmax
weight the merge stmts of that same loop step produce, so it cannot be hoisted
ahead of them. Composition therefore rides the sequence, not a hoisted operand
tuple, and uniform ``Stmt``-hood is what makes a node a legal member of a
``Body`` (generic walks reach its children through ``nested()`` like any block
stmt's). ``Map.sources`` are the exception on purpose: they are node EDGES,
reached by the node-aware walk (:func:`tree_nodes`), like a contraction operand.

The combine lives entirely in the ``op`` wrapper (the :class:`Map` /
:class:`Fold` / :class:`ContractionView` nodes here + ``ir/stmt/algebra``): a
node whose per-cell loop nest carries the role (``AxisRole``) + the decoupled
``Carrier`` (the ⊕ algebra). The algebra is **not stored as a node kind**; the
role/carrier are read off the node / annotated loop where a pass needs them
(``ops.axis_role`` / ``ops.reduce_loop``). ``lower(op)`` flattens the structural
tree back to the loop nest.

The structural nodes are the **post-schedule** skeleton a ``TileOp`` is built from:
each holds its own scheduling parameters (the contraction's :class:`~.schedule.TilePlan`)
plus its algebra params, with the derived geometry exposed as ``@property`` (so
``structural_key`` digests only the compact param fields and the ``--ir`` dumps stay
readable). The kernel materializer reads the schedule straight off the node — it never
re-recognizes structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expr import Expr, Literal
from emmy.compiler.ir.schedule import Placement, ReducePlan, Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carrier, Lambda, Load, Loop, RenderCtx, Stmt
from emmy.compiler.ir.stmt.body import effectful_lambda

if TYPE_CHECKING:
    from emmy.compiler.ir.atom import Atom


def _ext_expr(axis: Axis) -> Expr:
    """The axis extent as an ``Expr`` — a literal int (static) or the symbolic ``Dim`` expr."""
    return Literal(axis.extent.as_static(), "int") if axis.extent.is_static else axis.extent.expr


def _overhangs(axis: Axis, tile: int) -> bool:
    """True iff a ``tile``-wide CTA block overhangs ``axis`` (symbolic or non-divisible extent) —
    so its tail must be masked."""
    if tile <= 1:
        return False
    return not (axis.extent.is_static and axis.extent.as_static() % tile == 0)


def _splice_operands(operands: tuple, stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Splice each operand edge's producing stmts into ``stmts`` immediately BEFORE the first stmt
    that reads the operand's bound name (appended when nothing reads it), ties resolved in operand
    TUPLE order. This is the one lowering rule that turns the stored ``(operands, step)`` pair back
    into the flat loop body — deterministic, so the derived loop (and with it ``op_cache_key``)
    depends only on the stored params."""
    if not operands:
        return stmts
    at: dict[int, list] = {}
    for edge in operands:  # first-use indexes against the ORIGINAL stmts — ties keep tuple order
        name = _operand_name(edge)
        idx = next((i for i, st in enumerate(stmts) if name in _deep_reads([st])), len(stmts))
        at.setdefault(idx, []).append(edge)
    out: list[Stmt] = []
    for i, st in enumerate((*stmts, None)):
        for edge in at.get(i, ()):
            out.extend(_operand_body(edge))
        if st is not None:
            out.append(st)
    return tuple(out)


def _flatten_nodes(body: Body) -> tuple[Stmt, ...]:
    """Flatten any nested **structural node** — a :class:`ContractionView`, :class:`Fold`, or
    :class:`Map` — that sits as a stmt in ``body`` to its own lowered loop nest; plain stmts pass
    through. All three node kinds ARE ``Stmt``\\ s, which is what makes a composed step a legal member
    of a ``Body``: flash's ``Σ_dd Q·K`` score and its ``Σ_j P·V``, split-K's sliced contraction, and
    the fused sibling group (a ``Map``) inside a split reduce's partial. This is the single
    **node-walk** the ``.loop`` splice and the kernel materializer's recursion share, and it yields
    the same loop nest whether a node was pre-flattened or reached structurally.

    A composed step's POSITION in the body is semantic, not incidental: flash's PV reads the softmax
    weight the merge stmts of that same loop step produce, so the step cannot be hoisted ahead of
    them. That is why composition rides the sequence rather than a hoisted operand tuple."""
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    out: list[Stmt] = []
    for s in body:
        if isinstance(s, (Fold, Map)):
            out.extend(lower(s))
        else:
            out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class Fold(Stmt):
    """A scheduled reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/stmt/algebra``). It splits the reduce's **algebra** (the loop-carried
    :class:`~emmy.compiler.ir.stmt.algebra.Carrier` — degenerate ``id`` for a plain
    ``sum`` / ``max`` / ``mean``, twisted ``exp`` for online-softmax / flash) from its **structure**
    (the reduce ``axis`` + the per-element ``step`` it folds). Its :class:`AxisRole`
    (``PLANAR`` / ``TWISTED`` / ``CONTRACTION``) is **derived** from those params (:attr:`role`),
    never stored. The fold ``Loop`` is **synthesized on
    demand** (:attr:`loop`), never stored — so the same node tiles under any
    :class:`~emmy.compiler.ir.schedule.ReducePlan` (the reduce partition rides the node's
    ``reduce`` field, read via ``ops.reduce_plan``).

    A reduce whose per-step partial COMPOSES another node — split-K's ``Fold ⊃ ContractionView``
    (whose ``axis`` ``ksplit`` differs from the inner ``k_axis`` ``kslice``, so no double-reduce),
    flash's ``Σ Q·K`` score at the head of its kv step — spells it ONE way: the node sits in
    ``step`` and :func:`_flatten_nodes` flattens it in place. There is no second ``source`` edge.

    It holds **no projection**: a bare reduce (``sum`` / ``max``) is the kernel root (its grid ``Write``
    is glue); a reduce with a post-fold sweep (softmax / RMSNorm) is the ``source`` of a wrapping
    :class:`Map` whose body IS that projection. Like every structural node it IS a ``Stmt`` — that is
    what lets a composed step occupy a statement position in another node's body;
    :func:`ops.lower` flattens it to the synthesized loop (``[loop]``), so ``op_cache_key`` and the
    ``_factor._tile_reduce_axis`` expander stay byte-identical to the bare-loop form.

    The **scheduling param** is the ``reduce`` partition (:class:`ReducePlan` — GRID split / BLOCK coop
    / REG ILP), stamped onto the node by ``_schedule`` (inside ``010_recognize``) (its decided value lives **here** on the node
    — read via ``ops.reduce_plan``). ``lower`` ignores it (it's metadata the materializer / ``030_split_reduce``
    read), so it leaves ``op_cache_key`` byte-identical."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    carrier: Carrier  # the loop-carried ⊕ algebra (degenerate id / twisted exp / the product monoid)
    axis: Axis  # the reduce axis
    step: Body = field(default_factory=Body)  # the per-cell fold STEP (the reduce Loop's body, operands excluded)
    unroll: bool = False
    # The CLOSED inputs, each an operand edge (a gmem ``Load`` or an inline node) — the 1k fold
    # vocabulary. Sharing is edge reuse: the step reads an operand's bound name as many times as it
    # needs. ``lower`` splices each edge's body before its first use (:func:`_splice_operands`).
    operands: tuple = ()
    reduce: ReducePlan = field(default_factory=ReducePlan)  # the reduce partition (schedule slice), stamped by the _schedule helper
    # The output tile (schedule slice) of a role=CONTRACTION fold — the ``TilePlan`` the view
    # (``ops.contraction_view``) reads back; ``None`` for a plain / twisted reduce.
    tile: TilePlan | None = None
    # The operand smem pipeline this reduce runs under (schedule slice): the cooperative tier's
    # shared-row ``sync`` stage, or a warp-flash tree's resolved K/V stage. ``None`` = gmem-direct.
    stage: Stage | None = None
    # A cross-CTA SLICE of the stream (flash split-KV) is not spelled here: ``030_split_reduce``
    # shrinks ``axis`` to the slice length and the slice's absolute base / end ride that axis's
    # :class:`~emmy.compiler.ir.axis.Window` — ONE windowing vocabulary, the same one an axis's
    # split parentage uses, read by the realizer and the mask machinery alike.

    def __post_init__(self) -> None:
        if not isinstance(self.step, Body):
            object.__setattr__(self, "step", Body.coerce(self.step))

    @classmethod
    def from_loop(cls, loop: Loop) -> Fold:
        """Build a :class:`Fold` from an already-annotated reduce ``Loop`` (its ``carrier`` /
        ``axis`` / body — the role is DERIVED, see :attr:`role`, so the loop's own annotation is
        not captured) — the recognize-side constructor. :attr:`loop` reconstructs the exact same
        ``Loop`` whenever the captured annotation agrees with the derivation (every recognizer
        loop; any post-fold projection rides the wrapping ``Map``, not here)."""
        return cls(carrier=loop.carrier, axis=loop.axis, step=loop.body, unroll=loop.unroll)

    @property
    def role(self) -> AxisRole:
        """The fold's :class:`AxisRole`, DERIVED from the stored params (1l — never stored):

        - ``TWISTED`` iff the carrier's twist family is non-degenerate (``exp`` — online softmax /
          flash); the PLANAR/TWISTED half of the old stored role was redundant with the carrier.
        - ``CONTRACTION`` iff the step is the bilinear lift/fold form over hoisted operand edges
          (:func:`_parse_bilinear` — the ``as_fold`` storage shape), or composes exactly the
          sliced contraction fold (split-K's outer reduce).
        - ``PLANAR`` otherwise — which is where the old recognition-time DEMOTION now lives
          structurally: an unbindable contraction (matvec-shaped 1-D output, no ``(m, n)`` loads,
          the zero-legal-rows fallback) never hoists its operands, so its loads stay inline in
          ``step``, the parse declines, and the fold takes the reduce tiers at schedule dispatch.
          No role rewrite; the lowered loop's ``PLANAR`` annotation falls out of the same read."""
        if self.carrier.twist.family != "id":
            return AxisRole.TWISTED
        if _parse_bilinear(self) is not None:
            return AxisRole.CONTRACTION
        if len(self.step) == 1 and is_contraction_fold(self.step[0]):
            return AxisRole.CONTRACTION  # split-K: the outer additive reduce over the sliced fold
        return AxisRole.PLANAR

    @property
    def loop(self) -> Loop:
        """The synthesized annotated reduce ``Loop`` — reconstructed from the params. With a plain
        ``step`` it is byte-identical to the loop :meth:`from_loop` captured. Any **nested
        structural node inside the ``step``** — a :class:`ContractionView` / :class:`Fold` /
        :class:`Map` — is flattened to its own loop nest in place by the shared :func:`_flatten_nodes`
        node-walk (so flash's kv loop holds the head ``Σ Q·K`` score contraction loop and the embedded
        ``Σ_j P·V`` PV contraction loop, and split-K's kslice contraction its own nest — exactly the
        loop-in-body form the scalar tier expands): ONE structural rule for a reduce whose per-step
        partial composes other nodes. Operand edges splice in ahead of their first use
        (:func:`_splice_operands`), so a fold with hoisted inputs lowers byte-identically to the
        flat form that carried them in its body."""
        stmts = _splice_operands(self.operands, _flatten_nodes(self.step))
        return Loop(axis=self.axis, body=Body(stmts), unroll=self.unroll, role=self.role, carrier=self.carrier)

    def spliced_step(self) -> tuple[Stmt, ...]:
        """The step with every operand edge's body spliced before its first use — the stmt sequence
        the emit-side node walk consumes (nested structural nodes NOT flattened; :attr:`loop`
        additionally flattens them)."""
        return _splice_operands(self.operands, tuple(self.step))

    @property
    def out(self) -> str:
        """The bound output name — the carrier state's primary component (a bare reduce's grid
        ``Write`` is glue; a projected reduce's output name lives on the wrapping ``Map``)."""
        return self.carrier.out

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — just the synthesized reduce
        ``Loop`` (a wrapping ``Map`` appends its projection)."""
        return [self.loop]

    # ---- the stmt protocol: a composed step occupies a statement position, so generic body walks
    # must reach its children. ``defines`` stays the block-stmt default — the fold's names are bound
    # by the ``Accum``\\ s inside ``step``, exactly as for a plain reduce ``Loop``. ---------- #
    def nested(self) -> tuple[Body, ...]:
        return (self.step,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (partial,) = bodies
        return replace(self, step=partial)

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Fold must be lowered (ops.lower) before render")


@dataclass(frozen=True)
class Side:
    """One tiled output axis of a contraction — the outer ``m`` or inner ``n`` — paired with its
    derived per-CTA tile geometry. The two ride as a ``(m, n)`` pair (:attr:`ContractionView.mn`)
    mirroring the schedule's ``(m, n)`` tuples (``TilePlan.units`` / ``regs``), so the factorizer
    threads one object per axis instead of a dozen loose ``*_m`` / ``*_n`` args. The tile width,
    unit / register counts, and bound block/unit var names are stamped by :meth:`ContractionView._side`;
    the ``mask`` / ``ext`` are derived from the axis + width."""

    axis: Axis  # the output axis (a param)
    tile: int  # the per-CTA width = units · reg · atom_dim
    units: int  # parallel units on this axis (warps for mma / threads for scalar)
    reg: int  # register sub-cells per unit on this axis
    block: str  # the bound grid-block var (the axis name + ``_b``)
    unit: str  # the bound unit var (the axis name + ``_u``)

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def mask(self) -> bool:
        """True iff a ``tile``-wide CTA block overhangs the axis (its tail must be masked)."""
        return _overhangs(self.axis, self.tile)

    @property
    def ext(self) -> Expr:
        """The axis extent as an ``Expr`` — a static literal or the symbolic ``Dim`` expr."""
        return _ext_expr(self.axis)


def _deep_defines(s: Stmt) -> set[str]:
    """Every SSA name defined in ``s`` (deep — a stat reduce ``Loop``'s ``Accum`` counts)."""
    out = set(s.defines())
    for b in s.nested():
        for child in b:
            out |= _deep_defines(child)
    return out


def _deep_reads(stmts: list[Stmt]) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep)."""
    out: set[str] = set()
    for s in stmts:
        out |= set(s.deps())
        for b in s.nested():
            out |= _deep_reads(list(b))
    return out


def _stmt_axis_names(stmts) -> set[str]:
    """Every loop induction variable bound anywhere in ``stmts`` (deep). A composed structural node
    sitting in the body needs no special case — it is a ``Stmt``, so its children are reached through
    the same ``nested()`` walk as any block stmt's."""
    out: set[str] = set()
    for s in stmts:
        ax = getattr(s, "axis", None)
        if ax is not None and hasattr(ax, "name"):
            out.add(ax.name)
        for b in s.nested():
            out |= _stmt_axis_names(b)
    return out


def axis_names(root) -> set[str]:
    """Every ITERATION-SPACE name in ``root``'s tree — the structural nodes' axes plus every loop
    induction variable in their bodies. An induction variable is bound by the enclosing loop nest,
    not by any value tree, so a subtree reading one is never capturing a value."""
    out: set[str] = set()
    for node in tree_nodes(root):
        if isinstance(node, Fold):
            out.add(node.axis.name)
            out |= _stmt_axis_names(node.step)
        elif isinstance(node, Map):
            out |= _stmt_axis_names(node.body)
    return out


def captured_values(root, axes: set[str]) -> tuple[str, ...]:
    """The VALUE names ``root``'s subtree reads but does not itself define — its capture set, with
    iteration-space names (``axes``) excluded.

    Empty means the subtree is **closed**: it can be lifted out of its enclosing body and computed
    on its own, which is exactly what a placement cut does to a seam. A non-empty capture set is not
    an error — flash's ``P = exp(s − m)`` genuinely reads the online-softmax carrier's
    running max, updated by the merge stmts of the very loop step that consumes it (legal: an inline
    operand's one home is always inside the scope it captures from) — but it IS the
    reason such a seam is not cuttable. This is the predicate a placement cut asks before lifting
    any subtree into its own kernel.

    Returned sorted, so callers can put it straight into an error message."""
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    stmts = list(lower(root))
    defs: set[str] = set()
    for s in stmts:
        defs |= _deep_defines(s)
    return tuple(sorted(_deep_reads(stmts) - defs - axes))


def _operand_body(op) -> tuple[Stmt, ...]:
    """An operand edge's producing stmts — the singleton gmem ``Load``, or the inline node
    flattened."""
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    return (op,) if isinstance(op, Load) else tuple(lower(op))


def _operand_name(op) -> str:
    """An operand edge's bound SSA name — the inline node's ``out``, or the ``Load``'s def."""
    return op.defines()[-1] if isinstance(op, Load) else op.out


def _refs_axis(s: Stmt, name: str) -> bool:
    """``s`` references axis ``name`` in any index expr (deep)."""
    idx = getattr(s, "index", None)
    if idx and any(name in e.free_vars() for e in idx):
        return True
    return any(_refs_axis(child, name) for b in s.nested() for child in b)


def gmem_row_stride(load: Load, axis_name: str, inputs) -> int | None:
    """The flattened gmem stride between successive ``axis_name`` rows of ``load``'s buffer — the
    fragment loaders' row step (``ldm``). The axis var must appear in exactly ONE index component,
    affinely; the stride is its coefficient times the product of the buffer extents AFTER that
    component. ``head_dim`` for the canonical ``(batch…, row, dd)`` layout; ``H·D`` for an
    un-transposed ``(B, S, H, D)`` trace, where assuming the trailing extent read the WRONG rows
    (the gemma layer-0 NaN: resident-Q fragments at ``ldm=head_dim`` on a 4096-stride layout).
    ``None`` when underivable — axis absent or split across components, a non-affine use, an
    unknown buffer, or a symbolic trailing extent — the schedule gate's decline signal."""
    from emmy.compiler.ir.expr import affine_form  # noqa: PLC0415 — avoid a module-import cycle

    t = inputs.get(load.input) if inputs else None
    if t is None:
        return None
    positions = [i for i, e in enumerate(load.index) if axis_name in e.free_vars()]
    if len(positions) != 1:
        return None
    pos = positions[0]
    form = affine_form(load.index[pos], {axis_name})
    if form is None:
        return None
    coef = form[1].get(axis_name, 0)
    if coef <= 0:
        return None
    stride = coef
    for d in tuple(t.shape)[pos + 1 :]:
        if not d.is_static:
            return None
        stride *= d.as_static()
    return stride


@dataclass(frozen=True)
class Channel:
    """One product channel of a :class:`ContractionView` — the streamed K×N operand edge ``b`` plus the
    additive fold accumulator ``acc`` that channel produces. A plain matmul is one channel; the
    fused gate⊗up MLP edge is two channels over the node's single shared ``a`` (sharing is arity,
    not naming — the product-carrier contraction outputs a tuple)."""

    b: Load | Map | Fold | ContractionView  # the streamed operand edge — MATERIALIZED or COMPUTED
    acc: str  # this channel's fold accumulator


@dataclass(frozen=True)
class ContractionView:
    """The DERIVED bilinear view of a ``role=CONTRACTION`` fold — never stored (build it via
    :func:`contraction_view`; store :meth:`as_fold`). A contraction **before** atom factorization
    — built **recognize-side** at fork-emit
    (``_schedule._contraction_node`` resolves the operand→role binding via ``_atomize.semiring_binding``
    and stamps the resolved ``tile``), then expanded in ``010_materialize`` (``_factor.factorize``).
    :func:`ops.lower` / ``ops.reduce_loop`` flatten it back to the synthesized mul-add ``CONTRACTION``
    loop nest (:attr:`loop`), the same ``for k: v = a·b; acc += v`` form ``_atom._ScalarOps.reduce``
    register-tiles through the shared ``_contract_kloop`` skeleton. **ONE
    flat node** that cleanly splits the **algebra params** (what to contract) from the **schedule**
    (how to tile it): the params are the tiled output ``axes`` ``(m, n)``, the contraction ``k_axis``,
    the leading batch ``lead_axes``, the shared ``a`` operand edge and the product ``channels``
    ``(b_i, acc_i)``; the projection has ONE home — the wrapping :class:`Map`'s body — never a
    node field; the schedule is the
    one ``tile`` field — a resolved :class:`~emmy.compiler.ir.schedule.TilePlan` carrying the
    leaf ``atom`` (tensor-core :class:`AtomKind` or the ``1×1`` :class:`ScalarAtom`), the per-CTA
    **UNIT** grid + per-unit **REGISTER** sub-tile, and the K-chunk. Keeping the schedule a single
    swappable field is what lets the same operand/acc params be tiled by a different ``TilePlan`` (the
    flash inner QK/PV reuse). Arity N ≥ 2 is the fused sibling edge (gate⊗up) — a product semiring
    outputting N matrices over ONE shared A, scheduled and lowered as one unit under this node's
    single ``tile`` / ``stage`` row.

    The contraction itself is **never stored** — both tiers *synthesize* it from the operands:
    ``_atom.reduce_codegen`` lowers the mma atom into ``ldmatrix`` + ``mma.sync`` and the scalar atom into a
    ``for k: acc += a*b`` register-tiled loop — then run the projection peeled off the wrapping
    ``Map`` (``acc`` is the SSA name the synthesized reduce produces and the projection consumes). The
    operand buffers ride :meth:`external_reads`; the node has no nested ``Body``. ``_factor.factorize`` reads the
    **derived** geometry below (the ``(m, n)`` :class:`Side` pair — ``tile`` / ``mask`` / ``block`` /
    ``unit`` per axis — plus ``block_threads`` / ``b_trans``) straight off the node; it's ``@property``, so it stays out of the fields
    ``structural_key`` digests (the node IS keyed as an intermediate ``KernelOp``). The atom selects
    the codegen — there is no separate ``Leaf`` / per-atom subclass."""

    axes: tuple[Axis, Axis]  # the tiled output (m_axis, n_axis) — params
    k_axis: Axis  # the contraction axis — params
    # A and every channel's B are **operand edges** of the SAME type, whose two inhabitants are the
    # two things an input can be — MATERIALIZED (a gmem ``Load``) or COMPUTED (the node itself,
    # stored inline; ``_atomize.make_cone`` is the one cone builder). Tree ownership gives an inline
    # node exactly one consumer — sharing is this node's ARITY, never a name.
    #
    # The edges are SYMMETRIC in the algebra and asymmetric only in the SCHEDULE: A is the
    # M-resident operand (held across the K loop, so it can be compute-filled), each B the K×N
    # operand the K loop streams (staged through smem / TMA / cp.async, its buffer dtype selecting
    # the atom). That is a scheduling fact, so it lives in the schedule gates — every tier that
    # needs B's gmem address states ``isinstance(c.b, Load)`` as an explicit eligibility
    # precondition and declines otherwise — not in the structural type. A computed B lowers on the
    # gmem-direct scalar tier through the same ``contraction_loop`` builder a computed A does.
    a: Load | Map | Fold | ContractionView
    channels: tuple[Channel, ...]  # the product channels (b_i, acc_i) — params; arity 1 = plain matmul
    tile: TilePlan  # the schedule: leaf atom + unit/register widths + K-chunk
    lead_axes: tuple[Axis, ...] = ()  # params
    stage: Stage | None = None  # the schedule: the resolved operand smem pipeline (None = gmem-direct)

    @property
    def out(self) -> str:
        """The bound output name — the PRIMARY channel's fold accumulator (a bare contraction's grid
        ``Write`` stores it at the cell; a fused-projection contraction carries its own ``Write``)."""
        return self.channels[0].acc

    @property
    def acc(self) -> str:
        """The primary channel's fold accumulator (``channels[0]``) — the single-channel read every
        arity-1 consumer uses, and the group's primary component at arity N."""
        return self.channels[0].acc

    # ---- the operand edges, read through ONE set of accessors each. ``b``-prefixed reads are the
    # PRIMARY channel's: single-channel tiers read them under their own arity gates, and the layout
    # facts (``b_trans``) read them because channel agreement is a formation invariant. ---------- #
    @property
    def b(self):
        """The primary channel's streamed operand edge (``channels[0].b``)."""
        return self.channels[0].b

    @property
    def a_body(self) -> tuple[Stmt, ...]:
        """The A operand's producing stmts — a singleton gmem ``Load``, or the inline cone node
        flattened (a **register-resident** A: flash PV's ``P = exp(S − M)``, produced from an
        in-register score, not a gmem address). The last stmt's def is the operand value
        ``contraction_loop`` multiplies."""
        return _operand_body(self.a)

    @property
    def b_body(self) -> tuple[Stmt, ...]:
        """The primary B operand's producing stmts — symmetric to :attr:`a_body`. A gmem ``Load``
        for every B built today; a computed B (a fused per-column prologue — qk-norm / RoPE folded
        into a score, an on-the-fly dequant) would flatten here the same way, and the gmem-direct
        scalar tier consumes it through the same :attr:`loop` builder."""
        return _operand_body(self.b)

    @property
    def a_computed(self) -> bool:
        """True when A is a computed register-resident operand (an inline cone node), not a gmem
        ``Load`` — the mma tier reads it as a fragment, the scalar tier as the value."""
        return not isinstance(self.a, Load)

    @property
    def b_computed(self) -> bool:
        """True when the primary B is computed rather than a gmem ``Load``. Every staged tier
        (cp.async / TMA / the sync compute-fill's B channels) needs B's gmem address, so each gates
        on ``isinstance(c.b, Load)`` and declines here — the schedule states the asymmetry the
        structural type deliberately does not."""
        return not isinstance(self.b, Load)

    @property
    def a_name(self) -> str:
        """The A operand's bound SSA name — the inline node's ``out``, or the ``Load``'s def."""
        return _operand_name(self.a)

    @property
    def b_name(self) -> str:
        """The primary B operand's bound SSA name — symmetric to :attr:`a_name`."""
        return _operand_name(self.b)

    @property
    def loop(self) -> Loop:
        """The synthesized ``CONTRACTION`` reduce ``Loop`` — the canonical ``for k: v = a*b; acc += v``
        mul-add form (built by the shared ``ops.contraction_loop``, the same fold ``_factor``'s scalar
        contraction tier register-tiles). Lets :func:`ops.lower` / ``ops.reduce_loop`` flatten the node
        back to the loop nest; the node never stores the loop.

        At arity N the ONE fused group loop is DERIVED here (never stored, exactly like
        :attr:`Fold.loop`): the shared A is lifted once (the primary channel's loop), each
        further channel splices its ``b → ⊗ → ⊕`` triple after it reusing that A value, and the
        carrier is the N-COMPONENT identity-family state (one additive component per channel) — the
        true product-monoid state, so the cross-CTA split tier folds every channel's partials;
        ``carrier.out`` stays the primary component, and the coop tier remains contraction-blind."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from emmy.compiler.ir.tile.ops import contraction_loop  # noqa: PLC0415 — avoid an import cycle

        base = contraction_loop(
            lift=ElementwiseImpl("multiply"),
            fold=Accum(name=self.acc, value=f"{self.acc}__v", op=ElementwiseImpl("add"), axes=(self.k_axis.name,)),
            operand_bodies=(self.b_body, self.a_body),  # B[k, n], A[m, k] (or either's computed body) — keep B-then-A load reuse
            reduce_axis=self.k_axis,
        )
        if len(self.channels) == 1:
            return base
        from emmy.compiler.ir.stmt.algebra import State, Twist  # noqa: PLC0415 — algebra imports leaves
        from emmy.compiler.ir.stmt.carrier import Channel as CarrierChannel  # noqa: PLC0415

        a_name, k = self.a_name, self.k_axis.name
        extra: list[Stmt] = []
        for ch in self.channels[1:]:
            lift = Assign(name=f"{ch.acc}__v", op=ElementwiseImpl("multiply"), args=(a_name, _operand_name(ch.b)))
            extra += [*_operand_body(ch.b), lift, Accum(name=ch.acc, value=lift.name, op=ElementwiseImpl("add"), axes=(k,))]
        add = ElementwiseImpl("add")
        names = tuple(ch.acc for ch in self.channels)
        cchans = tuple(CarrierChannel(fold=add, term=f"{nm}__v", dtype=base.carrier.twist.channels[0].dtype) for nm in names)
        carrier = Carrier(state=State(names=names), twist=Twist(family="id", channels=cchans))
        return Loop(axis=base.axis, body=Body((*base.body, *extra)), unroll=base.unroll, role=base.role, carrier=carrier)

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body — just the synthesized reduce ``Loop`` (the derived product
        loop at arity N). A projection is NEVER
        here: it has one home, the wrapping :class:`Map`'s body. The materializer expands the node
        through ``_factor.factorize`` instead; this is the structural-key / dump path."""
        return [self.loop]

    def as_fold(self) -> Fold:
        """The STORED (1k) form of this view — a ``role=CONTRACTION`` :class:`Fold`/fold whose
        ``operands`` are the closed inputs and whose ``step`` is the pure lift/fold step. The
        operand tuple order and the lift arg spellings reproduce the historical synthesis exactly
        (primary lift ``(b, a)``, further channels ``(a, b_i)``), so :attr:`Fold.loop`'s
        first-use splice yields a loop byte-identical to :attr:`loop` — the round-trip
        ``contraction_view(as_fold()) == self`` (axes supplied by the caller's placement) is what
        keeps kernel identity fixed across the storage change."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

        mul, add = ElementwiseImpl("multiply"), ElementwiseImpl("add")
        prim = self.channels[0]
        k = self.k_axis.name
        step: list[Stmt] = [
            Assign(name=f"{prim.acc}__v", op=mul, args=(_operand_name(prim.b), self.a_name)),
            Accum(name=prim.acc, value=f"{prim.acc}__v", op=add, axes=(k,)),
        ]
        for ch in self.channels[1:]:
            step += [
                Assign(name=f"{ch.acc}__v", op=mul, args=(self.a_name, _operand_name(ch.b))),
                Accum(name=ch.acc, value=f"{ch.acc}__v", op=add, axes=(k,)),
            ]
        return Fold(
            carrier=self.loop.carrier,
            axis=self.k_axis,
            step=Body(tuple(step)),
            operands=(prim.b, self.a, *(ch.b for ch in self.channels[1:])),
            tile=self.tile,
            stage=self.stage,
        )

    # ---- params: the (m, n) output axes unpacked ---------- #
    @property
    def m_axis(self) -> Axis:
        return self.axes[0]

    @property
    def n_axis(self) -> Axis:
        return self.axes[1]

    @property
    def atom(self) -> Atom:
        return self.tile.atom

    # ---- the (m, n) output sides: each axis paired with its derived per-CTA tile geometry (width /
    # unit / register counts, read off the ``tile``) + the bound block/unit var names (the original
    # m/n names live in the operand indices, so the bound axes take a fresh ``_b`` / ``_u`` suffix).
    # ``_factor`` threads these as the ``(m, n)`` pair, mirroring the schedule's tuples. ---------- #
    @property
    def mn(self) -> tuple[Side, Side]:
        """The ``(m, n)`` output sides — each output axis paired with its derived per-CTA tile
        geometry (width / unit / register counts, read off the ``tile``). Built as the pair; :attr:`m`
        / :attr:`n` index it."""
        t = self.tile
        dims = ((t.tile_m, t.units_m, t.reg_m), (t.tile_n, t.units_n, t.reg_n))
        return tuple(
            Side(axis=ax, tile=tile, units=units, reg=reg, block=ax.name + "_b", unit=ax.name + "_u")
            for ax, (tile, units, reg) in zip(self.axes, dims, strict=True)
        )

    @property
    def m(self) -> Side:
        return self.mn[0]

    @property
    def n(self) -> Side:
        return self.mn[1]

    @property
    def block_threads(self) -> int | None:
        bt = self.tile.block_threads
        return bt if bt > 1 else None  # None ⇒ the scalar default block size

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical B[k, n] — read off the
        primary channel's load, the same test ``_atomize`` made when it bound the operand (channel
        layout agreement is a formation gate, so the primary speaks for the group). A gmem LAYOUT
        question, so it is meaningful only for a materialized B; a computed B has no gmem address and
        answers ``False`` (every tier that would act on the layout gates on ``isinstance(c.b, Load)``
        first)."""
        return isinstance(self.b, Load) and self.k_axis.name in self.b.index[-1].free_vars()

    # ---- reads the view consumers share (no stmt protocol — the view is never stored) ---------- #
    def defines(self) -> tuple[str, ...]:
        return tuple(ch.acc for ch in self.channels)

    def external_reads(self) -> tuple[str, ...]:
        def loads(stmts):
            for s in stmts:
                if isinstance(s, Load):
                    yield s.input
                for b in s.nested():
                    yield from loads(b)

        # Deep over EVERY operand edge — an inline cone may nest its loads (the fused norm→linear
        # cone's per-row statistic reduce ``Loop``).
        def edge(op):
            return loads(_operand_body(op))

        return tuple(dict.fromkeys((*edge(self.a), *(nm for ch in self.channels for nm in edge(ch.b)))))

    def pretty(self, indent: str = "") -> list[str]:
        t = " trans" if self.b_trans else ""
        src = lambda op, nm: op.input if isinstance(op, Load) else nm  # noqa: E731 — buffer name, else the bound SSA name
        bs = ",".join(src(ch.b, _operand_name(ch.b)) for ch in self.channels)
        accs = ",".join(ch.acc for ch in self.channels)
        ops = f"{src(self.a, self.a_name)} @ {bs}{t} -> {accs} ({self.atom.name})"
        return [f"{indent}ContractionView [{self.m_axis.name}, {self.n_axis.name}] {ops}"]


def _map_results(body: Body) -> tuple[str, ...]:
    """The synthesized ``results`` of a legacy ``Map(body=…)`` construction — the last defining
    stmt's name (the retired ``out`` last-def convention, run ONCE at construction instead of on
    every read). Empty when nothing in the body defines a name (a store-only projection tail)."""
    for s in reversed(body):
        d = s.defines()
        if d:
            return (d[-1],)
    return ()


@dataclass(frozen=True, init=False)
class Map(Stmt):
    """A pointwise lift / projection wrapper — ``fn: Lambda`` over zero or more reduction /
    contraction ``sources``, bound POSITIONALLY: ``sources[i]`` produces the value ``fn.params[i]``
    names (at construction the params ARE the sources' bound output names, so lowering splices the
    body verbatim). ``fn.results`` are the bound output names — they replace the retired ``out``
    last-def convention (``out`` reads ``fn.results[0]``).

    ``fn.body`` is the per-cell pointwise / projection compute: operand ``Load``\\ s, the lift
    ``Assign``\\ s, and — UNTIL 1q moves effects to the kernel boundary — the root output
    ``Write`` and ``030_split_reduce``'s sliced-partial ``Loop``\\ s (the one sanctioned impurity,
    built through ``effectful_lambda``). ``sources`` are the structural nodes it projects over
    (:class:`Fold` / :class:`ContractionView` — ``project ∘ reduce``), empty for a pure pointwise
    map; SEVERAL sources is the **fused sibling group** — N contractions over one shared A, the
    gate⊗up MLP edge; the group schedules and lowers as ONE unit. Every recognized contraction —
    per-cell scalar included — is stored as a ``role=CONTRACTION`` fold (``_nodify_contraction``
    in ``010_recognize``). It is itself a ``Stmt``, so it can occupy a statement position in
    another node's body (the fused sibling group inside a split reduce's ``step``); ``body`` is
    the compat read for ``fn.body``.

    The legacy ``Map(body=…, sources=…)`` construction shape keeps working: the binder is
    synthesized (params = the sources' output names; results = the last def, once, here)."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    fn: Lambda
    sources: tuple[Fold | ContractionView | Map, ...]

    def __init__(
        self,
        fn: Lambda | None = None,
        sources: tuple = (),
        *,
        body=None,
        results: tuple[str, ...] | None = None,
    ) -> None:
        sources = tuple(sources)
        if fn is None:
            b = Body.coerce(body) if body is not None else Body()
            params = tuple(_operand_name(s) for s in sources)
            if results is None:
                results = _map_results(b) or params[:1]
            fn = effectful_lambda(params, b, results)
        elif body is not None or results is not None:
            raise TypeError("Map: pass fn= xor (body= / results=), not both")
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "sources", sources)

    @property
    def body(self) -> Body:
        """The projection body — ``fn.body`` (the stmts live on the lambda)."""
        return self.fn.body

    @property
    def out(self) -> str:
        """The bound output name — the lambda's primary result (``fn.results`` replaced the
        last-def convention; an empty-body wrap's result is its primary param, i.e. the source's
        carried state)."""
        r = self.fn.results
        assert r and isinstance(r[0], str), f"Map has no named result: {r!r}"
        return r[0]

    def with_body(self, body) -> Map:
        """A copy with the lambda's body replaced (params / results preserved)."""
        return Map(fn=effectful_lambda(self.fn.params, Body.coerce(body), self.fn.results), sources=self.sources)

    # ---- the stmt protocol (see ``Fold``): ``sources`` are NOT nested bodies — they are node
    # edges, reached by the node-aware walk (:func:`tree_nodes`), the same way a ``ContractionView``'s
    # operand is. ---------- #
    def nested(self) -> tuple[Body, ...]:
        return (self.fn.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return self.with_body(body)

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Map must be lowered (ops.lower) before render")


# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural nodes' handlers here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Map, rename, sigma, axis_fn):
    # The binder renames in lockstep with the body and sources: params track the sources' output
    # names (positional binding), results track the projection defs.
    fn = effectful_lambda(
        tuple(rename(p) for p in s.fn.params),
        Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.fn.body)),
        tuple(rename(r) if isinstance(r, str) else r for r in s.fn.results),
    )
    return Map(fn=fn, sources=tuple(_rewrite(src, rename, sigma, axis_fn) for src in s.sources))


@_rewrite.register
def _(s: Fold, rename, sigma, axis_fn):
    # The schedule slices (``tile`` / ``reduce`` / ``stage``) pass through; the carrier renames in
    # lockstep with the partial's ``Accum``\\ s — the same rule the ``Loop`` handler applies
    # (``Carrier.rename``) — and every operand edge dispatches back through the registry.
    return replace(
        s,
        carrier=s.carrier.rename(rename),
        axis=axis_fn(s.axis),
        step=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.step)),
        operands=tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands),
    )


def _parse_bilinear(fold) -> tuple[object, tuple] | None:
    """Parse a fold's ``(operands, partial)`` back into ``(shared A edge, ((b_edge, acc), …))`` —
    the inverse of :meth:`ContractionView.as_fold`'s synthesis, keyed on its stored spellings: the
    primary lift's args are ``(b, a)``, further channels' ``(a, b_i)``, and at arity N the shared
    operand is the one name common to every lift. ``None`` when the fold is not that shape (a
    hand-built fold, a foreign step, a demoted matvec whose loads stay inline in the step) — the
    caller keeps the general fold reading. Success IS what makes a fold a contraction — the derived
    :attr:`Fold.role` reads this parse, so there is no stored role to gate on."""
    if not isinstance(fold, Fold) or not fold.operands:
        return None
    accums = [st for st in fold.step if isinstance(st, Accum)]
    lifts = {st.name: st for st in fold.step if isinstance(st, Assign)}
    if not accums or len(fold.step) != 2 * len(accums):
        return None
    by_name = {_operand_name(e): e for e in fold.operands}
    try:
        args = [lifts[acc.value].args for acc in accums]
    except KeyError:
        return None
    if any(len(ar) != 2 for ar in args):
        return None
    if len(accums) == 1:
        b_nm, a_nm = args[0]
    else:
        common = set(args[0])
        for ar in args[1:]:
            common &= set(ar)
        if len(common) != 1:
            return None
        (a_nm,) = common
    if a_nm not in by_name:
        return None
    chans = []
    for ar, acc in zip(args, accums, strict=True):
        b_nm = next((x for x in ar if x != a_nm), None)
        if b_nm is None or b_nm not in by_name:
            return None
        chans.append((by_name[b_nm], acc.name))
    return by_name[a_nm], tuple(chans)


def is_contraction_fold(s) -> bool:
    """``s`` is a stored fold whose DERIVED role is ``CONTRACTION`` — the one test every walk over
    a step sequence uses to spot a composed contraction (flash's QK / PV, split-K's sliced fold)."""
    return isinstance(s, Fold) and s.role is AxisRole.CONTRACTION


def shared_operand(fold):
    """The shared-multiplicand (A-role) operand edge of a bilinear ``role=CONTRACTION`` fold, or
    ``None`` — the placement-free read for callers that need only the edge (the cone), not the full
    view."""
    parsed = _parse_bilinear(fold)
    return parsed[0] if parsed is not None else None


def contraction_view(fold, m_axis: Axis, n_axis: Axis, lead_axes: tuple = ()) -> ContractionView | None:
    """The DERIVED bilinear reading of a stored ``role=CONTRACTION`` fold — a :class:`ContractionView`
    view over it, never stored. The output axes are the CALLER's placement facts (the trailing grid
    axes for a root kernel; a flash consumer supplies its own), which is exactly why they are
    parameters and not fold fields; everything else — the shared A edge, the ``(b, acc)`` channels,
    the k axis, the ``tile`` / ``stage`` slices — is read off the fold. ``None`` when the fold is not
    bilinear (the general fold reading stands alone)."""
    parsed = _parse_bilinear(fold)
    if parsed is None:
        return None
    a_edge, chans = parsed
    return ContractionView(
        axes=(m_axis, n_axis),
        k_axis=fold.axis,
        a=a_edge,
        channels=tuple(Channel(b=b, acc=acc) for b, acc in chans),
        tile=fold.tile if fold.tile is not None else TilePlan(),
        lead_axes=tuple(lead_axes),
        stage=fold.stage,
    )


def _stmt_nodes(s: Stmt):
    """Every structural node reachable from stmt ``s`` (deep through nested bodies)."""
    if isinstance(s, (Map, Fold)):
        yield from tree_nodes(s)
        return
    for b in s.nested():
        for child in b:
            yield from _stmt_nodes(child)


def tree_nodes(op):
    """Every structural node in ``op``'s tree, root first — the ONE node walk the path/seam
    enumerators share. An inline operand subtree has exactly one home (its edge), so the tree
    stays a tree and no visited set is needed."""
    if op is None:
        return
    yield op
    if isinstance(op, Map):
        for src in op.sources:
            yield from tree_nodes(src)
        for s in op.body:
            yield from _stmt_nodes(s)
    elif isinstance(op, Fold):
        for edge in op.operands:
            if isinstance(edge, (Map, Fold)):
                yield from tree_nodes(edge)
        for s in op.step:
            yield from _stmt_nodes(s)


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Map` /
    :class:`Fold` / :class:`ContractionView`, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``lower(op)``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.

    There is **no** let table: a computed operand is stored inline on its edge, and sharing is the
    product :class:`ContractionView`'s arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. There is **no** residual reduce-partition field
    either: EVERY partitioned reduce — a plain / twisted
    monoid, flash, a coop-K / split-K contraction — carries its :class:`~.schedule.ReducePlan` on its
    ``Fold`` node (read via ``ops.reduce_plan``); the flat-``Map`` coop-K contraction is nodified
    by ``ops.nodify_reduce`` at schedule / split time. The contraction operand→role binding is not a
    ``TileOp`` field either — a tiled contraction carries its A operand / channels on
    its ``ContractionView`` node (``op``), the single source of truth; ``_schedule._contraction_node``
    resolves them via ``_atomize.semiring_binding``."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    workers: WarpSpec | None = None

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from emmy.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        return "\n".join(pretty(self.op, "    "))


__all__ = [
    "Channel",
    "ContractionView",
    "Map",
    "Fold",
    "TileOp",
    "contraction_view",
    "is_contraction_fold",
    "shared_operand",
    "tree_nodes",
]
