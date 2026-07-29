"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine* — the :class:`Map` / :class:`Reduction` / :class:`Contraction`
nodes defined **in this module**, alongside ``ir/stmt/algebra``) directly,
plus the **let table** ``bindings`` (shared subtrees, keyed by the name each
one's root defines) and a thin set of **root-global schedule fields** — the
free-axis → grid :class:`~.schedule.Placement` (``place``) and the warp split
(``workers``). The
per-node schedule slices ride the structural nodes themselves (a
:class:`Contraction`'s ``tile``, a :class:`Reduction`'s
``reduce`` — EVERY reduce partition rides its node, none on the ``TileOp``); the residual root field
``stage`` holds the resolved operand pipeline (flash is now a
``Map(sources=(Reduction(partial=[Contraction(QK), …, Contraction(PV)]),))`` node tree, so its
partition rides the node). There is no per-kind kernel/schedule type: the algebra is read
structurally off the axes' :class:`~emmy.compiler.ir.axis.AxisRole`
(``ops.axis_role``), so MAP / MONOID / SEMIRING all ride the same ``TileOp``.

**Operand sharing is structural.** "These two matmuls read the same A" is a FACT
the tree states — N sibling :class:`Contraction`\\ s under one ``Map.sources``,
each naming the same bound subtree in ``bindings`` — separate from the
scheduling DECISION of how to lower them. ``ops.resolve`` inlines those
references before any lowering walk, so the walkers stay name-free.

The combine lives entirely in the ``op`` wrapper (the :class:`Map` /
:class:`Reduction` / :class:`Contraction` nodes here + ``ir/stmt/algebra``): a
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expr import Expr, Literal
from emmy.compiler.ir.schedule import Placement, ReducePlan, Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Accum, Body, Carrier, Load, Loop, RenderCtx, Stmt

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


def _flatten_nodes(body: Body) -> tuple[Stmt, ...]:
    """Flatten any nested **structural node** — a :class:`Contraction`, :class:`Reduction`, or
    :class:`Map` — that sits as a stmt in ``body`` to its own lowered loop nest; plain stmts pass
    through. ``Contraction`` is a ``Stmt``, so it can ride inside a reduce ``partial`` (the flash QK
    ``Σ_dd Q·K`` score AND the flash PV ``Σ_j P·V``); ``Reduction`` / ``Map`` ride here too once a
    per-step partial composes them. This is the single **node-walk** the ``.loop`` splice and the
    kernel materializer's recursion share: ``partial`` is a ``Body`` that may carry nodes, and the
    walk yields the same loop nest whether a node was pre-flattened or reached structurally."""
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    out: list[Stmt] = []
    for s in body:
        if isinstance(s, (Contraction, Reduction, Map)):
            out.extend(lower(s))
        else:
            out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class Reduction:
    """A scheduled ``PLANAR`` / ``TWISTED`` reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/stmt/algebra``). It splits the reduce's **algebra** (the loop-carried
    :class:`~emmy.compiler.ir.stmt.algebra.Carrier` — degenerate ``id`` for a plain
    ``sum`` / ``max`` / ``mean``, twisted ``exp`` for online-softmax / flash) from its **structure**
    (the reduce ``axis`` + the per-element ``partial`` it folds). The fold ``Loop`` is **synthesized on
    demand** (:attr:`loop`), never stored — so the same node tiles under any
    :class:`~emmy.compiler.ir.schedule.ReducePlan` (the reduce partition rides the node's
    ``reduce`` field, read via ``ops.reduce_plan``).

    A reduce whose per-step partial COMPOSES another node — split-K's ``Reduction ⊃ Contraction``
    (whose ``axis`` ``ksplit`` differs from the inner ``k_axis`` ``kslice``, so no double-reduce),
    flash's ``Σ Q·K`` score at the head of its kv step — spells it ONE way: the node sits in
    ``partial`` and :func:`_flatten_nodes` flattens it in place. There is no second ``source`` edge.

    It holds **no projection**: a bare reduce (``sum`` / ``max``) is the kernel root (its grid ``Write``
    is glue); a reduce with a post-fold sweep (softmax / RMSNorm) is the ``source`` of a wrapping
    :class:`Map` whose body IS that projection. It is NOT a ``Stmt``
    — like ``Map`` it is an op-tree node a :class:`TileOp` holds;
    :func:`ops.lower` flattens it to the synthesized loop (``[loop]``), so ``op_cache_key`` and the
    ``_factor._tile_reduce_axis`` expander stay byte-identical to the bare-loop form.

    The **scheduling param** is the ``reduce`` partition (:class:`ReducePlan` — GRID split / BLOCK coop
    / REG ILP), stamped onto the node by ``_schedule`` (inside ``010_recognize``) (its decided value lives **here** on the node
    — read via ``ops.reduce_plan``). ``lower`` ignores it (it's metadata the materializer / ``030_split_reduce``
    read), so it leaves ``op_cache_key`` byte-identical."""

    carrier: Carrier  # the loop-carried ⊕ algebra (degenerate id / twisted exp)
    axis: Axis  # the reduce axis
    partial: Body = field(default_factory=Body)  # the per-cell fold body (the reduce Loop's body)
    role: AxisRole = AxisRole.PLANAR  # PLANAR (plain) or TWISTED (online-softmax / flash)
    unroll: bool = False
    reduce: ReducePlan = field(default_factory=ReducePlan)  # the reduce partition (schedule slice), stamped by the _schedule helper
    # The stream's ABSOLUTE base for a cross-CTA slice partial (flash split-KV: ``030_split_reduce`` shrinks
    # ``axis`` to the slice length B and sets ``offset = _ksplit · B``). The fold walks its local
    # ``[0, B)`` window; a consumer needing the absolute reduce-axis coordinate (gmem/TMA operand
    # bases, the causal mask's key columns) adds this. ``None`` = the un-split stream (base 0).
    offset: Expr | None = None
    # The stream's ABSOLUTE end for a cross-CTA slice partial over a SYMBOLIC axis:
    # ``min((_ksplit + 1) · B, S)`` where ``S`` is the runtime extent (``axis`` then carries the
    # bn-aligned slice width ``B = ceil(S / (cta·bn)) · bn`` as a composite Dim). The realizer stops
    # the stream at ``bound − offset`` local steps and masks/clamps against it — a mid-tensor slice
    # end reads VALID keys belonging to the next slice, which the extent-only tail machinery would
    # not exclude. ``None`` = static slice (extent alone bounds it) or the un-split stream.
    bound: Expr | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.partial, Body):
            object.__setattr__(self, "partial", Body.coerce(self.partial))

    @classmethod
    def from_loop(cls, loop: Loop) -> Reduction:
        """Build a :class:`Reduction` from an already-annotated reduce ``Loop`` (its ``carrier`` /
        ``role`` / ``axis`` / body) — the recognize-side constructor. :attr:`loop` reconstructs the
        exact same ``Loop`` (any post-fold projection rides the wrapping ``Map``, not here)."""
        return cls(carrier=loop.carrier, axis=loop.axis, partial=loop.body, role=loop.role, unroll=loop.unroll)

    @property
    def loop(self) -> Loop:
        """The synthesized annotated reduce ``Loop`` — reconstructed from the params. With a plain
        ``partial`` it is byte-identical to the loop :meth:`from_loop` captured. Any **nested
        structural node inside the ``partial``** — a :class:`Contraction` / :class:`Reduction` /
        :class:`Map` — is flattened to its own loop nest in place by the shared :func:`_flatten_nodes`
        node-walk (so flash's kv loop holds the head ``Σ Q·K`` score contraction loop and the embedded
        ``Σ_j P·V`` PV contraction loop, and split-K's kslice contraction its own nest — exactly the
        loop-in-body form the scalar tier expands): ONE structural rule for a reduce whose per-step
        partial composes other nodes."""
        return Loop(axis=self.axis, body=Body(_flatten_nodes(self.partial)), unroll=self.unroll, role=self.role, carrier=self.carrier)

    @property
    def out(self) -> str:
        """The bound output name — the carrier state's primary component (a bare reduce's grid
        ``Write`` is glue; a projected reduce's output name lives on the wrapping ``Map``)."""
        return self.carrier.out

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — just the synthesized reduce
        ``Loop`` (a wrapping ``Map`` appends its projection)."""
        return [self.loop]


@dataclass(frozen=True)
class Side:
    """One tiled output axis of a contraction — the outer ``m`` or inner ``n`` — paired with its
    derived per-CTA tile geometry. The two ride as a ``(m, n)`` pair (:attr:`Contraction.mn`)
    mirroring the schedule's ``(m, n)`` tuples (``TilePlan.units`` / ``regs``), so the factorizer
    threads one object per axis instead of a dozen loose ``*_m`` / ``*_n`` args. The tile width,
    unit / register counts, and bound block/unit var names are stamped by :meth:`Contraction._side`;
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
class Contraction(Stmt):
    """A contraction **before** atom factorization — built **recognize-side** at fork-emit
    (``_schedule._contraction_node`` resolves the operand→role binding via ``_atomize.semiring_binding``
    and stamps the resolved ``tile``), then expanded in ``010_materialize`` (``_factor.factorize``).
    :func:`ops.lower` / ``ops.reduce_loop`` flatten it back to the synthesized mul-add ``CONTRACTION``
    loop nest (:attr:`loop`), the same ``for k: v = a·b; acc += v`` form ``_atom._ScalarOps.reduce``
    register-tiles through the shared ``_contract_kloop`` skeleton. **ONE
    flat node** that cleanly splits the **algebra params** (what to contract) from the **schedule**
    (how to tile it): the params are the tiled output ``axes`` ``(m, n)``, the contraction ``k_axis``,
    the leading batch ``lead_axes``, the A operand, the B operand (``b_load``) and the fold
    accumulator (``acc``); the projection has ONE home — the wrapping :class:`Map`'s body — never a
    node field; the schedule is the
    one ``tile`` field — a resolved :class:`~emmy.compiler.ir.schedule.TilePlan` carrying the
    leaf ``atom`` (tensor-core :class:`AtomKind` or the ``1×1`` :class:`ScalarAtom`), the per-CTA
    **UNIT** grid + per-unit **REGISTER** sub-tile, and the K-chunk. Keeping the schedule a single
    swappable field is what lets the same operand/acc params be tiled by a different ``TilePlan`` (the
    flash inner QK/PV reuse).

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
    # A: a gmem ``Load``, a computed register-resident ``Body`` (flash PV's ``P = exp(S − M)``), or the
    # SSA NAME of a let-bound subtree in the owning ``TileOp.bindings`` (the shared computed-A cone —
    # sibling channels referencing one name is what makes their operand sharing structural). A name
    # operand is inlined by ``ops.resolve`` before any lowering walk, so every accessor below sees the
    # ``Load`` / ``Body`` form. — params
    a_operand: Load | Body | str
    b_load: Load  # the B operand — params
    acc: str  # the fold accumulator this node produces — params
    tile: TilePlan  # the schedule: leaf atom + unit/register widths + K-chunk (the only schedule field)
    lead_axes: tuple[Axis, ...] = ()  # params

    @property
    def out(self) -> str:
        """The bound output name — the primary fold accumulator (a bare contraction's grid ``Write``
        stores it at the cell; a fused-projection contraction carries its own ``Write``)."""
        return self.acc

    @property
    def a_ref(self) -> str | None:
        """The binding NAME this node's A operand references, or ``None`` (an inline ``Load`` /
        ``Body``). Two siblings sharing one A are exactly two nodes with the same ``a_ref``."""
        return self.a_operand if isinstance(self.a_operand, str) else None

    @property
    def a_body(self) -> tuple[Stmt, ...]:
        """The A operand's producing stmts — a singleton gmem ``Load``, or a computed body's stmts
        (a **register-resident** A: flash PV's ``P = exp(S − M)``, produced from an in-register score,
        not a gmem address). The last stmt's def is the operand value ``contraction_loop`` multiplies.
        An unresolved binding NAME has no stmts here — ``ops.resolve`` inlines it first."""
        if isinstance(self.a_operand, str):
            raise AssertionError(f"Contraction.a_operand is the unresolved binding {self.a_operand!r} — call ops.resolve first")
        return (self.a_operand,) if isinstance(self.a_operand, Load) else tuple(self.a_operand)

    @property
    def a_computed(self) -> bool:
        """True when A is a computed register-resident operand (a ``Body``), not a gmem ``Load`` — the
        mma tier reads it as a fragment, the scalar tier as the value."""
        return not isinstance(self.a_operand, Load)

    @property
    def a_name(self) -> str:
        """The A operand's bound SSA name — the referenced binding's name, or the producing body's
        last def."""
        return self.a_operand if isinstance(self.a_operand, str) else self.a_body[-1].defines()[-1]

    def stat_prologue(self) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...], tuple[str, ...]]:
        """Split the computed-A body at the contraction-axis seam: the maximal leading run of stmts
        that never index the K axis is the **per-row statistic prologue** (the fused norm→linear
        cone's stat reduce ``Loop`` + scalar epilogue — or a broadcast row scale), the remainder the
        **per-cell** cone. Returns ``(prologue, cell, stats)`` — ``stats`` are the prologue defs the
        cell reads, the values bridged through the stat smem rows (a prologue whose defs go unread
        is dropped). The ONE seam both sides read: the scheduler sizes the stat rows into the sync
        stage's smem budget, the materializer fills them (``sync_stat_fill``)."""
        body = list(self.a_body)
        pro: list[Stmt] = []
        while body and not _refs_axis(body[0], self.k_axis.name):
            pro.append(body.pop(0))
        if not pro:
            return (), tuple(body), ()
        stats = tuple(sorted({nm for s in pro for nm in _deep_defines(s)} & _deep_reads(body)))
        return (tuple(pro), tuple(body), stats) if stats else ((), tuple(body), ())

    @property
    def loop(self) -> Loop:
        """The synthesized ``CONTRACTION`` reduce ``Loop`` — the canonical ``for k: v = a*b; acc += v``
        mul-add form (built by the shared ``ops.contraction_loop``, the same fold ``_factor``'s scalar
        contraction tier register-tiles). Lets :func:`ops.lower` / ``ops.reduce_loop`` flatten the node
        back to the loop nest; the node never stores the loop. A FUSED SIBLING GROUP (N channels over
        one shared A) derives its single loop from the group instead — ``ops.group_loop``."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
        from emmy.compiler.ir.tile.ops import contraction_loop  # noqa: PLC0415 — avoid an import cycle

        return contraction_loop(
            lift=ElementwiseImpl("multiply"),
            fold=Accum(name=self.acc, value=f"{self.acc}__v", op=ElementwiseImpl("add"), axes=(self.k_axis.name,)),
            operand_bodies=([self.b_load], self.a_body),  # B[k, n], A[m, k] (or A's computed body) — keep B-then-A load reuse
            reduce_axis=self.k_axis,
        )

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body — just the synthesized reduce ``Loop``. A projection is NEVER
        here: it has one home, the wrapping :class:`Map`'s body. The materializer expands the node
        through ``_factor.factorize`` instead; this is the structural-key / dump path."""
        return [self.loop]

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
        binding load, the same test ``_atomize`` made when it bound the operand."""
        return self.k_axis.name in self.b_load.index[-1].free_vars()

    # ---- the kernel-stmt protocol (no nested Body — a projection rides the wrapping ``Map``; the
    # operand buffers are external reads; the synthesized reduce produces ``acc``) ---------- #
    def nested(self) -> tuple[Body, ...]:
        return ()

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        assert not bodies, "Contraction has no nested body — a projection rides the wrapping Map"
        return self

    def defines(self) -> tuple[str, ...]:
        return (self.acc,)

    def external_reads(self) -> tuple[str, ...]:
        def loads(stmts):
            for s in stmts:
                if isinstance(s, Load):
                    yield s.input
                for b in s.nested():
                    yield from loads(b)

        # Deep over the A body — a computed cone may nest its loads (the fused norm→linear cone's
        # per-row statistic reduce ``Loop``). A binding REFERENCE reads no buffer here: the bound
        # subtree is the owning ``TileOp``'s, and its loads are reached through ``bindings``.
        a = () if isinstance(self.a_operand, str) else loads(self.a_body)
        return (*dict.fromkeys(a), self.b_load.input)

    def pretty(self, indent: str = "") -> list[str]:
        t = " trans" if self.b_trans else ""
        a_src = self.a_operand.input if isinstance(self.a_operand, Load) else self.a_name
        ops = f"{a_src} @ {self.b_load.input}{t} -> {self.acc} ({self.atom.name})"
        return [f"{indent}Contraction [{self.m_axis.name}, {self.n_axis.name}] {ops}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Contraction must be expanded by 010_materialize before render")


# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register ``Contraction``'s handler here, with the node.
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Contraction, rename, sigma, axis_fn):
    # Route the operand Loads + accumulator through the generic rewrite (SSA / Expr / axis
    # canonicalization); map the skeleton axes; pass the ``tile`` schedule through unchanged.
    # ``b_trans`` is derived from ``b_load`` (a property), so the rewritten load carries it.
    # A binding REFERENCE is a plain SSA name — it renames through the same map as any other, which
    # is what keeps references and their definitions (the ``bindings`` keys) in lockstep.
    if isinstance(s.a_operand, str):
        a_operand = rename(s.a_operand)
    elif isinstance(s.a_operand, Load):
        a_operand = _rewrite(s.a_operand, rename, sigma, axis_fn)
    else:
        a_operand = Body(tuple(_rewrite(c, rename, sigma, axis_fn) for c in s.a_operand))
    return Contraction(
        axes=tuple(axis_fn(a) for a in s.axes),
        k_axis=axis_fn(s.k_axis),
        a_operand=a_operand,
        b_load=_rewrite(s.b_load, rename, sigma, axis_fn),
        acc=rename(s.acc),
        tile=s.tile,
        lead_axes=tuple(axis_fn(a) for a in s.lead_axes),
    )


@dataclass(frozen=True)
class Map:
    """A pointwise lift / projection wrapper around a :class:`Body` of loop-IR stmts, optionally over
    one or more reduction / contraction ``sources``.

    ``body`` is the per-cell pointwise / projection compute: operand ``Load``\\ s, the lift
    ``Assign``\\ s, and — at the kernel root — the output ``Write``. ``sources`` are the structural
    nodes it projects over (:class:`Reduction` / :class:`Contraction` — ``project ∘ reduce``), empty
    for a pure pointwise map. A pure pointwise cell is a ``Map`` of plain stmts (``sources=()``);
    softmax / RMSNorm is a ``Map`` whose ``body`` is the post-fold sweep over a ``Reduction`` source.
    SEVERAL sources is the **fused sibling group** — N :class:`Contraction`\\ s over one shared A
    (referenced by binding name), the gate⊗up MLP edge; the group schedules and lowers as ONE unit.
    Every recognized contraction — per-cell scalar included — is a :class:`Contraction` node
    (``_nodify_contraction`` in ``010_recognize``); the only annotated reduce ``Loop``\\ s still riding
    a flat ``Map`` body are ``030_split_reduce``'s sliced partials. ``out`` is the bound output name (the
    body's last def, or the primary source's carried state for an empty-body wrap). It HAS a Body, not
    IS one. :attr:`source` is the len-≤1 compat read for the single-source forms."""

    body: Body = field(default_factory=Body)
    sources: tuple[Reduction | Contraction, ...] = ()  # the project∘reduce sources, () = pure pointwise

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))
        if not isinstance(self.sources, tuple):
            object.__setattr__(self, "sources", tuple(self.sources))

    @property
    def source(self) -> Reduction | Contraction | None:
        """The single projected source, or ``None`` (pure pointwise) — the compat read every
        single-source form uses. A fused sibling group (N sources) has no single source; read
        ``sources`` there."""
        assert len(self.sources) <= 1, f"Map has {len(self.sources)} sources — read `sources`, not `source`"
        return self.sources[0] if self.sources else None

    @property
    def out(self) -> str:
        """The bound output name. With no projection body it is the primary source's carried state;
        otherwise the last defining stmt's name (a pointwise lift / a post-reduce projection)."""
        if len(self.body) == 0 and self.sources:
            return self.sources[0].out
        return self.body[-1].defines()[-1]


def _stmt_nodes(s: Stmt):
    """Every structural node reachable from stmt ``s`` (deep through nested bodies)."""
    if isinstance(s, (Map, Reduction, Contraction)):
        yield from tree_nodes(s)
        return
    for b in s.nested():
        for child in b:
            yield from _stmt_nodes(child)


def tree_nodes(op):
    """Every structural node in ``op``'s tree, root first — the ONE node walk the name-uniqueness
    check, the binding resolver, and (later) the path/seam enumerators share. ``bindings`` are the
    owning :class:`TileOp`'s, walked separately: a bound subtree has exactly one home, so the tree
    stays a tree and no visited set is needed."""
    if op is None:
        return
    yield op
    if isinstance(op, Map):
        for src in op.sources:
            yield from tree_nodes(src)
        for s in op.body:
            yield from _stmt_nodes(s)
    elif isinstance(op, Reduction):
        for s in op.partial:
            yield from _stmt_nodes(s)
    elif isinstance(op, Contraction) and isinstance(op.a_operand, Body):
        for s in op.a_operand:
            yield from _stmt_nodes(s)


def _node_defines(node) -> list[str]:
    """The SSA names ``node`` itself binds — its own stmts' deep defs plus, for a
    :class:`Contraction`, the accumulators its SYNTHESIZED loop defines (no stored stmt binds them)."""
    out: list[str] = []
    if isinstance(node, Map):
        stmts = [s for s in node.body if not isinstance(s, (Map, Reduction, Contraction))]
    elif isinstance(node, Reduction):
        stmts = [s for s in node.partial if not isinstance(s, (Map, Reduction, Contraction))]
    else:
        cone = list(node.a_operand) if isinstance(node.a_operand, Body) else []
        stmts = [s for s in cone if not isinstance(s, (Map, Reduction, Contraction))]
        out += list(node.defines())
    for s in stmts:
        out += sorted(_deep_defines(s))
    return out


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Map` /
    :class:`Reduction` / :class:`Contraction`, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``lower(op)``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.
    - ``stage`` — the operand smem pipeline (:class:`~.schedule.Stage`); ``None`` = gmem-direct.

    ``bindings`` is the kernel's **let table** — shared subtrees keyed by the name each tree's root
    defines (its ``out``), referenced from an operand field by that plain SSA name
    (``Contraction.a_operand = "xhat"``). There is deliberately no ``Ref`` node kind: SSA names are
    already the IR's one reference mechanism, so ``deps`` / ``defines``, the rewrite rename maps and
    ``structural_key`` canonicalization all speak them unchanged. The invariant that replaces it is
    **binding-name uniqueness** (:meth:`validate_bindings`, run at construction): a binding's key is
    the only definition of that name anywhere in ``op`` + ``bindings``, and every name operand
    resolves to a binding — so a shared subtree has exactly one home, paths stay unique, and two
    references to one binding are distinguishable from two copies. ``ops.resolve`` inlines the
    references before any lowering walk.

    There is **no** residual reduce-partition field: EVERY partitioned reduce — a plain / twisted
    monoid, flash, a coop-K / split-K contraction — carries its :class:`~.schedule.ReducePlan` on its
    ``Reduction`` node (read via ``ops.reduce_plan``); the flat-``Map`` coop-K contraction is nodified
    by ``ops.nodify_reduce`` at schedule / split time. The contraction operand→role binding is not a
    ``TileOp`` field either — a tiled contraction carries its A/B operands / accumulator on
    its ``Contraction`` node (``op``), the single source of truth; ``_schedule._contraction_node``
    resolves them via ``_atomize.semiring_binding``."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    stage: Stage | None = None
    workers: WarpSpec | None = None
    bindings: dict = field(default_factory=dict)  # the let table: out-name → shared subtree

    def __post_init__(self) -> None:
        self.validate_bindings()

    def validate_bindings(self) -> None:
        """Enforce the binding invariants at CONSTRUCTION — a violation would otherwise surface as a
        missing (or silently wrong) operand several passes later:

        - every name operand resolves to a binding (no dangling reference);
        - a binding's key IS its tree's output name;
        - a binding's key is defined nowhere else — not in ``op``, not in another binding.

        The uniqueness check is scoped to the binding NAMES rather than every SSA name in the tree:
        loop IR legitimately binds one name from several stmts (a fold's ``Init`` seed plus its
        ``Accum`` steps, and split-K's outer reduce folding the accumulator its inner contraction
        also names). What the let table needs is that a REFERENCE is unambiguous, and that is
        exactly what this asserts."""
        refs: set[str] = set()
        homes: dict[str, list[str]] = {}
        keys = set(self.bindings)
        for home, root in (("op", self.op), *((f"bindings[{k!r}]", v) for k, v in self.bindings.items())):
            for node in tree_nodes(root):
                if isinstance(node, Contraction) and node.a_ref is not None:
                    refs.add(node.a_ref)
                for nm in _node_defines(node) if keys else ():  # the def walk only pays where a table exists
                    homes.setdefault(nm, []).append(home)
        if dangling := refs - set(self.bindings):
            raise AssertionError(f"TileOp: operand reference(s) {sorted(dangling)} resolve to no binding")
        for key, root in self.bindings.items():
            if root is None or key != root.out:
                raise AssertionError(f"TileOp.bindings key {key!r} is not its tree's output name ({root and root.out!r})")
            if other := sorted(set(homes.get(key, ())) - {f"bindings[{key!r}]"}):
                raise AssertionError(f"TileOp: binding name {key!r} is also defined in {other} — a reference must be unambiguous")

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from emmy.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        return "\n".join(pretty(self.op, "    ", self.bindings))


__all__ = ["Contraction", "Map", "Reduction", "TileOp", "tree_nodes"]
