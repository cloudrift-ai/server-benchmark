"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine* — the :class:`Fold`
nodes defined **in this module**, alongside ``ir/stmt/algebra``) directly,
plus a thin set of **root-global schedule fields** — the
free-axis → grid :class:`~.schedule.Placement` (``place``), the ONE worker inventory (``work``)
and the warp-spec split (``workers``). The
per-node schedule slices live in ``TileOp.schedule``: ``{codec key → resolved TilePlan /
ReducePlan / Stage}``, keyed by the tree-path codec's canonical key and read through
``ops.Sched`` — the stored term is pure algebra, IMMUTABLE across the whole schedule search
(flash is a ``Fold.projection(operands=(Fold(operands=(Fold.contraction(QK), Load(V)), lift, combine),))``
node tree whose hoisted score edge and DERIVED PV contraction are the ``TILE@dd`` / ``TILE@pj``
sites). There is no per-kind kernel/schedule type: dispatch reads the role
structurally off the node (``ops.axis_role`` — a contraction IS the
a bilinear ``Fold`` kind, a fold's role derives), so MAP / MONOID / SEMIRING all ride the
same ``TileOp``.

**Operand sharing is arity.** "These two matmuls read the same A" is ONE
a bilinear ``Fold`` whose output is a tuple: one ``a`` edge plus N product
:class:`Channel`\\ s ``(b_i, acc_i)``, folding the componentwise ``(+, ×)`` —
the N-component product monoid the derived :attr:`Fold.loop` carries.
There is no let table and no name-reference mechanism: the one in-tree relation
that had several consumers is a single edge, so a shared subtree has exactly one
home by construction.

**An operand is an edge with two inhabitants** — the two things an input can
be: MATERIALIZED (a gmem ``Load``) or COMPUTED (the node itself, stored inline
on the edge). Tree ownership gives an inline node exactly one consumer — its
parent — so sharing needs no reference arm. The node boundary a reader wants
(``ops.cone_seam``'s prologue / per-cell split) is read straight off the edge;
``lower`` flattens it once, at the point of use. A subtree that reads no value
name from its enclosing body is **closed**; closure is
the precondition for lifting any subtree into its own kernel (a placement cut),
and nothing else requires it — flash's ``P`` legitimately captures the running
max its own loop step updates, and that seam is simply not cuttable. The
predicate lives with its one consumer, the placement cut
(``passes/lowering/tile/_cut``), not here.

**Every structural node is a ``Stmt``.** A composed step occupies a statement
position in another node's body — flash's ``Σ_dd Q·K`` and ``Σ_j P·V`` in a
reduce ``step``, split-K's sliced contraction, the fused sibling group inside
a split reduce — and its POSITION there is semantic: flash's PV reads the softmax
weight the merge stmts of that same loop step produce, so it cannot be hoisted
ahead of them. Composition therefore rides the sequence, not a hoisted operand
tuple, and uniform ``Stmt``-hood is what makes a node a legal member of a
``Body`` (generic walks reach its children through ``nested()`` like any block
stmt's). ``operands`` are the exception on purpose: they are node EDGES,
reached by the node-aware walk (``path.sites`` — the ONE walk over the tree),
like a contraction operand.

The combine lives entirely in the ``op`` wrapper (the :class:`Fold` /
:class:`Fold` / a bilinear ``Fold``s here + ``ir/stmt/algebra``). A
fold's role is DERIVED (``Fold.role``), never stored; a contraction is the
a bilinear ``Fold`` kind itself. ``Fold.lower`` flattens the
structural tree back to the loop nest.

The stored nodes hold algebra params only; the contraction's placement/schedule
fields (``axes`` / ``tile`` / ``stage``) are stamped onto a ``replace()`` copy at
the point of use, with the derived geometry exposed as ``@property`` (so
``structural_key`` digests only the compact param fields and the ``--ir`` dumps stay
readable). The kernel materializer reads the schedule straight off the placed node —
it never re-recognizes structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.schedule import Placement, WarpSpec
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Lambda,
    Load,
    Loop,
    M,
    RenderCtx,
    Stmt,
    Write,
    component_ops,
    rename_combine,
)
from emmy.compiler.ir.stmt.body import _member_reads

if TYPE_CHECKING:
    pass


def _splice_operands(operands: tuple, stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Splice each operand edge's producing stmts into ``stmts`` immediately BEFORE the first stmt
    that reads the operand's bound name (appended when nothing reads it), ties resolved in operand
    TUPLE order. This is the one lowering rule that turns the stored operands + derived step back
    into the flat loop body — deterministic, so the derived loop (and with it ``op_cache_key``)
    depends only on the stored params."""
    if not operands:
        return stmts
    at: dict[int, list] = {}
    for edge in operands:  # first-use indexes against the ORIGINAL stmts — ties keep tuple order
        name = operand_name(edge)
        idx = next((i for i, st in enumerate(stmts) if name in deep_reads([st])), len(stmts))
        at.setdefault(idx, []).append(edge)
    out: list[Stmt] = []
    for i, st in enumerate((*stmts, None)):
        for edge in at.get(i, ()):
            out.extend(operand_body(edge))
        if st is not None:
            out.append(st)
    return tuple(out)


def _flatten_nodes(body: Body) -> tuple[Stmt, ...]:
    """Flatten any nested **structural node** — a a bilinear ``Fold``, :class:`Fold`, or
    :class:`Fold` — that sits as a stmt in ``body`` to its own lowered loop nest; plain stmts pass
    through. All three node kinds ARE ``Stmt``\\ s, which is what makes a composed step a legal member
    of a ``Body``: flash's ``Σ_dd Q·K`` score and its ``Σ_j P·V``, split-K's sliced contraction, and
    the fused sibling group (a zero-axis ``Fold``) inside a split reduce's partial. This is the single
    **node-walk** the ``.loop`` splice and the kernel materializer's recursion share, and it yields
    the same loop nest whether a node was pre-flattened or reached structurally.

    A composed step's POSITION in the body is semantic, not incidental: flash's PV reads the softmax
    weight the merge stmts of that same loop step produce, so the step cannot be hoisted ahead of
    them. That is why composition rides the sequence rather than a hoisted operand tuple."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Fold):
            out.extend(s.lower())
        else:
            out.append(s)
    return tuple(out)


def _derived_expect_fold(o: str, p_name: str, v_edge: Load) -> Fold:
    """The synthesized expectation contraction of a twisted fold's DERIVED blocked evaluation —
    flash's ``Oblk = Σ_j P·V``, a a bilinear ``Fold``. A is the
    register-resident softmax weight ``P`` — a one-stmt cone node whose one legal capture is the
    running max the same derived merge updates — and B is the fold's own expectation operand edge
    (the value ``Load``).

    A's ``copy`` is the REFERENCE, not a no-op: an operand edge's two inhabitants are a gmem
    ``Load`` and an inline node, with no let table and no name-reference arm, so pointing at an
    already-computed register value means wrapping it in the smallest node that yields one. The
    rename is load-bearing too — ``p_name`` is a positional temp of the generated twist program,
    while ``{o}__p`` is derived from the accumulator name and is stable.

    There is no wrapping ``Fold.projection(body=(), operands=(<this>,))`` cone shape around it: an empty cell
    carries no information (``cone_seam`` bridges no stats either way, both spellings lower to
    this one stmt, and the lowered CUDA is byte-identical), so the edge IS the one-stmt node."""
    cone = Fold.projection(body=Body((Assign(name=f"{o}__p", op="copy", args=(p_name,)),)))
    k = Axis(name="pj", extent=Dim(1))  # block=1: a singleton intra-block reduce
    return Fold.contraction(k_axis=k, a=cone, channels=(Channel(b=v_edge, acc=f"{o}__pv"),))


def _split_expect(merge: list[Stmt], o: str, v: str, v_edge: Load) -> list[Stmt]:
    """Split the generated streaming ``merge`` around the synthesized expectation contraction:
    the fused ``v·P`` product becomes :func:`_derived_expect_fold`'s output, and the state fold consumes it
    (``O = O·α + Oblk``; the ``O·α`` base is untouched)."""
    o_accum = next(s for s in merge if isinstance(s, Accum) and s.name == o)
    defs = {s.name: s for s in merge if isinstance(s, Assign)}
    prod = defs[o_accum.value]  # the fused ``v·P`` (multiply(v, P))
    p_name = next(a for a in prod.args if a != v)  # the softmax weight P (register-resident)
    pv = _derived_expect_fold(o, p_name, v_edge)
    out: list[Stmt] = []
    for s in merge:
        if s is prod:
            continue  # the inline v·P is dropped — the synthesized contraction computes it
        if s is o_accum:
            out.append(pv)
            out.append(replace(o_accum, value=f"{o}__pv"))
            continue
        out.append(s)
    return out


def _composes_state(inner, names: tuple[str, ...], ops) -> bool:
    """``inner`` carries EXACTLY the outer fold's accumulator state — the identity-lift
    reassociation's precondition (split-K): a a bilinear ``Fold`` whose (additive by
    construction) channel accumulators are the names and whose outer ⊕ is componentwise
    ``add``."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    return is_contraction(inner) and tuple(inner.defines()) == names and ops == (ElementwiseImpl("add"),) * len(names)


def _operand_binding(fold: Fold) -> dict:
    """The bound-name → operand-edge map (positional binding, one lift param per operand RESULT
    COMPONENT — a product edge binds every component to the same edge)."""
    out: dict = {}
    for e in fold.operands:
        for n in _operand_result_names(e):
            out[n] = e
    return out


def _twisted_derived_step(fold: Fold) -> tuple[Stmt, ...]:
    """The DERIVED blocked evaluation of a λ-spelled TWISTED fold: the INLINE-NODE operand edges
    at the head in operand order (flash's ``Σ_dd Q·K`` score, ahead of the lift body), the lift
    body (the scale / mask stmts), then the generated streaming merge
    with each Load-bound EXPECTATION operand split out as a synthesized contraction
    (:func:`_split_expect` — flash's PV). Deterministic from the stored params only; the operand
    edges consumed here are excluded from the generic first-use splice
    (:meth:`Fold._splice_edges`)."""
    from emmy.compiler.ir.stmt.carrier import exp_merge  # noqa: PLC0415

    lam = fold.lift
    names = tuple(fold.combine.results)
    terms = tuple(lam.results)
    merge = list(exp_merge(names, terms, key=names[0]))
    by_param = _operand_binding(fold)
    for i, (nm, term) in enumerate(zip(names, terms, strict=True)):
        if i == 0 or not isinstance(term, str):
            continue  # the pivot / a literal denominator — only EXPECTATION components split
        edge = by_param.get(term)
        if isinstance(edge, Load):
            merge = _split_expect(merge, nm, term, edge)
    head = tuple(e for e in fold.operands if isinstance(e, Fold))
    return (*head, *lam.body, *merge)


def _fold_derived_step(fold: Fold) -> tuple[Stmt, ...]:
    """The DERIVED serial step of a λ-spelled fold — ``s′ = combine(s, lift(k))``, the combine
    specialized at the singleton state the lift produces. For a DEGENERATE (componentwise)
    monoid that specialization IS the ``Accum`` form — ``sᵢ = ⊕ᵢ(sᵢ, liftᵢ)`` — and each
    component's ``Accum`` lands immediately after the lift stmt that defines its value (a
    literal / param component falls to the tail), exactly where the dissolved fold sat, so the
    derived loop is byte-identical to the historical spelling. A TWISTED (exp-family) monoid
    specializes through the family's own generator (``exp_merge`` — the streaming fold IS
    combine-at-the-singleton with the ψ-rescale simplification applied, temps renumbered by the
    same deterministic emitter that produced the stored combine), landing after the lift body —
    exactly where recognition's dissolved merge sat; a twisted fold with operand edges derives
    the full blocked evaluation (:func:`_twisted_derived_step` — flash). Deterministic from the
    stored params only — kernel identity (``op_cache_key``) depends on nothing else."""
    lam = fold.lift
    names = fold.combine.results
    ops = component_ops(fold.combine)
    if ops is None:  # the twisted (exp-family) serial step — the derived state's channels
        # carry the singleton terms (the lift results), so the generated streaming merge is the
        # singleton specialization of the stored combine, names included.
        return fold._derived_twisted
    if (
        not len(lam.body)
        and tuple(lam.results) == tuple(names)
        and len(fold.operands) == 1
        and _composes_state(fold.operands[0], tuple(names), ops)
    ):
        # The reassociation COMPOSITION (split-K's outer reduce): an IDENTITY lift over one
        # inline fold operand sharing the outer's exact accumulator state. Combine at that
        # singleton is the shared-accumulator simplification (a ×1 fold): the derived step is
        # the sliced fold in place — its own ``Accum``\\ s carry across both loops — with NO
        # outer folds.
        return (fold.operands[0],)
    accums = tuple(Accum(name=names[i], value=str(lam.results[i]), op=ops[i], axes=(fold.axis.name,)) for i in range(len(names)))
    after: dict[str, list[int]] = {}
    for i, r in enumerate(lam.results):
        if isinstance(r, str):
            after.setdefault(r, []).append(i)
    out: list[Stmt] = []
    placed: set[int] = set()
    for s in lam.body:
        out.append(s)
        for d in s.defines():
            for i in after.get(d, ()):
                out.append(accums[i])
                placed.add(i)
    out.extend(accums[i] for i in range(len(accums)) if i not in placed)
    return tuple(out)


@dataclass(frozen=True)
class Fold(Stmt):
    """A scheduled reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/stmt/algebra``). It splits the reduce's **algebra** (the loop-carried
    flat ⊕ — degenerate/componentwise for a plain
    ``sum`` / ``max`` / ``mean``, twisted (exp-family) for online-softmax / flash) from its **structure**
    (the reduce ``axis`` + the per-element ``step`` it folds). Its :class:`AxisRole`
    (``PLANAR`` / ``TWISTED`` / ``CONTRACTION``) is **derived** from those params (:attr:`role`),
    never stored. The fold ``Loop`` is **synthesized on
    demand** (:attr:`loop`), never stored — so the same node tiles under any
    :class:`~emmy.compiler.ir.schedule.ReducePlan` (the reduce partition rides the node's
    ``reduce`` field, read via ``ops.reduce_plan``).

    A reduce whose per-step partial COMPOSES another node — split-K's ``Fold ⊃ bilinear fold``
    (whose ``axis`` ``ksplit`` differs from the inner ``k_axis`` ``kslice``, so no double-reduce),
    flash's ``Σ Q·K`` score at the head of its kv step — spells it ONE way: the node sits in
    ``step`` and :func:`_flatten_nodes` flattens it in place. There is no second ``source`` edge.

    It holds **no projection**: a bare reduce (``sum`` / ``max``) is the kernel root (its grid ``Write``
    is glue); a reduce with a post-fold sweep (softmax / RMSNorm) is the ``source`` of a wrapping
    :class:`Fold` whose body IS that projection. Like every structural node it IS a ``Stmt`` — that is
    what lets a composed step occupy a statement position in another node's body;
    :meth:`lower` flattens it to the synthesized loop (``[loop]``), so ``op_cache_key`` and the
    ``_factor._tile_reduce_axis`` expander stay byte-identical to the bare-loop form.

    The **scheduling param** is the ``reduce`` partition (:class:`ReducePlan` — GRID split / BLOCK coop
    / REG ILP), stamped onto the node by the schedule (its decided value lives **here** on the node
    — read via ``ops.reduce_plan``). ``lower`` ignores it (it's metadata the materializer / ``030_split_reduce``
    read), so it leaves ``op_cache_key`` byte-identical."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    # The reduce axis — ``None`` is the ZERO-AXIS node (what zero-axis ``Fold`` was): no iteration, no monoid,
    # its ``lift`` the per-cell projection. bilinear ``Fold`` is a DERIVED READING of this
    # one stored kind (view classes below), never separate storage.
    axis: Axis | None = None
    unroll: bool = False
    # The CLOSED inputs, each an operand edge (a gmem ``Load`` or an inline node) — the 1k fold
    # vocabulary. Sharing is edge reuse: the step reads an operand's bound name as many times as it
    # needs. ``lower`` splices each edge's body before its first use (:func:`_splice_operands`).
    operands: tuple = ()
    # NO schedule fields: the ``tile`` / ``reduce`` / ``stage`` slices live in
    # ``TileOp.schedule``, keyed by the tree-path codec key — the term is pure algebra, IMMUTABLE
    # across the whole schedule search (a fork is a different map, never a rebuilt tree).
    # A cross-CTA SLICE of the stream (flash split-KV) is not spelled here: ``030_split_reduce``
    # shrinks ``axis`` to the slice length and the slice's absolute base / end ride that axis's
    # :class:`~emmy.compiler.ir.axis.Window` — ONE windowing vocabulary, the same one an axis's
    # split parentage uses, read by the realizer and the mask machinery alike.
    #
    # ---- the λ-foldMap spelling — the fold's storage: a PURE ``lift`` ``λ(k, v₁…vₙ) → S``
    # (params: the iteration var first, then one per operand edge, bound POSITIONALLY) plus the
    # TRUE monoid's flat ``(init, combine)`` pair whose combine carries the REAL accumulator
    # names (its results). The serial step, the ``Accum`` forms and the ``carrier`` annotation
    # are DERIVED (:func:`_fold_derived_step` / ``__post_init__``). ------------------------------ #
    lift: Lambda = field(kw_only=True)
    init: tuple = ()  # the ⊕ seeds — op identities for a plain fold; (−inf, 0, …) LSE
    combine: Lambda | None = field(kw_only=True, default=None)  # S × S → S — THE ⊕; None at zero axes

    def __post_init__(self) -> None:
        if not isinstance(self.init, tuple):
            object.__setattr__(self, "init", tuple(self.init))
        if self.axis is None:
            # The ZERO-AXIS node: no iteration and no monoid, so the only formation fact is the
            # positional binding — one lift param per operand RESULT COMPONENT, no leading
            # iteration var. (The projection (zero-axis) fold was exactly this, with ``fn`` for ``lift``.)
            assert self.combine is None and not self.init, "a zero-axis Fold carries no monoid"
            bound = tuple(n for e in self.operands for n in _operand_result_names(e))
            assert tuple(self.lift.params) == bound, f"lift params {self.lift.params} must bind the operands {bound} positionally"
            return
        # Formation validates the positional binding and the S × S → S arity; the ``carrier``
        # annotation is a DERIVED read (:attr:`carrier`), never a second stored spelling.
        n = len(self.init)
        if len(self.combine.params) != 2 * n or len(self.combine.results) != n:
            raise ValueError(f"Fold combine must be S × S → S at arity {n}: params={self.combine.params} results={self.combine.results}")
        lam = self.lift
        assert lam.params[:1] == (self.axis.name,), f"lift param 0 must be the iteration var {self.axis.name!r}: {lam.params}"
        # One lift param per operand RESULT COMPONENT (a product edge — split-K's sliced
        # multi-channel fold — binds every component), positionally.
        bound = tuple(n for e in self.operands for n in _operand_result_names(e))
        assert tuple(lam.params[1:]) == bound, f"lift params {lam.params[1:]} must bind the operand edges {bound} positionally"
        assert len(lam.results) == n, "one lift result per monoid component"
        if component_ops(self.combine) is not None:
            return  # DEGENERATE: the componentwise family — nothing further to validate
        # TWISTED: the family is selected STRUCTURALLY, never stored — the stored combine
        # must BE the exp/LSE generator's program over these state names (recognition built it
        # exactly there; a foreign twisted combine has no derivation yet and is rejected loudly).
        # The state-component ROLE decision is shape-derived off the lift's injected singleton,
        # no annotation: the pivot is component 0 (its injected term the score), a literal-1
        # injection is a denominator, a value injection an expectation.
        from emmy.compiler.ir.stmt.carrier import exp_combine_states  # noqa: PLC0415

        names = self.combine.results
        other = tuple(f"{nm}__o" for nm in names)
        expected = exp_combine_states(names, other)
        assert self.combine.params == names + other and tuple(self.combine.body) == tuple(expected), (
            "a twisted Fold's combine must be the generated exp/LSE-family program over its state names"
        )
        assert isinstance(lam.results[0], str), "the twisted lift's pivot component must inject the score name"

    @property
    def role(self) -> AxisRole:
        """The fold's :class:`AxisRole`, DERIVED from the stored params — never stored:

        - ``FREE`` iff there is no axis (the zero-axis node — a pure pointwise cell or the
          projection over a source node; what zero-axis ``Fold`` was).
        - ``TWISTED`` iff the stored combine's twist family is non-degenerate (``exp`` — online
          softmax / flash).
        - ``CONTRACTION`` iff the bilinear reading holds (:attr:`_contraction` — a ``⊗`` lift
          distributed over ≥ 2 operand edges under a componentwise-additive ⊕), or the step
          composes exactly the sliced contraction (split-K's outer reduce).
        - ``PLANAR`` otherwise — including an unbindable contraction (matvec-shaped 1-D output,
          no ``(m, n)`` loads, the zero-legal-rows fallback): recognition keeps its loads inline
          in the lift instead of building the node, so there are no edges for the bilinear
          reading to bind and the fold takes the reduce tiers at schedule dispatch. The demotion
          is a FORMATION fact and there is no role rewrite anywhere: recognition keeps an
          unbindable contraction's loads inline in the lift, so there are no edges to bind."""
        if self.axis is None:
            return AxisRole.FREE
        if component_ops(self.combine) is None:
            return AxisRole.TWISTED
        if self.composed is not None:
            return AxisRole.CONTRACTION  # split-K: the outer additive reduce over the sliced node
        if self._contraction is not None:
            return AxisRole.CONTRACTION  # the bilinear cell itself — the node kind that was
        return AxisRole.PLANAR

    @cached_property
    def _contraction(self) -> tuple[object, tuple[Channel, ...]] | None:
        """The BILINEAR reading — ``(a, channels)`` — or ``None`` when this fold is not a
        contraction. This is the derived successor of the stored bilinear ``Fold``'s ``a`` /
        ``channels`` fields: the shape it recognizes is exactly the
        one :meth:`Fold.contraction` builds, so ``a`` / ``channels`` / ``b_trans`` read back
        off any fold recognition stored as a contraction.

        The A/B split rides the OPERAND ORDER — ``(b₀, a, b₁…)`` — not the accesses. That is not
        a shortcut: node-locally the two are symmetric (matmul's ``A[m,k]`` and ``B[k,n]`` both
        carry ``k`` plus one free axis), and telling M from N needs the PLACEMENT, which is a
        caller fact living on the ``TileOp`` and deliberately absent here. Order is what
        ``as_fold`` always used and what the byte-identity gate pins."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

        if self.axis is None or len(self.operands) < 2 or self.combine is None:
            return None
        ops = component_ops(self.combine)
        n = len(self.combine.results)
        if ops != (ElementwiseImpl("add"),) * n or len(self.operands) != n + 1:
            return None
        body = tuple(self.lift.body)
        if len(body) != n or tuple(self.lift.results) != tuple(f"{r}__v" for r in self.combine.results):
            return None
        names = [operand_name(e) for e in self.operands]
        b0, a, rest = names[0], names[1], names[2:]
        want = [(b0, a), *((a, b) for b in rest)]
        for stmt, (x, y) in zip(body, want, strict=True):
            if not isinstance(stmt, Assign) or stmt.op != ElementwiseImpl("multiply") or stmt.args != (x, y):
                return None
        chans = tuple(Channel(b=e, acc=acc) for e, acc in zip((self.operands[0], *self.operands[2:]), self.combine.results, strict=True))
        return self.operands[1], chans

    @property
    def composed(self) -> Fold | None:
        """The single sliced a bilinear ``Fold`` this outer reduce COMPOSES (split-K's
        reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``), or ``None`` — the identity-lift
        λ spelling (one inline node operand carrying the outer's exact accumulator state). The ONE
        read ``030_split_reduce`` and the derived :attr:`role` share."""
        if len(self.lift.body) or len(self.operands) != 1:
            return None
        inner = self.operands[0]
        return inner if is_contraction(inner) else None

    # ---- the DERIVED READINGS. zero-axis ``Fold`` and bilinear ``Fold`` are no longer stored kinds (the
    # collapse); every field they carried reads back off the one stored term here, so their old
    # accessors keep their exact meanings and their consumers keep their exact spellings. ------- #
    @property
    def body(self) -> Body:
        """The projection body — ``lift.body`` (the stmts live on the lambda)."""
        return self.lift.body

    @property
    def a(self):
        """The shared M-resident operand edge of the bilinear reading (``operands[1]``)."""
        v = self._contraction
        assert v is not None, f"not a contraction fold (role={self.role.value}) — no `a` reading"
        return v[0]

    @property
    def channels(self) -> tuple[Channel, ...]:
        """The product channels ``(bᵢ, accᵢ)`` of the bilinear reading — arity 1 is a plain
        matmul, arity N the fused gate⊗up edge over ONE shared ``a``."""
        v = self._contraction
        assert v is not None, f"not a contraction fold (role={self.role.value}) — no `channels` reading"
        return v[1]

    @property
    def b(self):
        """The primary channel's streamed operand edge (``channels[0].b``)."""
        return self.channels[0].b

    @property
    def acc(self) -> str:
        """The primary channel's fold accumulator (``channels[0].acc``)."""
        return self.channels[0].acc

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical ``B[k, n]`` — a gmem
        LAYOUT question, so it is meaningful only for a materialized B; a computed B answers
        ``False`` (every tier that would act on the layout gates on ``isinstance(c.b, Load)``)."""
        return isinstance(self.b, Load) and self.axis.name in self.b.index[-1].free_vars()

    @classmethod
    def contraction(cls, *, k_axis: Axis, a, channels: tuple[Channel, ...]) -> Fold:
        """A BILINEAR fold — the matmul cell (what the bilinear fold kind named). Unlike
        :meth:`projection` this constructor GENERATES algebra: operands `(b₀, a, b₁…)`, the lift
        ``λ(k, b, a, b₂…). (b·a, a·b₂, …)`` and the componentwise-additive ⊕ over the channel
        accumulators. That generated shape is exactly what :attr:`_contraction` reads back, so
        ``a`` / ``channels`` / ``b_trans`` and the ``CONTRACTION`` role all follow from it.

        Arity N ≥ 2 is the fused sibling edge (gate⊗up): N matrices over ONE shared A, scheduled
        and lowered as one unit. Sharing is the arity — the shared edge simply appears in every
        lift term; there is no privileged slot and no let table.

        Placement and schedule live nowhere here: the ``(m, n)`` axes ride ``TileOp.place`` and the
        slices ``TileOp.schedule``, so a node's identity is its algebra alone."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

        mul = ElementwiseImpl("multiply")
        channels = tuple(channels)
        prim = channels[0]
        operands = (prim.b, a, *(ch.b for ch in channels[1:]))
        a_name = operand_name(a)
        body: list[Stmt] = [Assign(name=f"{prim.acc}__v", op=mul, args=(operand_name(prim.b), a_name))]
        body += [Assign(name=f"{ch.acc}__v", op=mul, args=(a_name, operand_name(ch.b))) for ch in channels[1:]]
        accs = tuple(ch.acc for ch in channels)
        lift = Lambda(
            params=(k_axis.name, *(operand_name(e) for e in operands)),
            body=Body(tuple(body)),
            results=tuple(f"{acc}__v" for acc in accs),
        )
        init, combine = M(*(["add"] * len(accs)), names=accs)
        return cls(axis=k_axis, operands=operands, lift=lift, init=init, combine=combine)

    @classmethod
    def projection(cls, fn: Lambda | None = None, operands: tuple = (), *, body=None, results: tuple[str, ...] | None = None) -> Fold:
        """A ZERO-AXIS fold — the pointwise / projection cell (what the zero-axis fold kind was).
        No axis and no monoid: ``fn`` becomes the ``lift``, and it IS the per-cell compute, so
        softmax's normalize, the relu epilogue and flash's ``divide(O, l)`` are this node over the
        reducing fold rather than a wrapper kind around it.

        ``operands`` bind POSITIONALLY, one lift param per RESULT COMPONENT — a product operand
        binds every channel accumulator, so the geglu combine's second read is a bound param and
        never a free name. The ``body=`` form synthesizes the binder (params from the operands'
        components, ``results`` from the last def) and is the arm the raw-loop-IR kernels take,
        where :func:`_loop_ir_fn` tolerates an impure body."""
        operands = tuple(operands)
        if fn is None:
            b = Body.coerce(body) if body is not None else Body()
            params = tuple(n for s in operands for n in _operand_result_names(s))
            if results is None:
                results = _map_results(b) or params[:1]
            fn = _loop_ir_fn(params, b, results)
        elif body is not None or results is not None:
            raise TypeError("Fold.projection: pass fn= xor (body= / results=), not both")
        return cls(axis=None, operands=operands, lift=fn)

    def with_axis(self, axis: Axis) -> Fold:
        """This fold over a different iteration ``axis`` — a pure ALGEBRA edit. The lift's
        leading param IS the iteration var (positional binding), so it renames in lockstep;
        nothing else moves. The warp-flash PV's intra-block → stream swap is the one caller: the
        scalar tree contracts one key per step, the fragment tier the whole block."""
        assert self.axis is not None, "with_axis: an iterating fold only"
        lift = _loop_ir_fn((axis.name, *self.lift.params[1:]), self.lift.body, self.lift.results)
        return replace(self, axis=axis, lift=lift)

    def external_reads(self) -> tuple[str, ...]:
        """Every input buffer read anywhere under this node's operand edges (deep — an inline
        cone may nest its own loads)."""

        def loads(stmts):
            for s in stmts:
                if isinstance(s, Load):
                    yield s.input
                for b in s.nested():
                    yield from loads(b)

        return tuple(dict.fromkeys(nm for e in self.operands for nm in loads(operand_body(e))))

    @cached_property
    def _derived_twisted(self) -> tuple[Stmt, ...]:
        """The MEMOIZED derived blocked evaluation of a λ-spelled twisted fold — cached so the
        synthesized expectation contraction (flash's PV) has ONE identity per stored fold: the
        path walker's derived sites, the schedule accessors (``Sched.tile_of``) and every
        realizer read the same node object (site matching is by identity)."""
        return _twisted_derived_step(self)

    def step_stmts(self) -> tuple[Stmt, ...]:
        """The DERIVED per-cell stmt sequence (:func:`_fold_derived_step`): the lift body with
        each component's ``Accum`` — the combine specialized at the singleton; the full blocked
        evaluation for a twisted fold with operand edges (flash's derived head is its score
        operand edge, its PV the memoized synthesized contraction); the embedded operand for
        the identity-lift composition."""
        return _fold_derived_step(self)

    def _splice_edges(self) -> tuple:
        """The operand edges the GENERIC first-use splice places — every edge not already
        consumed by the derived step: a λ-spelled TWISTED fold's derived blocked evaluation
        embeds its inline-node edges at the head and its Load-bound expectation edges inside
        the synthesized contraction (:func:`_twisted_derived_step`); the identity-lift
        composition (split-K) embeds its one fold operand verbatim. None of those splice
        twice."""
        consumed = {id(s) for s in self.step_stmts()}
        if component_ops(self.combine) is None:
            by_param = _operand_binding(self)
            for term in self.lift.results[1:]:  # EXPECTATION components: a str-injected non-pivot
                if isinstance(term, str):
                    edge = by_param.get(term)
                    if isinstance(edge, Load):
                        consumed.add(id(edge))
        return tuple(e for e in self.operands if id(e) not in consumed)

    @property
    def loop(self) -> Loop:
        """The synthesized annotated reduce ``Loop`` — reconstructed from the params: byte-identical
        to the loop :meth:`from_loop` captured (the λ spelling's construction gate guarantees it;
        a retained ``step`` reproduces trivially). Any **nested structural node inside the
        step** — a a bilinear ``Fold`` / :class:`Fold` /
        :class:`Fold` — is flattened to its own loop nest in place by the shared :func:`_flatten_nodes`
        node-walk (so flash's kv loop holds the head ``Σ Q·K`` score contraction loop and the embedded
        ``Σ_j P·V`` PV contraction loop, and split-K's kslice contraction its own nest — exactly the
        loop-in-body form the scalar tier expands): ONE structural rule for a reduce whose per-step
        partial composes other nodes. Operand edges splice in ahead of the first read of their
        bound param (:func:`_splice_operands` — positional binding names the edges, ties in
        operand order), so a fold with hoisted inputs lowers byte-identically to the flat form
        that carried them in its body."""
        stmts = _splice_operands(self._splice_edges(), _flatten_nodes(Body(self.step_stmts())))
        return Loop(axis=self.axis, body=Body(stmts), unroll=self.unroll, role=self.role)

    def spliced_step(self) -> tuple[Stmt, ...]:
        """The (derived) step with every operand edge's body spliced before its first use — the
        stmt sequence the emit-side node walk consumes (nested structural nodes NOT flattened;
        :attr:`loop` additionally flattens them). Edges the derived blocked evaluation already
        consumed (a twisted fold's head node / expectation Load) never splice twice."""
        return _splice_operands(self._splice_edges(), self.step_stmts())

    @property
    def out(self) -> str:
        """The bound output name — the carried state's primary component (the combine's first
        result; a bare reduce's grid ``Write`` is glue). At zero axes there is no carried state,
        so it is the projection's primary result (what what a projection's ``out`` read)."""
        if self.axis is None:
            r = self.lift.results
            assert r and isinstance(r[0], str), f"zero-axis Fold has no named result: {r!r}"
            return r[0]
        return self.combine.results[0]

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — the synthesized reduce ``Loop``,
        or, at zero axes, the operand nests followed by the projection body (a ``Load`` edge is
        the CUT TERMINAL: the seam value arrives materialized, so the "nest" is the load itself).

        The ONE lowering spelling: every caller that consumes a body calls this method (there is no
        free ``ops.lower`` wrapper duplicating it), which is also what keeps this module free of any
        import back into :mod:`~emmy.compiler.ir.tile.ops`."""
        if self.axis is None:
            prefix = [s for e in self.operands for s in operand_body(e)]
            return [*prefix, *self.body]
        return [self.loop]

    # ---- the stmt protocol: a composed node occupies a statement position, so generic body walks
    # must reach its children. ``defines`` stays the block-stmt default — the fold's names are bound
    # by the derived serial step's ``Accum``\\ s, exactly as for a plain reduce ``Loop``. The one
    # nested body is the lift's. ---------- #
    def defines(self) -> tuple[str, ...]:
        """The names this node BINDS. The block-stmt default for a plain fold — its names are
        bound by the derived serial step's ``Accum``\\ s, exactly as for a plain reduce ``Loop``
        — but the bilinear reading binds its channel accumulators directly, which is what the
        stored bilinear ``Fold`` did and what ``_operand_result_names`` reads."""
        return tuple(ch.acc for ch in self.channels) if self._contraction is not None else super().defines()

    def nested(self) -> tuple[Body, ...]:
        """The one nested body is the lift's — EXCEPT under the bilinear reading, whose lift is
        pure algebra reached through the operand edges (the stored bilinear ``Fold`` had no nested
        body, and generic walks must keep seeing none, or a contraction's multiply args start
        reading as body deps)."""
        return () if self._contraction is not None else (self.lift.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if not bodies:
            return self
        (partial,) = bodies
        return replace(self, lift=_loop_ir_fn(self.lift.params, Body.coerce(partial), self.lift.results))

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Fold must be lowered (Fold.lower) before render")


def is_contraction(x) -> bool:
    """The BILINEAR reading of ``x`` — the predicate that replaced ``isinstance(_, bilinear fold)``
    when the kind dissolved. A predicate, not a kind: it cannot be constructed, subclassed or
    annotated, and it answers ``False`` for a non-node (a ``Load`` edge, a plain stmt in a step
    stream) exactly as a type test would. Prefer this over a bare
    ``x.role is AxisRole.CONTRACTION`` anywhere ``x`` is not already known to be a ``Fold``."""
    return isinstance(x, Fold) and x._contraction is not None


def deep_defines(s: Stmt) -> set[str]:
    """Every SSA name defined in ``s`` (deep — a stat reduce ``Loop``'s ``Accum`` counts)."""
    out = set(s.defines())
    for b in s.nested():
        for child in b:
            out |= deep_defines(child)
    return out


def deep_reads(stmts: list[Stmt]) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep)."""
    out: set[str] = set()
    for s in stmts:
        out |= set(s.deps())
        for b in s.nested():
            out |= deep_reads(list(b))
    return out


def stmt_axis_names(stmts) -> set[str]:
    """Every loop induction variable bound anywhere in ``stmts`` (deep). A composed structural node
    sitting in the body needs no special case — it is a ``Stmt``, so its children are reached through
    the same ``nested()`` walk as any block stmt's."""
    out: set[str] = set()
    for s in stmts:
        ax = getattr(s, "axis", None)
        if ax is not None and hasattr(ax, "name"):
            out.add(ax.name)
        for b in s.nested():
            out |= stmt_axis_names(b)
    return out


def operand_body(op) -> tuple[Stmt, ...]:
    """An operand edge's producing stmts — the singleton gmem ``Load``, or the inline node
    flattened (:meth:`Fold.lower`). A free function, not a per-role ``a_body`` / ``b_body`` pair on
    the node: an edge is an edge, and which ROLE it plays (A vs B) is the caller's reading of the
    operand order, not a property of the edge itself."""
    return (op,) if isinstance(op, Load) else tuple(op.lower())


def operand_name(op) -> str:
    """An operand edge's bound SSA name — the inline node's ``out``, or the ``Load``'s def. Free
    for the same reason as :func:`operand_body`."""
    return op.defines()[-1] if isinstance(op, Load) else op.out


def _operand_result_names(op) -> tuple[str, ...]:
    """An operand edge's bound RESULT names — one per produced component. A single-valued edge
    (a gmem ``Load``) is the 1-tuple of :func:`operand_name`; a product fold / multi-channel
    node exposes EVERY component, so a wrapping zero-axis fold's synthesized params bind one name per
    component (the 1q params-flattening fix: the geglu combine reads BOTH ``acc_g`` and
    ``acc_u`` from one source — before the flattening the second component reached the lambda
    as a free name)."""
    if isinstance(op, Load):
        return (op.defines()[-1],)
    if isinstance(op, Fold):
        if op.axis is None:
            # The zero-axis reading: the projection's NAMED results (a store-only tail names
            # none, and falls back to its primary bound value).
            named = tuple(r for r in op.lift.results if isinstance(r, str))
            return named if named else (op.out,)
        return tuple(op.combine.results)
    return (operand_name(op),)


@dataclass(frozen=True)
class Store:
    """One ROOT-STORE decoration at the kernel boundary — the effect the stored term no
    longer carries. ``write`` is the store verbatim (target buffer, index template, stored value
    names, the atomic flag — holding the ``Write`` whole keeps every field lossless), and it is
    NOT part of the term: ``TileOp.stores`` owns the tuple, and consumers reconstitute the
    effectful stmt stream via :func:`effect_tail`. A ``sweep`` store's ``Write`` rides a per-cell output ``Loop`` over
    that axis (rms/softmax's normalize sweep, ``unroll`` preserved); the swept members are the
    trailing projection stmts reading the axis (:func:`_sweep_start`). Conversion sites go
    through :func:`split_effects`, whose reconstitution round-trip gate is what keeps kernel
    sources byte-identical to the stored-``Write`` era."""

    write: Write
    sweep: Axis | None = None
    unroll: bool = False


def _sweep_start(stmts, axis_name: str) -> int:
    """The first index of the trailing projection run a ``sweep`` store's output ``Loop``
    wraps — the earliest stmt reading the sweep axis (SSA deps + Expr free vars, deep). The
    trailing-RUN rule (everything from that stmt on is swept) is deliberately simple; the
    :func:`split_effects` round-trip gate is what proves it reproduces the captured loop."""
    for i, s in enumerate(stmts):
        if axis_name in _member_reads(s):
            return i
    return len(stmts)


def effect_tail(stmts, stores) -> list[Stmt]:
    """Reassemble the EFFECTFUL projection stmt stream from a pure projection body + the
    kernel-boundary ``stores`` — the ONE reconstitution rule the scheduler's tail gates, the
    materializer's zero-axis ``Fold`` peel and ``030_split_reduce`` share, so the lowered kernels stay
    byte-identical to the stored-``Write`` era. A plain store appends its ``Write``; a
    ``sweep`` store wraps the trailing run of stmts reading its axis (:func:`_sweep_start`)
    into the per-cell output ``Loop``, the ``Write`` last."""
    out = list(stmts)
    for st in stores:
        if st.sweep is None:
            out.append(st.write)
        else:
            i = _sweep_start(out, st.sweep.name)
            out = [*out[:i], Loop(axis=st.sweep, body=Body((*out[i:], st.write)), unroll=st.unroll)]
    return out


def split_effects(stmts) -> tuple[tuple[Stmt, ...], tuple[Store, ...]] | None:
    """Split an effectful projection stmt stream into ``(pure stmts, Store decorations)`` — the
    conversion-side inverse of :func:`effect_tail`, valid ONLY when the reconstitution
    round-trips byte-identically (checked here; ``None`` otherwise — the caller keeps the
    raw-loop-IR spelling, the 1o construction-gate pattern). Recognized shapes: a trailing run
    of top-level root ``Write``\\ s, or ONE trailing non-reduce output sweep ``Loop`` of pure
    stmts whose last stmt is the ``Write``. An already-pure stream returns ``(stmts, ())``."""
    original = list(stmts)
    rest = list(stmts)
    stores: list[Store] = []
    while rest and isinstance(rest[-1], Write):
        stores.insert(0, Store(write=rest.pop()))
    if not stores and rest and isinstance(rest[-1], Loop) and not rest[-1].is_reduce:
        loop = rest[-1]
        inner = list(loop.body)
        if inner and isinstance(inner[-1], Write) and all(s.pure for s in inner[:-1]):
            stores.insert(0, Store(write=inner[-1], sweep=loop.axis, unroll=loop.unroll))
            rest = [*rest[:-1], *inner[:-1]]
    if not all(s.pure for s in rest):
        return None
    if effect_tail(rest, stores) != original:
        return None
    return tuple(rest), tuple(stores)


def refs_axis(s: Stmt, name: str) -> bool:
    """``s`` references axis ``name`` in any index expr (deep)."""
    idx = getattr(s, "index", None)
    if idx and any(name in e.free_vars() for e in idx):
        return True
    return any(refs_axis(child, name) for b in s.nested() for child in b)


@dataclass(frozen=True)
class Channel:
    """One product channel of a a bilinear ``Fold`` — the streamed K×N operand edge ``b`` plus the
    additive fold accumulator ``acc`` that channel produces. A plain matmul is one channel; the
    fused gate⊗up MLP edge is two channels over the node's single shared ``a`` (sharing is arity,
    not naming — the product-carrier contraction outputs a tuple)."""

    b: Load | Fold  # the streamed operand edge — MATERIALIZED or COMPUTED
    acc: str  # this channel's fold accumulator


def _loop_ir_fn(params, body, results) -> Lambda:
    """The RAW-LOOP-IR formation arm — the ONE impure the zero-axis fold's ``lift`` builder left after 1q, for the
    kernels that are loop IR rather than recognized algebra: ``010_recognize``'s un-recognized
    flat escape cells (multi/nested reduces), ``030_split_reduce``'s finalize kernels (``Init``
    seeds + the un-annotated ``StateMerge`` merge ``Loop``), the prologue'd split partial, and
    the coop norm→linear/geglu sibling's composed contraction tail. A PURE body goes through
    strict :class:`Lambda` formation — every ROOT STORE left the term (``TileOp.stores``),
    so impurity here is only iteration/seed structure, never an effect on the output. Reached
    exclusively through zero-axis ``Fold``'s legacy ``body=`` construction / ``with_body`` / the rewrite
    handler — never build one by hand; the escape spelling dies when recognition becomes total
    (the "ONE algorithmic algebra recognizer" direction)."""
    body = Body.coerce(body)
    if all(s.pure for s in body):
        return Lambda(params=tuple(params), body=body, results=tuple(results))
    lam = object.__new__(Lambda)
    object.__setattr__(lam, "params", tuple(params))
    object.__setattr__(lam, "body", body)
    object.__setattr__(lam, "results", tuple(results))
    return lam


def _map_results(body: Body) -> tuple[str, ...]:
    """The synthesized ``results`` of a legacy ``Fold.projection(body=…)`` construction — the last defining
    stmt's name (the last-def convention, run ONCE at construction instead of on every read).
    Empty when nothing in the body defines a name (a store-only projection tail)."""
    for s in reversed(body):
        d = s.defines()
        if d:
            return (d[-1],)
    return ()


# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Fold, rename, sigma, axis_fn):
    # ONE handler for the one stored kind (the collapse retired the Map / bilinear fold arms —
    # singledispatch keys on the stored type, and there is now only one). Every operand edge
    # dispatches back through the registry; the fold renames its lift / monoid in lockstep
    # (params track the operand names positionally, the combine's results ARE the accumulator
    # names). At zero axes there is no iteration var to rename and no monoid to thread.
    operands = tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands)
    axis = axis_fn(s.axis) if s.axis is not None else None
    lead = (axis.name,) if axis is not None else ()
    lift = _loop_ir_fn(
        (*lead, *(rename(p) for p in s.lift.params[len(lead) :])),
        Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.lift.body)),
        tuple(rename(r) if isinstance(r, str) else r for r in s.lift.results),
    )
    combine = rename_combine(s.combine, rename) if s.combine is not None else None
    return replace(s, axis=axis, operands=operands, lift=lift, combine=combine)


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Fold` /
    :class:`Fold` / a bilinear ``Fold``, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``op.lower()``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.

    There is **no** let table: a computed operand is stored inline on its edge, and sharing is the
    product a bilinear ``Fold``'s arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. The per-node schedule SLICES live in
    ``schedule``: ``{codec key → resolved TilePlan / ReducePlan / Stage}``, keyed by the
    tree-path codec's canonical key (:mod:`~emmy.compiler.ir.tile.path` — a fold may carry all
    three families at once, so the path alone cannot key the map; the family selects the slice
    kind, so key and value agree by construction). The ``op`` term is pure algebra, IMMUTABLE
    across the whole schedule search — a fork is a different map, never a rebuilt tree. Read /
    write through :class:`~emmy.compiler.ir.tile.ops.Sched` (``ops.reduce_plan`` is the plan
    accessor); ``lower`` never sees the slices, so kernel identity (``op_cache_key``) is
    untouched. The contraction operand→role binding is not a
    ``TileOp`` field either — a tiled contraction carries its A operand / channels on
    its stored fold (``op``), the single source of truth, resolved recognize-side
    (``010_recognize._nodify_contraction``); the placed reading only PLACES that node."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    workers: WarpSpec | None = None
    schedule: dict = field(default_factory=dict)
    # The kernel's ROOT-STORE decorations (``Store``): the output ``Write``\\ s (and the
    # rms/softmax output-sweep spelling) — a kernel-boundary fact beside ``place``. Empty for a
    # bare reduction / contraction — its grid-cell store
    # stays the materializer's default glue (``_factor.with_store``). Consumers reconstitute
    # the effectful stmt stream via ``effect_tail`` — never read a ``Write`` out of the term.
    stores: tuple = ()
    # The ONE worker inventory (``ir.schedule.Workers``): the ``w``/``n`` worker
    # tokens factored out of the per-site TILE values, derived at option assembly
    # (``ops.Sched.seal_workers`` — loud on cross-site disagreement). ``None`` = the per-cell /
    # pure-reduce forms (derived launch geometry). The wire format still spells the embedded
    # tokens until the step-7 value-grammar split.
    work: object = None

    def pretty_body(self) -> str:
        """The structural dump — delegated to :mod:`~emmy.compiler.ir.tile._dump`, which owns
        every presentation concern in the layer."""
        from emmy.compiler.ir.tile._dump import tile_body  # noqa: PLC0415 — presentation, loaded on demand

        return tile_body(self)


__all__ = [
    "Channel",
    "Fold",
    "Store",
    "TileOp",
    "deep_defines",
    "deep_reads",
    "effect_tail",
    "operand_body",
    "operand_name",
    "refs_axis",
    "split_effects",
    "stmt_axis_names",
]
