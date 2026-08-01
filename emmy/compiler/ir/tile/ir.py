"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine* — the :class:`Map` / :class:`Fold` / :class:`Contraction`
nodes defined **in this module**, alongside ``ir/stmt/algebra``) directly,
plus a thin set of **root-global schedule fields** — the
free-axis → grid :class:`~.schedule.Placement` (``place``), the ONE worker inventory (``work``)
and the warp-spec split (``workers``). The
per-node schedule slices live in ``TileOp.schedule`` (1r): ``{codec key → resolved TilePlan /
ReducePlan / Stage}``, keyed by the tree-path codec's canonical key and read through
``ops.Sched`` — the stored term is pure algebra, IMMUTABLE across the whole schedule search
(flash is a ``Map(sources=(Fold(operands=(Contraction(QK), Load(V)), lift, combine),))``
node tree whose hoisted score edge and DERIVED PV contraction are the ``TILE@dd`` / ``TILE@pj``
sites). There is no per-kind kernel/schedule type: dispatch reads the role
structurally off the node (``ops.axis_role`` — a contraction IS the
:class:`Contraction` kind, a fold's role derives), so MAP / MONOID / SEMIRING all ride the
same ``TileOp``.

**Operand sharing is arity.** "These two matmuls read the same A" is ONE
:class:`Contraction` whose output is a tuple: one ``a`` edge plus N product
:class:`Channel`\\ s ``(b_i, acc_i)``, folding the componentwise ``(+, ×)`` —
the N-component product monoid the derived :attr:`Contraction.loop` carries.
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
stmt's). ``Map.sources`` are the exception on purpose: they are node EDGES,
reached by the node-aware walk (``path.sites`` — the ONE walk over the tree),
like a contraction operand.

The combine lives entirely in the ``op`` wrapper (the :class:`Map` /
:class:`Fold` / :class:`Contraction` nodes here + ``ir/stmt/algebra``). A
fold's role is DERIVED (``Fold.role``), never stored; a contraction is the
:class:`Contraction` node kind itself (1s). ``lower(op)`` flattens the
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
        name = _operand_name(edge)
        idx = next((i for i, st in enumerate(stmts) if name in deep_reads([st])), len(stmts))
        at.setdefault(idx, []).append(edge)
    out: list[Stmt] = []
    for i, st in enumerate((*stmts, None)):
        for edge in at.get(i, ()):
            out.extend(_operand_body(edge))
        if st is not None:
            out.append(st)
    return tuple(out)


def _flatten_nodes(body: Body) -> tuple[Stmt, ...]:
    """Flatten any nested **structural node** — a :class:`Contraction`, :class:`Fold`, or
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
        if isinstance(s, (Fold, Map, Contraction)):
            out.extend(lower(s))
        else:
            out.append(s)
    return tuple(out)


def _derived_expect_fold(o: str, p_name: str, v_edge: Load) -> Contraction:
    """The synthesized expectation contraction of a twisted fold's DERIVED blocked evaluation —
    flash's ``Oblk = Σ_j P·V``, a :class:`Contraction` node (1s). A is the
    register-resident softmax weight ``P`` — a one-stmt cone node whose one legal capture is the
    running max the same derived merge updates — and B is the fold's own expectation operand edge
    (the value ``Load``)."""
    prologue = Map(body=Body((Assign(name=f"{o}__p", op="copy", args=(p_name,)),)))
    cone = Map(body=Body(()), sources=(prologue,))
    k = Axis(name="pj", extent=Dim(1))  # block=1: a singleton intra-block reduce
    return Contraction(k_axis=k, a=cone, channels=(Channel(b=v_edge, acc=f"{o}__pv"),))


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
    reassociation's precondition (split-K): a :class:`Contraction` whose (additive by
    construction) channel accumulators are the names and whose outer ⊕ is componentwise
    ``add``."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    return isinstance(inner, Contraction) and tuple(inner.defines()) == names and ops == (ElementwiseImpl("add"),) * len(names)


def _operand_binding(fold: Fold) -> dict:
    """The bound-name → operand-edge map (positional binding, one lift param per operand RESULT
    COMPONENT — a product edge binds every component to the same edge)."""
    out: dict = {}
    for e in fold.operands:
        for n in _operand_result_names(e):
            out[n] = e
    return out


def _twisted_derived_step(fold: Fold) -> tuple[Stmt, ...]:
    """The DERIVED blocked evaluation of a λ-spelled TWISTED fold (step 7 — the composed ``step``
    sequence dissolved): the INLINE-NODE operand edges at the head in operand order (flash's
    ``Σ_dd Q·K`` score — its derived position, ahead of the lift body exactly where the stored
    step held it), the lift body (the scale / mask stmts), then the generated streaming merge
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
    head = tuple(e for e in fold.operands if isinstance(e, (Fold, Map, Contraction)))
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


def _extract_lift(loop: Loop, like: Fold | None = None) -> tuple[Lambda, tuple, Lambda] | None:
    """Read the λ spelling off a reduce ``Loop`` whose body is the dissolved ``[pure lift stmts…,
    Accum folds]`` sequence — ANNOTATION-FREE: every fact (accumulator names, channel ops, folded
    values) is read off the body's ``Accum``\\ s themselves, and threads into
    :func:`~emmy.compiler.ir.stmt.algebra.M` with the loop's REAL accumulator names. Returns
    ``(lift, init, combine)`` — the flat ⊕ pair — or ``None`` when the shape does not read
    off cleanly (a composed step, an effectful stmt, an identity-less op) — the caller keeps the
    raw-loop escape. A TWISTED (exp-family) loop has no self-describing ``Accum`` spelling (its
    merge interleaves ``base``-``Accum`` folds with ψ rescales), so it extracts only against a
    ``like`` fold carrying the algebra (:func:`_extract_twisted_lift` — the 030 split-partial
    path; recognition builds twisted folds directly, never through here). :meth:`Fold.from_loop`
    re-derives the loop and keeps the λ spelling only on byte-identity, so this extraction can
    stay shape-strict without a correctness burden."""
    if like is not None and component_ops(like.combine) is None:
        return _extract_twisted_lift(loop, like)
    accums = [s for s in loop.body if isinstance(s, Accum)]
    prefix = [s for s in loop.body if not isinstance(s, Accum)]
    if not accums:
        return None
    if any(a.base is not None for a in accums):
        # A ``base``-``Accum`` is the ψ-rescale signature of a dissolved exp-family merge — the
        # twisted spelling reconstructs from the body alone (the byte-compare is the validator).
        return _extract_twisted_self(loop)
    if any(not s.pure for s in prefix):
        return None  # an effectful / raw-block step is not λ-representable (the flat-Map escape)
    names = tuple(a.name for a in accums)
    try:
        lift = Lambda(params=(loop.axis.name,), body=Body(tuple(prefix)), results=tuple(a.value for a in accums))
        init, combine = M(*(a.op for a in accums), names=names)
    except ValueError:
        return None  # an undefined result / identity-less op — not the canonical dissolved shape
    if any(a.dtype is not None for a in accums):
        # A typed accumulator is PRECISION, which the λ spelling does not carry — the derived
        # ``Accum``\\ s come back dtype-free, so the byte-identity gate would reject the fold
        # anyway. Decline here and keep the raw-loop escape rather than silently dropping it.
        return None
    return lift, init, combine


def _extract_twisted_self(loop: Loop) -> tuple[Lambda, tuple, Lambda] | None:
    """Reconstruct the exp-family λ spelling from a TWISTED reduce ``Loop``'s body ALONE — no
    side-band algebra: the state is ``(the maximum-Accum, the add-Accums in body order)`` (the
    generator emits them exactly there), the pivot term is the maximum's folded score, and each
    non-pivot term is either the literal ``1.0`` (a denominator) or an external value name (an
    expectation) — resolved by regenerating the streaming merge per candidate and BYTE-COMPARING
    it against the body's tail (``exp_merge`` is deterministic, so equality is proof). The pure
    prefix ahead of the merge becomes the ``lift`` body. Returns ``None`` when no candidate
    matches — a composed step (flash's in-step folds) or a foreign merge spelling."""
    from itertools import product as _product  # noqa: PLC0415

    from emmy.compiler.ir.stmt.carrier import exp_combine_states, exp_merge  # noqa: PLC0415

    body = tuple(loop.body)
    accums = [s for s in body if isinstance(s, Accum)]
    maxes = [a for a in accums if a.op.reduce_canon == "maximum"]
    adds = [a for a in accums if a.op.reduce_canon == "add"]
    if len(maxes) != 1 or not adds or len(maxes) + len(adds) != len(accums):
        return None
    names = (maxes[0].name, *(a.name for a in adds))
    score = str(maxes[0].value)
    # A non-pivot term is the literal 1.0 (a denominator) or a value NAME — defined by a prefix
    # Load/Assign (flash's ``v``) or read from an enclosing scope. Every such name is a candidate;
    # the byte-compare below is the arbiter, so an over-wide pool costs regenerations, not
    # correctness (exp_merge embeds the term spelling, so no two candidates collide).
    defined = {s.name for s in body if isinstance(s, (Load, Assign))}
    read = {a for s in body for a in (getattr(s, "args", None) or ())}
    cands = sorted((defined | read) - {score} - set(names))
    for combo in _product([1.0, *cands], repeat=len(adds)):
        merge = tuple(exp_merge(names, (score, *combo), key=names[0]))
        if len(body) < len(merge) or body[-len(merge) :] != merge:
            continue
        prefix = body[: -len(merge)]
        if any(not isinstance(s, (Load, Assign)) for s in prefix):
            return None  # a composed step keeps the raw-loop escape
        try:
            lift = Lambda(params=(loop.axis.name,), body=Body(prefix), results=(score, *combo))
        except ValueError:
            return None
        other = tuple(f"{n}__o" for n in names)
        combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
        return lift, (float("-inf"),) + (0.0,) * len(adds), combine
    return None


def _extract_twisted_lift(loop: Loop, like: Fold) -> tuple[Lambda, tuple, Lambda] | None:
    """Read the λ spelling off an exp-family TWISTED reduce ``Loop`` against the ``like`` fold
    that carries its algebra (the 030 split partial — the sliced loop's state names are the
    original fold's): the body must be ``[pure score prefix…, the dissolved streaming merge]``
    verbatim (the merge regenerated from ``like``'s stored combine + injection terms), the prefix
    becomes the ``lift`` body and the injected terms its results (the singleton — ``(x, 1)``),
    and the flat ⊕ pair stores ``like``'s combine — the generated cross-partition program over
    the loop's REAL state names (the formation invariant the consuming ``Fold`` asserts).
    ``None`` when the shape does not read off cleanly — a composed step (flash's in-step QK / PV
    folds, which carry their own schedule slices), a foreign merge spelling, a role the singleton
    shape cannot carry. :meth:`Fold.from_loop`'s byte-identity gate stands behind this extraction
    like the degenerate one's."""
    from emmy.compiler.ir.stmt.carrier import exp_merge  # noqa: PLC0415

    names = tuple(like.combine.results)
    terms = tuple(like.lift.results)
    merge = tuple(exp_merge(names, terms, key=names[0]))
    body = tuple(loop.body)
    if len(body) < len(merge) or body[-len(merge) :] != merge:
        return None
    prefix = body[: -len(merge)]
    if any(not isinstance(s, (Load, Assign)) for s in prefix):
        return None  # a composed step keeps the step spelling
    if not terms or not isinstance(terms[0], str):
        return None
    try:
        lift = Lambda(params=(loop.axis.name,), body=Body(prefix), results=terms)
    except ValueError:
        return None
    return lift, (float("-inf"),) + (0.0,) * (len(names) - 1), like.combine


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

    A reduce whose per-step partial COMPOSES another node — split-K's ``Fold ⊃ Contraction``
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
    / REG ILP), stamped onto the node by ``020_schedule`` (its decided value lives **here** on the node
    — read via ``ops.reduce_plan``). ``lower`` ignores it (it's metadata the materializer / ``030_split_reduce``
    read), so it leaves ``op_cache_key`` byte-identical."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    axis: Axis  # the reduce axis
    unroll: bool = False
    # The CLOSED inputs, each an operand edge (a gmem ``Load`` or an inline node) — the 1k fold
    # vocabulary. Sharing is edge reuse: the step reads an operand's bound name as many times as it
    # needs. ``lower`` splices each edge's body before its first use (:func:`_splice_operands`).
    operands: tuple = ()
    # NO schedule fields (1r): the ``tile`` / ``reduce`` / ``stage`` slices live in
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
    combine: Lambda = field(kw_only=True)  # S × S → S — THE ⊕ (one program)

    def __post_init__(self) -> None:
        if not isinstance(self.init, tuple):
            object.__setattr__(self, "init", tuple(self.init))
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
        # TWISTED (1p): the family is selected STRUCTURALLY, never stored — the stored combine
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

    @classmethod
    def from_loop(cls, loop: Loop, like: Fold | None = None) -> Fold | None:
        """Build a :class:`Fold` from a reduce ``Loop`` — the loop-to-node constructor,
        ANNOTATION-FREE: every algebra fact reads off the body's own ``Accum``\\ s
        (:func:`_extract_lift`), so the loop carries no side-band algebra. A TWISTED loop has no
        self-describing spelling (its merge interleaves ``base``-``Accum`` folds with ψ
        rescales), so it extracts only against the ``like`` fold that carries its algebra —
        030's split partial, the sliced loop of a known fold. A canonical loop (the dissolved
        ``[pure lift stmts…, fold(s)]`` sequence) stores λ-spelled — ``lift`` + the flat
        ``(init, combine)`` pair threading the loop's real accumulator names — and the
        byte-identity gate settled at construction. A NON-canonical shape (an effectful /
        raw-block step, a non-reproducible derivation) returns ``None`` — the caller keeps the
        raw-loop-IR ``Map`` escape."""
        lifted = _extract_lift(loop, like)
        if lifted is None:
            return None
        lift, init, combine = lifted
        fold = cls(axis=loop.axis, unroll=loop.unroll, lift=lift, init=init, combine=combine)
        # The byte-identity gate — a non-reproducible BODY keeps the raw-loop escape. The role
        # annotation is the fold's own DERIVED read, deliberately excluded: an unbindable matvec
        # captures a CONTRACTION-shaped loop and derives PLANAR — the 1l demotion, now a
        # formation fact (its loads stay inline in the lift).
        derived = fold.loop
        if (derived.body, derived.axis, derived.unroll) != (loop.body, loop.axis, loop.unroll):
            return None
        return fold

    @property
    def role(self) -> AxisRole:
        """The fold's :class:`AxisRole`, DERIVED from the stored params — never stored:

        - ``TWISTED`` iff the stored combine's twist family is non-degenerate (``exp`` — online
          softmax / flash).
        - ``CONTRACTION`` iff the step composes exactly the sliced contraction node (split-K's
          outer reduce). A contraction cell itself is the :class:`Contraction` node (1s) —
          never a fold.
        - ``PLANAR`` otherwise — including an unbindable contraction (matvec-shaped 1-D output,
          no ``(m, n)`` loads, the zero-legal-rows fallback): recognition keeps its loads inline
          in the lift instead of building the node, and the fold takes the reduce tiers at
          schedule dispatch. No role rewrite anywhere."""
        if component_ops(self.combine) is None:
            return AxisRole.TWISTED
        if self.composed is not None:
            return AxisRole.CONTRACTION  # split-K: the outer additive reduce over the sliced node
        return AxisRole.PLANAR

    @property
    def composed(self) -> Contraction | None:
        """The single sliced :class:`Contraction` this outer reduce COMPOSES (split-K's
        reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``), or ``None`` — the identity-lift
        λ spelling (one inline node operand carrying the outer's exact accumulator state). The ONE
        read ``030_split_reduce`` and the derived :attr:`role` share."""
        if len(self.lift.body) or len(self.operands) != 1:
            return None
        inner = self.operands[0]
        return inner if isinstance(inner, Contraction) else None

    def demoted(self) -> Fold:
        """The operand hoist UNDONE — every edge moves INLINE into the lift body as a stmt (a
        materialized ``Load`` verbatim, a computed node as the structural NODE — a term is a value,
        legal in a pure ``Lambda``), each placed before the first read of its bound name (ties in
        operand order — the splice rule). With no operand edges the bilinear parse declines, so the
        fold DERIVES ``PLANAR`` and takes the reduce tiers at dispatch; ``_flatten_nodes`` flattens
        the inline node at lowering, so the derived loop is byte-identical to the hoisted
        spelling's (the demotion is a spelling change, never a semantics change)."""
        lam = self.lift
        assert lam is not None, "demoted: a λ-spelled fold only"
        body = list(lam.body)
        for edge in reversed(self.operands):
            name = _operand_name(edge)
            idx = next((i for i, st in enumerate(body) if name in deep_reads([st])), len(body))
            body.insert(idx, edge)
        lift = Lambda(params=(lam.params[0],), body=Body(tuple(body)), results=lam.results)
        return Fold(axis=self.axis, unroll=self.unroll, lift=lift, init=self.init, combine=self.combine)

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
        step** — a :class:`Contraction` / :class:`Fold` /
        :class:`Map` — is flattened to its own loop nest in place by the shared :func:`_flatten_nodes`
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
        result; a bare reduce's grid ``Write`` is glue; a projected reduce's output name lives on
        the wrapping ``Map``)."""
        return self.combine.results[0]

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — just the synthesized reduce
        ``Loop`` (a wrapping ``Map`` appends its projection)."""
        return [self.loop]

    # ---- the stmt protocol: a composed node occupies a statement position, so generic body walks
    # must reach its children. ``defines`` stays the block-stmt default — the fold's names are bound
    # by the derived serial step's ``Accum``\\ s, exactly as for a plain reduce ``Loop``. The one
    # nested body is the lift's. ---------- #
    def nested(self) -> tuple[Body, ...]:
        return (self.lift.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (partial,) = bodies
        lift = Lambda(params=self.lift.params, body=Body.coerce(partial), results=self.lift.results)
        return replace(self, lift=lift)

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Fold must be lowered (ops.lower) before render")


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


def _operand_body(op) -> tuple[Stmt, ...]:
    """An operand edge's producing stmts — the singleton gmem ``Load``, or the inline node
    flattened."""
    from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415 — avoid an import cycle

    return (op,) if isinstance(op, Load) else tuple(lower(op))


def _operand_name(op) -> str:
    """An operand edge's bound SSA name — the inline node's ``out``, or the ``Load``'s def."""
    return op.defines()[-1] if isinstance(op, Load) else op.out


def _operand_result_names(op) -> tuple[str, ...]:
    """An operand edge's bound RESULT names — one per produced component. A single-valued edge
    (a gmem ``Load``) is the 1-tuple of :func:`_operand_name`; a product fold / multi-channel
    node exposes EVERY component, so a wrapping ``Map``'s synthesized params bind one name per
    component (the 1q params-flattening fix: the geglu combine reads BOTH ``acc_g`` and
    ``acc_u`` from one source — before the flattening the second component reached the lambda
    as a free name)."""
    if isinstance(op, Load):
        return (op.defines()[-1],)
    if isinstance(op, Fold):
        return tuple(op.combine.results)
    if isinstance(op, Contraction):
        return op.defines()
    if isinstance(op, Map):
        named = tuple(r for r in op.fn.results if isinstance(r, str))
        return named if named else (op.out,)
    return (_operand_name(op),)


@dataclass(frozen=True)
class Store:
    """One ROOT-STORE decoration at the kernel boundary (1q) — the effect the stored term no
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
    materializer's ``Map`` peel and ``030_split_reduce`` share, so the lowered kernels stay
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
    """One product channel of a :class:`Contraction` — the streamed K×N operand edge ``b`` plus the
    additive fold accumulator ``acc`` that channel produces. A plain matmul is one channel; the
    fused gate⊗up MLP edge is two channels over the node's single shared ``a`` (sharing is arity,
    not naming — the product-carrier contraction outputs a tuple)."""

    b: Load | Map | Fold | Contraction  # the streamed operand edge — MATERIALIZED or COMPUTED
    acc: str  # this channel's fold accumulator


@dataclass(frozen=True)
class Contraction(Stmt):
    """The bilinear contraction node — the THIRD stored node kind (1s, next to :class:`Fold` /
    :class:`Map`): every recognized contraction — root matmul cell, fused computed-A edge, flash's
    hoisted QK operand edge, the derived PV, split-K's sliced partial — stores as this node, and
    the node kind IS the ``CONTRACTION`` role (no bilinear parse). The node is pure algebra AND
    NOTHING ELSE: the contraction ``k_axis`` (the node's own reduce axis), the shared ``a`` operand
    edge and the product ``channels`` ``(b_i, acc_i)``; the projection has ONE home — the wrapping
    :class:`Map`'s body — never a node field. Placement (the ``(m, n)`` output axes, the leading
    grid axes) and schedule (``TilePlan`` / ``Stage``) are CALLER facts and live nowhere on the
    node: the schedule slices' one home is ``TileOp.schedule`` (keyed by the tree-path codec
    against this very node, read through ``ops.Sched``), the placement's is ``TileOp.place``, and a
    tier binds both to the node at the point of use as a lowering-side VIEW
    (a placed ``TilePlan`` slice for the geometry, the node itself for the algebra). So a node's
    identity IS its algebra — ``==`` / ``hash`` / :func:`ops.term_key` cannot see a schedule, and
    no emission path can leak one into a stored term. :func:`ops.lower` / ``ops.reduce_loop``
    flatten it to the
    synthesized mul-add ``CONTRACTION`` loop nest (:attr:`loop`), the same ``for k: v = a·b; acc
    += v`` form ``_atom._ScalarOps.reduce`` register-tiles through the shared ``_contract_kloop``
    skeleton. Keeping the schedule OFF the node is what lets the same operand/acc params be tiled
    by a different ``TilePlan`` (the flash inner QK/PV reuse) — two views over one term. Arity N ≥ 2
    is the fused sibling edge (gate⊗up) — a product semiring outputting N matrices over ONE shared
    A, scheduled and lowered as one unit under a single ``TILE`` / ``STAGE`` slice.

    The contraction LOOP is never stored — both tiers *synthesize* it from the operands:
    ``_atom.reduce_codegen`` lowers the mma atom into ``ldmatrix`` + ``mma.sync`` and the scalar atom into a
    ``for k: acc += a*b`` register-tiled loop — then run the projection peeled off the wrapping
    ``Map`` (``acc`` is the SSA name the synthesized reduce produces and the projection consumes). The
    operand buffers ride :meth:`external_reads`; the node has no nested ``Body``. ``_factor.factorize``
    reads the placement-derived geometry (the ``(m, n)`` :class:`~emmy.compiler.ir.schedule.Side`
    pair — ``tile`` / ``mask`` /
    ``block`` / ``unit`` per axis — plus ``launch_threads``) off the placed ``TilePlan``; only
    ``b_trans``, a gmem LAYOUT fact of the stored ``b`` edge, stays on the node. The atom selects
    the codegen — there is no separate ``Leaf`` / per-atom subclass. :meth:`as_fold` remains the
    node's DERIVED λ reading — the flat ``(init, combine)`` algebra spelling ``Reduction`` (the
    cross-partition programs) and :meth:`Fold.demoted` consume."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    k_axis: Axis  # the contraction axis — the node's OWN (reduce) axis; params
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
    a: Load | Map | Fold | Contraction = field(kw_only=True)
    channels: tuple[Channel, ...] = field(kw_only=True)  # the product channels (b_i, acc_i) — params; arity 1 = plain matmul

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
        further channel splices its ``b → ⊗ → ⊕`` triple after it reusing that A value — the
        N-component identity-family state (one additive ``Accum`` per channel) is the true
        product-monoid state, so the cross-CTA split tier folds every channel's partials and the
        coop tier remains contraction-blind."""
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
        a_name, k = self.a_name, self.k_axis.name
        extra: list[Stmt] = []
        for ch in self.channels[1:]:
            lift = Assign(name=f"{ch.acc}__v", op=ElementwiseImpl("multiply"), args=(a_name, _operand_name(ch.b)))
            extra += [*_operand_body(ch.b), lift, Accum(name=ch.acc, value=lift.name, op=ElementwiseImpl("add"), axes=(k,))]
        return Loop(axis=base.axis, body=Body((*base.body, *extra)), unroll=base.unroll, role=base.role)

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body — just the synthesized reduce ``Loop`` (the derived product
        loop at arity N). A projection is NEVER
        here: it has one home, the wrapping :class:`Map`'s body. The materializer expands the node
        through ``_factor.factorize`` instead; this is the structural-key / dump path."""
        return [self.loop]

    def as_fold(self) -> Fold:
        """The node's DERIVED λ reading (1s — no longer the storage direction): a λ-spelled
        :class:`Fold` with ``operands`` the closed inputs bound positionally to the lift params,
        the ``lift`` the bilinear ``λ(k, b, a, b₂…). (b·a, a·b₂, …)``, and the flat ``(init,
        combine)`` the componentwise additive combine threading the channel accumulator names —
        the ONE algebra spelling the fold-generic machinery consumes: ``Reduction`` (the
        cross-partition merge programs) and :meth:`demoted` (the PLANAR demotion). The
        operand tuple order and the lift arg spellings reproduce the historical synthesis exactly
        (primary lift ``(b, a)``, further channels ``(a, b_i)``), so the derived serial step +
        first-use splice yield a loop whose body/axis are byte-identical to :attr:`loop`
        (unit-tested at both arities)."""
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

        mul = ElementwiseImpl("multiply")
        prim = self.channels[0]
        k = self.k_axis.name
        operands = (prim.b, self.a, *(ch.b for ch in self.channels[1:]))
        body: list[Stmt] = [Assign(name=f"{prim.acc}__v", op=mul, args=(_operand_name(prim.b), self.a_name))]
        body += [Assign(name=f"{ch.acc}__v", op=mul, args=(self.a_name, _operand_name(ch.b))) for ch in self.channels[1:]]
        accs = tuple(ch.acc for ch in self.channels)
        lift = Lambda(
            params=(k, *(_operand_name(e) for e in operands)),
            body=Body(tuple(body)),
            results=tuple(f"{acc}__v" for acc in accs),
        )
        init, combine = M(*(["add"] * len(accs)), names=accs)
        return Fold(
            axis=self.k_axis,
            operands=operands,
            lift=lift,
            init=init,
            combine=combine,
        )

    # ---- params: the node's own reduce axis + the (m, n) output axes unpacked ---------- #
    @property
    def axis(self) -> Axis:
        """The node's schedule-bearing axis — the contraction (K) axis, the same read a
        :class:`Fold`'s ``axis`` field answers (site keying, axis-name walks)."""
        return self.k_axis

    @property
    def role(self) -> AxisRole:
        """Always ``CONTRACTION`` — the node kind IS the role (the stored-node successor of the
        fold's bilinear-parse derivation)."""
        return AxisRole.CONTRACTION

    @property
    def composed(self) -> Contraction | None:
        """Always ``None`` — a contraction IS the cell, it composes no inner node. Present so a
        ``Fold | Contraction`` head reads :attr:`Fold.composed` without an isinstance guard."""
        return None

    def demoted(self) -> Fold:
        """:meth:`Fold.demoted` on this node's λ reading — the demotion works on the λ spelling,
        which is where the hoisted edges are there to move inline."""
        return self.as_fold().demoted()

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical B[k, n] — read off the
        primary channel's load, the same test ``_atomize`` made when it bound the operand (channel
        layout agreement is a formation gate, so the primary speaks for the group). A gmem LAYOUT
        question, so it is meaningful only for a materialized B; a computed B has no gmem address and
        answers ``False`` (every tier that would act on the layout gates on ``isinstance(c.b, Load)``
        first)."""
        return isinstance(self.b, Load) and self.k_axis.name in self.b.index[-1].free_vars()

    # ---- the stmt protocol (see ``Fold``): the operand edges are node EDGES, reached by the
    # node-aware walk (``path.sites``) — ``nested()`` stays the no-children default. ---------- #
    def defines(self) -> tuple[str, ...]:
        return tuple(ch.acc for ch in self.channels)

    def render(self, ctx: RenderCtx) -> list[str]:
        raise AssertionError("Contraction must be lowered (ops.lower) before render")

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
        # Algebra only — the placement / schedule the old stamped fields printed here belongs to
        # the placed ``TilePlan`` slice, and ``TileOp.pretty_body`` prints the schedule slices beside the term.
        ops = f"{src(self.a, self.a_name)} @ {bs}{t} -> {accs}"
        return [f"{indent}Contraction [Σ {self.k_axis.name}] {ops}"]


def _loop_ir_fn(params, body, results) -> Lambda:
    """The RAW-LOOP-IR formation arm — the ONE impure ``Map.fn`` builder left after 1q, for the
    kernels that are loop IR rather than recognized algebra: ``010_recognize``'s un-recognized
    flat escape cells (multi/nested reduces), ``030_split_reduce``'s finalize kernels (``Init``
    seeds + the un-annotated ``StateMerge`` merge ``Loop``), the prologue'd split partial, and
    the coop norm→linear/geglu sibling's composed contraction tail. A PURE body goes through
    strict :class:`Lambda` formation — every ROOT STORE left the term at 1q (``TileOp.stores``),
    so impurity here is only iteration/seed structure, never an effect on the output. Reached
    exclusively through ``Map``'s legacy ``body=`` construction / ``with_body`` / the rewrite
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
    """The synthesized ``results`` of a legacy ``Map(body=…)`` construction — the last defining
    stmt's name (the last-def convention, run ONCE at construction instead of on every read).
    Empty when nothing in the body defines a name (a store-only projection tail)."""
    for s in reversed(body):
        d = s.defines()
        if d:
            return (d[-1],)
    return ()


@dataclass(frozen=True, init=False)
class Map(Stmt):
    """A pointwise lift / projection wrapper — ``fn: Lambda`` over zero or more reduction /
    contraction ``sources``, bound POSITIONALLY: ``sources[i]``'s result components produce the
    values ``fn.params`` name (one param per component — a product source binds every channel
    accumulator, so the geglu combine's second read is a bound param, never a free name).
    ``fn.results`` are the bound output names (``out`` reads ``fn.results[0]``).

    ``fn.body`` is the per-cell pointwise / projection compute — operand ``Load``\\ s, the lift
    ``Assign``\\ s / ``Select``\\ s — and it is PURE for every recognized term: the root
    output ``Write`` (and the rms/softmax output-sweep ``Loop`` around it) rides ``TileOp.stores``
    at the kernel boundary, reconstituted on demand (``effect_tail``). The raw-loop-IR kernels
    that are NOT recognized algebra — the un-recognized flat escape cell, ``030_split_reduce``'s
    finalize (``Init`` seeds + the merge ``Loop``), the coop fused-tail sibling — still form an
    impure fn through the one ``_loop_ir_fn`` arm. ``sources`` are the structural nodes it
    projects over
    (:class:`Fold` / :class:`Contraction` — ``project ∘ reduce``), empty for a pure pointwise
    map; SEVERAL sources is the **fused sibling group** — N contractions over one shared A, the
    gate⊗up MLP edge; the group schedules and lowers as ONE unit. Every recognized contraction —
    per-cell scalar included — is stored as a ``role=CONTRACTION`` fold (``_nodify_contraction``
    in ``010_recognize``). It is itself a ``Stmt``, so it can occupy a statement position in
    another node's body (the fused sibling group inside a split reduce's ``step``); ``body`` is
    the compat read for ``fn.body``.

    The legacy ``Map(body=…, sources=…)`` construction shape keeps working: the binder is
    synthesized (params = the sources' result components; results = the last def, once, here)."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    fn: Lambda
    # A ``Load`` source is the CUT TERMINAL (phase 4): a placement cut materializes the seam
    # value to a buffer and the parent consumes it as a plain load — every edge admits ``Load``.
    sources: tuple[Fold | Contraction | Map | Load, ...]

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
            # One param per source RESULT COMPONENT (1q params flattening): a product fold /
            # multi-channel source produces N values and the projection may read all of them
            # (the geglu combine reads ``acc_g`` AND ``acc_u`` from one source), so every
            # component is a bound param — never a free name.
            params = tuple(n for s in sources for n in _operand_result_names(s))
            if results is None:
                results = _map_results(b) or params[:1]
            fn = _loop_ir_fn(params, b, results)
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
        return Map(fn=_loop_ir_fn(self.fn.params, Body.coerce(body), self.fn.results), sources=self.sources)

    # ---- the stmt protocol (see ``Fold``): ``sources`` are NOT nested bodies — they are node
    # edges, reached by the node-aware walk (``path.sites``), the same way a ``Contraction``'s
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
    fn = _loop_ir_fn(
        tuple(rename(p) for p in s.fn.params),
        Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.fn.body)),
        tuple(rename(r) if isinstance(r, str) else r for r in s.fn.results),
    )
    return Map(fn=fn, sources=tuple(_rewrite(src, rename, sigma, axis_fn) for src in s.sources))


@_rewrite.register
def _(s: Fold, rename, sigma, axis_fn):
    # Every operand edge dispatches back through the registry. The fold renames its lift /
    # monoid in lockstep (params track the operand names positionally, the combine's results
    # ARE the accumulator names) and re-derives the carrier.
    axis = axis_fn(s.axis)
    operands = tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands)
    lift = Lambda(
        params=(axis.name, *(rename(p) for p in s.lift.params[1:])),
        body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.lift.body)),
        results=tuple(rename(r) if isinstance(r, str) else r for r in s.lift.results),
    )
    return replace(s, axis=axis, operands=operands, lift=lift, combine=rename_combine(s.combine, rename))


@_rewrite.register
def _(s: Contraction, rename, sigma, axis_fn):
    # The operand edges dispatch back through the registry; the channel accumulator names rename
    # in lockstep with them (the accs ARE the node's bound results).
    return replace(
        s,
        k_axis=axis_fn(s.k_axis),
        a=_rewrite(s.a, rename, sigma, axis_fn),
        channels=tuple(replace(ch, b=_rewrite(ch.b, rename, sigma, axis_fn), acc=rename(ch.acc)) for ch in s.channels),
    )


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Map` /
    :class:`Fold` / :class:`Contraction`, or ``None`` for a
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
    product :class:`Contraction`'s arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. The per-node schedule SLICES live in
    ``schedule`` (1r): ``{codec key → resolved TilePlan / ReducePlan / Stage}``, keyed by the
    tree-path codec's canonical key (:mod:`~emmy.compiler.ir.tile.path` — a fold may carry all
    three families at once, so the path alone cannot key the map; the family selects the slice
    kind, so key and value agree by construction). The ``op`` term is pure algebra, IMMUTABLE
    across the whole schedule search — a fork is a different map, never a rebuilt tree. Read /
    write through :class:`~emmy.compiler.ir.tile.ops.Sched` (``ops.reduce_plan`` is the plan
    accessor); ``lower`` never sees the slices, so kernel identity (``op_cache_key``) is
    untouched. The contraction operand→role binding is not a
    ``TileOp`` field either — a tiled contraction carries its A operand / channels on
    its stored fold (``op``), the single source of truth, resolved recognize-side
    (``010_recognize._nodify_contraction``); ``_view.contraction_view`` only PLACES that node."""

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
    # The ONE worker inventory (1r in-memory — ``ir.schedule.Workers``): the ``w``/``n`` worker
    # tokens factored out of the per-site TILE values, derived at option assembly
    # (``ops.Sched.seal_workers`` — loud on cross-site disagreement). ``None`` = the per-cell /
    # pure-reduce forms (derived launch geometry). The wire format still spells the embedded
    # tokens until the step-7 value-grammar split.
    work: object = None

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering — plus the
        kernel-boundary ``stores`` (the root ``Write``\\ s live here since 1q, so a dump without
        them would hide where the kernel's output lands)."""
        from emmy.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        lines = pretty(self.op, "    ")
        for st in self.stores:
            sweep = f" sweep({st.sweep.name})" if st.sweep is not None else ""
            lines += [f"    store{sweep}: {line.strip()}" for line in st.write.pretty()]
        return "\n".join(lines)


__all__ = [
    "Channel",
    "Contraction",
    "Map",
    "Fold",
    "Store",
    "TileOp",
    "deep_defines",
    "deep_reads",
    "effect_tail",
    "refs_axis",
    "split_effects",
    "stmt_axis_names",
]
