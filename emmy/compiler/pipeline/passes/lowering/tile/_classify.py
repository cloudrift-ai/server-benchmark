"""Classification over the lifted Fold tree — recognition stated on ``Fold`` fields.

The total lift (:mod:`._lift`) stores what a loop SAYS; classification rewrites what it MEANS:
each stage is a pure, semantics-preserving tree rewrite whose condition reads only the node's
stored params (``lift`` / ``combine`` / ``operands``), never a raw loop stmt. A stage that
declines rewrites nothing — the fold already derives its fallback role (PLANAR) structurally,
so demotion is a no-op by construction, not a raise/catch.

Stage here today: **contraction binding** (:func:`bind_bilinear`). Pending (the rebuild
registry's other casualty classes): online-softmax pairing, the monoid-producer composition,
the placement cut fork.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.pure import component_ops
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.stmt import Assign, Body, Load, Select
from emmy.compiler.ir.tile.ops import make_cone


def classify(node, free):
    """Run the classification stages over the lifted root tree: the root's own fold and each
    operand fold of the root projection try every stage in place, then :func:`_legalize` demotes
    what downstream cannot consume yet. Needs the output-ordered ``free`` axes — role assignment
    is a whole-kernel fact (the trailing pair is the output's ``(m, n)``), which is exactly why
    this cannot live inside the loop parser."""
    if len(free) >= 2:
        m_name, n_name = free[-2].name, free[-1].name
        if isinstance(node, Fold) and node.axis is not None:
            node = bind_bilinear(node, m_name, n_name) or node
        elif isinstance(node, Fold) and node.axis is None and node.operands:
            ops = tuple((bind_bilinear(f, m_name, n_name) or f) if isinstance(f, Fold) and f.axis is not None else f for f in node.operands)
            if any(a is not b for a, b in zip(ops, node.operands, strict=True)):
                # The rebound channels keep their accumulator names, so the projection's λ params
                # (bound positionally to the operands' result components) are unchanged.
                node = replace(node, operands=ops)
    return _legalize(node)


def _legalize(node):
    """Demote what a NAMED downstream capability cannot consume — an explicit list that shrinks
    as stages land, never a parity list. Today's one entry: **multi-operand root projections** —
    materialization stamps ONE root site tree (``_schedule._materialize``) and the materializer
    recurses into ``operands[0]`` only, so the first operand stays hoisted and the rest are
    restored to their verbatim loops at the body head, in order. Scope holds by construction: a
    restored loop may read the kept operand's carried state (operands lower first), never the
    reverse — hoisting required reading nothing the body defines. The online-softmax pairing
    will CONSUME the sibling pair into one TWISTED fold ahead of this demotion once it lands."""
    if isinstance(node, Fold) and node.axis is None and len(node.operands) > 1:
        keep, rest = node.operands[0], node.operands[1:]
        return Fold.projection(body=Body((*(f.loop for f in rest), *node.body)), operands=(keep,))
    return node


def _idx_vars(load: Load) -> set[str]:
    """Every free Var name across a load's index exprs."""
    return {v for e in load.index or () for v in e.free_vars()}


def _cone(body: list, arg: str, avoid_name: str, k_name: str) -> list | None:
    """The backward cone of ``arg`` within a λ body, when it reads as a pure MAP producer for one
    contraction side: every member a ``Load`` / ``Assign`` / ``Select``, self-contained (no free
    SSA reads — a cross-operand read is another stage's shape), no member load indexed by the
    OTHER side's grid axis, and at least one K-indexed load (a k-invariant value is a hoistable
    factor, not an operand). ``None`` when the cone is not this shape."""
    cone = Body(tuple(body)).backward_cone([arg])
    stmts = list(cone.members)
    if not stmts or any(not isinstance(s, (Load, Assign, Select)) for s in stmts):
        return None
    if cone.external_reads:
        return None
    loads = [s for s in stmts if isinstance(s, Load)]
    if any(avoid_name in _idx_vars(ld) for ld in loads) or not any(k_name in _idx_vars(ld) for ld in loads):
        return None
    return stmts


def bind_bilinear(f: Fold, m_name: str, n_name: str) -> Fold | None:
    """Rebind a lifted additive fold as the bilinear contraction — the ONE contraction reading,
    off the λ spelling: each ``lift.results[i]`` must name a two-arg ``multiply``; its args
    classify by their loads' grid-axis indexing — B is the ``(n, k)``-indexed side and never
    reads ``m``, A the mirror (role-exclusive: a load carrying both axes is neither, and the
    fold stays PLANAR). One side may be a pure MAP cone (the computed operand: A rides
    :func:`make_cone`, which fixes the K seam on the node; B a plain projection evaluated per
    slab cell). N channels must share ONE A value — sharing is the node's arity. Every λ body
    stmt must be consumed by a lift or a cone (:meth:`Fold.contraction` REGENERATES the lift, so
    an unaccounted stmt would be silently dropped — decline instead). ``None`` when any
    condition fails; the caller keeps the PLANAR fold unchanged.

    Deliberately not yet bound here: the both-computed decode pair and the k-invariant factor
    hoist (the fp8 / W8A8 mul-hoist arm) — registered casualties until ported to the λ body."""
    if f.axis is None or f.combine is None or f.operands:
        return None  # a fold with operand edges already composes producers — another stage's shape
    ops = component_ops(f.combine)
    if ops is None or any(o.reduce_canon != "add" for o in ops):
        return None
    k_name = f.axis.name
    body = list(f.lift.body)
    defs = {s.name: s for s in body if isinstance(s, Assign)}
    loads = {s.names[0]: s for s in body if isinstance(s, Load)}

    def role_load(name: str, on: str, off: str) -> Load | None:
        ld = loads.get(name)
        return ld if ld is not None and on in _idx_vars(ld) and off not in _idx_vars(ld) else None

    # The per-channel ⊗ reads: each result's two-arg multiply, its directly-named B load (or
    # ``None`` when B rides a computed value), and the other argument — the channel's A value.
    reads: list[tuple[Assign, Load | None, str]] = []
    for res in f.lift.results:
        lift = defs.get(res) if isinstance(res, str) else None
        if lift is None or lift.op.name != "multiply" or len(lift.args) != 2:
            return None
        b_arg = next((a for a in lift.args if role_load(a, n_name, m_name) is not None), None)
        a_arg = next((a for a in lift.args if a != b_arg), None)
        if a_arg is None:
            return None  # a square product (both args one value) has no role split
        reads.append((lift, loads.get(b_arg) if b_arg is not None else None, a_arg))
    if len({a for _, _, a in reads}) != 1:
        return None  # channels do not share ONE A value — not the product-carrier shape
    a_arg = reads[0][2]
    consumed: set[int] = {id(lift) for lift, _, _ in reads}
    if all(b is not None for _, b, _ in reads):
        consumed.update(id(b) for _, b, _ in reads)
        a_edge = role_load(a_arg, m_name, n_name)
        if a_edge is not None:
            consumed.add(id(a_edge))
        else:
            cone = _cone([s for s in body if id(s) not in consumed], a_arg, n_name, k_name)
            if cone is None:
                return None  # a computed A that does not read cleanly — the fold stays PLANAR
            consumed.update(id(s) for s in cone)
            a_edge = make_cone(cone, k_name)
        b_edges: list = [b for _, b, _ in reads]
    elif len(reads) == 1 and (a_edge := role_load(a_arg, m_name, n_name)) is not None:
        # B rides a computed value (A is the direct load). A computed B is a closed zero-axis
        # operand node — no row-statistic seam to split; its whole MAP tree is evaluated at each
        # (k, n) slab cell.
        consumed.add(id(a_edge))
        lift = reads[0][0]
        b_arg = next(a for a in lift.args if a != a_arg)
        cone = _cone([s for s in body if id(s) not in consumed], b_arg, m_name, k_name)
        if cone is None:
            return None
        consumed.update(id(s) for s in cone)
        b_edges = [Fold.projection(body=Body(tuple(cone)))]
    else:
        return None  # both sides computed — the mul-hoist / decode-pair arm, not yet ported
    if any(id(s) not in consumed for s in body):
        return None  # an unaccounted λ stmt — a shape this binding does not understand
    # B-layout agreement across channels: one shared A fragment implies one slab orientation
    # (k-last vs k-first), so disagreeing loads never group.
    if len({k_name in b.index[-1].free_vars() if isinstance(b, Load) else True for b in b_edges}) != 1:
        return None
    return Fold.contraction(
        k_axis=f.axis,
        a=a_edge,
        channels=tuple(Channel(b=b, acc=acc) for b, acc in zip(b_edges, f.combine.results, strict=True)),
    )
