"""Hand-spelled tile terms for compiler tests.

The total lift (``lowering/tile/_fromloop``) forms every term the compiler schedules: a gmem read is a
:meth:`Fold.slab`, a reduce's product arguments arrive as operand edges, a projection is a zero-axis term
over its operands, and every extent lives in the kernel's axis table (``TileOp.axes``), never on the term.
A fixture that the lift cannot yet form from Loop IR — a contraction over a COMPUTED operand, a projection
over a hand-built reduce — is spelled here in that same vocabulary, so it reads exactly like a lifted term:
``operands[0]`` is A, a contraction's lift is nothing but its products, the accumulators are the combine's
results, and a term binds names only. Pass the reduce axis to ``TileOp(axes=...)`` beside the free axes.

A ``Load`` is accepted wherever an operand edge is expected and becomes its slab; there is no other
non-term input, so a reader of the built term never meets a statement in ``operands``.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.stmt import Assign, Body, Load


def term(edge: Fold | Load) -> Fold:
    """An operand edge as a term: a ``Load`` is its slab, a term is itself."""
    return edge if isinstance(edge, Fold) else Fold.slab(edge)


def slab(name: str, buffer: str, *index) -> Fold:
    """One gmem read ``buffer[index]`` as a term; a ``str`` coordinate spells the ``Var`` of that name."""
    return Fold.slab(Load(name=name, input=buffer, index=tuple(Var(i) if isinstance(i, str) else i for i in index)))


def _bound(operands: tuple[Fold, ...]) -> tuple[str, ...]:
    """The lift params the operands bind, positionally: one per operand result component."""
    return tuple(name for edge in operands for name in edge.exposes)


def _impl(op) -> ElementwiseImpl:
    return op if isinstance(op, ElementwiseImpl) else ElementwiseImpl(op)


def projection(operands: tuple = (), body=(), results: tuple[str, ...] | None = None) -> Fold:
    """A ZERO-AXIS term — the pointwise cell over ``operands``. ``body`` reads the operands' results by
    name, and every coordinate it indexes stays a free coordinate of the term; ``results`` default to the
    body's last definition, or with no body the first bound operand result (a pass-through)."""
    operands = tuple(term(edge) for edge in operands)
    body = Body.coerce(body)
    bound = _bound(operands)
    if results is None:
        last = next((stmt.defines()[-1] for stmt in reversed(body) if stmt.defines()), None)
        results = (last,) if last is not None else bound[:1]
    return Fold(operands=operands, lift=Lambda.closing(bound, body, tuple(results)))


def reduction(axis: Axis | str, operands: tuple, body, accs: tuple[str, ...], ops="add") -> Fold:
    """A reducing term over ``axis``: ``body`` is the per-step program over the operands' results, defining
    ``<acc>__v`` for every accumulator in ``accs``, folded through the componentwise ``ops`` (one op for
    all, or one per accumulator)."""
    operands = tuple(term(edge) for edge in operands)
    name = axis.name if isinstance(axis, Axis) else axis
    ops = tuple(_impl(op) for op in (ops if isinstance(ops, tuple) else (ops,) * len(accs)))
    lift = Lambda.closing((name, *_bound(operands)), body, tuple(f"{acc}__v" for acc in accs))
    return Fold(operands=operands, lift=lift, init=tuple(op.identity for op in ops), combine=Lambda.componentwise(ops, accs))


def contraction(axis: Axis | str, a: Fold | Load, *channels: tuple[Fold | Load, str], product="multiply", plus="add") -> Fold:
    """A BILINEAR term over ``axis`` — the matmul cell at the ``(multiply, add)`` semiring: one shared
    operand ``a`` and one ``(b, acc)`` channel per product, ``acc = ⊕_axis a ⊗ b``. Two channels over one
    ``a`` is the fused sibling edge (gate⊗up); a ``b`` reused by several channels occupies one operand
    slot. Formation orients the pair itself, so ``operands[0]`` is whichever edge it reads as A."""
    edges: dict[int, Fold] = {}  # one slab per Load object, however many channels spell it
    a, pairs = edges.setdefault(id(a), term(a)), tuple((edges.setdefault(id(b), term(b)), acc) for b, acc in channels)
    operands: list[Fold] = [a]
    for b, _ in pairs:
        if all(b is not edge for edge in operands):
            operands.append(b)
    products = tuple(Assign(name=f"{acc}__v", op=_impl(product), args=(a.exposes[0], b.exposes[0])) for b, acc in pairs)
    return reduction(axis, tuple(operands), products, tuple(acc for _, acc in pairs), plus)
