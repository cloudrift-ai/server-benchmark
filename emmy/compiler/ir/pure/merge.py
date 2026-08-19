"""``StateMerge`` — the cross-partition state⊕state combine, as a PURE term.

A partitioned reduce (the REG-tier ILP copies, the cooperative tree, the cross-CTA finalize) ends
holding several fully-reduced states that have to be folded into one. That fold is the fold's own
⊕ applied at ``S × S → S``: exactly the :class:`~emmy.compiler.ir.pure.lam.Lambda` a
:class:`Fold` already stores as its ``combine``, with the second operand renamed to the partial
being merged in. This module names that term and gives it its ONE statement realization.

It is a TERM, never a statement: :meth:`StateMerge.stmts` renders it into loop-IR stmts at the
point of use — the ``Assign`` temps unchanged, the per-component final writes as
``base``-``Accum``. Rendering to ``Accum`` is what makes the seed fall out for free: the identity
placement (``Loop.render`` / ``StridedLoop.render``) reads ``op.identity`` off an ``Accum`` and
declares the state there, so no neutral element has to travel on the term.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.pure.algebra import component_ops
from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign


@dataclass(frozen=True)
class StateMerge:
    """The state⊕state combine of a partitioned reduce — a ``combine`` :class:`Lambda` at
    ``S × S → S`` whose right operand names a second FULLY-REDUCED state (a REG copy, a
    tree-neighbour's partial, a workspace slice), plus the formation invariant that shape implies.

    Construction validates the S × S → S formation: the params are the merged state followed by
    the second operand, and the results ARE the merged state (the reassignment shape every
    combine generator emits). That check is the whole reason this is a named type rather than a
    bare ``Lambda`` — it is the invariant every consumer relies on when it reads
    :attr:`state` / :attr:`state_b` back off the params."""

    combine: Lambda

    def __post_init__(self) -> None:
        n = len(self.combine.results)
        if len(self.combine.params) != 2 * n:
            raise ValueError(f"StateMerge combine must be S × S → S: params={self.combine.params} results={self.combine.results}")
        if tuple(self.combine.params[:n]) != tuple(self.combine.results):
            raise ValueError(f"StateMerge combine must reassign its state: params={self.combine.params} results={self.combine.results}")

    @classmethod
    def of(cls, combine: Lambda, other: tuple[str, ...]) -> StateMerge:
        """This fold's stored ``combine`` with its second operand renamed to ``other`` — the
        partial actually being merged (a REG copy ``<n>__r1``, a workspace slice ``<n>__p``).

        A DEGENERATE program is substituted; a TWISTED one is REGENERATED keyed on ``other[0]``,
        because its temps are namespaced on the second operand's spelling and two folds merging
        different partials must not share them. Same rule, same reason as
        :func:`~emmy.compiler.ir.pure.algebra.rename_combine`."""
        names = tuple(combine.results)
        if component_ops(combine) is None:
            body = Body(exp_combine_states(names, other, key=other[0]))
        else:
            sub = dict(zip(combine.params[len(names) :], other, strict=True))
            body = Body(tuple(Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype) for a in combine.body))
        return cls(Lambda(params=names + tuple(other), body=body, results=names))

    @property
    def state(self) -> tuple[str, ...]:
        """The merged state's SSA names — the combine's results, reassigned in place."""
        return tuple(self.combine.results)  # type: ignore[arg-type]

    @property
    def state_b(self) -> tuple[str, ...]:
        """The second operand — the fully-reduced partial this term folds into :attr:`state`."""
        return tuple(self.combine.params[len(self.combine.results) :])

    def stmts(self, *, dtype: DataType = F32) -> tuple[Stmt, ...]:
        """The loop-IR realization — the ONE place a pure combine becomes statements.

        Both families land on the same shape: ``Assign`` temps (the ψ rescales, unchanged from
        the stored program) followed by one ``base``-``Accum`` per state component, which renders
        the in-place reassignment ``s = ⊕(base, value)`` AND carries the neutral element the
        identity placement seeds with. A DEGENERATE ⊕ needs no temps — one self-``Accum`` per
        component. A TWISTED one is REGENERATED in ``Accum`` form (the same regeneration rule
        :func:`~emmy.compiler.ir.pure.algebra.rename_combine` applies: the generated program is
        the deterministic function of its state names, so regenerating and patching agree)."""
        ops = component_ops(self.combine)
        if ops is not None:
            return tuple(Accum(name=s, value=o, op=op, dtype=dtype) for s, op, o in zip(self.state, ops, self.state_b, strict=True))
        return exp_combine_states(self.state, self.state_b, key=self.state_b[0], accum=True)

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}({', '.join(self.state)}) <- combine_states({', '.join(self.state_b)})"
        return [head, *(line for s in self.stmts() for line in s.pretty(indent + "    "))]
