"""Pure terms — the value vocabulary every IR level shares.

A pure term denotes a value. It binds names, it carries an algebra, it can be substituted and
compared up to α-renaming; it does not occupy a position in an instruction stream and it has no
scope of its own to seed or tear down. The statements live in :mod:`emmy.compiler.ir.stmt`; the
two vocabularies meet in exactly one direction — a term is RENDERED into statements at the point
of use (``StateMerge.stmts``), never spliced in as one.

- :mod:`.lam` — :class:`~emmy.compiler.ir.pure.lam.Lambda`, the ONE binder kind.
- :mod:`.algebra` — the TRUE monoid ``(init, combine)``: the free constructor ``M``, the
  DEGENERATE/TWISTED shape test, the rename lockstep, and the denotational foldMap spec oracle.
- :mod:`.carrier` — the exp/LSE-family combine GENERATORS (twisted monoid via ψ-conjugation) and
  the stability certificate.
- :mod:`.merge` — :class:`~emmy.compiler.ir.pure.merge.StateMerge`, the cross-partition
  state⊕state combine plus its one statement realization.

The invariant these modules exist to state lives in ``ir/ARCHITECTURE.md`` ("Pure terms vs statements").
"""

from emmy.compiler.ir.pure.algebra import (
    M,
    component_ops,
    degenerate,
    eval_lambda,
    foldmap_eval,
    rename_combine,
)
from emmy.compiler.ir.pure.carrier import UnstableCarrierError, exp_combine_states, exp_merge
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.pure.merge import StateMerge

__all__ = [
    "Lambda",
    "M",
    "StateMerge",
    "UnstableCarrierError",
    "component_ops",
    "degenerate",
    "eval_lambda",
    "exp_combine_states",
    "exp_merge",
    "foldmap_eval",
    "rename_combine",
]
