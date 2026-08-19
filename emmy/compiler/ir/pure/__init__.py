"""Pure terms — the value vocabulary every IR level shares.

A pure term denotes a value. It binds names, it carries an algebra, it can be substituted and
compared up to α-renaming; it does not occupy a position in an instruction stream and it has no
scope of its own to seed or tear down. The statements live in :mod:`emmy.compiler.ir.stmt`; the
two vocabularies meet in exactly one direction — a term is RENDERED into statements at the point
of use (``algebra.merge_stmts``), never spliced in as one.

- :mod:`.lam` — :class:`~emmy.compiler.ir.pure.lam.Lambda`, the ONE binder kind.
- :mod:`.fold` — :class:`~emmy.compiler.ir.pure.fold.Fold`, the ONE reduce term (what Tile IR
  stores) plus its derived readings and its render-to-statements (``lower`` / ``loop``).
- :mod:`.algebra` — the TRUE monoid ``(init, combine)``: the free constructor ``M``, the
  DEGENERATE/TWISTED shape test, the rename lockstep, ``merge_stmts`` (the cross-partition
  state⊕state combine's ONE statement realization) and the denotational foldMap spec oracle.
- :mod:`.carrier` — the exp/LSE-family combine GENERATORS (twisted monoid via ψ-conjugation) and
  the stability certificate.

The invariant these modules exist to state lives in ``ir/ARCHITECTURE.md`` ("Pure terms vs statements").
"""

from emmy.compiler.ir.pure.algebra import (
    M,
    component_ops,
    degenerate,
    eval_lambda,
    foldmap_eval,
    merge_stmts,
    rename_combine,
)
from emmy.compiler.ir.pure.carrier import UnstableCarrierError, exp_combine_states, exp_merge
from emmy.compiler.ir.pure.fold import (
    Channel,
    Fold,
    deep_defines,
    deep_reads,
    edge_refs_axis,
    is_contraction,
    operand_body,
    operand_name,
    refs_axis,
    splice_operands,
    stmt_axis_names,
)
from emmy.compiler.ir.pure.lam import Lambda

__all__ = [
    "Channel",
    "Fold",
    "Lambda",
    "M",
    "UnstableCarrierError",
    "component_ops",
    "deep_defines",
    "deep_reads",
    "edge_refs_axis",
    "degenerate",
    "eval_lambda",
    "exp_combine_states",
    "exp_merge",
    "foldmap_eval",
    "is_contraction",
    "merge_stmts",
    "operand_body",
    "operand_name",
    "refs_axis",
    "splice_operands",
    "stmt_axis_names",
    "rename_combine",
]
