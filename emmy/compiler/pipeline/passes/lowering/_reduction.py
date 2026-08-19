"""The lowering-side view of a fold's algebra — every cross-partition / cross-thread program the
materializers derive from a :class:`~emmy.compiler.ir.pure.fold.Fold`'s stored ``(combine, lift,
dtypes)``. This is a LOWERING helper, not IR vocabulary: the stored term keeps exactly one ⊕
program (the flat ``combine``), and everything here — the state⊕state re-emission, the one-shot
statement realization of the cross-partition combine, the twist facts —
is derived on demand at the two consumers, the kernel materializer (``lowering/kernel/_factor``
and friends) and the cross-CTA split (``lowering/tile/030_split_reduce``). Nothing else reads it.

Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.pure import component_ops, merge_stmts
from emmy.compiler.ir.stmt import Accum, Assign, Stmt


def loop_state_head(loop) -> str:
    """The carried state's primary component of an annotated reduce loop, read off its own body:
    the first ``Accum`` for a plain (componentwise) fold, the ``maximum``-fold pivot for a
    dissolved exp-family merge (whose ψ-rescale ``base``-``Accum``\\ s precede it in body order) —
    the same head the fold's ``combine.results[0]`` names."""
    accums = [s for s in loop.body if isinstance(s, Accum)]
    if not accums:
        return ""
    if any(a.base is not None for a in accums):
        mx = next((a for a in accums if a.op.reduce_canon == "maximum"), None)
        if mx is not None:
            return mx.name
    return accums[0].name


@dataclass(frozen=True)
class Reduction:
    """One fold node's algebra, read for lowering: wraps the ``Fold`` and answers the questions
    the partition machinery asks — the carried state ``names``, the second-operand ``state_b``
    spelling, the DEGENERATE/TWISTED discrimination (:attr:`ops` — the shape test on the stored
    combine, no family annotation), the cross-partition programs and the finalize seeds. All
    reads are off the node's stored params; nothing is stored back."""

    fold: object  # the Fold node (typed loosely — ir.tile imports nothing from here)

    @property
    def names(self) -> tuple[str, ...]:
        """The carried state's SSA names — the stored combine's results."""
        return tuple(self.fold.combine.results)

    @property
    def state_b(self) -> tuple[str, ...]:
        """The second-operand state names of the cross-partition combine — the combine's second
        param half (the ``"<n>__o"`` spelling both constructors use)."""
        return tuple(self.fold.combine.params[len(self.fold.combine.results) :])

    @property
    def terms(self) -> tuple:
        """The per-element injection singleton — the lift's results (a folded SSA name, or a
        float literal: softmax's denominator ``1.0``)."""
        return tuple(self.fold.lift.results)

    @cached_property
    def ops(self):
        """The per-component ⊕ handles of a DEGENERATE combine (:func:`component_ops`), or
        ``None`` for a twisted (exp-family) one — the family discriminator."""
        return component_ops(self.fold.combine)

    @property
    def twisted(self) -> bool:
        return self.ops is None

    @cached_property
    def combine_states(self) -> tuple[Assign, ...]:
        """The cross-partition state⊕state fold program — the stored combine's body, re-emitted
        for a degenerate ⊕ (whose stored program is dtype-free, like the fold itself)."""
        if self.ops is None:
            return tuple(self.fold.combine.body)
        return tuple(Assign(name=n, op=op, args=(n, o)) for n, op, o in zip(self.names, self.ops, self.state_b, strict=True))

    def merge_stmts(self, other: tuple[str, ...]) -> tuple[Stmt, ...]:
        """This fold's cross-partition combine against a second fully-reduced state named
        ``other``, as loop-IR statements (:func:`~emmy.compiler.ir.pure.algebra.merge_stmts` — the
        stored ``combine`` rendered to ``Assign`` temps plus one ``Accum`` per component, whose
        ``op.identity`` is the seed the ONE identity placement emits)."""
        return merge_stmts(self.fold.combine, other)

    @classmethod
    def of_cone_stat(cls, cone) -> Reduction | None:
        """The :class:`Reduction` of a computed-A cone's per-row statistic — the fold at the head
        of the cone's prologue node (``Fold.projection(body=cell, operands=(prologue,))``; the prologue itself
        a zero-axis ``Fold`` over the stat ``Fold``, or the bare fold). ``None`` when the cone carries no
        prologue (a gmem-``Load`` A) — the caller's serial fallback."""
        from emmy.compiler.ir.pure.fold import Fold  # noqa: PLC0415 — avoid an import cycle

        pro = cone.operands[0] if (isinstance(cone, Fold) and cone.axis is None) and cone.operands else None
        head = pro.operands[0] if (isinstance(pro, Fold) and pro.axis is None) and pro.operands else pro
        return cls(head) if isinstance(head, Fold) else None
