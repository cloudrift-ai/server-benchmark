"""The lowering-side view of a fold's algebra — every cross-partition / cross-thread program the
materializers derive from a :class:`~emmy.compiler.ir.tile.ir.Fold`'s stored ``(combine, lift,
dtypes)``. This is a LOWERING helper, not IR vocabulary: the stored term keeps exactly one ⊕
program (the flat ``combine``), and everything here — the state⊕state re-emission, the one-shot
:class:`~emmy.compiler.ir.stmt.algebra.StateMerge`, the per-component seeds, the twist facts —
is derived on demand at the two consumers, the kernel materializer (``lowering/kernel/_factor``
and friends) and the cross-CTA split (``lowering/tile/030_split_reduce``). Nothing else reads it.

Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.stmt import Accum, Assign, StateMerge, component_ops
from emmy.compiler.ir.stmt.carrier import exp_combine_states, exp_merge


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

    def __post_init__(self) -> None:
        from emmy.compiler.ir.tile.ir import Contraction  # noqa: PLC0415 — ir.tile imports nothing from here

        # A Contraction node answers through its λ-fold reading (``as_fold`` — the one
        # place the algebra machinery still derives the flat combine for a stored contraction).
        if isinstance(self.fold, Contraction):
            object.__setattr__(self, "fold", self.fold.as_fold())

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

    def state_merge(self, other: tuple[str, ...]) -> StateMerge:
        """A one-shot :class:`StateMerge` stmt folding this state with a second fully-reduced
        state named ``other`` (the cross-partition combine's right operand). The merge program IS
        :attr:`combine_states` with ``state_b`` renamed to ``other`` (a twisted program
        regenerates keyed on ``other[0]`` so distinct folds' temps never collide), so the
        cooperative-tree / cross-CTA reduce renders it through the same machinery as a streaming
        step."""
        if self.ops is None:
            merged: tuple = exp_combine_states(self.names, other, key=other[0])
        else:
            sub = dict(zip(self.state_b, other, strict=True))
            merged = tuple(
                Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype) for a in self.combine_states
            )
        return StateMerge(state=self.names, merge=merged, state_b=other)

    @classmethod
    def of_cone_stat(cls, cone) -> Reduction | None:
        """The :class:`Reduction` of a computed-A cone's per-row statistic — the fold at the head
        of the cone's prologue node (``Map(body=cell, sources=(prologue,))``; the prologue itself
        a ``Map`` over the stat ``Fold``, or the bare fold). ``None`` when the cone carries no
        prologue (a gmem-``Load`` A) — the caller's serial fallback."""
        from emmy.compiler.ir.tile.ir import Fold, Map  # noqa: PLC0415 — typing-only dependency

        pro = cone.sources[0] if isinstance(cone, Map) and cone.sources else None
        head = pro.sources[0] if isinstance(pro, Map) and pro.sources else pro
        return cls(head) if isinstance(head, Fold) else None

    def identities(self) -> dict[str, float]:
        """The per-state-component neutral element (a finalize seeds the state with it before
        folding partitions). A degenerate fold reads its channel ops' identities; a twisted fold
        the ``Accum`` folds of its generated streaming merge (``l``/``O`` → add → 0, ``m`` →
        maximum → −1e30 — the merge generator's spelling, NOT the stored ``init``'s −inf)."""
        if self.ops is not None:
            return {n: op.identity for n, op in zip(self.names, self.ops, strict=True)}
        return {s.name: s.op.identity for s in exp_merge(self.names, self.terms, key=self.names[0]) if isinstance(s, Accum)}
