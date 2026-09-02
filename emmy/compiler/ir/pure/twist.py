"""Twist recipes — the streaming carriers a two-pass reduce pair fuses into.

A *dependent* reduce ``F`` reads an earlier reduce ``G`` over the same stream: ``m = max_k s_k``
then ``l = Σ_k exp(s_k − m)``. After the lift that dependency is in the tree — ``G`` is an operand
of ``F`` — and a recipe is what says the pair fuses into ONE fold with state ``(m, l)``: which ⊕
the pivot folds, which per-element maps its channels recognize (over ROLES, never over any term's
names), what each channel injects at the singleton, and the fused ⊕ program (transport of
structure — the base componentwise monoid conjugated by the family's ψ; for the exp family the
generated, stabilized program of :mod:`~emmy.compiler.ir.pure.carrier`).

A recipe is DATA. The one generic algorithm that applies any of them is :meth:`Fold.twist`,
which finds the pivot among ``F``'s own operands: matching is alpha-invariant by construction —
``F``'s lift binds its operands positionally, so the pivot's state is the param bound to ``G``;
the score is whatever sub-cone of ``F``'s lift is
alpha-equal to ``G``'s own per-element map, operand for operand; and what remains of ``F``'s lift
with that cone cut out, its params in role order, compares with a channel's ``pattern`` by
canonical form. A click gives the role-to-name map, and the recipe is instantiated by renaming —
no algebra engine, no op-name table: either a recipe clicks or it does not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt import Assign, Body, Const, Stmt


@dataclass(frozen=True)
class Channel:
    """One dependent channel of a recipe.

    ``pattern`` is the per-element map a dependent fold's lift must spell, over the roles
    ``(score, pivot, *extras)``; ``injection`` is what the channel contributes at the injected
    singleton, over ``(score, *extras)`` — the pivot IS the score there, and the recipe's author has
    already simplified the pattern at that point (``exp(s − s)`` is ``1``)."""

    pattern: Lambda
    injection: Lambda


@dataclass(frozen=True)
class Recipe:
    """A twisted monoid as data: the pivot's ⊕ (``reduce_canon`` name), the channels' ⊕, the
    channel patterns, and the fused ⊕ program builder over ``(states, other)``. Applied by the one
    generic algorithm, :meth:`Fold.twist`."""

    name: str
    pivot: str
    plus: str
    channels: tuple[Channel, ...]
    combine: Callable[[tuple[str, ...], tuple[str, ...]], tuple[Stmt, ...]]

    def program(self, states: tuple[str, ...]) -> Lambda:
        """The fused ⊕ over these state names — ``S × S → S``, the second operand ``<n>__o``."""
        other = tuple(f"{name}__o" for name in states)
        return Lambda(params=states + other, body=Body(self.combine(states, other)), results=states)


def _lam(params: tuple[str, ...], body: tuple[Stmt, ...], result: str) -> Lambda:
    return Lambda(params=params, body=Body(body), results=(result,))


SOFTMAX = Recipe(
    name="softmax",
    pivot="maximum",
    plus="add",
    channels=(
        # The denominator: one weight per element, ``exp(s − m)``; at the singleton the pivot is
        # the score, so the channel injects ``1``.
        Channel(
            pattern=_lam(("s", "g"), (Assign("d", "subtract", ("s", "g")), Assign("w", "exp", ("d",))), "w"),
            injection=_lam(("s",), (Const(name="one", value=1.0),), "one"),
        ),
        # An expectation: the weight times a streamed value; it injects the value itself.
        Channel(
            pattern=_lam(
                ("s", "g", "v"), (Assign("d", "subtract", ("s", "g")), Assign("w", "exp", ("d",)), Assign("p", "multiply", ("w", "v"))), "p"
            ),
            injection=_lam(("s", "v"), (), "v"),
        ),
    ),
    combine=exp_combine_states,
)

RECIPES = (SOFTMAX,)

__all__ = ["RECIPES", "SOFTMAX", "Channel", "Recipe"]
