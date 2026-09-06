"""Twist recipes — the streaming carriers a two-pass reduce pair fuses into.

A *dependent* reduce ``F`` reads an earlier reduce ``G`` over the same stream: ``m = max_k s_k``
then ``l = Σ_k exp(s_k − m)``. After the lift that dependency is in the tree — ``G`` is an operand
of ``F`` — and a recipe is what says the pair fuses into ONE fold with state ``(m, l)``.

A recipe is a TWISTED MONOID stated as its mathematics — transport of structure: a componentwise
monoid ``base`` (one ⊕ per state, with ``lift`` the per-element contribution to each) conjugated
by a bijection ``psi`` on the carrier, ``x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))``, associative because the base
is. Online softmax is ``(max, Σeˢ, Σeˢv)`` under ``ψ(m, D, O) = (m, De⁻ᵐ, Oe⁻ᵐ)``; Welford's variance
is ``(Σx, Σ1, Σx, Σx²)`` under ``ψ(S, n, T, W) = (S, n, T/n, W − T²/n)``. What the tree is matched
on and what the kernel runs are DATA beside that definition, because stability is not preserved by
conjugation: the ``channels`` say which per-element maps a dependent's lift must spell (over ROLES,
never over any term's names) and what each state is at the singleton, and the fused ⊕ is spelled
in its numerically stable form (softmax's as a pivot advance and a per-channel rescale, so one
recipe serves any channel count; Welford's as one lambda). The definition certifies the data: the
program is the conjugate of the base on every state pair, the seeds are the base identities under
ψ⁻¹, the injections are the lift seen through ψ (``tests/compiler/ir/pure/test_twist.py``).

The one generic algorithm that applies any recipe is :meth:`Fold.fuse`, which finds the pivot
among ``F``'s own operands: matching is alpha-invariant by construction — ``F``'s lift binds its
operands positionally, so the pivot's state is the param bound to ``G``; the score is whatever
sub-cone of ``F``'s lift is alpha-equal to ``G``'s own per-element map, operand for operand; and
what remains of ``F``'s lift with that cone cut out, its params in role order, compares with a
channel's ``pattern`` by canonical form. A click gives the role-to-name map, and the recipe is
instantiated by renaming — no algebra engine, no op-name table: either a recipe clicks or it does
not.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt import Assign, Body, Const, Stmt


@dataclass(frozen=True)
class Channel:
    """One state of the fused carrier beyond the pivot's.

    ``injection`` is what the state is at the injected singleton, over ``(score, *extras)`` — the
    pivot IS the score there, and the recipe's author has already simplified the map at that point
    (``exp(s − s)`` is ``1``). With a ``pattern``, the state is a dependent fold's own, fused: the
    per-element map that fold's lift must spell, over the roles ``(score, pivot, *extras)`` — the
    extras operand-bound values, a streamed row or a scalar read from a buffer (Welford's ``1/N``)
    alike. Without one, the state is one the two-pass form never had (Welford's count and running
    mean): ``name`` suffixes the fused channel's state name, ``init`` seeds it."""

    injection: Lambda
    pattern: Lambda | None = None
    name: str = ""
    init: float = 0.0


@dataclass(frozen=True)
class Twist:
    """One :class:`Recipe` INSTANTIATED on a term — the schema, plus what this term calls the
    recipe's roles and which channel each of its carried states is.

    A recipe states its algebra over ROLE names: ``lift``'s params are ``(score, *extras)``, and
    every channel's ``pattern`` and ``injection`` bind those same roles. ``roles`` pairs each role
    the term bound with the term's own name for it, and ``channels`` says which recipe channel each
    carried state past the pivot is — the carrier grows one state per fusion, so its order is the
    order the tree was fused in and not the recipe's.

    Together they are what lets a :class:`~emmy.compiler.ir.pure.fold.Fold` DERIVE the two halves it
    does not store: the stable ⊕ (:meth:`program`) and the ψ-image of its own lift (:meth:`inject`).

    EMPTY ``roles`` says the term's elements are already carrier states, so ψ has nothing to do:
    a cross-CTA split's partial merge (:func:`_state_fold`) folds finished partials read out of a
    workspace, and it names the recipe only to reach the same ⊕ they were produced under.
    """

    recipe: Recipe
    roles: tuple[tuple[str, str], ...]
    channels: tuple[int, ...]

    @classmethod
    def merging(cls, twist: Twist) -> Twist:
        """The same ⊕ over values that are already carrier states — no per-element roles to bind."""
        return cls(recipe=twist.recipe, roles=(), channels=())

    @property
    def name(self) -> str:
        return self.recipe.name

    def program(self, states: tuple[str, ...]) -> Lambda:
        """``combine`` over these state names — ``psi(psi_inv(x) base psi_inv(y))``, in the stable
        spelling the recipe authored (:meth:`Recipe.program`)."""
        return self.recipe.program(states)

    def inject(self, roles: tuple[tuple[str, str], ...], states: tuple[str, ...]) -> Lambda:
        """``psi ∘ lift`` at the singleton — the score, then one channel injection per carried state
        past the pivot — over ``roles``, this term's spelling of the recipe's.

        Written out by the recipe's AUTHORED injections rather than by evaluating ``psi`` on the
        base contribution: at the singleton the pivot IS the score, so ``psi`` divides each channel
        by a factor the contribution already carries (softmax's ``exp(s)·v ↦ v``), and evaluating
        the two apart denotes ``exp(s)`` and overflows. That simplification is exactly what a
        channel's ``injection`` states, so lowering reads it instead of computing it.
        """
        spelled = dict(roles)
        body: list[Stmt] = []
        results = [spelled[self.recipe.lift.params[0]]]
        for state, index in zip(states[1:], self.channels, strict=True):
            injection = self.recipe.channels[index].injection
            names = {param: spelled[param] for param in injection.params}
            names.update((stmt.name, f"{state}__{stmt.name}") for stmt in injection.body)
            instance = injection.rename(names)
            body.extend(instance.body)
            results.extend(instance.results)
        return Lambda(params=tuple(name for _, name in roles), body=Body(tuple(body)), results=tuple(results))


@dataclass(frozen=True)
class Recipe:
    """A twisted monoid as data. ``base`` names the componentwise ⊕ of every carrier state — the
    pivot's first, then one per channel in recipe order — and ``lift`` is the base's per-element
    contribution over ``(score, *extras)``; ``psi`` / ``psi_inv`` conjugate it onto the carrier.
    The ``channels`` are every state beyond the pivot's, matched to a dependent fold or kept by the
    recipe, and the fused ⊕ program takes one of two spellings. ``advance`` / ``rescale`` serve any
    channel count: ``advance`` takes the pivot pair ``(g, g′)`` to the advanced pivot and the
    factors the move puts on every carried channel, ``rescale`` takes one channel pair and those
    factors ``(s, s′, *factors)`` to the channel's merged value. ``combine`` is one lambda over
    every state pair in role order — pivot, then the channels, then the same with ``__o`` — for a
    carrier of fixed arity. Applied by the one generic algorithm, :meth:`Fold.fuse`."""

    name: str
    base: tuple[str, ...]
    lift: Lambda
    psi: Lambda
    psi_inv: Lambda
    channels: tuple[Channel, ...]
    advance: Lambda | None = None
    rescale: Lambda | None = None
    combine: Lambda | None = None

    @property
    def pivot(self) -> str:
        """The pivot's ⊕ — the base monoid's first component."""
        return self.base[0]

    def program(self, states: tuple[str, ...]) -> Lambda:
        """The fused ⊕ over these state names — ``S × S → S``, the second operand ``<n>__o``. Temps
        are namespaced on the second pivot's name, so two merges into one state never collide. The
        advance/rescale spelling: the advance over the pivot pair, every channel rescaled by its
        factors, the pivot written last (the channels read the old pivot through the factors)."""
        other = tuple(f"{name}__o" for name in states)
        key = other[0]
        if self.combine is not None:
            roles = self.combine.results
            names = dict(zip((*roles, *(f"{role}__o" for role in roles)), (*states, *other), strict=True))
            return self.combine.rename(lambda name: names.get(name, f"{key}__{name}"))
        roles = dict(zip(self.advance.params, (states[0], key), strict=True))
        advance = self.advance.rename(lambda name: roles.get(name, f"{key}__{name}"))
        pivot, *factors = advance.results
        body = list(advance.body)
        for state, second in zip(states[1:], other[1:], strict=True):
            names = dict(zip(self.rescale.params, (state, second, *factors), strict=True))
            names[self.rescale.results[0]] = state
            body.extend(self.rescale.rename(lambda name, names=names, state=state: names.get(name, f"{key}__{state}_{name}")).body)
        body.append(Assign(name=states[0], op="copy", args=(pivot,)))
        return Lambda(params=states + other, body=Body(body), results=states)


def _lam(params: tuple[str, ...], body: tuple[Stmt, ...], *results: str) -> Lambda:
    return Lambda(params=params, body=Body(body), results=results)


def _const(name: str, value: float) -> Lambda:
    return _lam(("s",), (Const(name=name, value=value),), name)


SOFTMAX = Recipe(
    name="softmax",
    base=("maximum", "add", "add"),
    lift=_lam(("s", "v"), (Assign("e", "exp", ("s",)), Assign("ev", "multiply", ("e", "v"))), "s", "e", "ev"),
    psi=_lam(
        ("m", "D", "O"),
        (
            Assign("nm", "negative", ("m",)),
            Assign("f", "exp", ("nm",)),
            Assign("d", "multiply", ("D", "f")),
            Assign("o", "multiply", ("O", "f")),
        ),
        "m",
        "d",
        "o",
    ),
    psi_inv=_lam(
        ("m", "d", "o"),
        (Assign("f", "exp", ("m",)), Assign("D", "multiply", ("d", "f")), Assign("O", "multiply", ("o", "f"))),
        "m",
        "D",
        "O",
    ),
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
    # The pivot advances to the larger of the pair, and each side's factor is ``exp`` of its distance
    # below the new pivot — never positive, so the program cannot overflow.
    advance=_lam(
        ("g", "g_o"),
        (
            Assign("gn", "maximum", ("g", "g_o")),
            Assign("dg", "subtract", ("g", "gn")),
            Assign("alpha", "exp", ("dg",)),
            Assign("dg_o", "subtract", ("g_o", "gn")),
            Assign("beta", "exp", ("dg_o",)),
        ),
        "gn",
        "alpha",
        "beta",
    ),
    # A channel merges as the factor-weighted sum of its two sides.
    rescale=_lam(
        ("s", "s_o", "alpha", "beta"),
        (Assign("sa", "multiply", ("s", "alpha")), Assign("sb", "multiply", ("s_o", "beta")), Assign("sn", "add", ("sa", "sb"))),
        "sn",
    ),
)

# Welford's variance: ``mean = Σ_k x_k / N`` then ``Σ_k (x_k − mean)²``, fused into the carrier
# ``(sum, count, mean, M2)`` — the running mean beside the sum so the seed ``(0, 0, 0, 0)`` is a true
# identity (a mean of nothing is any value, a sum over nothing divided by its count is not), the sum
# kept so the pivot's readers keep their value. Chan's merge: ``δ = mean′ − mean``, ``mean +=
# δ·n′/n``, ``M2 += M2′ + δ²·n·n′/(n + n′)``.
WELFORD = Recipe(
    name="welford",
    base=("add", "add", "add", "add"),
    lift=_lam(("s", "c"), (Const(name="one", value=1.0), Assign("sq", "multiply", ("s", "s"))), "s", "one", "s", "sq"),
    psi=_lam(
        ("S", "n", "T", "W"),
        (Assign("mu", "divide", ("T", "n")), Assign("Tmu", "multiply", ("T", "mu")), Assign("M2", "subtract", ("W", "Tmu"))),
        "S",
        "n",
        "mu",
        "M2",
    ),
    psi_inv=_lam(
        ("S", "n", "mu", "M2"),
        (Assign("T", "multiply", ("n", "mu")), Assign("Tmu", "multiply", ("T", "mu")), Assign("W", "add", ("M2", "Tmu"))),
        "S",
        "n",
        "T",
        "W",
    ),
    channels=(
        # The count and the running mean: states the two-pass form never had, one element counting
        # one and averaging to itself.
        Channel(injection=_const("one", 1.0), name="n"),
        Channel(injection=_lam(("s",), (), "s"), name="mean"),
        # The squared deviation from the mean, the mean being the pivot scaled by ``1/N`` — a scalar
        # the tree reads like any operand; a single element deviates from its own mean by nothing.
        Channel(
            pattern=_lam(
                ("s", "g", "c"),
                (Assign("m", "multiply", ("g", "c")), Assign("d", "subtract", ("s", "m")), Assign("sq", "multiply", ("d", "d"))),
                "sq",
            ),
            injection=_const("zero", 0.0),
        ),
    ),
    combine=_lam(
        ("g", "n", "m", "q", "g__o", "n__o", "m__o", "q__o"),
        (
            Assign("tn", "add", ("n", "n__o")),
            Assign("d", "subtract", ("m__o", "m")),
            Assign("w", "divide", ("n__o", "tn")),
            Assign("dw", "multiply", ("d", "w")),
            Assign("nn", "multiply", ("n", "n__o")),
            Assign("r", "divide", ("nn", "tn")),
            Assign("dd", "multiply", ("d", "d")),
            Assign("corr", "multiply", ("dd", "r")),
            Assign("q1", "add", ("q", "q__o")),
            Assign("g", "add", ("g", "g__o")),
            Assign("m", "add", ("m", "dw")),
            Assign("q", "add", ("q1", "corr")),
            Assign("n", "copy", ("tn",)),
        ),
        "g",
        "n",
        "m",
        "q",
    ),
)

RECIPES = (SOFTMAX, WELFORD)

__all__ = ["RECIPES", "SOFTMAX", "WELFORD", "Channel", "Recipe", "Twist"]
