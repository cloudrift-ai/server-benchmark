"""The twist recipes, certified by their definition. A recipe IS a base componentwise monoid
conjugated by a bijection ψ (transport of structure); what it stores beside that — the stable ⊕
program, the channels' injections, the seeds — is checked against the conjugate here, on random
states, so associativity, the identity and the meaning of every state follow from the definition
rather than from a property test per recipe."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.pure.twist import RECIPES, SOFTMAX, WELFORD, Recipe
from emmy.compiler.ir.stmt import Const


def _eval(lam, args: tuple) -> tuple:
    """Evaluate a pure lambda on floats — the ANF chain of ``Assign``s and ``Const`` defs."""
    env = dict(zip(lam.params, args, strict=True))
    for stmt in lam.body:
        env[stmt.name] = stmt.value if isinstance(stmt, Const) else stmt.op(*(env[arg] for arg in stmt.args))
    return tuple(env[result] for result in lam.results)


def _sample(recipe: Recipe, rng, arity: int) -> tuple:
    """A random carrier state a partition could hold: Welford's needs a count of at least one."""
    state = list(rng.normal(size=arity))
    if recipe is WELFORD:
        state[1] = float(rng.integers(1, 6))
    return tuple(state)


def _carriers(recipe: Recipe):
    """Every arity the recipe's carrier takes — softmax with and without its expectation channel."""
    n = len(recipe.base)
    return [n] if recipe.combine is not None else list(range(2, n + 1))


@pytest.mark.parametrize("recipe", RECIPES, ids=[recipe.name for recipe in RECIPES])
def test_the_program_is_the_conjugate_of_its_base_monoid(recipe: Recipe) -> None:
    """``program(s, t) == ψ(ψ⁻¹(s) ⊕ ψ⁻¹(t))`` with ⊕ the base's componentwise monoid — the stable
    spelling is the transported monoid, so it is associative and its seeds are the base identities
    under ψ⁻¹. Checked at every arity the carrier takes; an absent channel rides at its identity."""
    ops = [ElementwiseImpl(name) for name in recipe.base]
    identities = tuple(op.identity for op in ops)
    rng = np.random.default_rng(0)
    for arity in _carriers(recipe):
        states = tuple(f"s{i}" for i in range(arity))
        program = recipe.program(states)
        assert program.params == (*states, *(f"{s}__o" for s in states)) and program.results == states
        seeds = (identities[0], *(channel.init for channel in recipe.channels[: arity - 1]))
        np.testing.assert_allclose(_eval(recipe.psi_inv, (*seeds, *identities[arity:]))[:arity], identities[:arity])
        for _ in range(50):
            s, t = _sample(recipe, rng, arity), _sample(recipe, rng, arity)
            base_s = _eval(recipe.psi_inv, (*s, *identities[arity:]))
            base_t = _eval(recipe.psi_inv, (*t, *identities[arity:]))
            merged = tuple(op(a, b) for op, a, b in zip(ops, base_s, base_t, strict=True))
            expected = _eval(recipe.psi, merged)[:arity]
            np.testing.assert_allclose(_eval(program, (*s, *t)), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("recipe", RECIPES, ids=[recipe.name for recipe in RECIPES])
def test_the_injections_are_the_lift_seen_through_psi(recipe: Recipe) -> None:
    """What a channel injects at the singleton is ψ of one element's base contribution — softmax's
    denominator injects ``1`` because ``eˢ·e⁻ˢ`` is ``1``, Welford's M2 injects ``0`` because one
    element deviates from its own mean by nothing."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        values = {param: float(rng.normal()) for param in recipe.lift.params}
        through = _eval(recipe.psi, _eval(recipe.lift, tuple(values[p] for p in recipe.lift.params)))
        for index, channel in enumerate(recipe.channels):
            injected = _eval(channel.injection, tuple(values[p] for p in channel.injection.params))
            np.testing.assert_allclose(injected, (through[1 + index],), rtol=1e-9, atol=1e-9)


def test_an_expectation_channel_joins_the_same_advance() -> None:
    """Online softmax and flash attention are ONE recipe: the three-state program is the two-state
    one plus a channel rescaled by the same two factors — no second advance, no second ``exp``."""
    softmax, flash = SOFTMAX.program(("m", "l")), SOFTMAX.program(("m", "o", "l"))
    assert set(softmax.body) < set(flash.body)
    assert sum(stmt.op.name == "exp" for stmt in flash.body) == 2
