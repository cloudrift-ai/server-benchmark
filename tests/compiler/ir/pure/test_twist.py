"""The twist recipes' fused ⊕ (``Recipe.program``): the program a recipe instantiates over a
fold's state names is a monoid — associative on random states, its seeds neutral — and one recipe
serves every channel count, an expectation channel joining the same pivot advance."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.pure.twist import SOFTMAX
from emmy.compiler.ir.stmt import Const


def _eval(lam, args: tuple) -> tuple:
    """Evaluate a pure lambda on floats — the ANF chain of ``Assign``s and ``Const`` defs."""
    env = dict(zip(lam.params, args, strict=True))
    for stmt in lam.body:
        env[stmt.name] = stmt.value if isinstance(stmt, Const) else stmt.op(*(env[arg] for arg in stmt.args))
    return tuple(env[result] for result in lam.results)


@pytest.mark.parametrize("states", [("m", "l"), ("m", "o", "l")], ids=["softmax", "flash"])
def test_the_program_is_an_associative_monoid_with_neutral_seeds(states: tuple[str, ...]) -> None:
    """The split / cooperative legality certificate: ``a ⊕ (b ⊕ c) == (a ⊕ b) ⊕ c`` on random
    states, and the fold's seeds ``(−inf, 0, …)`` are its neutral element on either side."""
    combine = SOFTMAX.program(states)
    assert combine.params == (*states, *(f"{state}__o" for state in states)) and combine.results == states
    seeds = (float("-inf"), *(0.0,) * (len(states) - 1))
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = (tuple(rng.normal(size=len(states))) for _ in range(3))
        lhs = _eval(combine, (*a, *_eval(combine, (*b, *c))))
        rhs = _eval(combine, (*_eval(combine, (*a, *b)), *c))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(_eval(combine, (*seeds, *a)), a)
        np.testing.assert_allclose(_eval(combine, (*a, *seeds)), a)


def test_an_expectation_channel_joins_the_same_advance() -> None:
    """Online softmax and flash attention are ONE recipe: the three-state program is the two-state
    one plus a channel rescaled by the same two factors — no second advance, no second ``exp``."""
    softmax, flash = SOFTMAX.program(("m", "l")), SOFTMAX.program(("m", "o", "l"))
    assert set(softmax.body) < set(flash.body)
    assert sum(stmt.op.name == "exp" for stmt in flash.body) == 2
