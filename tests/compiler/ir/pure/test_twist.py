"""The twist recipes' fused ⊕ (``Recipe.program``): the program a recipe instantiates over a
fold's state names is a monoid — associative on random states, its seeds neutral — one recipe
serves every channel count, an expectation channel joining the same pivot advance, and Welford's
fixed carrier streams the two-pass variance in one pass."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.pure.twist import SOFTMAX, WELFORD
from emmy.compiler.ir.stmt import Const


def _eval(lam, args: tuple) -> tuple:
    """Evaluate a pure lambda on floats — the ANF chain of ``Assign``s and ``Const`` defs."""
    env = dict(zip(lam.params, args, strict=True))
    for stmt in lam.body:
        env[stmt.name] = stmt.value if isinstance(stmt, Const) else stmt.op(*(env[arg] for arg in stmt.args))
    return tuple(env[result] for result in lam.results)


def _softmax_state(rng, n: int) -> tuple:
    return tuple(rng.normal(size=n))


def _welford_state(rng, n: int) -> tuple:
    # A non-empty partition: its sum, its count, its mean and a non-negative M2.
    return (rng.normal(), float(rng.integers(1, 6)), rng.normal(), abs(rng.normal()))


@pytest.mark.parametrize(
    ("recipe", "states", "seeds", "sample"),
    [
        (SOFTMAX, ("m", "l"), (float("-inf"), 0.0), _softmax_state),
        (SOFTMAX, ("m", "o", "l"), (float("-inf"), 0.0, 0.0), _softmax_state),
        (WELFORD, ("g", "n", "m", "q"), (0.0, 0.0, 0.0, 0.0), _welford_state),
    ],
    ids=["softmax", "flash", "welford"],
)
def test_the_program_is_an_associative_monoid_with_neutral_seeds(recipe, states: tuple[str, ...], seeds: tuple, sample) -> None:
    """The split / cooperative legality certificate: ``a ⊕ (b ⊕ c) == (a ⊕ b) ⊕ c`` on random
    states, and the fold's seeds are its neutral element on either side."""
    combine = recipe.program(states)
    assert combine.params == (*states, *(f"{state}__o" for state in states)) and combine.results == states
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = (sample(rng, len(states)) for _ in range(3))
        lhs = _eval(combine, (*a, *_eval(combine, (*b, *c))))
        rhs = _eval(combine, (*_eval(combine, (*a, *b)), *c))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(_eval(combine, (*seeds, *a)), a)
        np.testing.assert_allclose(_eval(combine, (*a, *seeds)), a)


def test_welford_program_streams_the_two_pass_variance() -> None:
    """Folded over a stream with the singleton injected as ``(x, 1, x, 0)``, the carrier lands on
    the sum, the count, the mean and the sum of squared deviations from that mean — the second
    pass's value, in one pass, with the seed ``(0, 0, 0, 0)`` a true identity."""
    combine = WELFORD.program(("g", "n", "m", "q"))
    rng = np.random.default_rng(1)
    xs = rng.normal(size=64)
    state = (0.0, 0.0, 0.0, 0.0)
    for x in xs:
        state = _eval(combine, (*state, x, 1.0, x, 0.0))
    np.testing.assert_allclose(state, (xs.sum(), 64.0, xs.mean(), ((xs - xs.mean()) ** 2).sum()), rtol=1e-9, atol=1e-9)


def test_an_expectation_channel_joins_the_same_advance() -> None:
    """Online softmax and flash attention are ONE recipe: the three-state program is the two-state
    one plus a channel rescaled by the same two factors — no second advance, no second ``exp``."""
    softmax, flash = SOFTMAX.program(("m", "l")), SOFTMAX.program(("m", "o", "l"))
    assert set(softmax.body) < set(flash.body)
    assert sum(stmt.op.name == "exp" for stmt in flash.body) == 2
