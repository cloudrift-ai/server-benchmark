"""Value semantics — ``Expr.symbolic`` / ``Expr.same_value`` / ``Body.factor`` / ``Body.linearize``.

The node tree is syntax, sympy is semantics. These tests state what each reading DECIDES; nothing
here builds IR from a normalized expression, which is the one-way rule.
"""

from __future__ import annotations

import subprocess
import sys

import sympy

from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.pure.twist import SOFTMAX, WELFORD
from emmy.compiler.ir.stmt import Assign
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.value import RING, exp, maximum, ring, values


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _hoist(op) -> bool:  # noqa: ANN001
    """The hoist's licence: only a product or a division reassociates under ``Σ``."""
    return op.semiring_product or op.name == "divide"


def _monomials(reading) -> dict:  # noqa: ANN001
    return dict(reading)


# --- factor -----------------------------------------------------------


def test_the_attention_summand_splits_into_its_invariant_and_varying_factors() -> None:
    """``Σ_k exp(s−m)·v / l`` is ``(Σ_k exp(s−m)·v) / l``: the normalizer is invariant along the
    fold axis and commutes out, the weight and the streamed row do not."""
    body = Body([_asn("d", "subtract", "s", "m"), _asn("w", "exp", "d"), _asn("p", "multiply", "w", "v"), _asn("q", "divide", "p", "l")])
    split = body.factor("q", lambda name: name in {"w", "v"}, _hoist)
    assert split.invariant == (("l", True),)
    assert sorted(split.varying) == ["v", "w"]
    assert [stmt.name for stmt in split.spine] == ["p", "q"]


def test_a_square_is_listed_twice() -> None:
    """Multiplicity is the point: a consumer must see ``x·x`` for what it is, since a square is the
    step that is NOT a semiring step and so carries no contraction reading."""
    body = Body([_asn("sq", "multiply", "x", "x"), _asn("ab", "multiply", "a", "b"), _asn("r", "multiply", "sq", "ab")])
    split = body.factor("r", lambda name: name == "x", _hoist)
    assert split.varying == ("x", "x")
    assert sorted(split.invariant) == [("a", False), ("b", False)]


def test_a_sum_is_not_a_product() -> None:
    assert Body([_asn("r", "add", "x", "y")]).factor("r", lambda name: name == "x", ring) is None


def test_a_varying_divisor_refuses() -> None:
    """``Σ x/c`` is ``(Σ x)/c`` only for an invariant ``c``; nothing licenses moving a fold into a
    denominator that streams."""
    assert Body([_asn("r", "divide", "x", "y")]).factor("r", lambda name: name in {"x", "y"}, _hoist) is None


def test_a_lone_atom_is_a_one_factor_product() -> None:
    """Structural refusals are the reading's; usefulness is the rule's. A cone that is one opaque
    weight splits into one varying factor and no invariant — the caller declines that, not this."""
    split = Body([_asn("w", "exp", "d")]).factor("w", lambda name: name == "w", _hoist)
    assert split.varying == ("w",)
    assert split.invariant == ()


# --- linearize --------------------------------------------------------


def test_welfords_residual_linearizes_to_its_monomials_in_the_stream() -> None:
    """``(s − g·c)²`` in the streamed ``s``. Each coefficient is an expression over the invariants
    by construction, which is what makes this the 'is the summand linear in what it streams?' read."""
    body = Body([_asn("m", "multiply", "g", "c"), _asn("d", "subtract", "s", "m"), _asn("sq", "multiply", "d", "d")])
    terms = _monomials(body.linearize("sq", lambda name: name == "s", ring))
    g, c = sympy.Symbol("g"), sympy.Symbol("c")
    assert terms[(("s", 2),)] == 1
    assert sympy.expand(terms[(("s", 1),)] + 2 * g * c) == 0
    assert sympy.expand(terms[()] - g**2 * c**2) == 0


def test_a_stream_under_an_uninterpreted_call_refuses() -> None:
    assert Body([_asn("w", "exp", "s")]).linearize("w", lambda name: name == "s", ring) is None


def test_nothing_streaming_refuses() -> None:
    assert Body([_asn("r", "multiply", "a", "b")]).linearize("r", lambda name: name == "s", ring) is None


# --- the readings over the real recipes --------------------------------


def test_the_softmax_carrier_reads_as_a_pivot_and_two_rescaled_channels() -> None:
    """The payoff. Each non-pivot state merges as ``α·s + β·s__o`` with the coefficients drawn from
    the pivot's own cone, and the pivot itself is not linear at all — read off the stored program,
    with no recipe field consulted."""
    program = SOFTMAX.program(("m", "d", "o"))
    assert program.body.linearize("m", lambda name: name in {"m", "m__o"}, ring) is None
    for state in ("d", "o"):
        terms = _monomials(program.body.linearize(state, lambda name, s=state: name in {s, f"{s}__o"}, ring))
        assert set(terms) == {((state, 1),), ((f"{state}__o", 1),)}
        assert {str(coeff) for coeff in terms.values()} == {"m__o__alpha", "m__o__beta"}


def test_welfords_variance_channel_carries_a_residual_the_other_states_feed() -> None:
    """The discriminator, structurally. ``mean`` is linear in its own pair; ``M2`` is linear too but
    carries a constant monomial reading ``mu`` — a state outside the pivot group — which is exactly
    what a consumer looking for a rescaling carrier must decline."""
    program = WELFORD.program(("g", "n", "mu", "q"))
    mean = _monomials(program.body.linearize("mu", lambda name: name in {"mu", "mu__o"}, ring))
    assert set(mean) == {(("mu", 1),), (("mu__o", 1),)}
    assert all(coeff.free_symbols <= {sympy.Symbol("n"), sympy.Symbol("n__o")} for coeff in mean.values())

    m2 = _monomials(program.body.linearize("q", lambda name: name in {"q", "q__o"}, ring))
    assert m2[(("q", 1),)] == 1 and m2[(("q__o", 1),)] == 1
    assert sympy.Symbol("mu") in m2[()].free_symbols


# --- symbolic / equal --------------------------------------------------


def test_an_uninterpreted_op_compares_structurally() -> None:
    """When nothing in a cone is interpretable the decision collapses to alpha-equality, so a
    reading never has to know every op to be sound."""
    a, b = values("a b")
    assert exp(a - b).expr.same_value(exp(a - b).expr)
    assert not exp(a - b).expr.same_value(exp(b - a).expr)
    assert not maximum(a, b).expr.same_value(maximum(b, a).expr)


def test_ring_identities_decide_equality() -> None:
    x, y = values("x y")
    assert ((x + y) * x).expr.same_value((x * x + y * x).expr)
    assert (x / y * y).expr.same_value(x.expr)


def test_an_index_expression_keeps_its_integer_meaning() -> None:
    """``apply_binop`` floor-divides, so the bridge must too — reading a coordinate as exact
    division would state an equality the emitted kernel does not honour."""
    assert (Var("i") / Literal(4, "int")).symbolic() == sympy.floor(sympy.Symbol("i") / 4)
    assert (Var("i") % Literal(4, "int")).symbolic() == sympy.Mod(sympy.Symbol("i"), 4)


def test_semantic_equality_never_replaces_the_structural_one() -> None:
    """``==`` stays the tree comparison: association and temps are what a kernel is spelled from,
    and both readings have callers."""
    x, y = values("x y")
    assert ((x + y) * x).expr != (x * x + y * x).expr


def test_a_value_reading_needs_no_torch() -> None:
    """sympy is a required dependency, not a ``compile`` extra: the reading decides value cones and
    must work on a host with no torch installed. A subprocess, because another test in this session
    may already have imported it."""
    probe = (
        "import sys;"
        "from emmy.compiler.ir.stmt import Assign;"
        "from emmy.compiler.ir.stmt.body import Body;"
        "from emmy.compiler.ir.value import ring;"
        "b = Body([Assign(name='p', op='multiply', args=('w', 'v'))]);"
        "assert b.linearize('p', lambda n: n == 'w', ring) is not None;"
        "assert 'torch' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_exp_is_not_in_the_ring() -> None:
    assert "exp" not in RING and RING == {"add", "subtract", "multiply", "divide", "negative"}
