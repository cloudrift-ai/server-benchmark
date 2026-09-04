""":class:`Lambda` — formation (the ``Stmt.pure`` gate, results-defined, closedness), α-invariance
by canonical renumbering, and the componentwise program a plain fold's combine is."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.twist import SOFTMAX
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Init, Load, Loop, Write

# --- Lambda formation: the Stmt.pure gate + results-defined ------------------------------------- #


def test_lambda_accepts_pure_stmts_and_defines_results() -> None:
    lam = Lambda(
        params=("k", "x"),
        body=Body((Assign(name="x2", op="multiply", args=("x", "x")),)),
        results=("x2",),
    )
    assert lam.results == ("x2",)
    # CLOSED: every name the body reads is a param or one of its own defs, so there is no
    # residual for the former to append and the two spellings coincide.
    assert lam.defined == frozenset({"k", "x", "x2"})
    assert Lambda.closing(("k", "x"), lam.body, lam.results) == lam


def test_lambda_post_init_canonicalizes_body_order() -> None:
    left = Load(name="left", input="x", index=(Var("m"), Var("k")))
    right = Load(name="right", input="w", index=(Var("n"), Var("k")))
    product = Assign(name="product", op="multiply", args=("left", "right"))

    first = Lambda.closing(("k",), Body((left, right, product)), ("product",))
    second = Lambda.closing(("k",), Body((right, left, product)), ("product",))

    assert first == second


@pytest.mark.parametrize(
    "stmt",
    [
        Accum(name="acc", value="x", op="add"),
        Write(output="out", index=(Var("m"),), values=("x",)),
        Init(name="acc", identity=0.0, dtype=None),
        Loop(axis=Axis("k", 4), body=Body(())),
    ],
    ids=lambda s: type(s).__name__,
)
def test_lambda_rejects_impure_stmt_kinds(stmt) -> None:
    """The conservative ``Stmt.pure`` default: effectful / scope-bound kinds never enter a stored
    lambda body — the purity-erosion guard."""
    with pytest.raises(ValueError, match="pure"):
        Lambda(params=("x",), body=Body((stmt,)), results=("x",))


def test_lambda_rejects_undefined_results() -> None:
    with pytest.raises(ValueError, match="not defined"):
        Lambda(params=("x",), body=Body(()), results=("y",))


def test_lambda_allows_param_results_and_constant_results() -> None:
    """ι is spelled in the lift: softmax's singleton is ``(x, 1)`` — the constant component is a
    ``Const`` def, a name like any other result."""
    lam = Lambda(params=("k", "x"), body=Body((Const(name="one", value=1.0),)), results=("x", "one"))
    assert lam.results == ("x", "one") and lam.defined == frozenset({"k", "x", "one"})


def test_lambda_refuses_an_open_body_and_closing_binds_the_residual() -> None:
    """A term carries no free names. What the retired ``free_names`` reported as a contextual read
    is now a FORMATION ERROR, and :meth:`Lambda.closing` is the former that turns it into a
    trailing param — trailing, never interleaved, so the operand prefix keeps its positions."""
    body = Body((Assign(name="y", op="add", args=("x", "outer")),))

    with pytest.raises(ValueError, match=r"reads \['outer'\]"):
        Lambda(params=("x",), body=body, results=("y",))

    lam = Lambda.closing(("x",), body, ("y",))
    assert lam.params == ("x", "outer")
    assert lam.defined == frozenset({"x", "outer", "y"})
    assert lam.results == ("y",)


# --- α-invariance: canonical renumbering -------------------------------------------------------- #


def test_cone_is_one_definition_closed_over_what_it_reads() -> None:
    """The cone of a result is the sub-lambda defining it — its dependence within the body, the
    reads it takes from the params as its own params; a param's cone is the identity on it."""
    fn = Lambda.closing(
        ("k", "a", "b"),
        Body((Assign("s", "multiply", ("a", "b")), Assign("d", "subtract", ("s", "k")), Assign("w", "exp", ("d",)))),
        ("w",),
    )
    score = fn.cone("s")
    assert score.params == ("a", "b") and [stmt.name for stmt in score.body] == ["s"] and score.results == ("s",)
    assert fn.cone("a") == Lambda(params=("a",), body=Body(()), results=("a",))
    assert set(fn.cone("w").params) == {"a", "b", "k"} and len(fn.cone("w").body) == 3


def test_rename_maps_params_body_and_results_in_lockstep() -> None:
    fn = Lambda(params=("m", "m__o"), body=Body((Assign("t", "maximum", ("m", "m__o")), Assign("m", "copy", ("t",)))), results=("m",))
    renamed = fn.rename({"m": "acc", "m__o": "acc__o", "t": "acc__t"})
    assert renamed.params == ("acc", "acc__o") and renamed.results == ("acc",)
    assert [(stmt.name, stmt.args) for stmt in renamed.body] == [("acc__t", ("acc", "acc__o")), ("acc", ("acc__t",))]
    assert fn.rename(lambda name: name) == fn
    assert fn.alpha_eq(renamed)


def test_alpha_equality_under_bound_renaming() -> None:
    a = Lambda(
        params=("k", "x"),
        body=Body((Assign(name="t", op="multiply", args=("x", "x")),)),
        results=("t",),
    )
    b = Lambda(
        params=("j", "v"),
        body=Body((Assign(name="sq", op="multiply", args=("v", "v")),)),
        results=("sq",),
    )
    assert a.alpha_eq(b)
    assert a.canonical() == b.canonical()
    assert hash(a.canonical()) == hash(b.canonical())


def test_alpha_inequality_on_structure_and_residual_arity() -> None:
    sq = Lambda(params=("x",), body=Body((Assign(name="t", op="multiply", args=("x", "x")),)), results=("t",))
    dbl = Lambda(params=("x",), body=Body((Assign(name="t", op="add", args=("x", "x")),)), results=("t",))
    closed = Lambda.closing(("x",), Body((Assign(name="t", op="multiply", args=("x", "w")),)), ("t",))
    renamed = Lambda.closing(("x",), Body((Assign(name="t", op="multiply", args=("x", "z")),)), ("t",))
    assert not sq.alpha_eq(dbl)
    # The residual read is BOUND, so it changes the arity — ``x·w`` is a two-param function.
    assert closed.params == ("x", "w") and not sq.alpha_eq(closed)
    # And being bound, it renumbers like any other param: its spelling no longer has to match.
    assert closed.alpha_eq(renamed)
    assert sq.canonical().canonical() == sq.canonical()  # idempotent


# --- the componentwise program: a plain fold's ⊕, and the shape read back off any combine -------- #


def test_componentwise_program_reads_back_its_ops() -> None:
    """``Lambda.componentwise`` spells one independent ⊕ per state, ``S × S → S`` with the second
    operand ``<n>__o``, and ``components`` reads the op vector back — in either argument order for
    a commutative ⊕, since a rename can reorder the sorted arguments."""
    combine = Lambda.componentwise(("add", "maximum"), ("acc", "mx"))
    assert combine.params == ("acc", "mx", "acc__o", "mx__o") and combine.results == ("acc", "mx")
    assert [op.name for op in combine.components()] == ["add", "maximum"]
    assert [op.name for op in combine.rename({"acc": "z"}).components()] == ["add", "maximum"]


def test_a_twisted_program_has_no_components() -> None:
    """A cross-component read — a recipe's rescale — fails the componentwise shape: the
    planar-vs-twisted reading every partition legality question asks, derived, never stored."""
    assert SOFTMAX.program(("m", "l")).components() is None
