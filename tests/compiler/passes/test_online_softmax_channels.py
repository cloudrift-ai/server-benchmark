"""The N-channel online-softmax pairing — expectation channels joining the ``(m, d)`` pair.

Unit tests for the generalized pairing in ``lowering/tile/_softmax.py``: the ``(m, d)`` pair
alone keeps its historical byte-exact spelling; a same-extent additive fold whose lifted value
is (the pair's per-element weight) × (a value cone) joins the SAME twisted loop as an expectation
channel, with any loop-invariant multiplicative factor split off (``Σ c·x = c·Σ x``) and
multiplied back after the loop; a foreign fold declines; the fused-matmul spelling (channels
inside a free output sweep) sinks the pair into the sweep; and the emitted body round-trips
``fold_from_loop`` byte-identically. The fusion pass's readable-seam refusal is exercised in
lockstep: the merged softmax·V region now splices while other multi-statistic entanglements
still refuse. Numeric equivalence of the merged region is pinned via ``NumpyBackend``.
"""

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Accum, Assign, Load, LoopOp, Write
from emmy.compiler.ir.stmt import Body, Loop
from emmy.compiler.ir.stmt.carrier import exp_merge
from emmy.compiler.ir.stmt.leaves import ElementwiseImpl
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from emmy.compiler.pipeline.passes.lowering.tile._softmax import _fuse, split_invariant_factors

# --------------------------------------------------------------------------------------------
# Fixtures — the merged loop-IR spelling the fusion pass produces for softmax(·V).
# --------------------------------------------------------------------------------------------


def _pair() -> tuple[Loop, Loop]:
    idx = (Var("a0"), Var("a1"))
    rowmax = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in0", input="x", index=idx), Accum(name="acc0", value="in0", op=ElementwiseImpl("maximum")))),
    )
    sumexp = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce(
            (
                Load(name="in1", input="x", index=idx),
                Assign(name="v0", op="subtract", args=("in1", "acc0")),
                Assign(name="v1", op="exp", args=("v0",)),
                Accum(name="acc1", value="v1", op=ElementwiseImpl("add")),
            )
        ),
    )
    return rowmax, sumexp


def _value_loop(input_buf: str = "x", value_input: str = "w") -> Loop:
    """``acc2 = Σₖ exp(score − acc0) · v2 · w[k]`` — the normalize-inside spelling the splicer
    produces for a fused softmax·V demand site (``v2`` is the hoisted ``1/d``, defined outside)."""
    return Loop(
        axis=Axis(name="a3", extent=Dim(128)),
        body=Body.coerce(
            (
                Load(name="in2", input=input_buf, index=(Var("a0"), Var("a3"))),
                Assign(name="v3", op="subtract", args=("in2", "acc0")),
                Assign(name="v4", op="exp", args=("v3",)),
                Assign(name="v5", op="multiply", args=("v2", "v4")),
                Load(name="in3", input=value_input, index=(Var("a3"),)),
                Assign(name="v6", op="multiply", args=("in3", "v5")),
                Accum(name="acc2", value="v6", op=ElementwiseImpl("add")),
            )
        ),
    )


def _recip() -> Assign:
    return Assign(name="v2", op="reciprocal", args=("acc1",))


# --------------------------------------------------------------------------------------------
# The (m, d) pair alone — the historical spelling is byte-pinned.
# --------------------------------------------------------------------------------------------


def test_pair_alone_keeps_the_md_spelling() -> None:
    fused, changed = _fuse(Body.coerce(_pair()))
    assert changed
    (loop,) = list(fused)
    expect = Body.coerce(
        (
            Load(name="acc0__osin", input="x", index=(Var("a0"), Var("a1"))),
            *exp_merge(("acc0", "acc1"), ("acc0__osin", 1.0), key="acc0"),
        )
    )
    assert isinstance(loop, Loop) and loop.role is AxisRole.TWISTED
    assert loop.body == expect


# --------------------------------------------------------------------------------------------
# An expectation channel joins the pair; the invariant factor moves to the epilogue.
# --------------------------------------------------------------------------------------------


def test_expectation_channel_joins_flat_sibling() -> None:
    fused, changed = _fuse(Body.coerce((*_pair(), _recip(), _value_loop())))
    assert changed
    stmts = list(fused)
    loops = [s for s in stmts if isinstance(s, Loop)]
    assert len(loops) == 1, "the pair and the value fold join into ONE twisted loop"
    loop = stmts[0]
    assert loop.role is AxisRole.TWISTED
    expect = Body.coerce(
        (
            Load(name="acc0__osin", input="x", index=(Var("a0"), Var("a1"))),
            Load(name="in3", input="w", index=(Var("a1"),)),  # the value cone, re-spelled on the pair's axis
            *exp_merge(("acc0", "acc1", "acc2__sum"), ("acc0__osin", 1.0, "in3"), key="acc0"),
        )
    )
    assert loop.body == expect
    # The held-back reciprocal re-emits after the loop; the invariant factor multiplies the
    # carried sum back into the original accumulator name (Σ c·x = c·Σ x).
    assert stmts[1:] == [_recip(), Assign(name="acc2", op="multiply", args=("acc2__sum", "v2"))]


def test_emitted_channel_body_round_trips_fold_from_loop() -> None:
    fused, _ = _fuse(Body.coerce((*_pair(), _recip(), _value_loop())))
    loop = list(fused)[0]
    fold = fold_from_loop(loop)
    assert fold is not None, "the emitted N-channel spelling must be λ-readable"
    assert fold.combine.results == ("acc0", "acc1", "acc2__sum")
    assert fold.lift.results == ("acc0__osin", 1.0, "in3")
    assert fold.init == (float("-inf"), 0.0, 0.0)
    assert fold.loop == loop, "the byte-identity gate: derivation reproduces the emitted loop"


def test_split_invariant_factors_reads_the_product() -> None:
    body = list(_value_loop().body)
    inv, local = split_invariant_factors(body, "v6", "a3")
    assert inv == ("v2",)
    assert local == ("in3", "v4")


def test_foreign_value_loop_declines() -> None:
    # The additive fold's weight reads a DIFFERENT score buffer — not the pair's weight: the
    # pair keeps its plain (m, d) spelling and the foreign loop rides untouched.
    fused, changed = _fuse(Body.coerce((*_pair(), _recip(), _value_loop(input_buf="y"))))
    assert changed
    stmts = list(fused)
    loops = [s for s in stmts if isinstance(s, Loop)]
    assert len(loops) == 2
    assert sum(1 for s in loops[0].body if isinstance(s, Accum)) == 2, "the pair fused alone"
    assert loops[1] == _value_loop(input_buf="y"), "the foreign fold is untouched"


# --------------------------------------------------------------------------------------------
# The fused-matmul spelling: the channels sit inside a free output sweep — the pair sinks in.
# --------------------------------------------------------------------------------------------


def _sweep_body() -> Body:
    sweep = Loop(
        axis=Axis(name="a2", extent=Dim(32)),
        body=Body.coerce(
            (
                _value_loop_2d(),
                Write(output="out", value="acc2", index=(Var("a0"), Var("a2"))),
            )
        ),
    )
    return Body.coerce((*_pair(), _recip(), sweep))


def _value_loop_2d() -> Loop:
    return Loop(
        axis=Axis(name="a3", extent=Dim(128)),
        body=Body.coerce(
            (
                Load(name="in2", input="x", index=(Var("a0"), Var("a3"))),
                Assign(name="v3", op="subtract", args=("in2", "acc0")),
                Assign(name="v4", op="exp", args=("v3",)),
                Assign(name="v5", op="multiply", args=("v2", "v4")),
                Load(name="in3", input="w", index=(Var("a3"), Var("a2"))),
                Assign(name="v6", op="multiply", args=("in3", "v5")),
                Accum(name="acc2", value="v6", op=ElementwiseImpl("add")),
            )
        ),
    )


def test_pair_sinks_into_the_free_sweep() -> None:
    fused, changed = _fuse(_sweep_body())
    assert changed
    stmts = list(fused)
    assert len(stmts) == 1 and isinstance(stmts[0], Loop) and not stmts[0].is_reduce, "everything sank into the sweep"
    inner = list(stmts[0].body)
    assert isinstance(inner[0], Loop) and inner[0].role is AxisRole.TWISTED
    assert sum(1 for s in inner[0].body if isinstance(s, Accum)) == 3, "per-cell (m, d, expectation) join"
    assert inner[1:] == [
        _recip(),
        Assign(name="acc2", op="multiply", args=("acc2__sum", "v2")),
        Write(output="out", value="acc2", index=(Var("a0"), Var("a2"))),
    ]
    assert fold_from_loop(inner[0]) is not None


def test_sink_declines_when_moved_state_is_read_downstream() -> None:
    # A trailing read of the denominator outside the sweep pins the pair at its level: the
    # sink would strand the reader, so only the plain (m, d) pairing happens.
    tail = Write(output="dsum", value="acc1", index=(Var("a0"),))
    fused, changed = _fuse(Body.coerce((*list(_sweep_body()), tail)))
    assert changed
    stmts = list(fused)
    assert isinstance(stmts[0], Loop) and stmts[0].role is AxisRole.TWISTED
    assert sum(1 for s in stmts[0].body if isinstance(s, Accum)) == 2, "plain (m, d) pairing only"
    assert stmts[-1] == tail


# --------------------------------------------------------------------------------------------
# The fusion pass in lockstep: the merged softmax·V region splices; other entanglements refuse.
# --------------------------------------------------------------------------------------------


def _softmax_matmul_graph(m: int = 8, k: int = 16, n: int = 4) -> Graph:
    from emmy.compiler.ir.frontend.ir import MatmulOp, SoftmaxOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (k, n)), node_id="w")
    g.add_node(SoftmaxOp(), ["x"], Tensor("p", (m, k)), node_id="p")
    g.add_node(MatmulOp(), ["p", "w"], Tensor("y", (m, n)), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def test_softmax_matmul_merges_to_one_kernel_with_matching_numerics() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    w = rng.standard_normal((16, 4)).astype(np.float32)
    e = np.exp(x - x.max(-1, keepdims=True))
    expect = (e / e.sum(-1, keepdims=True)) @ w

    merged = Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/fusion"]).run(_softmax_matmul_graph())
    kernels = [nd for nd in merged.nodes.values() if isinstance(nd.op, LoopOp)]
    assert len(kernels) == 1, "the fused softmax·V region splices into ONE kernel"
    backend = NumpyBackend()
    after = backend.run(backend.compile(merged), input_data={"x": x, "w": w})[0].outputs
    np.testing.assert_allclose(list(after.values())[0], expect, rtol=1e-5, atol=1e-5)


def test_multi_stat_entangled_with_expanding_tail_still_refuses() -> None:
    from importlib import import_module

    _entangled_multi_stat = import_module("emmy.compiler.pipeline.passes.loop.fusion.010_merge_loop_ops")._entangled_multi_stat

    # Pair + a free sweep whose nested fold is NOT flat-additive (a maximum) — still entangled.
    bad = Loop(
        axis=Axis(name="a2", extent=Dim(32)),
        body=Body.coerce(
            (
                Loop(
                    axis=Axis(name="a3", extent=Dim(128)),
                    body=Body.coerce(
                        (
                            Load(name="in2", input="x", index=(Var("a0"), Var("a3"))),
                            Accum(name="acc2", value="in2", op=ElementwiseImpl("maximum")),
                        )
                    ),
                ),
                Write(output="out", value="acc2", index=(Var("a0"), Var("a2"))),
            )
        ),
    )
    refused = LoopOp(body=Body.coerce((Loop(axis=Axis(name="a0", extent=Dim(8)), body=Body.coerce((*_pair(), bad))),)))
    readable = LoopOp(body=_wrap_rows(_sweep_body()))
    assert _entangled_multi_stat(refused)
    assert not _entangled_multi_stat(readable)


def _wrap_rows(body: Body) -> Body:
    return Body.coerce((Loop(axis=Axis(name="a0", extent=Dim(8)), body=body),))
