"""Standalone tests for ``splice_loop_ops``.

Each test builds producer/consumer ``LoopOp``s by hand and splices them
directly, asserting on the merged body structure. This skips the usual
frontend → tensor-IR → loop-IR pipeline so failures point at the splicer
itself rather than some upstream lowering quirk.
"""

from __future__ import annotations

import pytest

from emmy.compiler.dtype import F16, F32, DataType
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Write,
    splice_graph,
    splice_loop_ops,
    splice_loops,
)
from emmy.compiler.tensor import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_kind(op: LoopOp, cls: type) -> int:
    return sum(1 for s in op.body.iter() if isinstance(s, cls))


def _elementwise_fns(op: LoopOp) -> list[str]:
    return [s.op.name for s in op.body.iter() if isinstance(s, Assign)]


# Fixtures — shared axes
# ---------------------------------------------------------------------------


A0 = Axis("a0", 4)
A1 = Axis("a1", 8)
K = Axis("k", 16)


def _splice_graph_producer(
    producer: LoopOp,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    edge_dtype: DataType,
    shared_origin: bool = False,
    private_origin_output: bool = False,
) -> LoopOp:
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="value", input="producer", index=(Var("a0"),)),
                    Assign(name="activated", op="silu", args=("value",)),
                    Write(output="consumer", index=(Var("a0"),), value="activated"),
                ),
            ),
        ),
    )
    if shared_origin:
        origin = MatmulOp()
        producer.source = origin
        consumer.source = origin
    elif private_origin_output:
        origin = MatmulOp()
        origin.outputs["frontend_output"] = Tensor("frontend_output", output_shape, edge_dtype)
        producer.source = origin
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", input_shape, F16), node_id="x")
    graph.add_node(producer, ["x"], Tensor("producer", output_shape, edge_dtype), node_id="producer")
    graph.add_node(consumer, ["producer"], Tensor("consumer", output_shape, edge_dtype), node_id="consumer")
    graph.outputs = ["consumer"]
    result = splice_graph(graph)
    assert result is not None
    merged, external = result
    assert external == ["x"]
    return merged


# ---------------------------------------------------------------------------
# Basic pipeline: pointwise producer → pointwise consumer
# ---------------------------------------------------------------------------


def test_pointwise_chain():
    """producer: y = exp(x)  ;  consumer: z = add(y, y)."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="out_0", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", input="src_0", index=(Var("a0"),)),
                    Assign(name="z", op="add", args=("yv", "yv")),
                    Write(output="out_0", index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    fns = _elementwise_fns(merged)
    # exp from producer + copy alias for the Load + add from consumer.
    assert "exp" in fns
    assert "add" in fns
    # Producer's Load (consumer had no other inputs) survives as source 0.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 1
    assert loads[0].input == "src_0"
    # Single Write, referencing the final add.
    writes = [s for s in merged.body.iter() if isinstance(s, Write)]
    assert len(writes) == 1


# ---------------------------------------------------------------------------
# Shared intermediate: consumer uses producer output twice at same index → one emission
# ---------------------------------------------------------------------------


def test_shared_intermediate_deduped():
    """Two consumer refs to the same producer value under identity σ share one binding."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="out_0", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    # Consumer loads producer twice at the same index — both solve σ = {a0: a0}.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="ya", input="src_0", index=(Var("a0"),)),
                    Load(name="yb", input="src_0", index=(Var("a0"),)),
                    Assign(name="z", op="add", args=("ya", "yb")),
                    Write(output="out_0", index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # Only one exp in the merged body (producer chain materializes once).
    assert _elementwise_fns(merged).count("exp") == 1


# ---------------------------------------------------------------------------
# Different σs: two consumer loads at different indices → two emissions
# ---------------------------------------------------------------------------


def test_different_indices_emit_twice():
    """Consumer loads a 2D producer with two different index permutations —
    the two σs are distinct, so the producer's chain materializes twice."""
    # Producer axes (a0, a1); writes y[a0, a1] = exp(x[a0, a1]).
    A1_same = Axis("a1", 4)  # square so the transposed load is well-typed.
    A0_same = Axis("a0", 4)
    producer = LoopOp(
        body=(
            Loop(
                axis=A0_same,
                body=(
                    Loop(
                        axis=A1_same,
                        body=(
                            Load(name="x", input="src_0", index=(Var("a0"), Var("a1"))),
                            Assign(name="y", op="exp", args=("x",)),
                            Write(output="out_0", index=(Var("a0"), Var("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        ),
    )
    # Consumer reads the producer at (a0, a1) and (a1, a0) — transposed.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0_same,
                body=(
                    Loop(
                        axis=A1_same,
                        body=(
                            Load(name="ya", input="src_0", index=(Var("a0"), Var("a1"))),
                            Load(name="yb", input="src_0", index=(Var("a1"), Var("a0"))),
                            Assign(name="z", op="add", args=("ya", "yb")),
                            Write(output="out_0", index=(Var("a0"), Var("a1")), value="z"),
                        ),
                    ),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # Two distinct σs ({a0:a0,a1:a1}, {a0:a1,a1:a0}) → exp materializes twice.
    assert _elementwise_fns(merged).count("exp") == 2


# ---------------------------------------------------------------------------
# Reduction producer: sum over k, consumer is pointwise on the result
# ---------------------------------------------------------------------------


def test_reduction_producer():
    """Producer: s = sum_k x[a0,k]. Consumer: y = exp(s[a0])."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="src_0", index=(Var("a0"), Var("k"))),
                            Accum(name="s", value="x", op="add"),
                        ),
                    ),
                    Write(output="out_0", index=(Var("a0"),), value="s"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="sv", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("sv",)),
                    Write(output="out_0", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # The Accum survives in the merged body.
    assert _count_kind(merged, Accum) == 1
    # Elementwise chain includes the producer-load copy + the consumer's exp.
    assert "exp" in _elementwise_fns(merged)


@pytest.mark.parametrize(("edge_dtype", "expected_alias_dtype"), [(F16, F16), (F32, None)])
def test_graph_splice_preserves_distinct_origin_reduction_dtype(edge_dtype: DataType, expected_alias_dtype: DataType | None):
    """A reduction entering a distinct frontend operation crosses its tensor boundary."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="x", index=(Var("a0"), Var("k"))),
                            Accum(name="sum", value="x", op="add"),
                        ),
                    ),
                    Write(output="producer", index=(Var("a0"),), value="sum"),
                ),
            ),
        ),
    )
    merged = _splice_graph_producer(producer, input_shape=(4, 16), output_shape=(4,), edge_dtype=edge_dtype)
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    if expected_alias_dtype is None:
        assert aliases == []
    else:
        assert len(aliases) == 1
        assert aliases[0].dtype == expected_alias_dtype


def test_graph_splice_same_origin_reduction_is_private():
    """Nodes decomposed from one frontend operation may reconstruct their private edge directly."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="x", index=(Var("a0"), Var("k"))),
                            Accum(name="sum", value="x", op="add"),
                        ),
                    ),
                    Write(output="producer", index=(Var("a0"),), value="sum"),
                ),
            ),
        ),
    )
    merged = _splice_graph_producer(
        producer,
        input_shape=(4, 16),
        output_shape=(4,),
        edge_dtype=F16,
        shared_origin=True,
    )
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert aliases == []


def test_graph_splice_private_origin_output_is_private_in_mixed_fragment():
    """An internal decomposition output reconstructs after its consumer has mixed origins."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="x", index=(Var("a0"), Var("k"))),
                            Accum(name="sum", value="x", op="add"),
                        ),
                    ),
                    Write(output="producer", index=(Var("a0"),), value="sum"),
                ),
            ),
        ),
    )
    merged = _splice_graph_producer(
        producer,
        input_shape=(4, 16),
        output_shape=(4,),
        edge_dtype=F16,
        private_origin_output=True,
    )
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert aliases == []


def test_graph_splice_load_output_needs_no_conversion():
    """A pass-through value is already represented at the loaded input dtype."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="x", index=(Var("a0"),)),
                    Write(output="producer", index=(Var("a0"),), value="x"),
                ),
            ),
        ),
    )
    merged = _splice_graph_producer(producer, input_shape=(4,), output_shape=(4,), edge_dtype=F16)
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert aliases == []


@pytest.mark.parametrize(("edge_dtype", "expected_aliases"), [(F16, 2), (F32, 0)])
def test_graph_splice_preserves_multi_reduction_output_dtypes(edge_dtype: DataType, expected_aliases: int):
    """Sibling reductions cross their tensor boundaries before their shared projection."""

    def producer(input_name: str, output_name: str) -> LoopOp:
        return LoopOp(
            body=(
                Loop(
                    axis=A0,
                    body=(
                        Loop(
                            axis=K,
                            body=(
                                Load(name=f"{input_name}_value", input=input_name, index=(Var("a0"), Var("k"))),
                                Accum(name=f"{output_name}_sum", value=f"{input_name}_value", op="add"),
                            ),
                        ),
                        Write(output=output_name, index=(Var("a0"),), value=f"{output_name}_sum"),
                    ),
                ),
            ),
        )

    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="left_value", input="left", index=(Var("a0"),)),
                    Load(name="right_value", input="right", index=(Var("a0"),)),
                    Load(name="aux_value", input="aux", index=(Var("a0"),)),
                    Assign(name="activated", op="silu", args=("left_value",)),
                    Assign(name="product", op="multiply", args=("activated", "right_value")),
                    Assign(name="result", op="add", args=("product", "aux_value")),
                    Write(output="consumer", index=(Var("a0"),), value="result"),
                ),
            ),
        ),
    )
    pointwise = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="z_value", input="z", index=(Var("a0"),)),
                    Assign(name="exp_value", op="exp", args=("z_value",), dtype=F16),
                    Write(output="aux", index=(Var("a0"),), value="exp_value"),
                ),
            ),
        ),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 16), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("y", (4, 16), F16), node_id="y")
    graph.add_node(InputOp(), [], Tensor("z", (4,), F16), node_id="z")
    graph.add_node(producer("x", "left"), ["x"], Tensor("left", (4,), edge_dtype), node_id="left")
    graph.add_node(producer("y", "right"), ["y"], Tensor("right", (4,), edge_dtype), node_id="right")
    graph.add_node(pointwise, ["z"], Tensor("aux", (4,), F16), node_id="aux")
    graph.add_node(consumer, ["left", "right", "aux"], Tensor("consumer", (4,), F16), node_id="consumer")
    graph.outputs = ["consumer"]

    result = splice_graph(graph)
    assert result is not None
    merged, external = result
    assert external == ["x", "y", "z"]
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert len(aliases) == expected_aliases
    assert {alias.dtype for alias in aliases} <= {F16}
    pointwise_result = next(stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "exp")
    assert pointwise_result.dtype == F16


def test_graph_splice_typed_boundary_roundtrips():
    """Serialized Loop IR retains the conversion when source-chain metadata is absent."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="x", index=(Var("a0"), Var("k"))),
                            Accum(name="sum", value="x", op="add"),
                        ),
                    ),
                    Write(output="producer", index=(Var("a0"),), value="sum"),
                ),
            ),
        ),
    )
    merged = _splice_graph_producer(producer, input_shape=(4, 16), output_shape=(4,), edge_dtype=F16)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 16), F16), node_id="x")
    graph.add_node(merged, ["x"], Tensor("consumer", (4,), F16), node_id="consumer")
    graph.outputs = ["consumer"]

    restored = Graph.from_dict(graph.to_dict()).nodes["consumer"].op
    aliases = [stmt for stmt in restored.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert len(aliases) == 1
    assert aliases[0].dtype == F16


# ---------------------------------------------------------------------------
# Ordered prefix: a Write of the running accumulator cannot cross the splice
# ---------------------------------------------------------------------------


def _prefix_scan(*, source: str, output: str) -> LoopOp:
    return LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=A1,
                        body=(
                            Load(name="x", input=source, index=(Var("a0"), Var("a1"))),
                            Accum(name="prefix", value="x", op="add"),
                            Write(output=output, index=(Var("a0"), Var("a1")), value="prefix"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_ordered_prefix_root_declines_splice():
    """Rebuilding a root from its Writes must not move a prefix output after the reduce loop."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=A1,
                        body=(
                            Load(name="x", input="X", index=(Var("a0"), Var("a1"))),
                            Assign(name="y", op="negative", args=("x",)),
                            Write(output="P", index=(Var("a0"), Var("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert splice_loop_ops(producer, _prefix_scan(source="P", output="OUT"), source="P") is None


def test_ordered_prefix_producer_declines_splice():
    """Resolving a producer edge must not replace its prefix output with one final reduction."""
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=A1,
                        body=(
                            Load(name="x", input="P", index=(Var("a0"), Var("a1"))),
                            Assign(name="y", op="negative", args=("x",)),
                            Write(output="OUT", index=(Var("a0"), Var("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert splice_loop_ops(_prefix_scan(source="X", output="P"), consumer, source="P") is None


# ---------------------------------------------------------------------------
# Consumer has extra input: source-remapping puts consumer inputs after producer's
# ---------------------------------------------------------------------------


def test_consumer_extra_input_source_remap():
    """Consumer has producer at source 0 and an unrelated buffer at source 1;
    after splice, that unrelated buffer should load from merged source 1
    (producer's input is at 0)."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="out_0", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", input="src_0", index=(Var("a0"),)),  # from producer
                    Load(name="b", input="src_1", index=(Var("a0"),)),  # unrelated
                    Assign(name="z", op="add", args=("yv", "b")),
                    Write(output="out_0", index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    loads = {s.name: s for s in merged.body.iter() if isinstance(s, Load)}
    # Producer's Load survives at source 0; consumer's unrelated Load shifts to 1.
    assert any(ld.input == "src_0" for ld in loads.values())
    assert any(ld.input == "src_1" for ld in loads.values())
    assert len(tuple(merged.inputs)) == 2


# ---------------------------------------------------------------------------
# Literal in producer write index: no σ-binding needed, still splices
# ---------------------------------------------------------------------------


def test_live_axes_computed():
    """``LoopMeta.live_axes`` captures axes transitively used through each
    dep's Expr subtrees. For Accum, the reduce axis is excluded.

    Normalization canonicalizes SSA and axis names (``Load→in0``,
    ``Accum→acc0``; free axes first then reduce axes), so we assert on
    ``meta.scopes`` (now the post-reduce binding scope for Accum) and
    ``meta.reduce_axes`` rather than pre-normalize names.
    """
    op = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", input="src_0", index=(Var("a0"), Var("k"))),
                            Accum(name="s", value="x", op="add"),
                        ),
                    ),
                    Write(output="out_0", index=(Var("a0"),), value="s"),
                ),
            ),
        ),
    )
    meta = op.analyze()
    [(load_name, load_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Load)]
    [(acc_name, acc_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Accum)]
    # Load sees both loop axes, Accum binds post-reduce so its scope is shorter.
    assert {a.name for a in load_scope.enclosing} == meta.live_axes[load_name]
    assert acc_name in meta.reduce_axes
    reduce_axis = meta.reduce_axes[acc_name]
    assert reduce_axis.name not in meta.live_axes[acc_name]
    # Accum's binding scope = load's enclosing minus the reduce axis.
    assert meta.scopes[acc_name].enclosing == tuple(a for a in load_scope.enclosing if a.name != reduce_axis.name)


# ---------------------------------------------------------------------------
# N-way chain fusion via splice_loop_chain
# ---------------------------------------------------------------------------


def test_chain_three_loops():
    """Fuse a → b → c in one splice call. ``a`` is pointwise exp, ``b`` negates
    the result, ``c`` adds a bias. The merged kernel should contain exp, neg,
    add — with one Load for x (a's input) and one for bias (c's extra input)."""
    a = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="X", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="A", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    b = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="av", input="A", index=(Var("a0"),)),
                    Assign(name="y", op="negative", args=("av",)),
                    Write(output="B", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    c = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="bv", input="B", index=(Var("a0"),)),
                    Load(name="bias", input="BIAS", index=(Var("a0"),)),
                    Assign(name="y", op="add", args=("bv", "bias")),
                    Write(output="C", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )

    merged = splice_loops(
        loops={"a": a, "b": b, "c": c},
        splice_edges={("b", "A"): ("a", "A"), ("c", "B"): ("b", "B")},
    )
    assert merged is not None
    fns = _elementwise_fns(merged)
    assert "exp" in fns
    assert "negative" in fns
    assert "add" in fns
    # Two external Loads remain (X, BIAS); the splice edges collapsed into copy aliases.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 2
    assert {ld.input for ld in loads} == {"X", "BIAS"}
    assert len(tuple(merged.inputs)) == 2


# ---------------------------------------------------------------------------
# Multi-output support
# ---------------------------------------------------------------------------


def test_multi_output_root_preserves_all_writes():
    """A root whose body contains Writes at multiple output indices should
    seed each of them — the merged kernel carries all Writes."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="out_0", index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    # Consumer produces two outputs: a negation and an absolute value — both
    # reading the producer.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", input="out_0", index=(Var("a0"),)),
                    Assign(name="z0", op="negative", args=("yv",)),
                    Assign(name="z1", op="abs", args=("yv",)),
                    Write(output="C0", index=(Var("a0"),), value="z0"),
                    Write(output="C1", index=(Var("a0"),), value="z1"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="out_0")
    assert merged is not None
    writes = [s for s in merged.body.iter() if isinstance(s, Write)]
    assert {w.output for w in writes} == {"C0", "C1"}
    fns = _elementwise_fns(merged)
    # Producer chain materialized once (shared σ); both consumer ops present.
    assert fns.count("exp") == 1
    assert "negative" in fns
    assert "abs" in fns


def test_multi_output_splice_target():
    """A splice target with two outputs (output="out_0" and output="out_1") — distinct
    splice edges read each output. The target's expression chain reconstructs
    separately for each output via the standard worklist walk."""
    # Target: T0 = exp(x), T1 = neg(x). Two outputs.
    target = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="X", index=(Var("a0"),)),
                    Assign(name="y0", op="exp", args=("x",)),
                    Assign(name="y1", op="negative", args=("x",)),
                    Write(output="T0", index=(Var("a0"),), value="y0"),
                    Write(output="T1", index=(Var("a0"),), value="y1"),
                ),
            ),
        ),
    )
    # Sink: z = add(T0, T1). Two consumer Loads each pointing at one target output.
    sink = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="a", input="T0", index=(Var("a0"),)),
                    Load(name="b", input="T1", index=(Var("a0"),)),
                    Assign(name="z", op="add", args=("a", "b")),
                    Write(output="OUT", index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loops(
        loops={"target": target, "sink": sink},
        splice_edges={
            ("sink", "T0"): ("target", "T0"),
            ("sink", "T1"): ("target", "T1"),
        },
    )
    assert merged is not None
    fns = _elementwise_fns(merged)
    # Both target ops appear in the merged body once each.
    assert fns.count("exp") == 1
    assert fns.count("negative") == 1
    assert "add" in fns
    # Only one external Load (target's x); the splice edges replaced sink's Loads.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 1
    assert len(tuple(merged.inputs)) == 1


def test_shared_load_dedup_into_reduce():
    """Producer body has a Load reused across a long expression chain
    (mirrors silu: ``in = load x; v0 = neg(in); v1 = exp(v0); v2 = add(1, v1);
    v3 = recip(v2); v4 = mul(in, v3)``). The first and last uses of ``in`` sit
    at opposite ends of the chain. Splicing into a consumer that reads the
    producer's output inside a reduce loop exercises the dedup + emit-ordering
    path: when the late ``v4 = mul(in, v3)`` lands, ``in`` dedups to the
    existing binding emitted earlier for ``v0``; if the splicer prepends naïvely
    the new stmt lands above its own dep. The merged body must keep SSA in
    topological order across the reduce boundary."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="in_", input="src_0", index=(Var("a0"),)),
                    Assign(name="v0", op="negative", args=("in_",)),
                    Assign(name="v1", op="exp", args=("v0",)),
                    Assign(name="v2", op="add", args=("v1", "v1")),
                    Assign(name="v3", op="reciprocal", args=("v2",)),
                    Assign(name="v4", op="multiply", args=("in_", "v3")),
                    Write(output="out_0", index=(Var("a0"),), value="v4"),
                ),
            ),
        ),
    )
    # Consumer: matmul-style reduce over a1. The producer's output is loaded
    # inside the reduce loop, so the spliced chain has to be ordered correctly
    # within the (a0, a1) reduce scope.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=A1,
                        body=(
                            Load(name="pv", input="src_0", index=(Var("a1"),)),
                            Load(name="w", input="src_1", index=(Var("a1"), Var("a0"))),
                            Assign(name="prod", op="multiply", args=("pv", "w")),
                            Accum(name="acc", value="prod", op="add"),
                        ),
                    ),
                    Write(output="out_0", index=(Var("a0"),), value="acc"),
                ),
            ),
        ),
    )
    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None  # would have failed pre-fix with SSA validation error
    # Reduce + full silu chain + final multiply should all materialize once.
    assert _count_kind(merged, Accum) == 1
    fns = _elementwise_fns(merged)
    for op in ("negative", "exp", "add", "reciprocal", "multiply"):
        assert op in fns, f"missing {op} in merged body: {fns}"


def test_literal_producer_write_index():
    """Producer writes at (a0, 0); consumer reads at (a0, 0). The Literal dim
    contributes no σ binding and shouldn't block splicing."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", input="src_0", index=(Var("a0"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="out_0", index=(Var("a0"), Literal(0)), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", input="src_0", index=(Var("a0"), Literal(0))),
                    Assign(name="z", op="negative", args=("yv",)),
                    Write(output="out_0", index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    assert "exp" in _elementwise_fns(merged)
    assert "negative" in _elementwise_fns(merged)


def test_large_sigma_target_free_vars_walks_do_not_scale_with_producer_chain(monkeypatch):
    """A long producer chain reuses one large coordinate target without re-walking its tree."""
    from emmy.compiler.ir.expr import BinaryExpr

    producer_axis = Axis("p", 4096)
    producer_stmts = [Load(name="x", input="X", index=(Var("p"),))]
    previous = "x"
    for i in range(128):
        name = f"v{i}"
        producer_stmts.append(Assign(name=name, op="negative", args=(previous,)))
        previous = name
    producer_stmts.append(Write(output="P", index=(Var("p"),), value=previous))
    producer = LoopOp(body=(Loop(axis=producer_axis, body=tuple(producer_stmts)),))

    i_axis, j_axis = Axis("i", 8), Axis("j", 8)
    target = Var("i")
    for _ in range(128):
        target = target * 2 + Var("j")
    target = BinaryExpr("^", target, Literal(1, "int"))
    consumer = LoopOp(
        body=(
            Loop(
                axis=i_axis,
                body=(
                    Loop(
                        axis=j_axis,
                        body=(
                            Load(name="pv", input="P", index=(target,)),
                            Write(output="OUT", index=(Var("i"), Var("j")), value="pv"),
                        ),
                    ),
                ),
            ),
        )
    )

    root_walks = 0
    original = BinaryExpr.free_vars

    def counted_free_vars(expr):
        nonlocal root_walks
        if expr.op == "^":
            root_walks += 1
        return original(expr)

    monkeypatch.setattr(BinaryExpr, "free_vars", counted_free_vars)
    merged = splice_loops(loops={"producer": producer, "consumer": consumer}, splice_edges={("consumer", "P"): ("producer", "P")})

    assert merged is not None
    # Loop construction and normalization also inspect the coordinate expression, but the count
    # must not grow with the 128 producer dependencies (without the per-splice memo it exceeds 128).
    assert root_walks < 32


# ---------------------------------------------------------------------------
