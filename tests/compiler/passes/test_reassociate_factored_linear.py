"""Generic block-factor/linear reassociation tests."""

from __future__ import annotations

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import CastOp, ElementwiseOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import broadcast_to

_RULE = "033_reassociate_factored_linear"
_BLOCK = 128


def _add(graph: Graph, op, inputs, name: str, shape: tuple[int, ...], dtype: str) -> Node:
    nid = graph.add_node(op=op, inputs=inputs, output=Tensor(name, shape, dtype), node_id=name)
    return graph.nodes[nid]


def _factored_linear_graph(*, m: int = 3, k: int = 256, n: int = 128, bias: bool = False, shared_factor: bool = True) -> Graph:
    """Build the ordinary tensor algebra recognized by the rule, without loader metadata."""
    graph = Graph()
    x = _add(graph, InputOp(), [], "x", (m, k), "f16")
    core = _add(graph, InputOp(), [], "core", (k, n), "f16")
    factor = _add(graph, InputOp(), [], "factor", (_BLOCK, _BLOCK), "f64")
    factor_right = factor
    if not shared_factor:
        factor_right = _add(graph, InputOp(), [], "factor_right", (_BLOCK, _BLOCK), "f64")
    left_scale = _add(graph, InputOp(), [], "left_scale", (k,), "f16")
    right_scale = _add(graph, InputOp(), [], "right_scale", (n,), "f16")

    core64 = _add(graph, CastOp(dtype="f64"), [core], "core64", (k, n), "f64")
    left_blocks = _add(
        graph,
        ReshapeOp(shape=(k // _BLOCK, _BLOCK, n)),
        [core64],
        "left_blocks",
        (k // _BLOCK, _BLOCK, n),
        "f64",
    )
    left = _add(
        graph,
        MatmulOp(),
        [factor, left_blocks],
        "left",
        (k // _BLOCK, _BLOCK, n),
        "f64",
    )
    left_flat = _add(graph, ReshapeOp(shape=(k, n)), [left], "left_flat", (k, n), "f64")
    right_blocks = _add(
        graph,
        ReshapeOp(shape=(k, n // _BLOCK, _BLOCK)),
        [left_flat],
        "right_blocks",
        (k, n // _BLOCK, _BLOCK),
        "f64",
    )
    restored_blocks = _add(
        graph,
        MatmulOp(),
        [right_blocks, factor_right],
        "restored_blocks",
        (k, n // _BLOCK, _BLOCK),
        "f64",
    )
    restored = _add(graph, ReshapeOp(shape=(k, n)), [restored_blocks], "restored", (k, n), "f64")

    left64 = _add(graph, CastOp(dtype="f64"), [left_scale], "left64", (k,), "f64")
    left_col = _add(graph, ReshapeOp(shape=(k, 1)), [left64], "left_col", (k, 1), "f64")
    left_bc = broadcast_to(graph, left_col, (k, n))
    scaled_left = _add(graph, ElementwiseOp(op="multiply"), [restored, left_bc], "scaled_left", (k, n), "f64")

    right64 = _add(graph, CastOp(dtype="f64"), [right_scale], "right64", (n,), "f64")
    right_row = _add(graph, ReshapeOp(shape=(1, n)), [right64], "right_row", (1, n), "f64")
    right_bc = broadcast_to(graph, right_row, (k, n))
    scaled = _add(graph, ElementwiseOp(op="multiply"), [scaled_left, right_bc], "scaled", (k, n), "f64")
    rounded = _add(graph, CastOp(dtype="f16"), [scaled], "rounded", (k, n), "f16")
    weight = _add(graph, TransposeOp(axes=(1, 0)), [rounded], "weight", (n, k), "f16")

    inputs = [x, weight]
    graph_inputs = [x.id, core.id, factor.id, left_scale.id, right_scale.id]
    if not shared_factor:
        graph_inputs.append(factor_right.id)
    if bias:
        bias_node = _add(graph, InputOp(), [], "bias", (n,), "f16")
        inputs.append(bias_node)
        graph_inputs.append(bias_node.id)
    _add(graph, LinearOp(has_bias=bias), inputs, "out", (m, n), "f16")
    graph.inputs, graph.outputs = graph_inputs, ["out"]
    return graph


def _apply(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition"], select=[_RULE]).run(graph)


def _factor() -> np.ndarray:
    factor = np.ones((1, 1), dtype=np.float64)
    while factor.shape[0] < _BLOCK:
        factor = np.block([[factor, factor], [factor, -factor]])
    return factor / np.sqrt(_BLOCK)


def _inputs(*, m: int = 3, k: int = 256, n: int = 128, bias: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)
    values = {
        "x": (rng.standard_normal((m, k)) * 0.1).astype(np.float16),
        "core": (rng.standard_normal((k, n)) * 0.2).astype(np.float16),
        "factor": _factor(),
        "left_scale": (rng.standard_normal(k) * 0.01).astype(np.float16),
        "right_scale": rng.choice([-1.0, 1.0], n).astype(np.float16),
    }
    if bias:
        values["bias"] = (rng.standard_normal(n) * 0.01).astype(np.float16)
    return values


def test_reassociation_removes_dense_reconstruction_and_stays_generic():
    result = _apply(_factored_linear_graph())

    assert not any(isinstance(node.op, LinearOp) for node in result.nodes.values())
    matmuls = [node for node in result.nodes.values() if isinstance(node.op, MatmulOp)]
    assert [tuple(d.as_static() for d in node.output.shape) for node in matmuls] == [
        (3, 2, 1, 128),
        (3, 128),
        (3, 1, 1, 128),
    ]
    # The 256x128 dense f64 reconstruction is gone.  The only f64 matrix retained is the
    # shared 128x128 factor, which the rewrite casts once to the fp16 contraction dtype.
    assert not any(
        node.output.dtype.name == "f64" and tuple(d.as_static() for d in node.output.shape) == (256, 128) for node in result.nodes.values()
    )
    active = "\n".join(f"{type(node.op).__module__}.{type(node.op).__name__}" for node in result.nodes.values())
    assert "trellis" not in active.lower() and "exl3" not in active.lower()
    # Precision boundaries created by this rewrite remain typed generic
    # copies so they can fuse into the adjacent contractions. Frontend
    # CastOps elsewhere retain their ordinary backend/interpreter semantics.
    assert not any(isinstance(node.op, CastOp) for node in result.nodes.values())
    copies = [node for node in result.nodes.values() if isinstance(node.op, ElementwiseOp) and node.op.name == "copy"]
    assert {node.output.dtype.name for node in copies} == {"f16", "f32"}


def test_reassociation_tracks_materialized_fp16_weight_error():
    graph = _factored_linear_graph(bias=True)
    inputs = _inputs(bias=True)
    backend = NumpyBackend()
    reference = backend.run(graph, input_data=inputs)[0].outputs["out"].astype(np.float32)
    actual = backend.run(_apply(graph), input_data=inputs)[0].outputs["out"].astype(np.float32)

    error = actual - reference
    relative_rms = float(np.sqrt(np.mean(error * error)) / np.sqrt(np.mean(reference * reference)))
    relative_max = float(np.max(np.abs(error)) / np.max(np.abs(reference)))
    assert relative_rms < 3e-3
    assert relative_max < 3e-3


def test_reassociation_requires_one_shared_factor():
    graph = _factored_linear_graph(shared_factor=False)
    result = _apply(graph)
    assert any(isinstance(node.op, LinearOp) for node in result.nodes.values())


def test_reassociation_rejects_shared_dense_interior():
    graph = _factored_linear_graph()
    restored = graph.nodes["restored"]
    _add(graph, ElementwiseOp(op="negative"), [restored], "external", (256, 128), "f64")
    graph.outputs.append("external")

    result = _apply(graph)
    assert any(isinstance(node.op, LinearOp) for node in result.nodes.values())
