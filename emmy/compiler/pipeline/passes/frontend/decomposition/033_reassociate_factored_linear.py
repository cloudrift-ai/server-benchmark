"""Reassociate a block-factored weight reconstruction through ``LinearOp``.

This rule is deliberately representation-agnostic.  It recognizes only the ordinary tensor
algebra

``W = cast((D_left * reshape((H @ core) @ H)) * D_right).T``

where ``H`` is a shared 128x128 factor and the diagonal vectors arrive through explicit
broadcast index maps.  A linear consuming that weight can evaluate the equivalent factored
contraction

``x @ W.T = (((x * D_left) @ H) @ core) @ H * D_right``

without ever constructing the much larger dense ``W`` buffer.  Every emitted node is still an
ordinary cast, reshape, elementwise operation, or matmul; no storage-format operation or hint
crosses the frontend boundary.

The stored reconstruction rounds the complete matrix to fp16 before the linear.  Reassociation
moves that rounding boundary, so this optimization is intentionally restricted to an fp16
activation/core/result contract and spells the precision boundaries explicitly: scale and the
first block factor run in fp32, one cast supplies the fp16 tensor-core A operand, the core
contraction accumulates into fp32, and the second factor plus output scale remain fp32 until one
final cast.  Quantized-runtime parity tests pin the resulting error envelope against the
materialized-weight spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import Literal, placeholder
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import CastOp, ElementwiseOp, IndexMapOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import broadcast_to, open_fragment

_BLOCK = 128
PATTERN = [Pattern("linear", LinearOp)]


def _shape(node: Node) -> tuple[int, ...] | None:
    if any(not d.is_static for d in node.output.shape):
        return None
    return tuple(d.as_static() for d in node.output.shape)


def _input(graph: Graph, node: Node, index: int = 0) -> Node | None:
    if index >= len(node.inputs):
        return None
    return graph.producer(node.inputs[index])


def _is_op(node: Node | None, cls, *, shape: tuple[int, ...] | None = None, dtype: str | None = None) -> bool:
    return (
        node is not None
        and isinstance(node.op, cls)
        and (shape is None or _shape(node) == shape)
        and (dtype is None or node.output.dtype.name == dtype)
    )


def _is_elementwise(node: Node | None, name: str, *, shape: tuple[int, ...], dtype: str) -> bool:
    return _is_op(node, ElementwiseOp, shape=shape, dtype=dtype) and node.op.name == name


@dataclass(frozen=True)
class _BroadcastVector:
    leaf: Node
    interior: frozenset[str]


def _broadcast_vector(
    graph: Graph,
    node: Node | None,
    *,
    matrix_shape: tuple[int, int],
    vector_len: int,
    axis: int,
) -> _BroadcastVector | None:
    """Match ``cast(vector)->reshape->explicit broadcast`` for one diagonal factor."""
    if not _is_op(node, IndexMapOp, shape=matrix_shape, dtype="f64") or len(node.inputs) != 1:
        return None
    sources = node.op.sources
    if len(sources) != 1 or sources[0].input_idx != 0 or sources[0].select is not None:
        return None
    expected_coords = (placeholder(0), Literal(0, "int")) if axis == 0 else (Literal(0, "int"), placeholder(1))
    if sources[0].coord_map != expected_coords:
        return None

    reshaped = _input(graph, node)
    expected_reshape = (vector_len, 1) if axis == 0 else (1, vector_len)
    if not _is_op(reshaped, ReshapeOp, shape=expected_reshape, dtype="f64"):
        return None
    cast = _input(graph, reshaped)
    if not _is_op(cast, CastOp, shape=(vector_len,), dtype="f64") or cast.op.dtype != "f64":
        return None
    leaf = _input(graph, cast)
    if leaf is None or _shape(leaf) != (vector_len,) or leaf.output.dtype.name != "f16":
        return None
    return _BroadcastVector(leaf=leaf, interior=frozenset((node.id, reshaped.id, cast.id)))


@dataclass(frozen=True)
class _FactoredWeight:
    core: Node
    factor: Node
    left_scale: Node
    right_scale: Node
    interior: frozenset[str]
    k: int
    n: int


def _multiply_with_vector(
    graph: Graph,
    node: Node | None,
    *,
    matrix_shape: tuple[int, int],
    vector_len: int,
    axis: int,
) -> tuple[Node, _BroadcastVector] | None:
    if not _is_elementwise(node, "multiply", shape=matrix_shape, dtype="f64") or len(node.inputs) != 2:
        return None
    for matrix_index, vector_index in ((0, 1), (1, 0)):
        matrix = _input(graph, node, matrix_index)
        vector = _broadcast_vector(
            graph,
            _input(graph, node, vector_index),
            matrix_shape=matrix_shape,
            vector_len=vector_len,
            axis=axis,
        )
        if matrix is not None and vector is not None:
            return matrix, vector
    return None


def _factored_weight(graph: Graph, weight: Node | None) -> _FactoredWeight | None:
    """Recognize the exact generic factor graph, independent of names and provenance."""
    if not _is_op(weight, TransposeOp, dtype="f16") or weight.op.axes not in ((1, 0), (-1, -2)):
        return None
    weight_shape = _shape(weight)
    if weight_shape is None or len(weight_shape) != 2:
        return None
    n, k = weight_shape
    if not n or not k or n % _BLOCK or k % _BLOCK:
        return None
    matrix_shape = (k, n)

    rounded = _input(graph, weight)
    if not _is_op(rounded, CastOp, shape=matrix_shape, dtype="f16") or rounded.op.dtype != "f16":
        return None
    outer = _input(graph, rounded)
    outer_match = _multiply_with_vector(graph, outer, matrix_shape=matrix_shape, vector_len=n, axis=1)
    if outer_match is None:
        return None
    inner, right = outer_match
    inner_match = _multiply_with_vector(graph, inner, matrix_shape=matrix_shape, vector_len=k, axis=0)
    if inner_match is None:
        return None
    restored, left = inner_match

    if not _is_op(restored, ReshapeOp, shape=matrix_shape, dtype="f64"):
        return None
    right_mm = _input(graph, restored)
    if not _is_op(right_mm, MatmulOp, shape=(k, n // _BLOCK, _BLOCK), dtype="f64") or len(right_mm.inputs) != 2:
        return None
    right_blocks, factor_right = _input(graph, right_mm, 0), _input(graph, right_mm, 1)
    if not _is_op(right_blocks, ReshapeOp, shape=(k, n // _BLOCK, _BLOCK), dtype="f64"):
        return None
    left_flat = _input(graph, right_blocks)
    if not _is_op(left_flat, ReshapeOp, shape=matrix_shape, dtype="f64"):
        return None
    left_mm = _input(graph, left_flat)
    if not _is_op(left_mm, MatmulOp, shape=(k // _BLOCK, _BLOCK, n), dtype="f64") or len(left_mm.inputs) != 2:
        return None
    factor_left, left_blocks = _input(graph, left_mm, 0), _input(graph, left_mm, 1)
    if factor_left is None or factor_left.id != getattr(factor_right, "id", None):
        return None
    if _shape(factor_left) != (_BLOCK, _BLOCK) or factor_left.output.dtype.name != "f64":
        return None
    if not _is_op(left_blocks, ReshapeOp, shape=(k // _BLOCK, _BLOCK, n), dtype="f64"):
        return None
    core_cast = _input(graph, left_blocks)
    if not _is_op(core_cast, CastOp, shape=matrix_shape, dtype="f64") or core_cast.op.dtype != "f64":
        return None
    core = _input(graph, core_cast)
    if core is None or _shape(core) != matrix_shape or core.output.dtype.name != "f16":
        return None

    interior = {
        weight.id,
        rounded.id,
        outer.id,
        inner.id,
        restored.id,
        right_mm.id,
        right_blocks.id,
        left_flat.id,
        left_mm.id,
        left_blocks.id,
        core_cast.id,
        *left.interior,
        *right.interior,
    }
    return _FactoredWeight(
        core=core,
        factor=factor_left,
        left_scale=left.leaf,
        right_scale=right.leaf,
        interior=frozenset(interior),
        k=k,
        n=n,
    )


def _matmul(frag: Graph, a: Node | str, b: Node | str, *, name: str, shape: tuple, dtype: str) -> Node:
    nid = frag.add_node(op=MatmulOp(), inputs=[a, b], output=Tensor(name, shape, dtype))
    return frag.nodes[nid]


def _reshape(frag: Graph, x: Node | str, *, name: str, shape: tuple) -> Node:
    node = x if isinstance(x, Node) else frag.nodes[x]
    nid = frag.add_node(op=ReshapeOp(shape=shape), inputs=[node], output=Tensor(name, shape, node.output.dtype))
    return frag.nodes[nid]


def rewrite(match: Match, linear: Node, inp_x: Node, inp_w: Node, inp_b: Node | None, out: Tensor) -> Graph:
    factored = _factored_weight(match.graph, inp_w)
    x_shape = _shape(inp_x)
    if factored is None or x_shape is None or not x_shape or x_shape[-1] != factored.k:
        raise RuleSkipped("linear weight is not an owned fp16 block-factor reconstruction")
    if inp_x.output.dtype.name != "f16" or out.dtype.name != "f16":
        raise RuleSkipped("factored linear currently requires fp16 activation and result")

    consumed = set(factored.interior) | {linear.id}
    for nid in factored.interior:
        if nid in match.graph.outputs or not match.graph.users(nid) <= consumed:
            raise RuleSkipped(f"factor interior node {nid!r} is shared outside the linear")

    exts = [inp_x, factored.core, factored.factor, factored.left_scale, factored.right_scale]
    if inp_b is not None:
        exts.append(inp_b)
    frag = open_fragment(match.graph, exts)

    factor32_id = frag.add_node(
        op=CastOp(dtype="f32"),
        inputs=[factored.factor],
        output=Tensor(f"{out.name}_factor32", (_BLOCK, _BLOCK), "f32"),
    )
    factor32 = frag.nodes[factor32_id]

    x32_id = frag.add_node(op=CastOp(dtype="f32"), inputs=[inp_x], output=Tensor(f"{out.name}_x32", x_shape, "f32"))
    x32 = frag.nodes[x32_id]
    left_scale32_id = frag.add_node(
        op=CastOp(dtype="f32"),
        inputs=[factored.left_scale],
        output=Tensor(f"{out.name}_left_scale32", (factored.k,), "f32"),
    )
    left_bc = broadcast_to(frag, frag.nodes[left_scale32_id], x_shape)
    scaled_x_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x32, left_bc],
        output=Tensor(f"{out.name}_left_scaled", x_shape, "f32"),
    )
    scaled_x = frag.nodes[scaled_x_id]
    left_block_shape = x_shape[:-1] + (factored.k // _BLOCK, 1, _BLOCK)
    left_blocks = _reshape(frag, scaled_x, name=f"{out.name}_left_blocks", shape=left_block_shape)
    left = _matmul(frag, left_blocks, factor32, name=f"{out.name}_left_factor", shape=left_block_shape, dtype="f32")
    left_flat32 = _reshape(frag, left, name=f"{out.name}_left_flat32", shape=x_shape)
    left_flat_id = frag.add_node(
        op=CastOp(dtype="f16"),
        inputs=[left_flat32],
        output=Tensor(f"{out.name}_left_flat", x_shape, "f16"),
    )
    left_flat = frag.nodes[left_flat_id]

    core_shape = x_shape[:-1] + (factored.n,)
    core_out = _matmul(frag, left_flat, factored.core, name=f"{out.name}_core", shape=core_shape, dtype="f32")
    right_block_shape = core_shape[:-1] + (factored.n // _BLOCK, 1, _BLOCK)
    right_blocks = _reshape(frag, core_out, name=f"{out.name}_right_blocks", shape=right_block_shape)
    right = _matmul(frag, right_blocks, factor32, name=f"{out.name}_right_factor", shape=right_block_shape, dtype="f32")
    right_flat = _reshape(frag, right, name=f"{out.name}_right_flat", shape=core_shape)

    right_scale32_id = frag.add_node(
        op=CastOp(dtype="f32"),
        inputs=[factored.right_scale],
        output=Tensor(f"{out.name}_right_scale32", (factored.n,), "f32"),
    )
    right_bc = broadcast_to(frag, frag.nodes[right_scale32_id], core_shape)
    scaled_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[right_flat, right_bc],
        output=Tensor(f"{out.name}_scaled32", core_shape, "f32"),
    )
    scaled32 = frag.nodes[scaled_id]
    if inp_b is not None:
        bias32_id = frag.add_node(
            op=CastOp(dtype="f32"),
            inputs=[inp_b],
            output=Tensor(f"{out.name}_bias32", (factored.n,), "f32"),
        )
        bias_bc = broadcast_to(frag, frag.nodes[bias32_id], core_shape)
        biased_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[scaled32, bias_bc],
            output=Tensor(f"{out.name}_biased32", core_shape, "f32"),
        )
        scaled32 = frag.nodes[biased_id]
    final_id = frag.add_node(
        op=CastOp(dtype="f16"),
        inputs=[scaled32],
        output=Tensor(out.name, core_shape, "f16"),
    )
    final = frag.nodes[final_id]
    frag.outputs = [final.id]
    match.consumed = consumed
    return frag
