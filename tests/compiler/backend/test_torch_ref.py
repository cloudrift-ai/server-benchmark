"""torch_ref: the Graph→torch evaluator used as the eager reference for
``emmy run --ir``. Validated against each op's numpy ``forward()`` on
CPU (no GPU needed)."""

import numpy as np
import pytest

from emmy.compiler.backend import torch_ref
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, placeholder
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SoftmaxOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, IndexMapOp, IndexSource, RangeOp, ReduceOp, ScanOp

# torch is only needed to build reference tensors / call the evaluator; the
# emmy imports above are torch-free, so gate after them.
torch = pytest.importorskip("torch")


def _assert_matches_numpy(g: Graph, arrays: dict[str, np.ndarray]):
    """torch_ref output == numpy forward() output for graph ``g``."""
    be = NumpyBackend()
    npy = be.run(be.compile(g), input_data=arrays)[0].outputs[g.outputs[0]]

    tin = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
    fn, inputs = torch_ref.build_callable(g, tin)
    with torch.no_grad():
        tout = fn(*inputs).cpu().numpy()

    np.testing.assert_allclose(tout, npy, rtol=1e-4, atol=1e-4)


def _rng():
    return np.random.default_rng(0)


def test_rms_norm():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("o", (1, 4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"x": r.standard_normal((1, 4, 8)), "w": r.standard_normal((8,))})


def test_linear_and_elementwise():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (16, 8)), node_id="w")
    g.add_node(LinearOp(), ["x", "w"], Tensor("h", (4, 16)), node_id="h")
    g.add_node(ElementwiseOp(op="silu"), ["h"], Tensor("o", (4, 16)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"x": r.standard_normal((4, 8)), "w": r.standard_normal((16, 8))})


def test_multi_output_preserves_declared_order_and_single_output_tensor_contract():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 3)), node_id="x")
    g.add_node(ElementwiseOp(op="multiply"), ["x", "x"], Tensor("intermediate", (2, 3)), node_id="intermediate")
    g.add_node(ElementwiseOp(op="silu"), ["intermediate"], Tensor("hidden", (2, 3)), node_id="hidden")
    g.add_node(ElementwiseOp(op="add"), ["hidden", "x"], Tensor("final", (2, 3)), node_id="final")
    g.inputs = ["x"]
    # A working golden may expose a live intermediate after the semantic final
    # output so exact target replay can validate both buffers.
    g.outputs = ["final", "intermediate"]
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    expected = NumpyBackend().run(g, input_data={"x": x})[0].outputs
    fn, inputs = torch_ref.build_callable(g, {"x": torch.from_numpy(x)})

    actual = fn(*inputs)

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    for value, output_name in zip(actual, g.outputs, strict=True):
        np.testing.assert_allclose(value.numpy(), expected[output_name], rtol=1e-4, atol=1e-4)

    g.outputs = ["final"]
    single_fn, single_inputs = torch_ref.build_callable(g, {"x": torch.from_numpy(x)})
    single = single_fn(*single_inputs)

    assert torch.is_tensor(single)
    np.testing.assert_allclose(single.numpy(), expected["final"], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("dtype_name", "torch_dtype"), [("f16", torch.float16), ("f32", torch.float32)])
def test_zero_width_pad_is_runnable_identity(dtype_name, torch_dtype):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 3), dtype_name), node_id="x")
    g.add_node(ElementwiseOp(op="pad"), ["x"], Tensor("out", (2, 3), dtype_name), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    expected = torch.arange(6, dtype=torch_dtype).reshape(2, 3)

    assert torch_ref.is_runnable(g)
    fn, inputs = torch_ref.build_callable(g, {"x": expected})
    actual = fn(*inputs)

    assert actual.dtype == torch_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_source_free_range_is_runnable_on_cpu():
    g = Graph()
    g.add_node(RangeOp(stop=4, dtype="i32"), [], Tensor("axis", (4,), "i32"), node_id="axis")
    g.inputs, g.outputs = [], ["axis"]

    assert torch_ref.is_runnable(g)
    fn, inputs = torch_ref.build_callable(g, {})

    assert inputs == []
    torch.testing.assert_close(fn(), torch.arange(4, dtype=torch.int32), rtol=0, atol=0)


def test_scan_sum_is_runnable_and_matches_torch():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4)), node_id="x")
    g.add_node(ScanOp(op="sum", axis=-1), ["x"], Tensor("out", (2, 4)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    assert torch_ref.is_runnable(g)
    fn, inputs = torch_ref.build_callable(g, {"x": x})
    actual = fn(*inputs)

    torch.testing.assert_close(actual, torch.cumsum(x, dim=-1), rtol=0, atol=0)


def test_where_preserves_bool_condition_dtype_and_broadcasts():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("condition", (2, 1), "bool"), node_id="condition")
    g.add_node(InputOp(), [], Tensor("left", (2, 3)), node_id="left")
    g.add_node(InputOp(), [], Tensor("right", (1, 3)), node_id="right")
    g.add_node(ElementwiseOp(op="where"), ["condition", "left", "right"], Tensor("out", (2, 3)), node_id="out")
    g.inputs, g.outputs = ["condition", "left", "right"], ["out"]
    arrays = {
        "condition": np.array([[True], [False]]),
        "left": np.arange(6, dtype=np.float32).reshape(2, 3),
        "right": np.array([[10, 20, 30]], dtype=np.float32),
    }
    expected = NumpyBackend().run(g, input_data=arrays)[0].outputs["out"]
    tensors = {name: torch.from_numpy(array) for name, array in arrays.items()}
    fn, inputs = torch_ref.build_callable(g, tensors)

    actual = fn(*inputs)

    assert torch_ref.is_runnable(g)
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 3)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(("dtype_name", "value", "expected_dtype"), [("bool", 1.0, torch.bool), ("i64", 7.0, torch.int64)])
def test_indexmap_scalar_preserves_declared_dtype(dtype_name, value, expected_dtype):
    g = Graph()
    g.add_node(ConstantOp(name="scalar", value=value), [], Tensor("scalar", (1,), dtype_name), node_id="scalar")
    g.add_node(
        IndexMapOp(out_shape=(2, 2), sources=(IndexSource(input_idx=0, coord_map=(Literal(0, "int"),)),)),
        ["scalar"],
        Tensor("out", (2, 2), dtype_name),
        node_id="out",
    )
    g.outputs = ["out"]

    fn, inputs = torch_ref.build_callable(g, {})
    actual = fn(*inputs)

    assert torch_ref.is_runnable(g)
    assert actual.dtype == expected_dtype
    assert actual.tolist() == [[int(value), int(value)], [int(value), int(value)]]


def test_where_accepts_bool_scalar_broadcast_through_triangular_indexmap():
    g = Graph()
    g.add_node(ConstantOp(name="one", value=1.0), [], Tensor("one", (1,), "bool"), node_id="one")
    g.add_node(ConstantOp(name="zero", value=0.0), [], Tensor("zero", (1,), "bool"), node_id="zero")
    g.add_node(
        IndexMapOp(
            out_shape=(2, 2),
            sources=(
                IndexSource(
                    input_idx=0,
                    coord_map=(Literal(0, "int"),),
                    select=BinaryExpr(">=", placeholder(1), placeholder(0)),
                ),
                IndexSource(input_idx=1, coord_map=(Literal(0, "int"),)),
            ),
        ),
        ["one", "zero"],
        Tensor("condition", (2, 2), "bool"),
        node_id="condition",
    )
    g.add_node(InputOp(), [], Tensor("left", (2, 2)), node_id="left")
    g.add_node(InputOp(), [], Tensor("right", (2, 2)), node_id="right")
    g.add_node(ElementwiseOp(op="where"), ["condition", "left", "right"], Tensor("out", (2, 2)), node_id="out")
    g.inputs, g.outputs = ["left", "right"], ["out"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    left = torch.full((2, 2), 1.0, device=device)
    right = torch.full((2, 2), -1.0, device=device)

    fn, inputs = torch_ref.build_callable(g, {"left": left, "right": right})
    actual = fn(*inputs)

    assert torch_ref.is_runnable(g)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=device), rtol=0, atol=0)


def test_declared_dtype_cast_is_enforced():
    """The trace folds HF's explicit casts (e.g. the fp32 RMSNorm body casting
    back to fp16) into each node's declared output dtype; ``build_callable``
    must cast accordingly — torch promotion alone would carry the f16×f32 mix
    forward at fp32 and ``F.linear`` would reject the f32×f16 operands."""
    from emmy.compiler.dtype import F16, F32

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("s", (4, 8), F32), node_id="s")
    g.add_node(ElementwiseOp(op="multiply"), ["x", "s"], Tensor("m", (4, 8), F16), node_id="m")  # declared f16 = the cast
    g.add_node(InputOp(), [], Tensor("w", (16, 8), F16), node_id="w")
    g.add_node(LinearOp(), ["m", "w"], Tensor("o", (4, 16), F16), node_id="o")
    g.inputs, g.outputs = ["x", "s", "w"], ["o"]

    tin = {
        "x": torch.randn(4, 8, dtype=torch.float16),
        "s": torch.randn(4, 8),
        "w": torch.randn(16, 8, dtype=torch.float16),
    }
    fn, inputs = torch_ref.build_callable(g, tin)
    with torch.no_grad():
        out = fn(*inputs)
    assert out.dtype == torch.float16


def _dynamic_w8a8_graph() -> Graph:
    """A post-spelling dynamic-activation FP8 projection target."""
    from emmy.compiler.dtype import F8E4M3, F16, F32

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("x_scale", (4, 1), F32), node_id="x_scale")
    g.add_node(ElementwiseOp(op="divide"), ["x", "x_scale"], Tensor("x_normalized", (4, 8), F32), node_id="x_normalized")
    g.add_node(ElementwiseOp(op="to_f8e4m3"), ["x_normalized"], Tensor("x_bits", (4, 8), F8E4M3), node_id="x_bits")
    g.add_node(ElementwiseOp(op="from_f8e4m3"), ["x_bits"], Tensor("x_decoded", (4, 8), F16), node_id="x_decoded")
    g.add_node(ElementwiseOp(op="multiply"), ["x_decoded", "x_scale"], Tensor("x_quantized", (4, 8), F16), node_id="x_quantized")
    g.add_node(InputOp(), [], Tensor("w_bits", (6, 8), F8E4M3), node_id="w_bits")
    g.add_node(InputOp(), [], Tensor("w_scale", (6, 1), F32), node_id="w_scale")
    g.add_node(ElementwiseOp(op="from_f8e4m3"), ["w_bits"], Tensor("w_decoded", (6, 8), F16), node_id="w_decoded")
    g.add_node(ElementwiseOp(op="multiply"), ["w_decoded", "w_scale"], Tensor("w_quantized", (6, 8), F16), node_id="w_quantized")
    g.add_node(LinearOp(), ["x_quantized", "w_quantized"], Tensor("o", (4, 6), F16), node_id="o")
    g.inputs, g.outputs = ["x", "x_scale", "w_bits", "w_scale"], ["o"]
    return g


def _dynamic_w8a8_inputs():
    from emmy.compiler.dtype import encode_f8

    x = torch.linspace(-2.0, 2.0, 32, dtype=torch.float16).reshape(4, 8)
    x_scale = x.abs().amax(dim=-1, keepdim=True).float() / 448.0
    weight = np.linspace(-3.0, 3.0, 48, dtype=np.float32).reshape(6, 8)
    w_scale = np.max(np.abs(weight), axis=-1, keepdims=True).astype(np.float32) / 448.0
    w_bits = encode_f8(weight / w_scale, "f8e4m3")
    return {"x": x, "x_scale": x_scale, "w_bits": torch.from_numpy(w_bits), "w_scale": torch.from_numpy(w_scale)}


def test_dynamic_w8a8_eager_preserves_fp8_bit_semantics():
    g = _dynamic_w8a8_graph()
    tensors = _dynamic_w8a8_inputs()
    fn, inputs = torch_ref.build_callable(g, tensors)

    out = fn(*inputs)
    x_bits = (tensors["x"] / tensors["x_scale"]).to(torch.float8_e4m3fn).view(torch.uint8)
    x_quantized = (x_bits.view(torch.float8_e4m3fn).to(torch.float16) * tensors["x_scale"]).to(torch.float16)
    w_quantized = (tensors["w_bits"].view(torch.float8_e4m3fn).to(torch.float16) * tensors["w_scale"]).to(torch.float16)
    expected = torch.nn.functional.linear(x_quantized, w_quantized)

    assert torch_ref.is_runnable(g)
    assert tensors["w_bits"].dtype == torch.uint8
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_dynamic_w8a8_torch_compile_is_fullgraph():
    g = _dynamic_w8a8_graph()
    fn, inputs = torch_ref.build_callable(g, _dynamic_w8a8_inputs())
    eager = fn(*inputs)

    compiled = torch.compile(fn, fullgraph=True)
    actual = compiled(*inputs)

    torch.testing.assert_close(actual, eager, rtol=1e-3, atol=1e-3)


def test_matmul_softmax():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 4)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("s", (4, 4)), node_id="s")
    g.add_node(SoftmaxOp(axis=-1), ["s"], Tensor("o", (4, 4)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"a": r.standard_normal((4, 8)), "b": r.standard_normal((8, 4))})


def test_reduce_sum_keepdim():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", (4, 1)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    _assert_matches_numpy(g, {"x": _rng().standard_normal((4, 8))})


def _imap_graph(in_shapes, out_shape, sources) -> Graph:
    g = Graph()
    names = [f"in{i}" for i in range(len(in_shapes))]
    for n, shp in zip(names, in_shapes, strict=True):
        g.add_node(InputOp(), [], Tensor(n, shp), node_id=n)
    g.add_node(IndexMapOp(out_shape=out_shape, sources=sources), names, Tensor("o", out_shape), node_id="o")
    g.inputs, g.outputs = names, ["o"]
    return g


def test_indexmap_transpose():
    # output (4,3)[a0,a1] = in0[a1,a0]
    g = _imap_graph([(3, 4)], (4, 3), (IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),))
    _assert_matches_numpy(g, {"in0": _rng().standard_normal((3, 4))})


def test_indexmap_broadcast():
    # (8,) → (4,8): every row reads in0[a1]
    g = _imap_graph([(8,)], (4, 8), (IndexSource(input_idx=0, coord_map=(placeholder(1),)),))
    _assert_matches_numpy(g, {"in0": _rng().standard_normal((8,))})


def test_indexmap_cat_with_select():
    # output (4,4): a1<2 → in0[a0,a1]; a1>=2 → in1[a0,a1-2]
    s0 = IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1)), select=placeholder(1).lt(Literal(2, "int")))
    s1 = IndexSource(
        input_idx=1,
        coord_map=(placeholder(0), placeholder(1) - Literal(2, "int")),
        select=BinaryExpr(">=", placeholder(1), Literal(2, "int")),
    )
    g = _imap_graph([(4, 2), (4, 2)], (4, 4), (s0, s1))
    _assert_matches_numpy(g, {"in0": _rng().standard_normal((4, 2)), "in1": _rng().standard_normal((4, 2))})


def test_indexmap_recurrence_patch_uses_tensor_ternary_coordinates():
    # Insert a two-cell recurrence update into columns 1:3 of a larger carried state.
    column = placeholder(1)
    bounds = BinaryExpr("&&", BinaryExpr(">=", column, Literal(1, "int")), column.lt(Literal(3, "int")))
    in_patch = BinaryExpr("&&", Literal(True, "bool"), bounds)
    patch_column = TernaryExpr(in_patch, column - Literal(1, "int"), Literal(0, "int"))
    sources = (
        IndexSource(input_idx=0, coord_map=(placeholder(0), patch_column), select=in_patch),
        IndexSource(input_idx=1, coord_map=(placeholder(0), column)),
    )
    g = _imap_graph([(2, 2), (2, 4)], (2, 4), sources)
    patch = np.array([[10, 11], [20, 21]], dtype=np.float32)
    carried = np.arange(8, dtype=np.float32).reshape(2, 4)

    _assert_matches_numpy(g, {"in0": patch, "in1": carried})


@pytest.mark.parametrize(("condition", "expected"), [(1, 7), (0, 9)])
def test_index_expr_ternary_preserves_scalar_condition(condition, expected):
    expr = TernaryExpr(Literal(condition, "int"), Literal(7, "int"), Literal(9, "int"))

    assert torch_ref._idx_expr(expr, {}) == expected


def test_symbolic_shapes_resolve_from_input_tensors():
    """A symbolic-``Dim`` graph (dynamic-trace reproducer shape) evaluates with
    concrete tensors: ``build_callable`` binds ``seq_len`` from the supplied
    tensor's shape, and shape-resolving ops (``ReshapeOp`` target) eval through
    that env instead of raising on ``as_static``."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.frontend.ir import ReshapeOp

    s = Dim("seq_len")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(1), s, Dim(8))), node_id="x")
    g.add_node(ElementwiseOp(op="exp"), ["x"], Tensor("e", (Dim(1), s, Dim(8))), node_id="e")
    g.add_node(ReshapeOp(shape=(1, -1)), ["e"], Tensor("o", (Dim(1), s * Dim(8))), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]

    x = torch.randn(1, 6, 8)
    fn, inputs = torch_ref.build_callable(g, {"x": x})
    with torch.no_grad():
        out = fn(*inputs)
    assert out.shape == (1, 48)
    np.testing.assert_allclose(out.numpy(), torch.exp(x).reshape(1, 48).numpy(), rtol=1e-5, atol=1e-6)


def test_is_runnable_accepts_indexmap():
    g = _imap_graph([(8,)], (4, 8), (IndexSource(input_idx=0, coord_map=(placeholder(1),)),))
    assert torch_ref.is_runnable(g)


def test_is_runnable_rejects_gather():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("idx", (4, 8)), node_id="idx")
    g.add_node(GatherOp(axis=0), ["x", "idx"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "idx"], ["o"]
    assert not torch_ref.is_runnable(g)


def test_is_runnable_accepts_frontend():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("o", (1, 4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    assert torch_ref.is_runnable(g)


def test_is_runnable_rejects_unmapped_elementwise():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ElementwiseOp(op="square"), ["x"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    assert not torch_ref.is_runnable(g)
