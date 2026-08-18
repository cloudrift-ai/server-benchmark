import numpy as np
import pytest

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.loader.physical import spell_physical_inputs
from emmy.compiler.loader.sm70_fp8 import expected_sm70_fp8_metadata, retained_sm70_fp8_storage
from emmy.compiler.tensor import Tensor

_PROFILES = (
    (1536, 4096, False, None),
    (4096, 1024, False, None),
    (1024, 4096, False, 0),
    (8192, 1024, False, None),
    (512, 4096, True, None),
    (4096, 256, False, None),
)


def _evaluate(coords, row, col):
    env = {
        f"{PLACEHOLDER_PREFIX}0": np.asarray(row, dtype=np.int64),
        f"{PLACEHOLDER_PREFIX}1": np.asarray(col, dtype=np.int64),
    }
    return tuple(np.asarray(expr.eval(env), dtype=np.int64) for expr in coords)


def _projection_graph(m, n, k, interleaved, group):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (m, k), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("weight", (n, k), "f16"), node_id="weight")
    graph.add_node(LinearOp(), ["x", "weight"], Tensor("output", (m, n), "f16"), node_id="output")
    graph.inputs = ["x", "weight"]
    graph.outputs = ["output"]
    weight_shape = (k, n) if group is None else (group + 1, k, n)
    scale_shape = (k // 128, n) if group is None else (group + 1, k // 128, n)
    storage = retained_sm70_fp8_storage(
        (n, k),
        weight_shape=weight_shape,
        scale_shape=scale_shape,
        metadata=expected_sm70_fp8_metadata((n, k)),
        interleave_halves=interleaved,
        group_index=group,
    )
    spell_physical_inputs(graph, {"weight": storage})
    graph.validate()
    return graph


@pytest.mark.parametrize(("n", "k", "interleaved", "group"), _PROFILES)
def test_all_live_profiles_validate_exported_carrier_shapes_and_metadata(n, k, interleaved, group):
    weight_shape = (k, n) if group is None else (1, k, n)
    scale_shape = (k // 128, n) if group is None else (1, k // 128, n)
    storage = retained_sm70_fp8_storage(
        (n, k),
        weight_shape=weight_shape,
        scale_shape=scale_shape,
        metadata=expected_sm70_fp8_metadata((n, k)),
        interleave_halves=interleaved,
        group_index=group,
    )

    assert storage.logical_shape == (n, k)
    assert storage.carriers[0].shape == weight_shape
    assert storage.carriers[1].shape == scale_shape
    assert storage.output == "scaled"


@pytest.mark.parametrize(("n", "k", "interleaved", "group"), _PROFILES)
def test_all_live_profiles_lower_to_one_volta_mma_kernel_without_a_decoded_weight(monkeypatch, n, k, interleaved, group):
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline

    for name, value in {
        "WORK": "w1x1",
        "TILE": "mma_m8n8k4_f16_f32/f4x4/k8",
        "REDUCE": "",
        "STAGE": "d1/smem",
        "LOOPIFY": "0",
        "RASTER": "",
    }.items():
        monkeypatch.setenv(f"EMMY_{name}", value)

    result = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(
        _projection_graph(8, n, k, interleaved, group),
        ctx=Context.from_target((7, 0)),
    )
    sources = [node.op.kernel_source for node in result.nodes.values() if isinstance(node.op, CudaOp)]

    assert len(sources) == 1
    source = sources[0]
    assert "mma.sync.aligned.m8n8k4" in source
    assert "uint2 _v_" not in source
    assert source.count("weight[") == 8
    assert source.count("weight_scale[") == (8 if interleaved else 1)
    assert source.count("emmy_mma884_load_b_smem_trans(") == 1


def test_hmma884_weight_map_uses_fixed_256_byte_tiles_not_scale_leading_dimension():
    n, k = 128, 256
    storage = retained_sm70_fp8_storage(
        (n, k),
        weight_shape=(k, n),
        scale_shape=(k // 128, n),
        metadata=(32 * k, n),
    )
    row, col = np.indices((n, k), dtype=np.int64)
    physical_row, physical_col = _evaluate(storage.carriers[0].coord_map, row, col)
    actual = physical_row * n + physical_col
    lane = col % 8
    permuted_lane = (lane // 4) * 4 + (lane % 2) * 2 + (lane // 2) % 2
    expected = ((row // 32) * (k // 8) + col // 8) * 256 + (row % 32) * 8 + permuted_lane

    np.testing.assert_array_equal(actual, expected)
    assert np.unique(actual).size == n * k
    # The rejected formula would use meta[1]=N=128 here and collide tiles.
    rejected = (row // 32) * (32 * k) + (col // 8) * n + (row % 32) * 8 + permuted_lane
    assert np.unique(rejected).size < n * k


def test_gated_half_interleave_applies_to_both_carriers_and_group_axis_is_fixed():
    n, k = 256, 128
    storage = retained_sm70_fp8_storage(
        (n, k),
        weight_shape=(1, k, n),
        scale_shape=(1, k // 128, n),
        metadata=(32 * k, n),
        interleave_halves=True,
        group_index=0,
    )
    rows = np.arange(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    weight_group, _, _ = _evaluate(storage.carriers[0].coord_map, rows, cols)
    scale_group, scale_row, scale_col = _evaluate(storage.carriers[1].coord_map, rows, cols)

    expected_rows = 2 * (rows % (n // 2)) + rows // (n // 2)
    np.testing.assert_array_equal(weight_group, 0)
    np.testing.assert_array_equal(scale_group, 0)
    np.testing.assert_array_equal(scale_row, 0)
    np.testing.assert_array_equal(scale_col, expected_rows)


@pytest.mark.parametrize("interleaved", (False, True))
def test_spelled_retained_layout_executes_in_stock_output_order_without_a_decoded_weight_input(interleaved):
    from emmy.compiler.backend.numpy.backend import NumpyBackend
    from emmy.compiler.dtype import decode_f8

    m, n, k = 2, 128, 256
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (m, k), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("weight", (n, k), "f16"), node_id="weight")
    graph.add_node(LinearOp(), ["x", "weight"], Tensor("output", (m, n), "f16"), node_id="output")
    graph.inputs = ["x", "weight"]
    graph.outputs = ["output"]
    storage = retained_sm70_fp8_storage(
        (n, k),
        weight_shape=(k, n),
        scale_shape=(k // 128, n),
        metadata=(32 * k, n),
        interleave_halves=interleaved,
    )
    spell_physical_inputs(graph, {"weight": storage})
    graph.validate()

    logical_bits = ((np.arange(n * k, dtype=np.uint32).reshape(n, k) % 120) + 1).astype(np.uint8)
    row, col = np.indices((n, k), dtype=np.int64)
    weight_row, weight_col = _evaluate(storage.carriers[0].coord_map, row, col)
    physical_bits = np.empty((k, n), dtype=np.uint8)
    physical_bits[weight_row, weight_col] = logical_bits
    logical_scale = np.array([[0.5, 1.5], [2.0, 0.25]], dtype=np.float16)
    logical_scale = np.resize(logical_scale, (n, k // 128))
    scale_row, scale_col = _evaluate(storage.carriers[1].coord_map, row, col)
    physical_scale = np.empty((k // 128, n), dtype=np.float16)
    physical_scale[scale_row, scale_col] = logical_scale[row, col // 128]
    x = np.linspace(-0.25, 0.25, m * k, dtype=np.float32).reshape(m, k).astype(np.float16)

    backend = NumpyBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data={"x": x, "weight": physical_bits, "weight_scale": physical_scale})
    decoded = decode_f8(logical_bits, "f8e4m3").astype(np.float16)
    expected = (x @ (decoded * logical_scale.repeat(128, axis=1)).T).astype(np.float16)
    np.testing.assert_allclose(result.outputs[compiled.outputs[0]], expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight_shape": (256, 256), "scale_shape": (2, 128), "metadata": (8192, 128)}, "weight carrier"),
        ({"weight_shape": (256, 128), "scale_shape": (1, 128), "metadata": (8192, 128)}, "scale carrier"),
        ({"weight_shape": (256, 128), "scale_shape": (2, 128), "metadata": (4096, 128)}, "metadata"),
    ],
)
def test_export_contract_rejects_any_carrier_or_metadata_mismatch(kwargs, message):
    with pytest.raises(ValueError, match=message):
        retained_sm70_fp8_storage((128, 256), **kwargs)
