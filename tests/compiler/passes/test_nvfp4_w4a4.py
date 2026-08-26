"""The declared W4A4 program — static 4-bit activations beside packed 4-bit weights.

The static activation speller (``spell_static_fp4_activations``) writes the checkpoint-declared
quantize→dequantize round trip into the graph as shared-vocabulary algebra, next to the weight
cone the weight speller spells. The program's own meaning then becomes Σ x̂·ŵ, and the numpy
backend is the parity oracle for every lowering of it. These tests hold the CUDA lowering to that
oracle: no pass below the frontend knows the checkpoint format, and no lowering code is specific
to this chain — the generic readings compute it. The block-scaled tensor-core atom that would
multiply the packed codes directly exists and is not offered yet; when its offer lands, these
same oracles hold it.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.loader.quant import spell_quantized_constants, spell_static_fp4_activations
from emmy.compiler.tensor import Tensor
from tests.compiler.helpers import requires_cuda
from tests.compiler.loader.test_quant import _w4a4_checkpoint

pytest.importorskip("torch")


def _w4a4_linear(tmp_path, *, m, n, k, dtype="f16", input_scale=0.02):
    """One spelled W4A4 linear over a synthetic marked checkpoint: weight trio + ``input_scale``."""
    _w4a4_checkpoint(tmp_path, {"layer": (n, input_scale)}, k=k)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), dtype), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="layer", source_path="layer.weight", source_shape=(n, k), source_dtype=dtype),
        inputs=[],
        output=Tensor("layer_w", (n, k), dtype),
        node_id="layer_w",
    )
    g.add_node(op=LinearOp(), inputs=["x", w], output=Tensor("y", (m, n), dtype), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    assert spell_static_fp4_activations(g, str(tmp_path)) == 1
    return g


def test_the_spelled_w4a4_program_lowers_whole(tmp_path):
    """The full CUDA pass list consumes the spelled chain — quantize, pack, decode and matmul all
    land in kernels, with the e2m1 encode reaching the emitted source. Structure only; which
    kernel carries which piece is the scheduler's call and not pinned here."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    g = _w4a4_linear(tmp_path, m=32, n=128, k=128)
    lowered = Pipeline.build(CUDA_PASSES).run(g, ctx=Context.from_target((12, 0)))
    sources = [s for node in lowered.nodes.values() if (s := getattr(node.op, "kernel_source", None))]
    assert sources, "no kernel came out of the lowering"
    leftovers = [nid for nid, n in lowered.nodes.items() if type(n.op).__name__ not in ("CudaOp", "ConstantOp", "InputOp")]
    assert not leftovers, f"lowering left tensor ops behind: {leftovers}"
    assert any("emmy_to_f4e2m1" in s for s in sources), "the activation encode never reached a kernel"


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_shared_quantized_activation_behind_a_norm_compiles_and_holds_flip_bounded_parity(tmp_path):
    """q/k/v's serving shape at toy size: one RMSNorm output, quantized once, read by three
    marked linears. Two contracts in one shape.

    COMPILES: the norm statistic's fold is re-evaluated at the quantize's per-element sites
    through projection edges, and the edge copies' accumulators must α-rename — before that fix
    the rendered kernel declared the statistic's accumulator twice in one scope and nvcc refused
    it, so this compile IS the regression pin.

    FLIP-BOUNDED PARITY: behind a COMPUTED producer, numpy and CUDA reach the e2m1/e4m3 encodes
    with slightly different upstream values (the norm's rsqrt differs at the f16 epsilon), and
    where a block's scale ratio lands within that epsilon of a rounding boundary the code flips —
    a step function amplifies an epsilon into one code step. Both backends compute the declared
    program correctly; they disagree only AT boundaries, so the parity contract here is
    distributional: almost every element tight, the flipped remainder rare and bounded by the
    quantization step (an e4m3 ulp compounded with the e2m1 re-round stays under 25%). The
    direct-feed sibling below keeps the tight bound, because there the encode inputs are
    bit-identical by construction."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    m, n, k = 16, 128, 128
    g = _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=m, n=n, k=k)
    encodes = [nd for nd in g.nodes.values() if type(nd.op).__name__ == "ElementwiseOp" and nd.op.name == "to_f4e2m1"]
    assert len(encodes) == 1, "equal input_scale values must share ONE quantize chain"

    x = (np.random.default_rng(5).standard_normal((m, k)) * 0.5).astype(np.float16)
    feed = {"x": x, "norm_w": np.ones(k, dtype=np.float16)}
    data = load_constants_from_safetensors(g, str(tmp_path))
    ref, _ = NumpyBackend().run(g, input_data={**data, **feed})
    backend = CudaBackend()
    compiled = backend.compile(g)  # the regression pin: this raised on the duplicate accumulator
    got, _ = backend.run(compiled, input_data={**data, **feed})
    for out in g.outputs:
        r = ref.outputs[out].astype(np.float32).reshape(-1)
        c = np.asarray(got.outputs[out]).astype(np.float32).reshape(-1)
        rel = np.abs(c - r) / max(float(np.abs(r).max()), 1e-9)
        assert float(np.median(rel)) == 0.0, f"{out}: a systematic gap, not boundary flips"
        assert float(np.quantile(rel, 0.99)) < 2e-2, f"{out}: too many elements off the tight band"
        assert float((rel > 5e-3).mean()) < 0.05, f"{out}: flipped fraction too large"
        assert float(rel.max()) < 0.25, f"{out}: an outlier beyond one quantization step"


@requires_cuda
@pytest.mark.parametrize(("m", "n", "k"), [(32, 128, 128), (16, 512, 2048), (256, 512, 2048)])
@pytest.mark.xdist_group("cuda")
def test_the_spelled_w4a4_program_matches_numpy_on_device(tmp_path, m, n, k):
    """numpy-vs-CUDA parity on the declared W4A4 program. numpy evaluates the same graph the CUDA
    backend compiles, so any gap is a lowering defect. The deep-K decode shape and a prefill-wide
    M ride along — split-K and the wider schedules are where the packed lane has been bitten
    before. Bound is roughly 3x the measured error on an RTX 5090 (4.8e-4 – 6.3e-4)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    g = _w4a4_linear(tmp_path, m=m, n=n, k=k)
    x = (np.random.default_rng(3).standard_normal((m, k)) * 0.05).astype(np.float16)
    data = load_constants_from_safetensors(g, str(tmp_path))
    ref, _ = NumpyBackend().run(g, input_data={**data, "x": x})
    backend = CudaBackend()
    compiled = backend.compile(g)
    got, _ = backend.run(compiled, input_data={**data, "x": x})
    r = ref.outputs["y"].astype(np.float32)
    c = np.asarray(got.outputs["y"]).reshape(m, n).astype(np.float32)
    denom = max(float(np.abs(r).max()), 1e-9)
    assert float(np.abs(c - r).max()) / denom < 2e-3


def _w4a4_shared_linears(tmp_path, names, *, m, n, k):
    """One RMSNorm output quantized once and read by several marked linears — q/k/v's shape.
    Equal ``input_scale`` values, so the checkpoint's fused projection group calibrates to one
    scale and the consumers share one quantize."""
    from emmy.compiler.ir.frontend.ir import RmsNormOp

    _w4a4_checkpoint(tmp_path, {name: (n, 0.02) for name in names}, k=k)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
    nw = g.add_node(op=InputOp(), inputs=[], output=Tensor("norm_w", (k,), "f16"), node_id="norm_w")
    h = g.add_node(op=RmsNormOp(), inputs=["x", nw], output=Tensor("h", (m, k), "f16"), node_id="h")
    outs = []
    for name in names:
        w = g.add_node(
            op=ConstantOp(name=name, source_path=f"{name}.weight", source_shape=(n, k), source_dtype="f16"),
            inputs=[],
            output=Tensor(f"{name}_w", (n, k), "f16"),
            node_id=f"{name}_w",
        )
        outs.append(g.add_node(op=LinearOp(), inputs=[h, w], output=Tensor(f"{name}_y", (m, n), "f16"), node_id=f"{name}_y"))
    g.inputs, g.outputs = ["x", "norm_w"], outs
    assert spell_quantized_constants(g, str(tmp_path)) == len(names)
    assert spell_static_fp4_activations(g, str(tmp_path)) == len(names)
    return g


def _folds(node):
    """Every ``Fold`` in a tile tree, the root first."""
    from emmy.compiler.ir.pure.fold import Fold

    return [node, *(f for e in node.operands for f in _folds(e))] if isinstance(node, Fold) else []


def test_a_shared_activation_reaches_its_matmuls_as_packed_codes(tmp_path):
    """The boundary the two-half spelling buys. Several linears read one quantized activation, so
    loop fusion materializes what they SHARE — the packed codes, beside the raw e4m3 block scales
    a packed weight constant also stores. Before the split the shared point was the
    reconstruction: a dense 16-bit buffer reached memory with the codes dissolved into the
    producer, leaving nothing downstream to read as a 4-bit operand."""
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.stmt import Load
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

    g = _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=16, n=128, k=128)
    lowered = Pipeline.build(LOOP_PASSES).run(g)
    codes = [t.name for node in lowered.nodes.values() if isinstance(node.op, LoopOp) for t in node.outputs if t.dtype.logical_elems == 2]
    assert len(codes) == 1, f"the activation's packed codes must materialize exactly once, got {codes}"
    readers = [
        nid
        for nid, node in lowered.nodes.items()
        if isinstance(node.op, LoopOp) and any(isinstance(s, Load) and s.input == codes[0] for s in node.op.body.iter())
    ]
    assert len(readers) == 3, f"each marked linear must read the shared codes, got {readers}"


def test_the_marked_matmul_binds_both_operands_as_packed_decode_chains(tmp_path):
    """The contraction reading the block-scaled atom's staging needs: one fold over two operand
    edges, each a packed-pair k-block decode chain over 16-element blocks. The atom is not
    offered yet, so the scheduler still deploys the demoted planar view; what this pins is the
    bound shape, which is what an offer would read."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.pure.fold import Fold, is_contraction, operand_body
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.pipeline.passes.lowering.tile._packed import match_packed_kblock_b

    g = _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=16, n=128, k=128)
    # Recognition alone: the scheduler's chosen VIEW decides what a node stores, and with no atom
    # offered for a packed pair it deploys the demoted planar one.
    looped = Pipeline.build(LOOP_PASSES).run(g)
    tiled = Pipeline.build(["lowering/tile"], select={"recognize"}).run(looped, ctx=Context.from_target((12, 0)))
    tiles = [node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp)]
    bound = [(tile, t) for tile in tiles for t in _folds(tile.op) if is_contraction(t)]
    assert bound, "no marked matmul bound as a contraction"
    for tile, con in bound:
        assert len(con.channels) == 1
        edges = (con.a, con.channels[0].b)
        assert all(isinstance(e, Fold) and e.axis is None for e in edges), "both operands must be plain operand edges"
        reads = [match_packed_kblock_b(list(operand_body(e)), con.axis.name, tile.inputs) for e in edges]
        assert all(r is not None for r in reads), "both operands must read as packed decode chains"
        assert {r.block for r in reads} == {16}
