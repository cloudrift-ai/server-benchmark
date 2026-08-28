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


def _w4a4_shared_linears(tmp_path, names, *, m, n, k, norm=True):
    """One activation quantized once and read by several marked linears — q/k/v's shape. Equal
    ``input_scale`` values, so the checkpoint's fused projection group calibrates to one scale and
    the consumers share one quantize.

    ``norm=False`` feeds the linears the graph INPUT directly. That is the shape a tight parity
    bound needs: behind a computed producer the two backends reach the encodes with
    epsilon-different upstream values and a block at a rounding boundary flips one code, so parity
    there is distributional. Fed directly, both encode bit-identical codes."""
    from emmy.compiler.ir.frontend.ir import RmsNormOp

    _w4a4_checkpoint(tmp_path, {name: (n, 0.02) for name in names}, k=k)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (m, k), "f16"), node_id="x")
    h = "x"
    if norm:
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
    g.inputs, g.outputs = (["x", "norm_w"] if norm else ["x"]), outs
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


def _bound_contractions(tmp_path):
    """The marked linears' contractions after the tile LIFT, as ``(tile, contraction)`` pairs.

    The lift alone, not the whole tile pass list: the scheduler's chosen view decides what a node
    stores, and with no atom offered for a packed pair it deploys the demoted planar one. What
    these tests read is the bound shape a block-scaled offer would see.
    """
    from emmy.compiler.context import Context
    from emmy.compiler.ir.pure.fold import is_contraction
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

    g = _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=16, n=128, k=128)
    looped = Pipeline.build(LOOP_PASSES).run(g)
    tiled = Pipeline.build(["lowering/tile"], select={"lift"}).run(looped, ctx=Context.from_target((12, 0)))
    tiles = [node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp)]
    return [(tile, t) for tile in tiles for t in _folds(tile.op) if is_contraction(t)]


def _edge_readings(tile, con):
    """``(activation, weight)`` — each operand edge's packed k-block reading, or ``None`` where the
    edge does not read as one."""
    from emmy.compiler.ir.pure.fold import operand_body
    from emmy.compiler.pipeline.passes.lowering.tile._packed import match_packed_kblock_b

    return tuple(match_packed_kblock_b(list(operand_body(e)), con.axis.name, tile.inputs) for e in (con.a, con.channels[0].b))


def test_the_marked_matmul_binds_one_contraction_per_linear_over_plain_operand_edges(tmp_path):
    """The half of the block-scaled atom's staging need that holds today: one fold per marked
    linear, one channel, both operands plain operand edges, and the WEIGHT edge reading as a
    packed-pair decode chain over 16-element blocks."""
    from emmy.compiler.ir.pure.fold import Fold

    bound = _bound_contractions(tmp_path)
    assert bound, "no marked matmul bound as a contraction"
    for tile, con in bound:
        assert len(con.channels) == 1
        edges = (con.a, con.channels[0].b)
        assert all(isinstance(e, Fold) and e.axis is None for e in edges), "both operands must be plain operand edges"
        weight = _edge_readings(tile, con)[1]
        assert weight is not None, "the weight edge must read as a packed decode chain"
        assert weight.block == 16


def test_the_marked_matmul_binds_its_activation_edge_as_a_packed_decode_chain(tmp_path):
    """The other half, and what the block-scaled cell waits on: the ACTIVATION edge reading as a
    packed-pair decode chain over the same 16-element blocks as the weight.

    It reads that way because the fusion boundary leaves the activation's raw e4m3 block scales
    standing and puts the per-consumer decode beside its matmul. Materializing the fused f16 scale
    instead — one value per logical element rather than one per block — is what used to hide the
    block structure from this reading."""
    bound = _bound_contractions(tmp_path)
    assert bound, "no marked matmul bound as a contraction"
    for tile, con in bound:
        activation = _edge_readings(tile, con)[0]
        assert activation is not None, "the activation edge must read as a packed decode chain"
        assert activation.block == 16


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_block_scaled_cell_runs_and_holds_its_declared_tolerance(tmp_path):
    """The native fp4 path end to end: both operands packed, the block-scaled cell selected, and
    the result within the gap PR decision 18 accepts.

    That gap is not rounding noise, so this is a tolerance and not the exact oracle every other
    lowering answers to. The declared program applies ``f16(block_scale x tensor_scale)`` per
    element (decision 3's single fused rounding); the instruction applies the RAW e4m3 block
    scale itself and the tensor scale rides the epilogue, and no reassociation connects the two.
    The two differ by one f16 rounding of a per-block constant on each side — about 2^-11
    relative per side — which is what the bound below is sized for.

    K is 512 because the scale slab's cp.async fill copies 16 B chunks: at 16-element blocks a
    chunk spans 256 logical k, so the tile — and therefore K — is a multiple of 256. Narrower
    shapes decline the stage and keep the generic reading."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    m, n, k = 16, 128, 512
    g = _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=m, n=n, k=k, norm=False)
    feed = {"x": (np.random.default_rng(11).standard_normal((m, k)) * 0.5).astype(np.float16)}
    data = load_constants_from_safetensors(g, str(tmp_path))
    ref, _ = NumpyBackend().run(g, input_data={**data, **feed})
    backend = CudaBackend()
    compiled = backend.compile(g)
    sources = [s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None))]
    assert any("emmy_mma_m16n8k64_e2m1_f32(" in s for s in sources), "the block-scaled cell was never selected"
    native = next(s for s in sources if "emmy_mma_m16n8k64_e2m1_f32(" in s)
    assert "emmy_mma_load_sfa_f4" in native and "emmy_mma_load_sfb_f4" in native, "the cell ran without its scale operands"
    assert "EMMY_F4_LUT" not in native, "a native cell must not decode either operand through the value table"

    got, _ = backend.run(compiled, input_data={**data, **feed})
    for out in g.outputs:
        r = ref.outputs[out].astype(np.float32).reshape(-1)
        c = np.asarray(got.outputs[out]).astype(np.float32).reshape(-1)
        rel = np.abs(c - r) / max(float(np.abs(r).max()), 1e-9)
        # Every element carries the gap, so the median is small but not zero — unlike the W4A16
        # drain, which reproduces the declared value exactly. Measured on a 5090: median 1.8e-5,
        # max 5.6e-4; the bounds are roughly 5x those.
        assert float(np.median(rel)) < 1e-4, f"{out}: a systematic shift, not the fused-scale rounding"
        assert float(rel.max()) < 2e-3, f"{out}: past one fused-scale rounding per side"


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_one_consumer_block_scaled_linear_reads_its_codes_from_memory(tmp_path):
    """Coverage's other half. A linear with ONE consumer reaches the same shape as the shared one
    above: the activation quantize is a kernel of its own writing the packed codes, the matmul
    copies those codes into its slab, and the cell still fires against the numeric oracle.

    Consumer count decides nothing here. Loop fusion stops at a computed packed buffer whether one
    linear reads it or three, because splicing it away would leave the codes as a value with no
    stored extent — see ``_packed_readers`` in the merge rule.
    """
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    m, n, k = 16, 128, 256
    g = _w4a4_linear(tmp_path, m=m, n=n, k=k)
    x = (np.random.default_rng(23).standard_normal((m, k)) * 0.5).astype(np.float16)
    data = load_constants_from_safetensors(g, str(tmp_path))
    ref, _ = NumpyBackend().run(g, input_data={**data, "x": x})
    backend = CudaBackend()
    compiled = backend.compile(g)
    sources = [s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None))]
    native = [s for s in sources if "emmy_mma_m16n8k64_e2m1_f32(" in s]
    assert native, "the block-scaled cell was never selected on a one-consumer linear"
    assert "emmy_to_f4e2m1" not in native[0], "the quantize belongs in its own kernel, not inside the matmul"
    assert any("emmy_to_f4e2m1" in s for s in sources if s not in native), "no kernel encodes the activation at all"
    assert "EMMY_F4_LUT" not in native[0], "a native cell must not decode either operand through the value table"

    got, _ = backend.run(compiled, input_data={**data, "x": x})
    r = ref.outputs["y"].astype(np.float32).reshape(-1)
    c = np.asarray(got.outputs["y"]).astype(np.float32).reshape(-1)
    rel = np.abs(c - r) / max(float(np.abs(r).max()), 1e-9)
    assert float(np.median(rel)) < 1e-4, "a systematic shift, not the fused-scale rounding"
    assert float(rel.max()) < 2e-3, "past one fused-scale rounding per side"


@pytest.mark.parametrize("consumers", [1, 3])
def test_the_quantized_activation_survives_loop_fusion_whatever_its_consumer_count(tmp_path, consumers):
    """The packed activation buffer is a fusion boundary: the merge leaves it standing whether one
    linear reads it or three, so both reach the block-scaled cell with the codes in memory.

    Structural and deliberately GPU-free — the shape this pins is decided in the loop dialect,
    long before a kernel exists. ``test_a_one_consumer_block_scaled_linear_reads_its_codes_from_memory``
    carries the same contract down to emitted CUDA and holds it to the numeric oracle.

    A merge that spliced this buffer away would still compute the declared program. What it would
    lose is the object the packed dtype describes: the stored byte whose extent is half the
    logical one. The codes would survive only as a register value, and the consumer's index would
    name a nibble of one rather than a whole element.
    """
    from emmy.compiler.context import Context
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

    g = (
        _w4a4_linear(tmp_path, m=64, n=128, k=128)
        if consumers == 1
        else _w4a4_shared_linears(tmp_path, ("q", "kp", "v"), m=64, n=128, k=128)
    )
    lowered = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((12, 0)))

    # The stored weights are packed too, but they arrive as constants; the activation's codes are
    # the only packed buffer any kernel COMPUTES, so a surviving LoopOp output is exactly the boundary.
    codes = [t.name for node in lowered.nodes.values() if isinstance(node.op, LoopOp) for t in node.outputs if t.dtype.logical_elems > 1]
    assert len(codes) == 1, f"expected one computed packed buffer, got {codes} among {sorted(lowered.nodes)}"
    readers = [nid for nid, node in lowered.nodes.items() if codes[0] in node.inputs]
    assert len(readers) == consumers, f"every linear should read the one materialized codes buffer, got {readers}"
