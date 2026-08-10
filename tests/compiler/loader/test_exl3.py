"""EXL3 trellis-coded weight decode (``loader.exl3``): bit-window extraction against a bitwise
reference, byte-exact pack/unpack roundtrip, the 3INST computed codebook against a scalar
reference, mma-fragment tile placement, the Hadamard/sign fold, and — when the pinned
GLM-4.5-Air-exl3 checkpoint is in the local HF cache — internal invariants on real tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emmy.compiler.loader.exl3 import (
    CODEBOOK_SCALE,
    codebook_values,
    decode_exl3_blocks,
    decode_exl3_linear,
    decode_trellis,
    fold_hadamard,
    pack_trellis,
    trellis_windows,
)

from ..conftest import requires_cuda

rng = np.random.default_rng(7)


def _random_trellis(kt, nt, K):
    """Any random int16 words form a valid circular code stream — the format has no invalid bytes."""
    return rng.integers(-(2**15), 2**15, (kt, nt, 16 * K)).astype(np.int16)


# ===================================================================
# Scalar references, straight from the format (independent of the module's vectorization)
# ===================================================================


def _ref_windows(tile_u16, K):
    """Bitwise window reference: expand the stream to individual bits, read each window MSB-first."""
    u32 = [int(tile_u16[2 * i]) | (int(tile_u16[2 * i + 1]) << 16) for i in range(8 * K)]
    nbits = 256 * K
    bits = [(u32[j // 32] >> (31 - j % 32)) & 1 for j in range(nbits)]
    wins = []
    for t in range(256):
        w = 0
        for b in range(16):
            w = (w << 1) | bits[((t + 1) * K - 16 + b) % nbits]
        wins.append(w)
    return np.array(wins, dtype=np.uint16)


def _f16_bits(u):
    return np.array(u, dtype=np.uint16).view(np.float16)


def _ref_codebook(w, cb):
    """Scalar 3INST reference: python ints mod 2^32, numpy fp16 scalar arithmetic."""
    m = (1 << 32) - 1
    x = int(w)
    if cb == 0:
        x = (x * 89226354 + 64248484) & m
    elif cb == 1:
        x = (x * 0xCBAC1FED) & m
    elif cb == 2:
        x = (x * 0x83DCD12D) & m
        s = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + 0x6400) & 0xFFFF
        return np.float16(np.float64(_f16_bits(s)) * np.float64(_f16_bits(0x1EEE)) + np.float64(_f16_bits(0xC931)))
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return _f16_bits(x & 0xFFFF) + _f16_bits(x >> 16)


def _ref_decode_tile(tile_i16, K, cb):
    """Scalar tile decode: windows → codebook → the mma B-fragment (row, col) placement."""
    wins = _ref_windows(tile_i16.view(np.uint16), K)
    tile = np.zeros((16, 16), dtype=np.float16)
    for lane in range(32):
        for j in range(8):
            r = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
            c = lane // 4 + 8 * (j >> 2)
            tile[r, c] = _ref_codebook(wins[8 * lane + j], cb)
    return tile


# ===================================================================
# Bit windows and the packed stream
# ===================================================================


@pytest.mark.parametrize("K", range(1, 9))
def test_windows_match_bitwise_reference(K):
    tr = _random_trellis(2, 3, K)
    wins = trellis_windows(tr)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(wins[i, j], _ref_windows(tr[i, j].view(np.uint16), K))


@pytest.mark.parametrize("K", range(1, 9))
def test_windows_overlap_invariant(K):
    """Tail-biting: window(t) >> K == window(t-1) mod 2^(16-K), circularly (t=0 wraps to t=255)."""
    wins = trellis_windows(_random_trellis(3, 2, K)).astype(np.uint32)
    prev = np.roll(wins, 1, axis=-1)
    np.testing.assert_array_equal(wins >> K, prev & ((1 << (16 - K)) - 1))


@pytest.mark.parametrize("K", range(1, 9))
def test_pack_roundtrip_byte_exact(K):
    """unpack → repack reproduces the stored int16 words exactly — pins every bit's placement."""
    tr = _random_trellis(2, 3, K)
    np.testing.assert_array_equal(pack_trellis(trellis_windows(tr), K), tr)


def test_trellis_validation_errors():
    with pytest.raises(ValueError, match="3-D"):
        trellis_windows(np.zeros((4, 32), dtype=np.int16))
    with pytest.raises(ValueError, match="int16"):
        trellis_windows(np.zeros((1, 1, 32), dtype=np.int32))
    with pytest.raises(ValueError, match="integer K"):
        trellis_windows(np.zeros((1, 1, 24), dtype=np.int16))  # 24 = 16*1.5
    with pytest.raises(ValueError, match="integer K"):
        trellis_windows(np.zeros((1, 1, 16 * 9), dtype=np.int16))  # K = 9
    with pytest.raises(ValueError, match="last dim must be 256"):
        pack_trellis(np.zeros((2, 2, 128), dtype=np.uint16), 2)
    with pytest.raises(ValueError, match=r"K must be in \[1, 8\]"):
        pack_trellis(np.zeros((2, 2, 256), dtype=np.uint16), 9)


# ===================================================================
# 3INST computed codebook
# ===================================================================


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_codebook_matches_scalar_reference(cb):
    wins = np.concatenate([np.array([0, 1, 0x7FFF, 0x8000, 0xFFFF]), rng.integers(0, 65536, 512)]).astype(np.uint16)
    ref = np.array([_ref_codebook(w, cb) for w in wins], dtype=np.float16)
    got = codebook_values(wins, cb)
    np.testing.assert_array_equal(got.view(np.uint16), ref.view(np.uint16))  # bit-exact, not tolerance


def test_codebook_full_table_pins_the_scale():
    """Full-table stats over all 65536 windows: mean ~0, std == the encoder's CODEBOOK_SCALE
    constant (1.24371088) — the anchor for the rms sanity checks on real decoded tiles."""
    vals = codebook_values(np.arange(65536, dtype=np.uint16)).astype(np.float32)
    assert abs(float(vals.std()) - CODEBOOK_SCALE) < 1e-4
    assert abs(float(vals.mean())) < 1e-2
    assert not np.isnan(vals).any() and not np.isinf(vals).any()


def test_codebook_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown codebook id"):
        codebook_values(np.zeros(4, dtype=np.uint16), 3)


# ===================================================================
# Tile placement and whole-tensor decode
# ===================================================================


@pytest.mark.parametrize(("K", "cb"), [(2, 0), (6, 0), (2, 1), (3, 2)])
def test_decode_trellis_matches_scalar_reference(K, cb):
    """Vectorized decode == the scalar per-tile reference (windows, codebook, placement), exactly.
    K=2 and K=6 are the two rungs the pinned GLM-4.5-Air 2.0bpw checkpoint uses (body / lm_head)."""
    tr = _random_trellis(2, 2, K)
    got = decode_trellis(tr, cb)
    assert got.shape == (32, 32) and got.dtype == np.float16
    for i in range(2):
        for j in range(2):
            ref = _ref_decode_tile(tr[i, j], K, cb)
            np.testing.assert_array_equal(got[16 * i : 16 * i + 16, 16 * j : 16 * j + 16].view(np.uint16), ref.view(np.uint16))


# ===================================================================
# Hadamard / sign fold
# ===================================================================


def _ref_sylvester(n):
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def test_fold_hadamard_basis_element():
    """A single 1 at (0, 0) spreads to a constant 1/128 over its own 128x128 block and exactly
    zero outside — 1/128 is exact in fp16, so this is an exact expectation."""
    w_hat = np.zeros((256, 256), dtype=np.float16)
    w_hat[0, 0] = 1.0
    out = fold_hadamard(w_hat, np.ones(256, dtype=np.float16), np.ones(256, dtype=np.float16))
    np.testing.assert_array_equal(out[:128, :128], np.full((128, 128), 1 / 128, dtype=np.float16))
    np.testing.assert_array_equal(out[128:, :], 0)
    np.testing.assert_array_equal(out[:, 128:], 0)


def test_fold_hadamard_matches_dense_reference():
    """Blocked fold == the dense block-diagonal H128 sandwich computed independently in float64.
    The comparison tolerance is the fold's own contract — fp16 rounding of the fold is
    implementation-defined (exllamav3's fused-vs-reference test tolerates 2e-3 relative)."""
    w_hat = (rng.standard_normal((256, 384)) * CODEBOOK_SCALE).astype(np.float16)
    suh = (rng.choice([-1.0, 1.0], 256) * 0.01).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], 384).astype(np.float16)
    h = _ref_sylvester(128) / np.sqrt(128)
    hk = np.kron(np.eye(2), h)  # block-diagonal over the 256 rows
    hn = np.kron(np.eye(3), h)  # block-diagonal over the 384 cols
    ref = np.diag(suh.astype(np.float64)) @ hk @ w_hat.astype(np.float64) @ hn @ np.diag(svh.astype(np.float64))
    out = fold_hadamard(w_hat, suh, svh)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), rtol=2e-3, atol=1e-6)


def test_fold_hadamard_validation_errors():
    with pytest.raises(ValueError, match="not a multiple of 128"):
        fold_hadamard(np.zeros((64, 128), dtype=np.float16), np.ones(64, np.float16), np.ones(128, np.float16))
    with pytest.raises(ValueError, match="do not match"):
        fold_hadamard(np.zeros((128, 128), dtype=np.float16), np.ones(64, np.float16), np.ones(128, np.float16))


# ===================================================================
# One-linear decode (the sibling-tensor entry point)
# ===================================================================


def test_decode_exl3_linear_composes_decode_and_fold():
    tr = _random_trellis(8, 8, 2)
    suh = (rng.choice([-1.0, 1.0], 128) * 0.01).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], 128).astype(np.float16)
    out = decode_exl3_linear(tr, suh, svh)
    ref = fold_hadamard(decode_trellis(tr, 0), suh, svh)
    np.testing.assert_array_equal(out.view(np.uint16), ref.view(np.uint16))


def test_decode_exl3_linear_marker_selects_codebook():
    """Codebook selection is by marker PRESENCE; the stored marker value is never read."""
    tr = _random_trellis(8, 8, 2)
    suh, svh = np.ones(128, dtype=np.float16), np.ones(128, dtype=np.float16)
    via_mcg = decode_exl3_linear(tr, suh, svh, mcg=np.array(0xCBAC1FED, dtype=np.uint32).view(np.int32))
    np.testing.assert_array_equal(via_mcg.view(np.uint16), fold_hadamard(decode_trellis(tr, 1), suh, svh).view(np.uint16))
    via_mul1 = decode_exl3_linear(tr, suh, svh, mul1=np.array(0, dtype=np.int32))  # value ignored
    np.testing.assert_array_equal(via_mul1.view(np.uint16), fold_hadamard(decode_trellis(tr, 2), suh, svh).view(np.uint16))


def test_decode_exl3_linear_rejects_both_markers():
    tr = _random_trellis(8, 8, 2)
    ones = np.ones(128, dtype=np.float16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        decode_exl3_linear(tr, ones, ones, mcg=np.array(1, np.int32), mul1=np.array(1, np.int32))


@pytest.mark.parametrize("cols", [128, 256, 300, 4096])
def test_decode_exl3_blocks_tile_the_whole_decode(cols):
    """Blocked reconstruction matches the whole-tensor path at every complete 128-column tile."""
    tr = _random_trellis(8, 40, 3)
    suh = (rng.choice([-1.0, 1.0], 128) * 0.01).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], 640) * 0.5).astype(np.float16)
    ref, ref_hat = decode_exl3_linear(tr, suh, svh), decode_trellis(tr)
    seen = 0
    for lo, hi, block in decode_exl3_blocks(tr, suh, svh, cols=cols):
        assert lo == seen and block.shape == (128, hi - lo)
        np.testing.assert_array_equal(
            decode_trellis(tr[:, lo // 16 : hi // 16]).view(np.uint16),
            ref_hat[:, lo:hi].view(np.uint16),
        )
        np.testing.assert_allclose(block.astype(np.float32), ref[:, lo:hi].astype(np.float32), rtol=1e-3, atol=0)
        seen = hi
    assert seen == 640


def test_decode_exl3_blocks_honors_the_codebook_markers():
    tr = _random_trellis(8, 16, 2)
    suh, svh = np.ones(128, dtype=np.float16), np.ones(256, dtype=np.float16)
    ref = decode_exl3_linear(tr, suh, svh, mul1=np.array(0, np.int32))
    got = np.concatenate(
        [block for _, _, block in decode_exl3_blocks(tr, suh, svh, mul1=np.array(0, np.int32), cols=128)],
        axis=1,
    )
    np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=1e-3, atol=0)
    assert not np.allclose(
        got.astype(np.float32),
        decode_exl3_linear(tr, suh, svh).astype(np.float32),
        rtol=1e-3,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        next(decode_exl3_blocks(tr, suh, svh, mcg=np.array(1, np.int32), mul1=np.array(1, np.int32)))


# ===================================================================
# Real checkpoint invariants (skips cleanly when the pinned snapshot is not cached)
# ===================================================================

_REPO = "turboderp/GLM-4.5-Air-exl3"
_REVISION = "a1adde54568f29a04c4c369180be2c17286dbec6"  # the 2.0bpw rung, pinned


def _cached_snapshot() -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    p = try_to_load_from_cache(_REPO, "model.safetensors.index.json", revision=_REVISION)
    return Path(p).parent if isinstance(p, str) else None


_SNAPSHOT = _cached_snapshot()

requires_glm_exl3 = pytest.mark.skipif(_SNAPSHOT is None, reason="pinned GLM-4.5-Air-exl3 2.0bpw snapshot not in the HF cache")


def _load_real(name: str, tile_slice=None):
    """Read one tensor (or a leading-axes tile slice of it) from the cached snapshot."""
    from safetensors import safe_open

    index = json.loads((_SNAPSHOT / "model.safetensors.index.json").read_text())
    with safe_open(str(_SNAPSHOT / index["weight_map"][name]), framework="numpy") as f:
        if tile_slice is None:
            return f.get_tensor(name)
        return f.get_slice(name)[tile_slice]


# Dense-layer, MoE-expert, attention, and head tensors; lm_head covers the K=6 rung.
_REAL_NAMES = [
    "model.layers.0.mlp.up_proj",
    "model.layers.5.mlp.experts.0.down_proj",
    "model.layers.5.self_attn.q_proj",
    "lm_head",
]


@requires_glm_exl3
@pytest.mark.parametrize("name", _REAL_NAMES)
def test_real_tensor_windows_and_repack(name):
    """On real checkpoint tiles: the tail-biting overlap invariant holds, and repacking the
    extracted windows reproduces the stored bytes exactly (pins alignment and endianness)."""
    tr = _load_real(name + ".trellis", np.s_[:8, :8])
    K = tr.shape[-1] // 16
    wins = trellis_windows(tr)
    w32 = wins.astype(np.uint32)
    np.testing.assert_array_equal(w32 >> K, np.roll(w32, 1, axis=-1) & ((1 << (16 - K)) - 1))
    np.testing.assert_array_equal(pack_trellis(wins, K), tr)


@requires_glm_exl3
def test_real_tensor_decode_statistics():
    """Decoded hat-basis values have rms near CODEBOOK_SCALE; the folded weight has a plausible
    fp16 weight scale, with suh carrying the magnitude and svh nearly pure signs."""
    name = "model.layers.0.mlp.up_proj"
    tr = _load_real(name + ".trellis", np.s_[:8, :8])
    suh = _load_real(name + ".suh", np.s_[:128])
    svh = _load_real(name + ".svh", np.s_[:128])
    w_hat = decode_trellis(tr)

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a.astype(np.float32)))))

    assert 1.0 < rms(w_hat) < 1.5  # ~ CODEBOOK_SCALE
    w = fold_hadamard(w_hat, suh, svh)
    assert 0.001 < rms(w) < 0.1
    assert rms(suh) < 0.1  # magnitude side
    assert 0.5 < rms(svh) < 2.0  # ~ pure signs


# ===================================================================
# Checkpoint ingestion: generic spelling → fold → bind.
# The speller lives in ``loader.quant`` (the one quantization-concept module);
# its tests sit here because the fixtures and the decode reference are EXL3's.
# ===================================================================

_FOLD_RULE = "032_fold_constant_subgraphs"


def _exl3_linear_tensors(base: str, n: int, k: int, K: int = 2, cb: int = 0):
    """Mint one EXL3-coded linear: random windows packed into real trellis words, fp16
    ``suh``/``svh`` at the encoder's padded extents (roundup to 128), plus the marker sibling
    for ``cb``. Returns ``(tensors, ref)`` with ``ref`` the LOGICAL ``(n, k)`` fp16 weight
    (decode → transpose → slice — the reference math for encode padding)."""
    torch = pytest.importorskip("torch")

    n_pad, k_pad = -(-n // 128) * 128, -(-k // 128) * 128
    wins = rng.integers(0, 1 << 16, (k_pad // 16, n_pad // 16, 256)).astype(np.uint16)
    tr = pack_trellis(wins, K)
    suh = (rng.standard_normal(k_pad) * 0.01).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], n_pad).astype(np.float16)
    tensors = {
        f"{base}.trellis": torch.from_numpy(tr),
        f"{base}.suh": torch.from_numpy(suh),
        f"{base}.svh": torch.from_numpy(svh),
    }
    if cb == 1:
        tensors[f"{base}.mcg"] = torch.tensor(0x7BAC1FED, dtype=torch.int32)  # value never read
    elif cb == 2:
        tensors[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)
    ref = fold_hadamard(decode_trellis(tr, cb), suh, svh).T[:n, :k]
    return tensors, ref


_EXL3_QC = {"quant_method": "exl3", "version": "0.0.5", "bits": 2.0, "head_bits": 6}


def _write_exl3_checkpoint(dirpath, tensors, quant_config=_EXL3_QC):
    """Single-shard safetensors dir + config.json declaring the EXL3 scheme."""
    import json as _json

    from safetensors.torch import save_file

    save_file({k: v.clone() for k, v in tensors.items()}, str(dirpath / "model.safetensors"))
    cfg = {"model_type": "test"}
    if quant_config is not None:
        cfg["quantization_config"] = quant_config
    (dirpath / "config.json").write_text(_json.dumps(cfg))


def _weight_graph(shape, dtype="f32", source_path="layer.weight"):
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import ConstantOp

    g = Graph()
    g.add_node(
        op=ConstantOp(name="p_w", source_path=source_path, source_shape=shape, source_dtype=dtype),
        inputs=[],
        output=Tensor("p_w", shape, dtype),
        node_id="p_w",
    )
    g.inputs, g.outputs = [], ["p_w"]
    return g


def _spelled_exl3(tmp_path, *, n=256, k=128, K=2, cb=0, dtype="f32"):
    from emmy.compiler.loader.quant import spell_trellis_constants

    tensors, ref = _exl3_linear_tensors("layer", n, k, K=K, cb=cb)
    _write_exl3_checkpoint(tmp_path, tensors)
    g = _weight_graph((n, k), dtype=dtype)
    assert spell_trellis_constants(g, str(tmp_path)) == 1
    return g, ref


def _fold(graph):
    from emmy.compiler.pipeline import Pipeline

    return Pipeline.build(["frontend/decomposition"], select=[_FOLD_RULE]).run(graph)


def _bind_and_run_spelled(graph, checkpoint):
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    graph = _fold(graph)
    feed = load_constants_from_safetensors(graph, str(checkpoint))
    result, _ = NumpyBackend().run(graph, input_data=feed)
    return result.outputs[graph.outputs[0]]


def test_spell_trellis_builds_generic_cone(tmp_path):
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.ir.tensor.ir import BitcastOp, GatherOp, IndexMapOp, RangeOp

    g, _ref = _spelled_exl3(tmp_path)
    consts = {nn.op.source_path: nn for nn in g.nodes.values() if isinstance(nn.op, ConstantOp) and nn.op.source_path is not None}
    assert set(consts) == {"layer.trellis", "layer.suh", "layer.svh"}
    assert consts["layer.trellis"].output.dtype.name == "i16" and consts["layer.trellis"].op.source_dtype == "i16"
    assert consts["layer.suh"].output.dtype.name == "f16"
    # Hadamard, the one per-step shift, and tile permutation remain generic
    # source-free range algebra. Packed-word indices are address expressions,
    # not materialized tensors.
    assert any(isinstance(nn.op, RangeOp) for nn in g.nodes.values())
    assert any(isinstance(nn.op, BitcastOp) for nn in g.nodes.values())
    assert sum(isinstance(nn.op, GatherOp) for nn in g.nodes.values()) == 1
    direct_maps = {
        nn.output.name: nn.op
        for nn in g.nodes.values()
        if isinstance(nn.op, IndexMapOp) and nn.output.name.endswith(("window_lo", "window_hi"))
    }
    assert len(direct_maps) == 2
    assert all(any(op in src.coord_map[-1].pretty() for op in ("%", "//")) for src in direct_maps.values() for src in src.sources)
    assert sum(isinstance(nn.op, MatmulOp) for nn in g.nodes.values()) == 2
    # Interface invariance: the cone's output is exactly what the trace promised.
    out = g.nodes[g.outputs[0]].output
    assert out.dtype.name == "f32" and tuple(d.as_static() for d in out.shape) == (256, 128)
    assert g.outputs == ["p_w"]


def test_input_spelling_is_generic_and_matches_reference():
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.loader.quant import spell_trellis_inputs

    tensors, ref = _exl3_linear_tensors("layer", 128, 128, K=2, cb=0)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("w", (128, 128), "f16"), node_id="w")
    graph.inputs = graph.outputs = ["w"]

    spell_trellis_inputs(graph, {"w": (0, tuple(tensors["layer.trellis"].shape))})
    result, _ = NumpyBackend().run(
        graph,
        input_data={
            "w": tensors["layer.trellis"].numpy(),
            "w_suh": tensors["layer.suh"].numpy(),
            "w_svh": tensors["layer.svh"].numpy(),
        },
    )
    np.testing.assert_array_equal(result.outputs[graph.outputs[0]], ref)


def test_input_spelling_reaches_cuda_source_without_format_ir():
    """A runtime-coded dense contraction lowers entirely through generic integer IR."""
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.quant import spell_trellis_inputs
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (1, 128), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (128, 128), "f16"), node_id="w")
    graph.add_node(LinearOp(), ["x", "w"], Tensor("y", (1, 128), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    spell_trellis_inputs(graph, {"w": (0, (8, 8, 32))})

    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context(compute_capability=(12, 0)))
    cuda = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    from emmy.compiler.backend.plan import plan_from_graph

    plan = plan_from_graph(lowered)
    assert plan.launches and plan.weights
    assert all(weight.generated is not None and weight.load_ops == () for weight in plan.weights.values())
    for table in ("bit_start", "word_idx", "next_word"):
        assert f"w_decoded_{table}" not in plan.weights
    index_tables = {
        name: spec
        for name, spec in plan.weights.items()
        if name.startswith("w_decoded_") and spec.generated is not None and spec.generated[1] == (256,)
    }
    assert set(index_tables) == {"w_decoded_shift_step", "w_decoded_tile_step"}
    assert any(spec.generated is not None and spec.generated[1] == (128, 128) for spec in plan.weights.values())
    from emmy.serving.gen_runner import _bind_plan_constants

    assert set(_bind_plan_constants(plan, {}, cache=None)) == set(plan.weights)
    source = "\n".join(op.kernel_source for op in cuda)
    assert "emmy_bitcast" in source and "#include <cuda_fp16.h>" in source
    for spelling in (" % ", " << ", " >> ", " & ", " | ", " ^ "):
        assert spelling in source
    assert "unsigned int" in source and "unsigned long long" in source
    active_ir = "\n".join(f"{nid} {type(node.op).__module__} {type(node.op).__name__}" for nid, node in lowered.nodes.items())
    assert "trellis" not in active_ir.lower() and "exl3" not in active_ir.lower()


def test_input_spelling_streams_computed_b_through_tensor_cores():
    """A generic expanding B cone compute-fills a canonical slab and drains through mma.sync."""
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.quant import spell_trellis_inputs
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import flatten_leaves
    from emmy.compiler.pipeline.pipeline import Run

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16, 128), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (128, 128), "f16"), node_id="w")
    graph.add_node(LinearOp(), ["x", "w"], Tensor("y", (16, 128), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    spell_trellis_inputs(graph, {"w": (0, (8, 8, 32))})

    def choose_sync_mma(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if str(row.get("WORK", "")).startswith("w") and row.get("STAGE") == "d1/sync":
                return leaf
        return leaves[0]

    lowered, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(graph, choose_sync_mma)
    cuda = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    streamed = [op for op in cuda if "emmy_bitcast" in op.kernel_source and "mma.sync.aligned" in op.kernel_source]
    assert len(streamed) == 1
    source = streamed[0].kernel_source
    assert "cp.async" in source and "ldmatrix" in source
    assert "__launch_bounds__(32)" in source
    assert not any("const float* y_factor32" in op.kernel_source and "cp.async" in op.kernel_source for op in cuda)
    active_ir = "\n".join(f"{nid} {type(node.op).__module__} {type(node.op).__name__}" for nid, node in lowered.nodes.items())
    assert "trellis" not in active_ir.lower() and "exl3" not in active_ir.lower()


@requires_cuda
def test_input_spelling_computed_b_mma_matches_decoded_linear():
    """The streamed generic B cone agrees with the decoded-weight linear on CUDA."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import spell_trellis_inputs
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import flatten_leaves
    from emmy.compiler.pipeline.pipeline import Run

    tensors, decoded = _exl3_linear_tensors("layer", 128, 128, K=2, cb=0)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16, 128), "f16"), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (128, 128), "f16"), node_id="w")
    graph.add_node(LinearOp(), ["x", "w"], Tensor("y", (16, 128), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    spell_trellis_inputs(graph, {"w": (0, tuple(tensors["layer.trellis"].shape))})

    def choose_sync_mma(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", {}) or {})
            if str(row.get("WORK", "")).startswith("w") and row.get("STAGE") == "d1/sync":
                return leaf
        return leaves[0]

    lowered, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(graph, choose_sync_mma)
    sources = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert any("emmy_bitcast" in source and "mma.sync.aligned" in source for source in sources)
    lowered.outputs = [nid for nid, node in lowered.nodes.items() if isinstance(node.op, CudaOp)]

    x = (rng.standard_normal((16, 128)) * 0.25).astype(np.float16)
    feed = bind_constants(lowered, {})
    assert "y_factor32" in feed and np.any(feed["y_factor32"])
    decoded_constants = [name for name in feed if name.startswith("w_decoded_")]
    assert set(decoded_constants) == {"w_decoded_shift_step", "w_decoded_tile_step"}
    assert all(np.any(feed[name]) for name in decoded_constants)
    feed.update(
        {
            "x": x,
            "w": tensors["layer.trellis"].numpy(),
            "w_suh": tensors["layer.suh"].numpy(),
            "w_svh": tensors["layer.svh"].numpy(),
        }
    )
    backend = CudaBackend(debug=True)
    run_outputs = backend.run(lowered, input_data=feed)[0].outputs
    got = run_outputs["y"].astype(np.float32)
    ref = x.astype(np.float32) @ decoded.astype(np.float32).T
    err = got - ref
    rel_rms = float(np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(ref**2)), 1e-12))
    max_ratio = float(np.max(np.abs(err)) / max(np.max(np.abs(ref)), 1e-12))
    assert np.any(run_outputs["y_left_flat"])
    assert np.any(run_outputs["y_core_reduce"])
    assert rel_rms < 2e-2 and max_ratio < 2e-2, (
        rel_rms,
        max_ratio,
        float(got.min()),
        float(got.max()),
        float(ref.min()),
        float(ref.max()),
        {name: (float(np.asarray(value).min()), float(np.asarray(value).max())) for name, value in run_outputs.items()},
    )


@pytest.mark.parametrize("cb", [1, 2])
def test_spell_trellis_marker_selects_codebook(tmp_path, cb):
    from emmy.compiler.ir.base import ConstantOp

    g, _ref = _spelled_exl3(tmp_path, cb=cb)
    values = {nn.op.value for nn in g.nodes.values() if isinstance(nn.op, ConstantOp) and nn.op.value is not None}
    assert (0xCBAC1FED if cb == 1 else 0x83DCD12D) in values


def test_spell_trellis_is_idempotent(tmp_path):
    from emmy.compiler.loader.quant import spell_trellis_constants

    g, _ref = _spelled_exl3(tmp_path)
    nodes_after_first = set(g.nodes)
    assert spell_trellis_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == nodes_after_first


def test_spell_trellis_noop_without_exl3_config(tmp_path):
    from emmy.compiler.loader.quant import spell_trellis_constants

    tensors, _ref = _exl3_linear_tensors("layer", 256, 128)
    _write_exl3_checkpoint(tmp_path, tensors, quant_config=None)
    g = _weight_graph((256, 128))
    assert spell_trellis_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_trellis_leaves_plain_weight_alone(tmp_path):
    """A sensitivity-kept fp16 linear (real precedent: GLM-4.5-Air layer 0 ``o_proj``) has a
    plain ``.weight`` and no trellis sibling — it must load as an ordinary tensor."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.loader.quant import spell_trellis_constants

    tensors, _ref = _exl3_linear_tensors("layer", 256, 128)
    tensors["other.weight"] = torch.ones(8, 16, dtype=torch.float16)
    _write_exl3_checkpoint(tmp_path, tensors)
    g = _weight_graph((8, 16), source_path="other.weight")
    assert spell_trellis_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_trellis_skips_legacy_packed_signs(tmp_path):
    """Old checkpoints store packed ``su``/``sv`` sign words instead of ``suh``/``svh`` —
    unsupported; the constant stays un-spelled with a warning, never a compile error."""
    tensors, _ref = _exl3_linear_tensors("layer", 256, 128)
    del tensors["layer.suh"], tensors["layer.svh"]
    _write_exl3_checkpoint(tmp_path, tensors)
    g = _weight_graph((256, 128))
    from emmy.compiler.loader.quant import spell_trellis_constants

    assert spell_trellis_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_trellis_shape_mismatch_left_alone(tmp_path):
    """Sibling extents must be exactly the traced dims' roundups to 128."""
    from emmy.compiler.loader.quant import spell_trellis_constants

    tensors, _ref = _exl3_linear_tensors("layer", 256, 128)
    _write_exl3_checkpoint(tmp_path, tensors)
    g = _weight_graph((512, 128))  # traced n does not round up to the stored 256
    assert spell_trellis_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_fold_preserves_storage_expanding_trellis_cone(tmp_path):
    from emmy.compiler.ir.base import ConstantOp

    g, _ref = _spelled_exl3(tmp_path)
    folded = _fold(g)
    paths = {op.source_path for _nid, op in folded.loadable_constants() if op.source_path is not None}
    assert paths == {"layer.trellis", "layer.suh", "layer.svh"}
    assert not isinstance(folded.nodes["p_w"].op, ConstantOp)
    records = [op.source_graph for _nid, op in folded.loadable_constants() if op.source_graph is not None]
    assert records
    assert all(leaf.source_path is None and not leaf.source_parts for record in records for _leaf_id, leaf in record.loadable_constants())


def test_storage_expanding_checkpoint_trunk_compiles_plans_and_rebinds(tmp_path):
    """The serving trunk keeps compact source leaves all the way into its execution plan."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.loader.quant import spell_trellis_constants
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.serving.gen_runner import _bind_plan_constants, _plan_sources

    tensors, _ref = _exl3_linear_tensors("layer", 128, 128, K=2, cb=2)
    _write_exl3_checkpoint(tmp_path, tensors)
    graph = _weight_graph((128, 128), dtype="f16")
    graph.add_node(InputOp(), [], Tensor("x", (1, 128), "f16"), node_id="x")
    graph.add_node(LinearOp(), ["x", "p_w"], Tensor("y", (1, 128), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    assert spell_trellis_constants(graph, str(tmp_path)) == 1

    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context(compute_capability=(12, 0)))
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    plan = plan_from_graph(lowered)
    paths = {weight.source_path for weight in plan.weights.values() if weight.source_path is not None}
    assert paths == {"layer.trellis", "layer.suh", "layer.svh"}
    assert all(weight.load_ops is not None for weight in plan.weights.values())
    assert any(weight.generated is not None for weight in plan.weights.values())
    for table in ("bit_start", "word_idx", "next_word"):
        assert f"p_w_{table}" not in plan.weights
    assert {"p_w_shift_step", "p_w_tile_step"} <= set(plan.weights)
    assert any(spec.generated is not None and spec.generated[1] == (128, 128) for spec in plan.weights.values())

    sources = _plan_sources(plan, torch.nn.Module(), np.float16, str(tmp_path), {})
    bound = _bind_plan_constants(plan, sources, cache=None)
    assert set(bound) == set(plan.weights)
    by_path = {weight.source_path: bound[nid] for nid, weight in plan.weights.items() if weight.source_path is not None}
    for path in paths:
        np.testing.assert_array_equal(by_path[path], tensors[path].numpy())


@pytest.mark.parametrize(
    ("n", "k", "K", "cb"),
    [(128, 128, 1, 0), (256, 128, 2, 0), (192, 128, 3, 1), (128, 128, 6, 2)],
)
def test_loader_binds_spelled_trellis_weight_exact(tmp_path, n, k, K, cb):
    """spell → bind → generic evaluation reproduces the direct decode exactly (the fp16 decode cast to the
    traced f32 — one widening, bit-preserving). ``n=192`` exercises the encode padding slice;
    ``K=6`` is the lm_head rung."""
    g, ref = _spelled_exl3(tmp_path, n=n, k=k, K=K, cb=cb)
    out = _bind_and_run_spelled(g, tmp_path)
    assert out.dtype == np.float32 and out.shape == (n, k)
    np.testing.assert_array_equal(out, ref.astype(np.float32))


def test_loader_binds_f16_graph_dtype_bit_exact(tmp_path):
    """At a traced f16 dtype the bound weight is bit-identical to the direct decode."""
    g, ref = _spelled_exl3(tmp_path, dtype="f16")
    out = _bind_and_run_spelled(g, tmp_path)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out.view(np.uint16), ref.view(np.uint16))


def test_expanded_trellis_graph_survives_json_roundtrip(tmp_path):
    import json as _json

    from emmy.compiler.graph import Graph

    g, ref = _spelled_exl3(tmp_path)
    folded = _fold(g)
    g2 = Graph.from_dict(_json.loads(_json.dumps(folded.to_dict())))
    assert set(g2.nodes) == set(folded.nodes)
    assert any(op.source_graph is not None for _nid, op in g2.loadable_constants())
    out = _bind_and_run_spelled(g2, tmp_path)
    np.testing.assert_array_equal(out, ref.astype(np.float32))


def test_load_dequantized_state_dict_exl3(tmp_path):
    """The eager twin's state dict: sibling tensors decode to ``<module>.weight`` fp16 values
    in the HF ``(out, in)`` orientation (still padded — the twin loader trims); the consumed
    siblings drop; plain tensors pass through."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.loader.quant import load_dequantized_state_dict

    tensors, ref = _exl3_linear_tensors("layer", 192, 128)  # padded to (256, 128)
    tensors["norm.weight"] = torch.ones(16, dtype=torch.float16) * 2
    _write_exl3_checkpoint(tmp_path, tensors)
    sd = load_dequantized_state_dict(tmp_path)
    assert set(sd) == {"layer.weight", "norm.weight"}
    assert sd["layer.weight"].dtype == np.float16 and sd["layer.weight"].shape == (256, 128)
    np.testing.assert_array_equal(sd["layer.weight"][:192].view(np.uint16), ref.view(np.uint16))


def test_quantized_checkpoint_dir_detects_exl3(tmp_path):
    from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

    tensors, _ref = _exl3_linear_tensors("layer", 128, 128)
    quantized = tmp_path / "exl3"
    plain = tmp_path / "plain"
    quantized.mkdir()
    plain.mkdir()
    _write_exl3_checkpoint(quantized, tensors)
    _write_exl3_checkpoint(plain, tensors, quant_config=None)
    assert quantized_checkpoint_dir(str(quantized)) == quantized
    assert quantized_checkpoint_dir(str(plain)) is None


@requires_glm_exl3
def test_real_tensor_bind_time_decode_matches_direct():
    """D.1 pin on the real checkpoint: spell → bind of a real expert linear equals the
    direct Phase-1 decode bit-exactly (uint16 view of the fp16 values)."""
    from emmy.compiler.loader.quant import spell_trellis_constants

    name = "model.layers.5.mlp.experts.0.down_proj"
    tr, suh, svh = (_load_real(name + s) for s in (".trellis", ".suh", ".svh"))
    ref = decode_exl3_linear(tr, suh, svh).T
    g = _weight_graph(ref.shape, dtype="f16", source_path=name + ".weight")
    assert spell_trellis_constants(g, str(_SNAPSHOT)) == 1
    out = _bind_and_run_spelled(g, _SNAPSHOT)
    np.testing.assert_array_equal(out.view(np.uint16), ref.view(np.uint16))


# ===================================================================
# emmy compile / run wiring: whole-model trace of an EXL3 checkpoint on CUDA
# (mirrors test_quant's fp8 e2e; the EXL3 twin carries the decoded real weights)
# ===================================================================


def _tiny_exl3_checkpoint(dirpath):
    """Tiny Llama-architecture EXL3 checkpoint: every decoder-layer projection trellis-coded
    (the k/v projections at out=64 exercise the encode padding), embeddings / norms / lm_head
    fp16. Returns ``(config, ref_sd)`` with the decoded torch f32 reference state dict."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).to(torch.float16).eval()
    tensors: dict = {}
    ref_sd: dict = {}
    for name, t in model.state_dict().items():
        t = t.detach().cpu()
        if name.endswith(".weight") and t.ndim == 2 and ".layers." in name:  # the linear projections
            base = name[: -len(".weight")]
            n, k = t.shape
            coded, ref = _exl3_linear_tensors(base, n, k)
            tensors.update(coded)
            ref_sd[name] = torch.from_numpy(ref.astype(np.float32))
        else:
            tensors[name] = t.to(torch.float16) if t.is_floating_point() else t
            ref_sd[name] = t.float() if t.is_floating_point() else t
    _write_exl3_checkpoint(dirpath, tensors)
    import json as _json

    cfg = config.to_dict()
    cfg["quantization_config"] = dict(_EXL3_QC)
    (dirpath / "config.json").write_text(_json.dumps(cfg))
    return config, ref_sd


def test_exl3_twin_carries_decoded_weights(tmp_path):
    """The compile/run seam: the traced twin's parameters equal the decode reference (padding
    trimmed), and the trellis cones are spelled on exactly the coded projections."""
    torch = pytest.importorskip("torch")

    from emmy.commands.compile import _trace_model
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.binder import bind_constants_from_module

    _config, ref_sd = _tiny_exl3_checkpoint(tmp_path)
    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)
    records = [op for _nid, op in graph.loadable_constants() if isinstance(op, ConstantOp) and op.source_graph is not None]
    assert not records  # spelled, not yet folded — the fold runs inside the pipeline
    coded = [op for _nid, op in graph.loadable_constants() if op.source_path and op.source_path.endswith(".trellis")]
    assert len(coded) == 7  # q,k,v,o + gate,up,down
    params = dict(wrapper.named_parameters())
    for name, ref in ref_sd.items():
        got = params["model." + name]  # the trace wrapper nests the CausalLM under .model
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    # Wrapper-owned rotary buffers are absent from safetensors. Their trace paths must remain
    # wrapper-relative so the CUDA runtime binds the live module values rather than zeros.
    bound = bind_constants_from_module(graph, wrapper)
    wrapper_state = wrapper.state_dict()
    rotary = [(nid, op) for nid, op in graph.loadable_constants() if op.source_path and ".rotary_emb." in op.source_path]
    assert {op.source_path for _nid, op in rotary} == {
        "model.model.rotary_emb.cos",
        "model.model.rotary_emb.sin",
    }
    for nid, op in rotary:
        assert nid in bound
        np.testing.assert_array_equal(bound[nid], wrapper_state[op.source_path].detach().float().numpy())


@requires_cuda
def test_exl3_checkpoint_e2e_cuda(tmp_path):
    """Whole tiny EXL3 model through the same seam ``emmy compile`` / ``emmy run`` use, compiled
    on the CUDA backend: generic reconstruction cones remain source-bindable, and the output matches the
    decoded eager reference."""
    from emmy.compiler.ir.tensor.ir import BitcastOp, RangeOp

    from .test_quant import _assert_e2e_gate, _run_e2e

    config, ref_sd = _tiny_exl3_checkpoint(tmp_path)
    emmy_logits, ref_logits, compiled = _run_e2e(tmp_path, config, ref_sd)
    assert not any(isinstance(nn.op, (BitcastOp, RangeOp)) for nn in compiled.nodes.values())
    _assert_e2e_gate(emmy_logits, ref_logits, "generic reconstruction fold")


# ===================================================================
# Weight-free allocation metadata
# ===================================================================


def _sidecar(tmp_path: Path, storage: dict, *, method: str = "exl3") -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": method, "bits": 2.26}}))
    (tmp_path / "quantization_config.json").write_text(json.dumps({"quant_method": method, "tensor_storage": storage}))
    return tmp_path


def _coded(base: str, bits: int, *, leaves=("trellis", "suh", "svh")) -> dict:
    return {
        "quant_format": "exl3",
        "bits_per_weight": bits,
        "stored_tensors": {f"{base}.{leaf}": {"shape": [8]} for leaf in leaves},
    }


def test_coded_tensor_storage_reads_the_allocation_weight_free(tmp_path):
    """The architecture twin can read rate and sibling shapes without loading a shard."""
    from emmy.compiler.loader.exl3 import coded_tensor_storage

    root = _sidecar(
        tmp_path,
        {
            "model.layers.0.self_attn.q_proj": _coded("model.layers.0.self_attn.q_proj", 4),
            "model.layers.0.mlp.gate_proj": _coded("model.layers.0.mlp.gate_proj", 2),
            "model.embed_tokens": {"stored_tensors": {"model.embed_tokens.weight": {"shape": [32, 8]}}},
        },
    )
    got = coded_tensor_storage(str(root))
    assert sorted(got) == ["model.layers.0.mlp.gate_proj", "model.layers.0.self_attn.q_proj"]
    assert got["model.layers.0.mlp.gate_proj"]["bits_per_weight"] == 2


def test_coded_tensor_storage_skips_incomplete_siblings(tmp_path):
    from emmy.compiler.loader.exl3 import coded_tensor_storage

    root = _sidecar(tmp_path, {"m.proj": _coded("m.proj", 2, leaves=("trellis", "su", "sv"))})
    assert coded_tensor_storage(str(root)) == {}


def test_coded_tensor_storage_is_empty_off_an_exl3_checkpoint(tmp_path):
    from emmy.compiler.loader.exl3 import coded_tensor_storage

    root = _sidecar(tmp_path, {"m.proj": _coded("m.proj", 2)}, method="fp8")
    assert coded_tensor_storage(str(root)) == {}
    assert coded_tensor_storage(str(tmp_path / "nonexistent")) == {}
