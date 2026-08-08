"""The computed-B trellis decode (VQ Phase 3.1): warp-tier in-kernel decode of EXL3 codes.

A matmul whose B is a trellis-coded weight binds its decode as a COMPUTED-B cone (the
``bind_contraction`` trellis arm over the per-element :class:`TrellisLoad` leaf), schedules
warp-only over the mandatory ``sync`` compute-fill (the fill decodes each B tile into its slab
while the packed codes stay compressed in gmem), and keeps the COLLAPSE reading as its
reduce-tier fallback. Accuracy is checked in the HAT BASIS (``W_hat`` — no ``suh``/``svh``
channel vectors and no Hadamard fold; the activation-side basis restore is later work) against
the numpy reference decode, on synthetic codes and on real GLM-4.5-Air-exl3 tensors.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import LinearOp, TrellisDecodeOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, TrellisLoad
from emmy.compiler.loader.exl3 import decode_trellis
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile import _schedule as sched
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_contraction, make_cone
from emmy.compiler.pipeline.pipeline import LoweringError

from ..conftest import requires_cuda, requires_sm90

# The pinned checkpoint the plan's accuracy bar names (2.0bpw rung: everything K=2 cb0, lm_head K=6).
_GLM_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--turboderp--GLM-4.5-Air-exl3/snapshots/a1adde54568f29a04c4c369180be2c17286dbec6"
)


def _codes(rng, k_pad: int, n_pad: int, kbits: int) -> np.ndarray:
    """Random packed codes — every int16 bit pattern is a valid trellis stream."""
    return rng.integers(-(2**15), 2**15, (k_pad // 16, n_pad // 16, 16 * kbits), dtype=np.int64).astype(np.int16)


def _trellis_linear_graph(m: int, n: int, k: int, kbits: int, cb: int = 0) -> tuple[Graph, tuple[int, int, int]]:
    """``x @ W_hat.T`` with W the HAT-BASIS decode of input-fed codes — the kernel-path shape
    (an input-rooted cone is not a constant subgraph, so it survives 032 unconditionally)."""
    n_pad, k_pad = -(-n // 128) * 128, -(-k // 128) * 128
    t_shape = (k_pad // 16, n_pad // 16, 16 * kbits)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k), "f16"), node_id="x")
    g.add_node(InputOp(), [], Tensor("codes", t_shape, "i16"), node_id="codes")
    w = g.add_node(
        op=TrellisDecodeOp(cb=cb, out_features=n, in_features=k, hadamard=False),
        inputs=["codes"],
        output=Tensor("w_hat_t", (n, k), "f16"),
    )
    g.add_node(LinearOp(), inputs=["x", w], output=Tensor("y", (m, n), "f16"), node_id="y")
    g.inputs, g.outputs = ["x", "codes"], ["y"]
    return g, t_shape


def _ref(x: np.ndarray, codes: np.ndarray, cb: int, n: int, k: int) -> np.ndarray:
    w_hat = decode_trellis(codes, cb)  # (k_pad, n_pad) f16 — the hat basis
    return x.astype(np.float32) @ w_hat[:k, :n].astype(np.float32)


# ===================================================================
# The TrellisLoad leaf: subclass integrity through the rewrite machinery
# ===================================================================


def _tl(k="k", n="n") -> TrellisLoad:
    return TrellisLoad(name="w", input="codes", index=(Var(k), Var(n)), cb=1, k_bits=6, n_tiles=8)


def test_trellis_load_rewrite_keeps_the_subclass_and_params():
    """A σ-rewrite must keep the decode leaf a ``TrellisLoad`` (its cb/K/tile params intact) —
    the base ``Load`` handler would degrade it to a plain codes-word load."""
    out = _tl().rewrite(lambda nm: f"{nm}__c0" if nm == "w" else nm, Sigma({"k": Var("k2")}))
    assert type(out) is TrellisLoad
    assert (out.cb, out.k_bits, out.n_tiles) == (1, 6, 8)
    assert out.names == ("w__c0",) and out.index[0] == Var("k2")


def test_trellis_load_identity_keys_the_decode_params():
    """Two decodes differing only in K (or cb) must never share kernel identity — the params ride
    the dataclass repr the structural key digests."""
    a = make_cone([_tl()], "k")
    b = make_cone([TrellisLoad(name="w", input="codes", index=(Var("k"), Var("n")), cb=1, k_bits=2, n_tiles=8)], "k")
    assert a.structural_key() != b.structural_key()


def test_trellis_load_is_a_load_but_never_an_exact_one():
    """The generic machinery (``map_cone``, the recognizer's k-load scan, the structural key's
    buffer walk) treats the decode leaf as a Load; the exact-type guards (``050_vectorize_loads``,
    the fill's vector-merge plan) must be able to tell it apart."""
    assert isinstance(_tl(), Load) and type(_tl()) is not Load


def test_raw_decode_op_matches_hat_basis_reference():
    """The hat-basis ``TrellisDecodeOp.forward`` is exactly ``decode_trellis(...).T`` sliced."""
    rng = np.random.default_rng(0)
    codes = _codes(rng, 128, 256, 3)
    op = TrellisDecodeOp(cb=2, out_features=200, in_features=100, hadamard=False)
    np.testing.assert_array_equal(op.forward(codes), decode_trellis(codes, 2).T[:200, :100])


# ===================================================================
# bind_contraction: the computed-B trellis arm
# ===================================================================


def _trellis_matmul_loop(*, wrapped: bool = False, m_indexed: bool = False, plain_cone: bool = False) -> Loop:
    """The fused matmul loop body loop fusion produces: ``acc += x[m,k] · decode(codes @ (k,n))``,
    optionally with a pointwise wrapper on the decode (``wrapped``) or defects the arm must
    refuse (an m-indexed load riding the cone, a decode-free computed B)."""
    k = Axis("k", 64)
    stmts: list = [Load(name="a", input="x", index=(Var("m"), Var("k")), dtype=F16)]
    if plain_cone:
        stmts += [Load(name="w0", input="wbuf", index=(Var("k"), Var("n")), dtype=F16), Assign(name="w", op="relu", args=("w0",))]
        b_name = "w"
    elif wrapped:
        stmts.append(_tl())
        if m_indexed:
            stmts.append(Load(name="sm", input="mrow", index=(Var("m"),), dtype=F16))
            stmts.append(Assign(name="wsc", op="multiply", args=("w", "sm")))
        else:
            stmts.append(Assign(name="wsc", op="negative", args=("w",)))
        b_name = "wsc"
    else:
        stmts.append(_tl())
        b_name = "w"
    stmts += [
        Assign(name="v", op="multiply", args=("a", b_name)),
        Accum(name="acc", value="v", op=ElementwiseImpl("add")),
    ]
    return Loop(axis=k, body=Body(tuple(stmts)), role=AxisRole.CONTRACTION)


def test_direct_trellis_b_binds_as_a_cone():
    a, b, acc, _epi = bind_contraction(_trellis_matmul_loop(), "m", "n", Body())
    assert isinstance(a, Load) and a.input == "x"
    assert isinstance(b, list) and len(b) == 1 and type(b[0]) is TrellisLoad  # a cone list, never a materialized Load
    assert acc == "acc"


def test_wrapped_trellis_b_binds_as_a_cone():
    _a, b, _acc, _epi = bind_contraction(_trellis_matmul_loop(wrapped=True), "m", "n", Body())
    assert isinstance(b, list) and any(type(s) is TrellisLoad for s in b)


def test_m_indexed_trellis_cone_declines():
    """An m-indexed load riding the B cone makes the streamed operand M-resident — refuse, so the
    recognizer demotes the cell to PLANAR (the guardrail contract)."""
    with pytest.raises(LoweringError, match="computed cone"):
        bind_contraction(_trellis_matmul_loop(wrapped=True, m_indexed=True), "m", "n", Body())


def test_decode_free_computed_b_still_declines():
    """An arbitrary f16 producer cone on B stays unbound — offering warp rows there would
    re-schedule existing models' shapes that today demote to PLANAR."""
    with pytest.raises(LoweringError, match="computed cone"):
        bind_contraction(_trellis_matmul_loop(plain_cone=True), "m", "n", Body())


# ===================================================================
# The schedule: warp-only over the mandatory sync fill, collapse fallback
# ===================================================================


def _rows_for(m=32, n=128, k=128, kbits=2):
    g, _ = _trellis_linear_graph(m, n, k, kbits)
    from emmy.compiler.pipeline import TILE_PASSES
    from emmy.compiler.pipeline.fork import flatten_leaves
    from emmy.compiler.pipeline.pipeline import Run

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        rows.extend(dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(g, decide)
    return rows


def test_computed_b_schedules_warp_only_over_the_sync_fill():
    rows = _rows_for()
    warp = [r for r in rows if (r.get("WORK") or "").startswith("w")]
    assert warp, "no warp rows — the computed-B contraction did not reach the mma tier"
    # Every warp row rides the MANDATORY sync compute-fill; no copy transport can evaluate the
    # decode cone, and split-K has no gmem index to σ-reindex on a computed B.
    for r in warp:
        assert (r.get("STAGE") or "").endswith("/sync"), r
        assert not (r.get("REDUCE") or "").startswith("g"), r
        assert "a:" not in (r.get("TILE") or "") and (r.get("TILE") or "").startswith("mma_"), r
    # The COLLAPSE sibling carries the reduce tiers (the per-cell decode fallback).
    assert any(not (r.get("WORK") or "").startswith("w") and (r.get("REDUCE") or "") for r in rows), "no collapse reduce rows"


def test_computed_b_readings_never_collide():
    """Reading identity survives into the prior's key space without an ``S_*`` stamp: the base
    (computed-B contraction) spells warp TILE rows only and the collapse spells the reduce
    partition — ``_enumerate`` raises on any cross-reading ident collision, so a clean pass IS
    the assertion. Checked at the enumeration layer to pin the reading count too."""
    g, _ = _trellis_linear_graph(32, 128, 128, 2)
    from emmy.compiler.pipeline import TILE_PASSES
    from emmy.compiler.pipeline.pipeline import Run

    seen: list[int] = []
    orig = sched._enumerate

    def spy(terms):
        seen.append(len(terms))
        return orig(terms)

    sched._enumerate = spy
    try:
        from emmy.compiler.pipeline.fork import flatten_leaves

        Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(g, lambda fp: flatten_leaves(fp.options)[0])
    finally:
        sched._enumerate = orig
    assert 2 in seen, f"the computed-B term should enumerate two readings (base + collapse), saw {seen}"


# ===================================================================
# 032: the hat-basis fold gate
# ===================================================================


def _const_raw_cone_graph() -> Graph:
    g = Graph()
    codes = g.add_node(
        op=ConstantOp(name="p_codes", source_path="layer.trellis", source_shape=(8, 8, 32), source_dtype="i16"),
        inputs=[],
        output=Tensor("p_codes", (8, 8, 32), "i16"),
    )
    g.add_node(
        op=TrellisDecodeOp(cb=0, out_features=128, in_features=128, hadamard=False),
        inputs=[codes],
        output=Tensor("p_w", (128, 128), "f16"),
        node_id="p_w",
    )
    g.inputs, g.outputs = [], ["p_w"]
    return g


def _fold(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition"], select=["032_fold_constant_subgraphs"]).run(graph)


def test_hat_basis_constant_cone_folds_by_default():
    folded = _fold(_const_raw_cone_graph())
    assert folded.nodes["p_w"].op.source_graph is not None


def test_hat_basis_constant_cone_survives_under_the_gate(monkeypatch):
    monkeypatch.setenv("EMMY_TRELLIS_EXPAND", "1")
    folded = _fold(_const_raw_cone_graph())
    assert any(isinstance(nd.op, TrellisDecodeOp) for nd in folded.nodes.values()), "the hat-basis cone must stay in-graph"


def test_checkpoint_basis_cone_folds_even_under_the_gate(monkeypatch):
    """Only the hat-basis form has a kernel realization — a ``hadamard=True`` cone must fold
    regardless, or the pipeline is handed an op no lowering rule knows."""
    monkeypatch.setenv("EMMY_TRELLIS_EXPAND", "1")
    g = Graph()
    leaves = {}
    for nm, shape, dt in (("trellis", (8, 8, 32), "i16"), ("suh", (128,), "f16"), ("svh", (128,), "f16")):
        leaves[nm] = g.add_node(
            op=ConstantOp(name=f"p_{nm}", source_path=f"layer.{nm}", source_shape=shape, source_dtype=dt),
            inputs=[],
            output=Tensor(f"p_{nm}", shape, dt),
        )
    g.add_node(
        op=TrellisDecodeOp(cb=0, out_features=128, in_features=128, hadamard=True),
        inputs=[leaves["trellis"], leaves["suh"], leaves["svh"]],
        output=Tensor("p_w", (128, 128), "f16"),
        node_id="p_w",
    )
    g.inputs, g.outputs = [], ["p_w"]
    folded = _fold(g)
    assert folded.nodes["p_w"].op.source_graph is not None


# ===================================================================
# CUDA: hat-basis accuracy through the full pipeline
# ===================================================================


def _run_cuda(m, n, k, kbits, pins=None, cb=0, codes=None, seed=3):
    from emmy.commands.run import _pinned_knobs
    from emmy.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(seed)
    g, t_shape = _trellis_linear_graph(m, n, k, kbits, cb=cb)
    codes = codes if codes is not None else _codes(rng, t_shape[0] * 16, t_shape[1] * 16, kbits)
    x = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)
    be = CudaBackend()
    if pins is not None:
        with _pinned_knobs(pins):
            compiled = be.compile(g)
    else:
        compiled = be.compile(g)
    kernels = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values() if getattr(nd.op, "kernel_source", None)]
    assert sum("emmy_trellis_decode" in s for s in kernels) == 1, "exactly one kernel decodes in-kernel"
    assert len(kernels) == 1, "the decode fuses into the matmul — no separate dequant kernel"
    result, _ = be.run(compiled, input_data={"x": x, "codes": codes})
    y = result.outputs["y"].reshape(m, n).astype(np.float32)
    ref = _ref(x, codes, cb, n, k)
    denom = max(float(np.abs(ref).max()), 1e-9)
    assert float(np.abs(y - ref).max()) / denom < 2e-3, "hat-basis matmul off the f16 tolerance"


_WARP_PINS = {"TILE": "mma_m16n8k16_f16_f32/f2x2/k2", "WORK": "w1x4", "REDUCE": "", "STAGE": ""}


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(("m", "n", "k", "kbits"), [(32, 128, 128, 2), (32, 128, 128, 6), (33, 128, 128, 2)])
def test_trellis_matmul_matches_hat_basis_reference_cuda(m, n, k, kbits):
    """The pinned warp row: K=2 (the GLM 2.0bpw rung), K=6 (the lm_head rung), and a masked M."""
    _run_cuda(m, n, k, kbits, _WARP_PINS)


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
def test_trellis_matmul_encode_padding_cuda():
    """Logical dims below the 128-padded codes grid — the decode slices the top-left submatrix."""
    _run_cuda(48, 96, 192, 2, {"TILE": "mma_m16n8k16_f16_f32/f2x3/k2", "WORK": "w1x2", "REDUCE": "", "STAGE": ""})


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
def test_trellis_matmul_greedy_cuda():
    """The unpinned greedy deploy stays correct — whichever reading it picks."""
    _run_cuda(32, 128, 128, 2, None)


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.skipif(not os.path.isdir(_GLM_SNAPSHOT), reason="pinned GLM-4.5-Air-exl3 snapshot not in the HF cache")
def test_trellis_matmul_real_checkpoint_cuda():
    """Real GLM-4.5-Air 2.0bpw codes (q_proj, K=2) through the kernel vs the numpy hat-basis
    reference — the Phase 3.1 accuracy bar, stated in the W_hat basis."""
    from safetensors import safe_open

    key = "model.layers.1.self_attn.q_proj.trellis"
    codes = None
    for shard in glob.glob(_GLM_SNAPSHOT + "/*.safetensors"):
        with safe_open(shard, framework="numpy") as f:
            if key in f.keys():
                codes = f.get_tensor(key)
                break
    assert codes is not None, f"{key} not found in the snapshot"
    codes = np.ascontiguousarray(codes[:, :64])  # 4096 x 1024 slice keeps the test quick
    k_in, n_out = codes.shape[0] * 16, codes.shape[1] * 16
    _run_cuda(32, n_out, k_in, 2, _WARP_PINS, codes=codes)


# ===================================================================
# CUDA: the activation-side basis restore (VQ Phase 3.3) end to end
# ===================================================================


def _exl3_checkpoint(tmp_path, n, k, kbits, *, seed=0):
    """A one-linear EXL3 checkpoint at the PADDED extents, plus its checkpoint-basis weight in
    the traced HF ``(out, in)`` orientation."""
    from safetensors.numpy import save_file

    from emmy.compiler.loader.exl3 import decode_trellis, fold_hadamard

    rng = np.random.default_rng(seed)
    k_pad, n_pad = -(-k // 128) * 128, -(-n // 128) * 128
    codes = _codes(rng, k_pad, n_pad, kbits)
    suh = (rng.standard_normal(k_pad) * 0.012).astype(np.float16)
    svh = np.sign(rng.standard_normal(n_pad)).astype(np.float16)
    save_file({"m.proj.trellis": codes, "m.proj.suh": suh, "m.proj.svh": svh}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "exl3"}}')
    return fold_hadamard(decode_trellis(codes, 0), suh, svh).T[:n, :k]


def _run_activation_cuda(tmp_path, monkeypatch, m, n, k, ref_w):
    """Spell → compile → run the activation-side chain over the EXL3 checkpoint written to
    ``tmp_path``; compare against ``x @ ref_w`` — the CHECKPOINT basis (the ``fold_hadamard``
    weight, the Phase-2 correctness lane's own value)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Tensor
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import spell_trellis_constants
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    monkeypatch.setenv("EMMY_TRELLIS_EXPAND", "1")

    from emmy.compiler.ir.frontend.ir import LinearOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k), "f16"), node_id="x")
    g.add_node(
        ConstantOp(name="w", source_path="m.proj.weight", source_shape=(n, k), source_dtype="f16"),
        [],
        Tensor("w", (n, k), "f16"),
        node_id="w",
    )
    g.add_node(LinearOp(), ["x", "w"], Tensor("y", (m, n), "f16"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    assert spell_trellis_constants(g, str(tmp_path)) == 1

    be = CudaBackend()
    compiled = be.compile(g)
    kernels = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values() if getattr(nd.op, "kernel_source", None)]
    assert sum("emmy_trellis_decode" in s for s in kernels) == 1, "exactly one kernel decodes in-kernel"
    consts = load_constants_from_safetensors(compiled, str(tmp_path))
    consts.update(bind_constants(compiled, {}))  # the zero-leaf Hadamard record
    x = (np.random.default_rng(7).standard_normal((m, k))).astype(np.float16)
    result, _ = be.run(compiled, input_data={"x": x, **consts})
    y = np.asarray(result.outputs["y"]).reshape(m, n).astype(np.float32)
    ref = x.astype(np.float32) @ ref_w.astype(np.float32).T
    assert float(np.abs(y - ref).max()) / max(float(np.abs(ref).max()), 1e-9) < 2e-3


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(("m", "n", "k", "kbits"), [(32, 256, 256, 2), (1, 512, 512, 2), (32, 128, 128, 6), (8, 200, 300, 2)])
def test_activation_side_trellis_linear_cuda(tmp_path, monkeypatch, m, n, k, kbits):
    """The kernel path end to end: decode M=1 and prefill M, the lm_head K=6 rung, and a
    doubly encode-padded shape (200→256 out, 300→384 in)."""
    _run_activation_cuda(tmp_path, monkeypatch, m, n, k, _exl3_checkpoint(tmp_path, n, k, kbits))


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.skipif(not os.path.isdir(_GLM_SNAPSHOT), reason="pinned GLM-4.5-Air-exl3 snapshot not in the HF cache")
def test_activation_side_real_checkpoint_cuda(tmp_path, monkeypatch):
    """Real GLM-4.5-Air 2.0bpw siblings (q_proj, K=2) through the activation-side chain vs the
    checkpoint-basis reference — the Phase 3.3 accuracy bar, stated in the ORIGINAL basis."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    base = "model.layers.1.self_attn.q_proj"
    got: dict[str, np.ndarray] = {}
    for shard in glob.glob(_GLM_SNAPSHOT + "/*.safetensors"):
        with safe_open(shard, framework="numpy") as f:
            for leaf in ("trellis", "suh", "svh"):
                if f"{base}.{leaf}" in f.keys():
                    got[leaf] = f.get_tensor(f"{base}.{leaf}")
    assert set(got) == {"trellis", "suh", "svh"}, f"{base} siblings not found in the snapshot"
    codes = np.ascontiguousarray(got["trellis"][:, :64])  # a 4096 x 1024 window keeps the test quick
    k_in, n_out = codes.shape[0] * 16, codes.shape[1] * 16
    save_file(
        {"m.proj.trellis": codes, "m.proj.suh": got["suh"], "m.proj.svh": np.ascontiguousarray(got["svh"][:n_out])},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "exl3"}}')

    from emmy.compiler.loader.exl3 import decode_trellis, fold_hadamard

    ref_w = fold_hadamard(decode_trellis(codes, 0), got["suh"], got["svh"][:n_out]).T
    _run_activation_cuda(tmp_path, monkeypatch, 32, n_out, k_in, ref_w)


# ===================================================================
# 3.2 — the reduce/gemv tier: the decode band and its run fusion
# ===================================================================


def _matvec_term(n=11008, k=4096, kbits=2):
    """The M=1 reading pair for a decoded-B linear — the decode-phase matvec shape."""
    g, _ = _trellis_linear_graph(1, n, k, kbits)
    from emmy.compiler.pipeline import TILE_PASSES
    from emmy.compiler.pipeline.fork import flatten_leaves
    from emmy.compiler.pipeline.pipeline import Run

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        rows.extend(dict(getattr(leaf, "knobs", {}) or {}) for leaf in leaves)
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(g, decide)
    return rows


def test_decoded_b_matvec_offers_the_tile_band_alone():
    """A decoded B answers the reduce partition with the decode band only: every reduce row is the
    transposed coop band at the tile's 16 register rows, over a cross-CTA split. The per-element
    rows are what the pre-3.2 schedule deployed, and they are 2-8x off."""
    rows = [r for r in _matvec_term() if r.get("REDUCE")]
    assert rows, "no reduce rows for the M=1 decoded-B matvec"
    for r in rows:
        assert r["REDUCE"].endswith("/coop-t/r16"), r
        assert r["REDUCE"].startswith("g"), r
        assert r.get("WORK") == "t32", r
    # Option-0 (the first offered row) is the WIDEST split — the deploy every prior-free path takes.
    first = next(r for r in _matvec_term() if r.get("REDUCE"))
    assert first["REDUCE"] == "g32k/coop-t/r16", first


def test_decoded_b_declines_the_b_orientation_gate():
    """``_matvec_b_kstride`` reads a gmem row stride off a STORED B. A ``TrellisLoad``'s index is
    the logical ``(k, n)`` of the weight while the buffer is the codes grid, so the stride would
    describe the wrong array — the whole layout gate must answer ``None`` there. Left answering
    ``False`` (what the codes grid's stride says) the plain band is refused AND option-0 collapses
    to the serial one-thread-per-output fold, which is 4-20x off at the GLM projection shapes."""
    from emmy.compiler.ir.stmt import Load

    class _Term:
        place = type("P", (), {"free": (Axis(name="n", extent=1024),)})()
        tile = type(
            "T", (), {"op": None, "inputs": {"codes": Tensor("codes", (32, 64, 32), "i16"), "wbuf": Tensor("wbuf", (512, 1024), "f16")}}
        )()

    carrier = type("C", (), {"axis": Axis(name="k", extent=512)})()
    decode = TrellisLoad(name="w", input="codes", index=(Var("k"), Var("n")), cb=0, k_bits=2, n_tiles=64)
    row = Load(name="xv", input="x", index=(Var("k"),))
    monkeyed = [decode, row]
    real = sched._node_loads
    sched._node_loads = lambda _op: monkeyed
    try:
        assert sched._matvec_b_kstride(_Term(), carrier) is None
        monkeyed = [Load(name="w", input="wbuf", index=(Var("k"), Var("n"))), row]
        assert sched._matvec_b_kstride(_Term(), carrier) is not None, "a STORED B still classifies"
    finally:
        sched._node_loads = real


def test_tile_band_fuses_the_column_into_one_run():
    """``055_fuse_trellis_runs`` turns the band's 16 per-element decodes into ONE tile-column run:
    the kernel calls ``emmy_trellis_decode_col`` and no scalar decode survives in the body."""
    from emmy.commands.run import _pinned_knobs
    from emmy.compiler.backend.cuda.backend import CudaBackend

    g, _ = _trellis_linear_graph(1, 1024, 512, 2)
    with _pinned_knobs({"REDUCE": "g4k/coop-t/r16", "WORK": "t32"}):
        compiled = CudaBackend().compile(g)
    partial = next(
        s for nd in compiled.nodes.values() if (s := getattr(nd.op, "kernel_source", None)) and "__partial" in (nd.op.kernel_name or "")
    )
    body = partial.split('extern "C"', 1)[1]
    assert body.count("emmy_trellis_decode_col<2, 0>(") == 1, body
    assert "emmy_trellis_decode(" not in body, "a scalar decode survived the run fusion"


def test_run_fusion_declines_an_unaligned_anchor():
    """The run's element ``i`` IS the weight at ``k_lo == i``, so an anchor the pass cannot prove
    tile-aligned must not fuse."""
    import importlib

    mod = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.055_fuse_trellis_runs")
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.stmt import StridedLoop

    def column(start: int, step: int) -> Body:
        loads = tuple(
            TrellisLoad(
                name=f"w{i}",
                input="codes",
                index=(BinaryExpr("+", Var("kk"), Literal(i, "int")) if i else Var("kk"), Var("n")),
                cb=0,
                k_bits=2,
                n_tiles=4,
            )
            for i in range(16)
        )
        loop = StridedLoop(axis=Axis(name="kk", extent=512), start=Literal(start, "int"), step=Literal(step, "int"), body=Body(loads))
        return mod._fuse(Body((loop,)), {})

    fused = column(0, 16)[0].body
    assert len(fused) == 1 and not fused[0].is_scalar, "an aligned anchor should fuse to one run"
    for start, step in ((8, 16), (0, 8)):
        kept = column(start, step)[0].body
        assert len(kept) == 16, f"start={start} step={step} is not tile-aligned and must not fuse"


def test_shape_key_separates_a_decoded_b_from_its_f16_twin():
    """``ShapeKey`` is layout- and storage-blind unless told: without the trellis class an f16
    matvec's golden / DB row joins a trellis matvec at the same ``(M, N, K)`` and deploys a plan
    measured on a wholly different kernel."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey

    base = {"S_ext_free_prod": 11008, "S_ext_reduce_max": 4096, "S_ext_free_max": 11008, "S_dtype_f16": 1.0}
    f16 = ShapeKey.from_s_features(base)
    coded = ShapeKey.from_s_features({**base, "S_dtype_i16": 1.0})
    assert f16.dtype_class == "" and coded.dtype_class == "trellis"
    assert f16 != coded and not coded.joins(ShapeKey.from_matmul(1, 11008, 4096, "fp16"))


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(("n", "k", "kbits"), [(1024, 512, 2), (512, 1024, 6), (1024, 512, 3)])
def test_trellis_matvec_matches_hat_basis_reference_cuda(n, k, kbits):
    """The decode-phase matvec on the greedy deploy — the decode band plus its split finalize."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(5)
    g, t_shape = _trellis_linear_graph(1, n, k, kbits)
    codes = _codes(rng, t_shape[0] * 16, t_shape[1] * 16, kbits)
    x = (rng.standard_normal((1, k)) * 0.05).astype(np.float16)
    be = CudaBackend()
    compiled = be.compile(g)
    sources = [s for nd in compiled.nodes.values() if (s := getattr(nd.op, "kernel_source", None))]
    assert any("emmy_trellis_decode_col" in s for s in sources), "the matvec did not reach the fused decode band"
    y = be.run(compiled, input_data={"x": x, "codes": codes})[0].outputs["y"].reshape(1, n).astype(np.float32)
    ref = _ref(x, codes, 0, n, k)
    assert float(np.abs(y - ref).max()) / max(float(np.abs(ref).max()), 1e-9) < 2e-3


# ===================================================================
# 3.4 — the expert / serving path: weights as program INPUTS
# ===================================================================


def _coded_expert(rng, hidden, inter, kbits):
    """One EXL3 expert's three coded linears: the per-input feed plus the decoded reference
    weights in the traced ``(out, in)`` orientation."""
    from emmy.compiler.loader.exl3 import fold_hadamard

    feed, ref, specs = {}, {}, {}
    for name, (n, k) in (("w_gate", (inter, hidden)), ("w_up", (inter, hidden)), ("w_down", (hidden, inter))):
        k_pad, n_pad = -(-k // 128) * 128, -(-n // 128) * 128
        codes = _codes(rng, k_pad, n_pad, kbits)
        suh = (rng.standard_normal(k_pad) * 0.012).astype(np.float16)
        svh = np.sign(rng.standard_normal(n_pad)).astype(np.float16)
        feed[name] = codes
        feed[f"{name}_suh"] = suh.reshape(k_pad // 128, 128)
        feed[f"{name}_svh"] = np.ascontiguousarray(svh[:n])
        ref[name] = fold_hadamard(decode_trellis(codes, 0), suh, svh).T[:n, :k]
        specs[name] = (0, codes.shape)
    return feed, ref, specs


def _expert_graph(m, hidden, inter):
    """``down(silu(gate(x)) * up(x))`` with all three weights as forward-argument INPUTS — the
    shape ``build_moe_split_wrapper(..., split_gate_up=True)`` traces to."""
    from emmy.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, hidden), "f16"), node_id="x")
    for name, (n, k) in (("w_gate", (inter, hidden)), ("w_up", (inter, hidden)), ("w_down", (hidden, inter))):
        g.add_node(InputOp(), [], Tensor(name, (n, k), "f16"), node_id=name)
    g.add_node(LinearOp(), ["x", "w_gate"], Tensor("gate", (m, inter), "f16"), node_id="gate")
    g.add_node(LinearOp(), ["x", "w_up"], Tensor("up", (m, inter), "f16"), node_id="up")
    g.add_node(ElementwiseOp(op="silu"), ["gate"], Tensor("act", (m, inter), "f16"), node_id="act")
    g.add_node(ElementwiseOp(op="multiply"), ["act", "up"], Tensor("glu", (m, inter), "f16"), node_id="glu")
    g.add_node(LinearOp(), ["glu", "w_down"], Tensor("y", (m, hidden), "f16"), node_id="y")
    g.inputs = ["x", "w_gate", "w_up", "w_down"]
    g.outputs = ["y"]
    return g


def _assert_expert_close(y, x, ref):
    """Against the decoded reference in f32. The bar is looser than the single-linear one (2e-3)
    because the program chains THREE f16 matmuls through a SiLU, and the synthetic codes give a
    near-zero output scale where that rounding is worst — the real-checkpoint accuracy numbers
    live in the serving probe, not here."""
    expected = _expert_reference(x, ref)
    assert float(np.abs(y - expected).max()) / max(float(np.abs(expected).max()), 1e-9) < 1e-2


def _expert_reference(x, ref):
    gate = x.astype(np.float32) @ ref["w_gate"].astype(np.float32).T
    up = x.astype(np.float32) @ ref["w_up"].astype(np.float32).T
    return (gate / (1.0 + np.exp(-gate)) * up) @ ref["w_down"].astype(np.float32).T


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("m", [1, 8, 64])
def test_input_rooted_expert_program_cuda(m):
    """The MoE expert program with its weights kept COMPRESSED: three input-rooted trellis
    cones in one program (gate/up/down), decoded in-kernel, against the decoded reference.
    M=1 is the fixed-slot decode shape, 8 the decode bucket, 64 a prefill chunk."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import spell_trellis_inputs

    hidden, inter = 256, 128
    rng = np.random.default_rng(3)
    feed, ref, specs = _coded_expert(rng, hidden, inter, 2)
    g = _expert_graph(m, hidden, inter)
    spell_trellis_inputs(g, specs)

    be = CudaBackend()
    compiled = be.compile(g)
    sources = [getattr(nd.op, "kernel_source", "") or "" for nd in compiled.nodes.values()]
    assert sum("emmy_trellis_decode" in s for s in sources) >= 3, "each coded linear decodes in-kernel"
    x = rng.standard_normal((m, hidden)).astype(np.float16) * 0.2
    result, _ = be.run(compiled, input_data={"x": x, **feed, **bind_constants(compiled, {})})
    y = np.asarray(result.outputs["y"]).reshape(m, hidden).astype(np.float32)
    _assert_expert_close(y, x, ref)


@requires_cuda
@requires_sm90
@pytest.mark.xdist_group("cuda")
def test_trellis_expert_program_round_trips_through_a_pack(tmp_path):
    """The Phase-5 gate: a trellis serving program SAVES to a pack, reloads, and rebuilds with
    the same output. Two things have to hold — the load-op chains stay inside the pack
    vocabulary (one that does not disables pack writing for the WHOLE program set), and the
    basis-restore Hadamard, which has no checkpoint key at all, rebinds from the plan alone."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.pack import load_pack, save_pack
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.loader.quant import spell_trellis_inputs
    from emmy.serving.gen_runner import _bind_plan_constants

    hidden, inter, m = 256, 128, 8
    rng = np.random.default_rng(5)
    feed, ref, specs = _coded_expert(rng, hidden, inter, 2)
    g = _expert_graph(m, hidden, inter)
    spell_trellis_inputs(g, specs)
    plan = plan_from_graph(CudaBackend().compile(g))
    assert all(w.load_ops is not None for w in plan.weights.values()), "a chain outside the pack vocabulary"
    assert any(w.source_op is not None for w in plan.weights.values()), "the Hadamard must project as a computed constant"

    key = {"kind": "trellis-expert-test"}
    save_pack(tmp_path, {"moe.expert.bucket": plan}, key=key)
    stored = load_pack(tmp_path, key=key)["moe.expert.bucket"]

    x = rng.standard_normal((m, hidden)).astype(np.float16) * 0.2
    with gpu_lock():
        # No ``sources``: every weight this program binds is computed, not read from a module.
        const_feed = _bind_plan_constants(stored, {}, None)
        program = CompiledProgram.build_from_plan(stored, {**const_feed, "x": x, **feed})
        program.run_once()
        y = np.asarray(program.outputs()["y"]).reshape(m, hidden).astype(np.float32)
    _assert_expert_close(y, x, ref)


# ===================================================================
# 4 — the golden-side spelling: key agreement and the coded snippet
# ===================================================================


def _deploy_keys(graph) -> list:
    """Every deploy-time :class:`ShapeKey` a greedy resolve of ``graph`` would build — the key
    ``_golden_pick`` looks a golden up by, reconstructed off the same fork rows and base stamps
    (no CUDA: this is the lowering's fork structure, not a compile)."""
    from emmy.compiler.pipeline import TILE_PASSES
    from emmy.compiler.pipeline.fork import flatten_leaves
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.policy.greedy import _fork_shape_key

    keys: list = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        rows = [dict(getattr(leaf, "knobs", None) or {}) for leaf in leaves]
        keys.append(_fork_shape_key(rows, base={**fp.ctx.features(), **dict(fp.root_op.knobs)}))
        return leaves[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(graph, decide)
    return keys


def _spelled_linear_graph(tmp_path, m: int, n: int, k: int, kbits: int) -> Graph:
    """The IN-MODEL form: a plain f16 constant-weight linear rewritten by the EXL3 speller into
    codes + ``suh``/``svh`` under the activation-side basis restore (Phase 3.3)."""
    from safetensors.numpy import save_file

    from emmy.compiler.loader.quant import spell_trellis_constants

    rng = np.random.default_rng(0)
    save_file(
        {
            "m.proj.trellis": _codes(rng, k, n, kbits),
            "m.proj.suh": (rng.standard_normal(k) * 0.012).astype(np.float16),
            "m.proj.svh": np.sign(rng.standard_normal(n)).astype(np.float16),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "exl3"}}')
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k), "f16"), node_id="x")
    g.add_node(
        ConstantOp(name="w", source_path="m.proj.weight", source_shape=(n, k), source_dtype="f16"),
        [],
        Tensor("w", (n, k), "f16"),
        node_id="w",
    )
    g.add_node(LinearOp(), ["x", "w"], Tensor("y", (m, n), "f16"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    assert spell_trellis_constants(g, str(tmp_path)) == 1
    return g


def test_trellis_shape_key_agrees_across_both_constructors():
    """``from_matmul("trellis")`` must produce the key ``from_s_features`` builds off the stamped
    op — the golden ↔ measured join. ``is_warp`` is FORCED on both sides: it names the dtype family
    (fp32 scalar tier vs the 16-bit world), not the deployed tier, so it holds for the M=1 decode
    band as much as for the prefill mma — and forcing it makes the two sides agree even when the
    basis-restore chain puts an f32 constant on the contraction's A cone."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey

    base = {"S_ext_free_prod": 11008, "S_ext_reduce_max": 4096, "S_ext_free_max": 11008, "S_dtype_i16": 1.0}
    golden = ShapeKey.from_matmul(1, 11008, 4096, "trellis")
    assert golden.dtype_class == "trellis" and golden.is_warp
    assert ShapeKey.from_s_features(base).joins(golden)
    # An f32 leaf in the cone must not flip the op side off the golden's key.
    assert ShapeKey.from_s_features({**base, "S_dtype_f32": 2.0}).joins(golden)


@pytest.mark.parametrize(("m", "n", "k"), [(1, 11008, 4096), (32, 1024, 512), (256, 512, 512)])
def test_golden_snippet_key_joins_the_in_model_key(tmp_path, monkeypatch, m, n, k):
    """The Phase 4 enablement bar, at BOTH tiers (M=1 decode band, prefill mma): the key a golden
    records (``MatmulGoldenConfig.shape_key`` → ``from_matmul``), the key its own torch snippet
    lowers to, and the key the EXL3-SPELLED in-model graph lowers to must all be one key. A golden
    whose snippet keys differently from the deployed program is the "matched but never deploys"
    failure this repo has hit repeatedly."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig

    monkeypatch.setenv("EMMY_TRELLIS_EXPAND", "1")
    cfg = MatmulGoldenConfig(name="probe", M=m, N=n, K=k, dtype="trellis", trans_b=True, k_bits=2, cb=0)
    golden = cfg.shape_key()

    snippet_graph, _, _ = graph_from_code(cfg.snippet())
    snippet_keys = _deploy_keys(snippet_graph)
    assert any(key.joins(golden) for key in snippet_keys), f"{golden} not in {snippet_keys}"

    in_model_keys = _deploy_keys(_spelled_linear_graph(tmp_path, m, n, k, 2))
    assert any(key.joins(golden) for key in in_model_keys), f"{golden} not in {in_model_keys}"


def test_coded_prefill_fork_is_not_rekeyed_as_a_computed_a_cone():
    """``_fork_shape_key`` rebuilds a ``d*/sync``-offering fork as ``kind="fused"`` because only a
    computed-A cone used to spell that transport. The trellis compute fill spells it too, so a
    prefill coded contraction must be excluded — left in, every one of them re-keyed to ``fused``
    AND dropped its storage class, colliding with real RMSNorm→linear goldens of equal extents."""
    g, _ = _trellis_linear_graph(256, 512, 512, 2)
    keys = [key for key in _deploy_keys(g) if key.dtype_class == "trellis"]
    assert keys, "the coded contraction's fork lost its storage class"
    assert all(key.kind == "" for key in keys), keys


def test_coded_snippet_traces_to_the_in_kernel_decode():
    """The snippet's codes must reach the graph as an INPUT under a hat-basis ``TrellisDecodeOp``
    — minted in a preamble statement exactly like the fp8 arm's weight. Folded to a constant
    instead (``032``), the decode would vanish and the entry would record a plain f16 matmul."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline.search.golden import matmul_snippet

    graph, _, _ = graph_from_code(matmul_snippet(1, 1024, 512, "trellis", True, k_bits=2, cb=0))
    decodes = [nd for nd in graph.nodes.values() if isinstance(nd.op, TrellisDecodeOp)]
    assert len(decodes) == 1 and not decodes[0].op.hadamard
    assert (decodes[0].op.cb, decodes[0].op.out_features, decodes[0].op.in_features) == (0, 1024, 512)
    assert graph.nodes["codes"].output.dtype.name == "i16" and "codes" in graph.inputs
