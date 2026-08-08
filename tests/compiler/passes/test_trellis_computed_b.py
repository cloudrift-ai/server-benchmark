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
