r"""Attention coverage — flash (the twisted ``(m, l, O)`` MONOID on the streaming schedule), one file.

Attention is the hybrid algebra: a SEMIRING contraction (QK^T, P@V) wrapped in a MONOID streaming
softmax reduce. This file pins every tier of it:

- **scalar-tier flash** (``FLASH`` knob, the Loop-IR ``025_recognize_flash`` pass) — non-causal /
  causal / GQA / additive-mask SDPA fuses to ONE streaming online-softmax kernel matching torch,
  static AND dynamic (symbolic ``seq_len``); KV tiling; the default-path guards. This is the ONLY
  flash tier that lowers today — the two-``Contraction`` ``TWISTED`` reduce tree at block=1, through
  the one ``_factor`` contraction path.
- **tensor-core flash** — RECOVERED through the one emitter: ``_schedule._twisted_warp_option`` stamps
  the mma ``TilePlan``\ s on the Q@K / P@V ``Contraction``\ s and the tree realizes at fragment
  residence (``_twist``) — no private emitter. The ``test_generated_tensorcore_flash_*`` /
  ``test_warp_chain_*`` cases assert that warp chain.
- **cooperative-KV flash** (``BR``) — the KV axis split across threads, partial ``(m, l, O)`` states
  merged via the monoid combine. Xfailed pending the rebuild.
- **validated FA-2 reference** — a hand-written fused tensor-core flash kernel, the executable spec a
  future through-the-contraction-path tensor-core flash tier must reproduce.
- **model attention chains** — TinyLlama ``LlamaAttention`` bisection (chained Linears → QKV+SDPA →
  full RoPE attention) that localizes a whole-block accuracy regression.

GPU accuracy in the correctness lane; the warp-tier needs sm_90+ where pinned.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ..conftest import from_pretrained_or_skip, requires_cuda


class _Sdpa(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class _Causal(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


class _Gqa(torch.nn.Module):
    """GQA SDPA. ``enable_gqa=True`` is a bool kwarg the tracer's is_causal scan grabs (the default
    ``is_causal=False`` is dropped by dynamo), so this traces as GQA **and** causal — the only GQA
    form reachable through the public torch API here, and the Qwen3-Embedding layer-0 shape."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)


class _Masked(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _trace(module, args, dynamic_shapes=None):
    """Trace + compile ``module``; return ``(backend, compiled, graph, kernel_node_ids)``."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    graph = trace_module(module.cpu(), args, dynamic_shapes=dynamic_shapes)
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, graph, kernels


def _max_diff(backend, compiled, feed: dict, ref_fn) -> float:
    """Run emmy + the torch eager ``ref_fn`` under one GPU-lock window; return max|Δ|."""
    run_result, eager = backend.run(compiled, input_data=feed, pre_run=ref_fn)
    got = list(run_result.outputs.values())[0].flatten()
    assert got.shape == eager.shape
    assert not np.any(np.isnan(got)), "emmy output has NaN"
    return float(np.max(np.abs(got - eager)))


# =========================================================================== #
# Scalar-tier flash (the FLASH knob).
# =========================================================================== #

# (variant): module factory, torch-SDPA ref kwargs, and the list of static configs to sweep.
# plain/causal/mask use (B, H, S, D); gqa uses (Hq, Hkv, S, D) with B=1.
_FLASH_VARIANTS = {
    "plain": (_Sdpa, {}, [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)]),
    "causal": (_Causal, {"is_causal": True}, [(1, 2, 16, 8)]),
    "gqa": (_Gqa, {"is_causal": True, "enable_gqa": True}, [(4, 2, 16, 8), (16, 8, 32, 16)]),
    "mask": (_Masked, {}, [(1, 2, 16, 8)]),
}


def _flash_feed(variant, B_or_Hq, H_or_Hkv, S, D):
    """Build (module, feed, ref_fn) for one static config of ``variant``."""
    if variant == "gqa":
        Hq, Hkv = B_or_Hq, H_or_Hkv
        q = torch.randn(1, Hq, S, D)
        k, v = (torch.randn(1, Hkv, S, D) for _ in range(2))
        feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    elif variant == "mask":
        B, H = B_or_Hq, H_or_Hkv
        q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
        mask = torch.zeros(1, 1, S, S)
        mask[0, 0, :, S // 2 :] = float("-inf")
        feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}
    else:
        B, H = B_or_Hq, H_or_Hkv
        q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
        feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}

    module_cls, kwargs, _ = _FLASH_VARIANTS[variant]
    module = module_cls()
    cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

    def ref():
        with torch.no_grad():
            if variant == "mask":
                out = F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"], attn_mask=cuda["mask"])
            else:
                out = F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"], **kwargs)
            return out.cpu().flatten().numpy()

    args = (cuda["q"].cpu(), cuda["k"].cpu(), cuda["v"].cpu()) + ((cuda["mask"].cpu(),) if variant == "mask" else ())
    return module, args, feed, ref


@requires_cuda
@pytest.mark.parametrize("variant", list(_FLASH_VARIANTS))
def test_scalar_flash_matches_torch(monkeypatch, variant):
    """With ``FLASH`` on, an SDPA variant (non-causal / causal / GQA / explicit additive mask) fuses
    to ONE streaming online-softmax kernel and matches torch SDPA across the variant's static
    configs. The non-causal kernel carries the streaming softmax markers (``fmaxf`` + ``expf``);
    causal/mask/GQA recognize their per-element guard structurally from the fused body."""
    torch.manual_seed(0)
    for cfg in _FLASH_VARIANTS[variant][2]:
        module, args, feed, ref = _flash_feed(variant, *cfg)
        backend, compiled, _graph, kernels = _trace(module, args)
        assert len(kernels) == 1, f"{variant}{cfg}: flash should fuse to one kernel, got {len(kernels)}"
        if variant == "plain" and cfg == _FLASH_VARIANTS["plain"][2][0]:
            src = compiled.nodes[kernels[0]].op.kernel_source
            assert "fmaxf" in src and "expf" in src, "fused kernel should carry the streaming softmax (max + exp)"
        md = _max_diff(backend, compiled, feed, ref)
        assert md < 1e-4, f"{variant}{cfg}: flash vs torch max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize("variant", ["plain", "gqa", "mask"])
def test_scalar_flash_dynamic_matches_torch(monkeypatch, variant):
    """Symbolic ``seq_len`` (Q/K/V dim -2): ONE cached kernel carrying ``int seq_len`` serves every
    runtime size — flash's single dynamic axis lands on the masked-row M, the symbolic reduce, and
    (for GQA) the causal guard at once. Accurate vs torch at seq ∈ {8, 16, 37}."""
    torch.manual_seed(0)
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    module_cls, kwargs, _ = _FLASH_VARIANTS[variant]
    Hq, Hkv, D = (4, 2, 8) if variant == "gqa" else (2, 2, 8)

    if variant == "mask":
        ds = {"q": {2: seq}, "k": {2: seq}, "v": {2: seq}, "mask": {2: seq, 3: seq}}
        seed_args = (torch.randn(1, 2, 16, D), torch.randn(1, 2, 16, D), torch.randn(1, 2, 16, D), torch.zeros(1, 1, 16, 16))
    else:
        ds = {"q": {2: seq}, "k": {2: seq}, "v": {2: seq}}
        seed_args = (torch.randn(1, Hq, 16, D), torch.randn(1, Hkv, 16, D), torch.randn(1, Hkv, 16, D))
    backend, compiled, _graph, kernels = _trace(module_cls(), seed_args, dynamic_shapes=ds)
    assert len(kernels) == 1, f"dynamic {variant} flash should fuse to one kernel, got {len(kernels)}"
    assert "int seq_len" in compiled.nodes[kernels[0]].op.kernel_source, "dynamic kernel must carry the runtime seq_len arg"

    for s in (8, 16, 37):
        if variant == "mask":
            q, k, v = (torch.randn(1, 2, s, D) for _ in range(3))
            mask = torch.zeros(1, 1, s, s)
            mask[0, 0, :, s // 2 :] = float("-inf")
            feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}
            cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

            def ref(c=cuda):
                with torch.no_grad():
                    return F.scaled_dot_product_attention(c["q"], c["k"], c["v"], attn_mask=c["mask"]).cpu().flatten().numpy()
        else:
            q = torch.randn(1, Hq, s, D)
            k, v = (torch.randn(1, Hkv, s, D) for _ in range(2))
            feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
            cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

            def ref(c=cuda, kw=kwargs):
                with torch.no_grad():
                    return F.scaled_dot_product_attention(c["q"], c["k"], c["v"], **kw).cpu().flatten().numpy()

        md = _max_diff(backend, compiled, feed, ref)
        assert md < 1e-4, f"dynamic {variant} flash seq={s} max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize("bk", [2, 4])
def test_scalar_flash_kv_tile_matches_torch(monkeypatch, bk):
    """KV tiling: a ``EMMY_BK`` pin re-brackets the streaming reduce ``S_k → S_k/BK · BK``
    (serial within the tile). The fused flash kernel must still fuse to one kernel and match torch.
    ``S=32`` / ``D=16`` are divisible by both 2 and 4, so the pin is honored."""
    monkeypatch.setenv("EMMY_BK", str(bk))
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 32, 16) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"flash should still fuse to one kernel under BK={bk}, got {len(kernels)}"
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv).cpu().flatten().numpy()

    md = _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 1e-4, f"BK={bk} KV-tiled flash vs torch max_diff={md:.6e}"


@requires_cuda
def test_flash_causal_and_gqa_match_torch(monkeypatch):
    """Scalar flash keeps the causal / GQA masks in the ``d``-invariant score prefix, so masked +
    grouped-head flash also matches torch (one streaming online-softmax kernel)."""
    torch.manual_seed(0)

    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Causal(), (q, k, v))
    assert len(kernels) == 1
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def rc():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv, is_causal=True).cpu().flatten().numpy()

    assert _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, rc) < 1e-4

    qg = torch.randn(1, 4, 16, 8)
    kg, vg = (torch.randn(1, 2, 16, 8) for _ in range(2))
    backend, compiled, _graph, _kernels = _trace(_Gqa(), (qg, kg, vg))
    cqg, ckg, cvg = qg.cuda(), kg.cuda(), vg.cuda()

    def rg():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cqg, ckg, cvg, is_causal=True, enable_gqa=True).cpu().flatten().numpy()

    assert _max_diff(backend, compiled, {"q": qg.numpy(), "k": kg.numpy(), "v": vg.numpy()}, rg) < 1e-4


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)])
def test_flash_chain_matches_torch(monkeypatch, B, H, S, D):
    """The FA-2 shared-score scalar chain — the P@V output ``d`` rides a register vector ``O[BM, D]``,
    the QK^T score computed once per KV step and shared across ``d`` (one kernel, scalar FMA P@V).
    **Xfailed:** this scalar-chain restructuring is not yet rebuilt (no ``O_i_0`` register vector is
    emitted today), so the ``O_i_0`` assertion below fails."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"chain flash should fuse to one kernel, got {len(kernels)}"
    assert "O_i_0" in compiled.nodes[kernels[0]].op.kernel_source, "chain form must carry the O[d] register vector"
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv).cpu().flatten().numpy()

    md = _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 1e-4, f"chain flash max_diff={md:.6e}"


# =========================================================================== #
# Tensor-core flash — the fragment-resident warp tier.
# =========================================================================== #
# These cases expect a fp16/bf16 SDPA to lower to a single ``mma.sync`` kernel (the warp chain:
# tiled + atomized contractions, fragment online-softmax, C->A register repack) — realized through the
# ONE pipeline: ``_schedule._twisted_warp_options`` stamps the mma ``TilePlan``\ s on the Q@K / P@V
# ``Contraction``\ s and ``_bind``'s reduce arm realizes the TWISTED carrier at fragment residence
# (``_twist``). No private emitter exists; a bespoke path would be the mandate violation the
# demolition removed. Unpinned, the warp rows are fork SIBLINGS of the chain / reduce-partition
# forms (the flash-form fork) — the cold ``AnalyticPrior`` pick stays warp-when-eligible, which is
# what these cold-compile cases pin.


def _compile_tc(q, k, v, module=None):
    return _trace(module if module is not None else _Sdpa(), (q, k, v))


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (2, 3, 64, 32), (1, 4, 128, 64), (1, 1, 16, 16)])
def test_generated_tensorcore_flash_matches_torch(monkeypatch, B, H, S, D):
    torch.manual_seed(S + D)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"fused TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_mma_m16n8k16_f16" in src and "dpl_ldmatrix_x4" in src, "the generated kernel must use the shared tensor-core ops"
    assert "dpl_c_to_a" in src, "the generated kernel must be the fused warp-chain (C->A register repack)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"generated TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


def test_flash_form_fork_offers_geometry_grid():
    """The flash-form fork's enumerated rows (live-fork capture, no GPU): fp16 offers the full
    warp move grid (every divisibility-legal ``(warps_m, key_atoms)`` point), each geometry crossed
    with its K/V operand-stage candidates (gmem-direct option-0 + the resolver-gated cp.async ring
    depths), plus the chain and the per-cell serial escape — every row spelling the SAME
    ``TILE@dd`` / ``TILE@pj`` / ``REDUCE@kv`` / ``STAGE@kv`` key set (the evidence pick's
    prefix-consistency); f32 (no mma atom) offers chain + serial."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.schedule import is_warp_codec  # noqa: PLC0415
    from emmy.compiler.pipeline.search.analytic import enumerate_graph  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import twisted_warp_moves  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    ctx = Context.from_target((12, 0))
    for dtype, want_warp in ((torch.float16, len(twisted_warp_moves())), (torch.float32, 0)):
        q, k, v = (torch.randn(1, 4, 128, 64, dtype=dtype) for _ in range(3))
        graph = trace_module(_Sdpa().cpu(), (q, k, v))
        rows = [r for r in enumerate_graph(graph, ctx) if "TILE@dd" in r or "TILE@pj" in r or "REDUCE@kv" in r]
        assert all({"TILE@dd", "TILE@pj", "REDUCE@kv", "STAGE@kv"} <= set(r) for r in rows), "flash rows must spell one uniform key set"
        warp = [r for r in rows if is_warp_codec(r["TILE@dd"])]
        chain = [r for r in rows if not is_warp_codec(r["TILE@dd"]) and r["TILE@pj"]]
        serial = [r for r in rows if not r["TILE@dd"] and not r["TILE@pj"]]
        # (1, 4, 128, 64): every (warps_m, key_atoms) point is divisibility-legal (128 % (um·16) == 0,
        # 128 % (nt·8) == 0), so the fp16 pool spans the whole geometry grid; each geometry offers a
        # gmem-direct row plus at least one resolved cp.async stage row (the exact stage count is
        # budget-dependent — the depth clamp dedups on the resolved spelling).
        geoms = {r["TILE@dd"] for r in warp}
        assert len(geoms) == want_warp, f"{dtype}: expected {want_warp} warp geometries, got {len(geoms)}"
        for g in geoms:
            stages = {r["STAGE@kv"] for r in warp if r["TILE@dd"] == g}
            assert "" in stages, f"{dtype} {g}: the gmem-direct option-0 row is missing"
            assert any("cp" in s for s in stages), f"{dtype} {g}: no resolved cp.async stage row"
        assert all(not r["STAGE@kv"] for r in [*chain, *serial]), "chain/serial rows stamp the decided-empty stage"
        assert len(chain) == 1 and len(serial) >= 1, f"{dtype}: chain/serial siblings missing ({len(chain)}/{len(serial)})"


@requires_cuda
@pytest.mark.parametrize(("um", "nt"), [(2, 4), (4, 8)])
def test_warp_flash_geometry_pin_matches_torch(monkeypatch, um, nt):
    """A pinned non-conservative warp-flash geometry — several warps per CTA, a wide streaming key
    block (the per-step C→A slices) — still lowers through the one fragment realizer and matches
    torch. Pins the move grid's realizability, not just the conservative option-0 the cold pick takes."""
    monkeypatch.setenv("EMMY_TILE", f"a:mma_m16n8k16_f16/w{um}x1/f1x{nt}/k4")
    torch.manual_seed(um * 10 + nt)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"pinned warp flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_c_to_a" in src, "must be the fused warp form (C->A register repack)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"pinned warp flash w{um}/f1x{nt} max_diff={max_diff:.2e}"


# --------------------------------------------------------------------------- #
# Staged K/V (the ``STAGE@<kv>`` cp.async stream) — Moves 4/5 on the warp tier.
# --------------------------------------------------------------------------- #


def _run_flash(backend, compiled, graph, tensors) -> np.ndarray:
    data = {n: t.numpy() for n, t in zip(graph.inputs, tensors, strict=True)}
    run_result, _ = backend.run(compiled, input_data=data)
    return list(run_result.outputs.values())[0].flatten()


@requires_cuda
@pytest.mark.parametrize("stage", ["d1/cp", "d2/cp/ring", "d3/cp/ring", "d1/tma", "d2/tma/ring"])
def test_staged_warp_flash_matches_torch(monkeypatch, stage):
    """A pinned K/V operand ``STAGE`` on the warp-flash stream: the kernel fills per-block K/V smem
    slabs (cooperative cp.async into padded rows, or rank-N TMA box copies into dense
    hardware-swizzled slabs — the batched operands encode with leading extent-1 box dims) and
    drains them via the staged ldmatrix variants. ``d1`` single-buffer; ``d2+/ring`` the prefetch
    ring overlapping the next block's loads with this block's mma work. Matches torch."""
    monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(7)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"staged warp flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "_k_smem" in src and "_v_smem" in src, "the staged stream must fill K/V slabs"
    if "tma" in stage:
        assert "cp_async_bulk_tensor_4d" in src, "the batched K/V must box-copy via the rank-4 TMA descriptor"
    else:
        assert "cp.async" in src, "the cp transport must fill via cp.async"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"staged ({stage}) warp flash max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("stage", ["d1/cp", "d2/cp/ring"])
def test_staged_warp_flash_bit_identical_to_gmem_direct(monkeypatch, stage):
    """Staging is a pure perf transform (the matmul tier's invariant, carried to the stream): the
    K/V slab fills are verbatim row copies and the mma order is unchanged, so the staged kernel's
    output is BIT-identical to its gmem-direct sibling on the same inputs and geometry."""
    torch.manual_seed(11)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    monkeypatch.setenv("EMMY_TILE", "a:mma_m16n8k16_f16/w2x1/f1x4/k4")
    backend, compiled, graph, _ = _compile_tc(q, k, v)
    base = _run_flash(backend, compiled, graph, (q, k, v))
    monkeypatch.setenv("EMMY_STAGE", stage)
    backend2, compiled2, graph2, kernels2 = _compile_tc(q, k, v)
    assert "cp.async" in compiled2.nodes[kernels2[0]].op.kernel_source
    staged = _run_flash(backend2, compiled2, graph2, (q, k, v))
    assert np.array_equal(base, staged), f"staged ({stage}) output differs from its gmem-direct sibling"


@requires_cuda
@pytest.mark.parametrize("stage", ["d2/cp/ring", "d2/tma/ring"])
def test_staged_warp_flash_causal_and_gqa_match_torch(monkeypatch, stage):
    """The staged stream composes with the fragment causal mask and the GQA ``head // group`` K/V
    indexing — both ride the slab fill's operand index verbatim (the σ passes batch/head terms
    through; the TMA transport carries them as box-origin coords), so no staging-side special
    case exists to regress."""
    monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(13)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v, module=_Causal())
    assert ("cp_async_bulk_tensor" if "tma" in stage else "cp.async") in compiled.nodes[kernels[0]].op.kernel_source

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert float(np.max(np.abs(got - eager))) < 5e-3, "staged causal warp flash drifted from torch"

    qg = torch.randn(1, 4, 128, 32, dtype=torch.float16)
    kg, vg = (torch.randn(1, 2, 128, 32, dtype=torch.float16) for _ in range(2))
    backend, compiled, graph, kernels = _compile_tc(qg, kg, vg, module=_Gqa())
    assert ("cp_async_bulk_tensor" if "tma" in stage else "cp.async") in compiled.nodes[kernels[0]].op.kernel_source

    def rg():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(qg.cuda(), kg.cuda(), vg.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (qg.numpy(), kg.numpy(), vg.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=rg)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert float(np.max(np.abs(got - eager))) < 5e-3, "staged GQA warp flash drifted from torch"


@requires_cuda
def test_staged_flash_symbolic_declines_to_gmem_direct(monkeypatch):
    """A ``STAGE`` pin on a SYMBOLIC ``seq_len`` flash declines (the resolver stages a static,
    block-divisible kv only — the masked gmem-direct fragment loads keep the symbolic path correct)
    and the kernel still compiles and matches torch — the standard pin-validity degrade."""
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(_Sdpa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_c_to_a" in src, "must still be the fused warp form"
    assert "_k_smem" not in src and "_v_smem" not in src, "a symbolic stream must decline the K/V stage"

    torch.manual_seed(37)
    q, k, v = (torch.randn(B, H, 37, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert float(np.max(np.abs(got - eager))) < 5e-3, "declined-stage symbolic flash drifted from torch"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (1, 4, 128, 64)])
def test_generated_tensorcore_flash_bf16_matches_torch(monkeypatch, B, H, S, D):
    """bf16 in, f32 accumulate. Same fused warp-chain as fp16 (the 16-bit operand dtype only swaps
    the mma atom / PTX dtype field); validated vs torch SDPA."""
    torch.manual_seed(S + D + 1)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.bfloat16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"fused TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_mma_m16n8k16_bf16" in src, "the bf16 flash must use the bf16 mma atom"
    assert "dpl_c_to_a" in src, "the generated kernel must be the fused warp-chain (C->A register repack)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t.view(torch.uint16).numpy() for n, t in zip(graph.inputs, (q, k, v), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got_bits = list(run_result.outputs.values())[0].flatten().astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy()
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-2, f"generated bf16 TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (1, 4, 128, 64)])
def test_generated_tensorcore_flash_causal_bf16_matches_torch(monkeypatch, B, H, S, D):
    """The cross-product: bf16 operands AND the fragment causal mask, together. The softmax realizer
    is dtype-agnostic (f32 algebra) and causal is a score-partial mask, so the two compose with no
    special-casing — validated vs torch's bf16 is_causal SDPA."""
    torch.manual_seed(S + D + 2)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.bfloat16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v, module=_Causal())
    assert len(kernels) == 1, f"fused causal bf16 TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_mma_m16n8k16_bf16" in src and "dpl_c_to_a" in src

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t.view(torch.uint16).numpy() for n, t in zip(graph.inputs, (q, k, v), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got_bits = list(run_result.outputs.values())[0].flatten().astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy()
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-2, f"generated causal bf16 TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (2, 3, 64, 32), (1, 4, 128, 64), (1, 1, 16, 16)])
def test_generated_tensorcore_flash_causal_matches_torch(monkeypatch, B, H, S, D):
    """Causal masking at the fragment tier. The fused warp-chain inserts a per-element
    ``FragmentMask`` (causal) on the score fragment (strict upper triangle → ``-1e30`` before the
    rowmax), matching torch's ``is_causal=True`` SDPA."""
    torch.manual_seed(S + D)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v, module=_Causal())
    assert len(kernels) == 1, f"fused causal TC flash should be one kernel, got {len(kernels)}"
    assert "dpl_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"generated causal TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_dynamic_matches_torch(monkeypatch, seq):
    """Symbolic ``seq_len`` warp-chain flash. ONE cached fused-TC kernel carrying ``int seq_len``
    serves every runtime size: the partial final KV / query tile (seq=37 straddles both) is masked
    at the score fragment, its K/V gmem loads clamped, its output store guarded. Matches torch SDPA
    at seq ∈ {8, 16, 37, 64}."""
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(_Sdpa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    assert len(kernels) == 1, f"dynamic warp-chain flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_c_to_a" in src, "the symbolic flash must be the fused warp-chain (C->A register repack)"
    assert "int seq_len" in src, "the symbolic warp-chain must carry the runtime seq_len arg"

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_causal_dynamic_matches_torch(monkeypatch, seq):
    """Symbolic ``seq_len`` warp-chain flash with **causal** masking (equal-head). The causal
    score-fragment mask (``kv_col > q_row`` → ``-1e30``) composes with the symbolic boundary mask
    (both write soft −inf before the rowmax). Matches torch ``is_causal=True`` SDPA."""
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(_Causal(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    assert len(kernels) == 1, f"dynamic causal warp-chain flash should fuse to one kernel, got {len(kernels)}"
    assert "dpl_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"causal symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"causal symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(("Hq", "Hkv", "S", "D"), [(4, 2, 32, 16), (16, 8, 32, 32)])
def test_warp_chain_gqa_static_matches_torch(monkeypatch, Hq, Hkv, S, D):
    """STATIC ``S`` warp-chain flash with GQA (``head // group`` K/V indexing). ``_Gqa`` traces as
    GQA+causal; the fused warp-chain reads K/V at the kv-head with no materialized broadcast."""
    torch.manual_seed(S + D)
    q = torch.randn(1, Hq, S, D, dtype=torch.float16)
    k, v = (torch.randn(1, Hkv, S, D, dtype=torch.float16) for _ in range(2))
    backend, compiled, graph, kernels = _compile_tc(q, k, v, module=_Gqa())
    assert len(kernels) == 1, f"static GQA warp-chain flash should be one kernel, got {len(kernels)}"
    assert "dpl_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"static GQA warp-chain flash {(Hq, Hkv, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_gqa_dynamic_matches_torch(monkeypatch, seq):
    """Symbolic ``seq_len`` warp-chain flash with **GQA** (``Hq=4 / Hkv=2``, group 2). ``_Gqa`` traces
    as GQA+causal, so this also exercises the causal mask composed with the symbolic boundary mask.
    ONE cached kernel carrying ``int seq_len``; matches torch GQA+causal SDPA at seq ∈ {8,16,37,64}."""
    B, Hq, Hkv, D = 1, 4, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = (
        torch.randn(B, Hq, 16, D, dtype=torch.float16),
        torch.randn(B, Hkv, 16, D, dtype=torch.float16),
        torch.randn(B, Hkv, 16, D, dtype=torch.float16),
    )
    backend, compiled, graph, kernels = _trace(_Gqa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    assert len(kernels) == 1, f"dynamic GQA warp-chain flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_c_to_a" in src and "int seq_len" in src, "must be the symbolic fused warp-chain"

    torch.manual_seed(seq)
    q = torch.randn(B, Hq, seq, D, dtype=torch.float16)
    k, v = (torch.randn(B, Hkv, seq, D, dtype=torch.float16) for _ in range(2))

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"GQA symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"GQA symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


# =========================================================================== #
# Cooperative-KV flash (BR) — the KV axis split across threads, monoid combine.
# =========================================================================== #


@requires_cuda
@pytest.mark.parametrize("br", ["32", "64"])
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 64, 16), (1, 4, 128, 32)])
def test_cooperative_flash_matches_torch(monkeypatch, br, B, H, S, D):
    """A cooperative-KV flash (BR>1) fuses to one kernel carrying the monoid cross-thread combine
    (``__shfl_xor_sync`` for BR≤32, a per-component smem tree for BR>32) and matches torch SDPA —
    the KV parallelization is accuracy-preserving (the LSE monoid is associative + commutative)."""
    monkeypatch.setenv("EMMY_REDUCE", f"b{br}")
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "__shfl_xor_sync" in src or "_smem" in src, "cooperative-KV flash must carry the cross-thread monoid combine"
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def eager():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv).cpu().flatten().numpy()

    assert _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, eager) < 1e-4


# =========================================================================== #
# Validated FA-2 reference kernel — the executable spec the warp-chain must reproduce.
# =========================================================================== #
# A hand-written FA-2 kernel (NOT compiler output) proving the design works end-to-end on real
# hardware and pinning the lane-layout contracts the warp-chain codegen relies on: the Q ldmatrix.x4
# A fragment, the transposed-B native K pack, the fragment online-softmax (rowmax/rowsum + the
# 4-lane butterfly), the C→A smem handoff, the canonical-B V load. One warp / 16 query rows, D=16.

_KERNEL = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ void mma_m16n8k16(float* d, const unsigned* a, const unsigned* b, const float* c){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]),
      "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
// A (m16k16): ldmatrix.x4 — row=lane%16, k-block=(lane/16)*8.
__device__ __forceinline__ void ldm_a(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; unsigned addr=__cvta_generic_to_shared(sm + (lane%16)*ldm + (lane/16)*8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(addr));
}
// canonical B[k,n] k-major: ldmatrix.x2.trans -> col-major; row=lane%16.
__device__ __forceinline__ void ldm_b_trans(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; unsigned addr=__cvta_generic_to_shared(sm + (lane%16)*ldm);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
    :"=r"(r[0]),"=r"(r[1]):"r"(addr));
}
// transposed-B (Q@K^T) native col-major: manual pack. n=lane/4, k=(lane%4)*2{+8}.
__device__ __forceinline__ void load_b_native(unsigned* r, const __half* sm, int ldm){
  int lane=threadIdx.x&31; int n=lane/4; int kb=(lane%4)*2;
  __half2 h0=__halves2half2(sm[n*ldm+kb+0], sm[n*ldm+kb+1]);
  __half2 h1=__halves2half2(sm[n*ldm+kb+8+0], sm[n*ldm+kb+8+1]);
  r[0]=*reinterpret_cast<unsigned*>(&h0); r[1]=*reinterpret_cast<unsigned*>(&h1);
}

extern "C" __global__ void fa2(const __half* Q,const __half* K,const __half* V,float* O,int S,float scale){
  int qb = blockIdx.x; int lane=threadIdx.x&31; const int D=16;
  __shared__ __half qs[16*16], ks[16*16], vs[16*16], ps[16*16];
  for(int i=lane;i<16*D;i+=32){ qs[i]=Q[(qb*16)*D + i]; }
  __syncwarp();
  unsigned qa[4]; ldm_a(qa, qs, D);                 // Q -> A fragment, once per query tile
  float m0=-1e30f,m1=-1e30f,l0=0,l1=0;              // online stats, rows g / g+8 per lane
  float Of[2][4]={{0,0,0,0},{0,0,0,0}};             // O[16,D] accumulator (2 N-tiles of d)
  int g = lane/4;
  for(int kv0=0; kv0<S; kv0+=16){                   // KV stream
    for(int i=lane;i<16*D;i+=32){ ks[i]=K[(kv0)*D+i]; vs[i]=V[(kv0)*D+i]; }
    __syncwarp();
    float Sf[2][4];                                 // QK^T mma -> score C-fragments
    for(int nt=0;nt<2;nt++){
      unsigned kb[2]; load_b_native(kb, ks + nt*8*D, D);
      float z[4]={0,0,0,0}; mma_m16n8k16(Sf[nt], qa, kb, z);
      for(int e=0;e<4;e++) Sf[nt][e]*=scale;
    }
    float r0=fmaxf(fmaxf(Sf[0][0],Sf[0][1]),fmaxf(Sf[1][0],Sf[1][1]));   // fragment rowmax
    float r1=fmaxf(fmaxf(Sf[0][2],Sf[0][3]),fmaxf(Sf[1][2],Sf[1][3]));
    r0=fmaxf(r0,__shfl_xor_sync(-1,r0,2)); r0=fmaxf(r0,__shfl_xor_sync(-1,r0,1));
    r1=fmaxf(r1,__shfl_xor_sync(-1,r1,2)); r1=fmaxf(r1,__shfl_xor_sync(-1,r1,1));
    float mn0=fmaxf(m0,r0), mn1=fmaxf(m1,r1);
    float a0=__expf(m0-mn0), a1=__expf(m1-mn1);      // α rescale (combine_states)
    float Pf[2][4]; float s0=0,s1=0;
    for(int nt=0;nt<2;nt++){
      Pf[nt][0]=__expf(Sf[nt][0]-mn0); Pf[nt][1]=__expf(Sf[nt][1]-mn0);
      Pf[nt][2]=__expf(Sf[nt][2]-mn1); Pf[nt][3]=__expf(Sf[nt][3]-mn1);
      s0+=Pf[nt][0]+Pf[nt][1]; s1+=Pf[nt][2]+Pf[nt][3];
    }
    s0+=__shfl_xor_sync(-1,s0,2); s0+=__shfl_xor_sync(-1,s0,1);   // fragment rowsum
    s1+=__shfl_xor_sync(-1,s1,2); s1+=__shfl_xor_sync(-1,s1,1);
    l0=l0*a0+s0; l1=l1*a1+s1;
    for(int nt=0;nt<2;nt++){ Of[nt][0]*=a0;Of[nt][1]*=a0;Of[nt][2]*=a1;Of[nt][3]*=a1; }
    int c0=(lane%4)*2;                               // C->A handoff: P C-frag -> smem row-major
    for(int nt=0;nt<2;nt++){
      ps[g*16 + nt*8 + c0+0]=__float2half(Pf[nt][0]);  ps[g*16 + nt*8 + c0+1]=__float2half(Pf[nt][1]);
      ps[(g+8)*16 + nt*8 + c0+0]=__float2half(Pf[nt][2]); ps[(g+8)*16 + nt*8 + c0+1]=__float2half(Pf[nt][3]);
    }
    __syncwarp();
    unsigned pa[4]; ldm_a(pa, ps, 16);               // P@V mma: A=P (ldmatrix), B=V canonical
    for(int nt=0;nt<2;nt++){
      unsigned vb[2]; ldm_b_trans(vb, vs + nt*8, D);
      mma_m16n8k16(Of[nt], pa, vb, Of[nt]);
    }
    m0=mn0; m1=mn1;
  }
  int c0=(lane%4)*2;                                 // epilogue O/l + store (C-frag layout)
  for(int nt=0;nt<2;nt++){
    O[((qb*16)+g)*D + nt*8 + c0+0]=Of[nt][0]/l0;   O[((qb*16)+g)*D + nt*8 + c0+1]=Of[nt][1]/l0;
    O[((qb*16)+g+8)*D + nt*8 + c0+0]=Of[nt][2]/l1; O[((qb*16)+g+8)*D + nt*8 + c0+1]=Of[nt][3]/l1;
  }
}
"""


@requires_cuda
@pytest.mark.parametrize("S", [16, 32, 64, 128])
def test_fused_tensorcore_flash_reference_matches_torch(S):
    """The hand-written fused tensor-core flash matches torch SDPA across the KV stream (1–8 tiles).
    The validated spec for the warp-chain codegen — every lane layout (A/B fragments, the
    C-fragment row reduction, the C→A handoff) is exercised here."""
    import cupy as cp  # noqa: PLC0415

    from emmy.compiler.backend.cuda import nvcc  # noqa: PLC0415

    fn = nvcc.load_function(_KERNEL, "fa2", "", uses_tma=False)
    torch.manual_seed(S)
    D = 16
    q, k, v = (torch.randn(S, D, dtype=torch.float16) for _ in range(3))
    dq, dk, dv = (cp.asarray(t.numpy()) for t in (q, k, v))
    d_out = cp.zeros((S, D), cp.float32)
    fn((S // 16,), (32,), (dq, dk, dv, d_out, np.int32(S), np.float32(1.0 / np.sqrt(D))))
    got = torch.from_numpy(cp.asnumpy(d_out))
    ref = torch.nn.functional.scaled_dot_product_attention(q.cuda().float(), k.cuda().float(), v.cuda().float()).cpu()
    max_diff = float((got - ref).abs().max())
    assert max_diff < 2e-3, f"fused TC flash S={S} max_diff={max_diff:.2e}"


# =========================================================================== #
# Model attention chains — TinyLlama LlamaAttention bisection.
# =========================================================================== #
# When ``test_block_accuracy::test_tinyllama_block_accuracy[cuda]`` fails, these bisect WHERE in the
# attention sub-block the divergence appears: chained Linears (every matmul) → QKV + masked SDPA (no
# RoPE) → the real LlamaAttention (Q/K/V + RoPE + masked SDPA + O). Random fp32 weights hit the same
# magnitude regime as the block test; thresholds are tight (1e-4) — a larger drift is a real bug.


@pytest.fixture
def _chain_tile_pins(monkeypatch):
    """Pin a small, budget-safe scalar tile + a fixed seed for the model-chain tests. These chains
    compile the real attention path UNPINNED, which relied on the retired prior to pick an
    in-smem-budget tile; the cold emission-order pick can choose an over-budget tile and hard-fail.
    The tile is irrelevant to the accuracy checks (legacy env pins route through the ingest mapper)."""
    torch.manual_seed(42)
    for k, v in (("BN", "16"), ("BM", "8"), ("FN", "2"), ("FM", "2"), ("BK", "8"), ("BR", "4")):
        monkeypatch.setenv(f"EMMY_{k}", v)


def _run_module_with_eager(module: torch.nn.Module, args: tuple, inputs_by_name: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Trace + compile ``module``, then run the emmy kernels and the torch eager reference under
    one ``backend.run`` GPU-lock window via ``pre_run``. Returns ``(emmy_flat, eager_flat)``."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.loader.binder import apply_load_ops  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    graph = trace_module(module.cpu(), args)
    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_set = set(compiled.inputs)
    feed: dict[str, np.ndarray] = {}
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if nid in input_set and nid in inputs_by_name:
            feed[nid] = inputs_by_name[nid]
        elif isinstance(node.op, ConstantOp):
            n = 1
            for d in node.output.shape:
                n *= d.as_static()
            for key, p in module.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                # Match by the ConstantOp's stored name (which carries the placeholder identity
                # through ``004a`` const-fold, even when the graph node id changes).
                if safe_key.endswith(node.op.name[2:]) and p.numel() == n:
                    arr = p.detach().cpu().numpy()
                    feed[nid] = apply_load_ops(arr, node.op.load_ops)
                    break
            if nid not in feed and node.op.value is not None:
                feed[nid] = np.array([node.op.value], dtype=np.float32)

    cuda_module = module.cuda()
    cuda_args = tuple(a.cuda() for a in args)

    def eager_pre_run() -> np.ndarray:
        with torch.no_grad():
            out = cuda_module(*cuda_args)
        if isinstance(out, tuple):
            out = out[0]
        return out.cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data=feed, pre_run=eager_pre_run)
    dpd = list(run_result.outputs.values())[0].flatten()
    return dpd, eager


def _assert_close(emmy: np.ndarray, eager: np.ndarray, threshold: float = 1e-4) -> None:
    assert emmy.shape == eager.shape, f"shape: {emmy.shape} vs {eager.shape}"
    assert not np.any(np.isnan(emmy)), "emmy output has NaN"
    max_diff = float(np.max(np.abs(emmy - eager)))
    mean_diff = float(np.mean(np.abs(emmy - eager)))
    max_eager = float(np.max(np.abs(eager)))
    assert max_diff < threshold, f"max_diff={max_diff:.6f} >= {threshold} (mean={mean_diff:.6f}, max_eager={max_eager:.3f})"


class _StackedLinears(torch.nn.Module):
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o(self.q(x))


@requires_cuda
def test_two_linears_tinyllama_shape(_chain_tile_pins):
    """Two chained 2048×2048 Linears at TinyLlama hidden size and seq=32. Confirms basic
    matmul-chain accuracy — if this fails, every matmul is broken."""
    m = _StackedLinears().eval()
    x = torch.randn(1, 32, 2048)
    dpd, eager = _run_module_with_eager(m, (x,), {"x": x.numpy()})
    _assert_close(dpd, eager)


class _QKVAttnNoRope(torch.nn.Module):
    def __init__(self, hidden: int = 2048, n_heads: int = 32, head_dim: int = 64):
        super().__init__()
        self.h, self.d = n_heads, head_dim
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, hidden, bias=False)
        self.v = torch.nn.Linear(hidden, hidden, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, S, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, S, self.h, self.d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o(out)


@requires_cuda
def test_qkv_attn_no_rope(_chain_tile_pins):
    """Q/K/V Linears + causal SDPA + O Linear, no RoPE. Confirms the matmul-chain + masked-SDPA
    composition is numerically sound on its own."""
    m = _QKVAttnNoRope().eval()
    x = torch.randn(1, 32, 2048)
    dpd, eager = _run_module_with_eager(m, (x,), {"x": x.numpy()})
    _assert_close(dpd, eager)


class _SdpaExplicitMask(torch.nn.Module):
    """SDPA fed an explicit additive float ``attn_mask`` (the way HF passes its precomputed causal
    mask) rather than ``is_causal=True``."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@requires_cuda
@pytest.mark.parametrize("n_heads,seq_len", [(1, 32), (16, 32)])
def test_sdpa_explicit_additive_mask(_chain_tile_pins, n_heads: int, seq_len: int):
    """SDPA with an explicit additive float mask must apply the mask, not silently drop it.
    Regression for the tracer capturing only ``Q/K/V`` and discarding ``attn_mask`` — which turned
    whole-model causal attention into full bidirectional attention. Uses varying random Q/K/V and a
    tight threshold to actually exercise masking (a ``(1,1,S,S)`` additive bias)."""
    head_dim = 128
    m = _SdpaExplicitMask().eval()
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)
    mask = torch.zeros((seq_len, seq_len))
    mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    mask = mask[None, None]
    dpd, eager = _run_module_with_eager(m, (q, k, v, mask), {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()})
    _assert_close(dpd, eager)


def _run_self_attn_tinyllama(seq_len: int, threshold: float = 1e-4) -> None:
    """Run TinyLlama's ``LlamaAttention`` sub-module at ``seq_len`` and verify emmy matches
    eager (forced MATH SDPA backend) within ``threshold``."""
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    config.num_hidden_layers = 1
    block = AutoModelForCausalLM.from_config(config).float().model.layers[0].eval()
    attn = block.self_attn

    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads

    x = torch.randn(1, seq_len, hidden)
    cos = torch.randn(1, 1, seq_len, head_dim)
    sin = torch.randn(1, 1, seq_len, head_dim)

    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.loader.binder import apply_load_ops  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    attn_cpu = attn.cpu()
    graph = trace_module(attn_cpu, (x,), kwargs={"position_embeddings": (cos, sin)})
    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_set = set(compiled.inputs)
    feed: dict[str, np.ndarray] = {}
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if nid in input_set:
            if nid == "hidden_states":
                feed[nid] = x.numpy()
            elif nid == "position_embeddings_0":
                feed[nid] = cos.numpy()
            elif nid == "position_embeddings_1":
                feed[nid] = sin.numpy()
        elif isinstance(node.op, ConstantOp):
            n = 1
            for d in node.output.shape:
                n *= d.as_static()
            for key, p in attn_cpu.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(node.op.name[2:]) and p.numel() == n:
                    arr = p.detach().cpu().numpy()
                    feed[nid] = apply_load_ops(arr, node.op.load_ops)
                    break
            if nid not in feed and node.op.value is not None:
                feed[nid] = np.array([node.op.value], dtype=np.float32)

    attn_cuda = attn.cuda()
    x_cuda, cos_cuda, sin_cuda = x.cuda(), cos.cuda(), sin.cuda()

    def eager_pre_run() -> np.ndarray:
        # Force the math (naive) SDPA backend so eager and emmy compare the same algorithm —
        # flash re-orders FMAs and would drift O(0.5 × max_eager) at seq ≥ 512. Runs inside
        # ``backend.run``'s GPU lock so eager + emmy share one uninterrupted GPU window.
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = attn_cuda(x_cuda, position_embeddings=(cos_cuda, sin_cuda))[0]
        return out.cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data=feed, pre_run=eager_pre_run)
    dpd = list(run_result.outputs.values())[0].flatten()
    _assert_close(dpd, eager, threshold=threshold)


@requires_cuda
def test_full_self_attn_tinyllama(_chain_tile_pins):
    """The real ``LlamaAttention`` from a TinyLlama config — the smallest scope that includes Q/K/V
    Linears, RoPE, masked SDPA, and O Linear. If this fails while the two simpler chains pass, the
    regression is in the RoPE elementwise kernel or its interaction with the attention numerics."""
    _run_self_attn_tinyllama(seq_len=32, threshold=1e-4)


@requires_cuda
def test_full_self_attn_tinyllama_seq512(_chain_tile_pins):
    """Same at seq_len=512 — the shape that makes the SDPA P@V kernel the dominant cost (32 MB
    materialized score matrix, one CTA per output element). Pins correctness so future fusion /
    cooperative-output-tiling doesn't regress accuracy. Threshold is loose (2.0 ≈ 90% of max_eager):
    at seq=512 with random fp32 weights the naive-vs-naive comparison drifts substantially, and TMA
    (default on sm_90+) reorders FMAs vs cp.async. Catches order-of-magnitude regressions, not
    bit-equivalence."""
    _run_self_attn_tinyllama(seq_len=512, threshold=2.0)
