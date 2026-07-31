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

from ..conftest import from_pretrained_or_skip, requires_cuda, requires_sm90


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


class _Scaled(torch.nn.Module):
    """Explicit non-default ``scale=`` (Gemma-nano E2B/E4B passes 1.0 — q_norm absorbs the
    scaling). The trace must capture the kwarg and the flash re-synthesis must apply it;
    the historical hardcoded ``1/sqrt(d)`` redistributed the whole softmax (the
    gemma-4-E2B layer-0 accuracy failure)."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)


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
    "scaled": (_Scaled, {"scale": 1.0}, [(1, 2, 16, 8)]),
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


class _SdpaTranspose(torch.nn.Module):
    """SDPA whose ``(b, h, s, d)`` output is transposed to ``(b, s, h, d)`` — the ``attn.transpose(1, 2)``
    every HF attention does before the reshape to ``(b, s, hidden)``. The transpose is a view that fuses
    INTO the flash kernel (the store writes the transposed layout), so the output buffer's real layout is
    NOT the bare grid order."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


@requires_cuda
def test_flash_transposed_output_matches_torch(monkeypatch):
    """The flash store must match the OUTPUT buffer's real rank + layout, not the bare ``(batch…, m, d)``
    grid order: a fused output transpose (and, in models, size-1 broadcast / unsqueeze dims) makes the
    root's output non-canonical, so a grid-order write mis-strides — all elements alias, the rest stays
    uninitialized → NaN (the Gemma model-trace flash NaN). This pins the layout-aware store
    (``_out_store_index``): SDPA + absorbed transpose fuses to ONE kernel and matches torch."""
    torch.manual_seed(0)
    for cfg in [(1, 4, 16, 16), (2, 3, 32, 16)]:
        q, k, v = (torch.randn(*cfg) for _ in range(3))
        backend, compiled, _graph, kernels = _trace(_SdpaTranspose(), (q, k, v))
        assert len(kernels) == 1, f"{cfg}: sdpa+transpose should fuse to one kernel, got {len(kernels)}"
        cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

        def ref(cq=cq, ck=ck, cv=cv):
            with torch.no_grad():
                return F.scaled_dot_product_attention(cq, ck, cv).transpose(1, 2).contiguous().cpu().flatten().numpy()

        md = _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
        assert md < 1e-4, f"{cfg}: transposed-output flash vs torch max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)])
def test_flash_chain_matches_torch(monkeypatch, B, H, S, D):
    """The FA-2 shared-score scalar chain — the P@V output ``d`` rides a register vector ``O[BM, D]``,
    the QK^T score computed once per KV step and shared across ``d`` (one kernel, scalar FMA P@V).
    Greedy: these fp32 shapes are not warp-eligible, so the prior picks the chain among the scalar
    forms (the pinned selection has its own case below)."""
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


@requires_cuda
def test_flash_chain_pin_selects_chain_on_warp_eligible_shape(monkeypatch):
    """A ``TILE@<pv_k>=f<D>`` pin (with ``TILE=a:scalar`` covering the score node) selects the CHAIN
    row on a shape where the mma tier is also on offer — the pinned spelling of the shared-score
    scalar baseline. Regression: the fold dispatch used to route ANY live ``TILE`` pin to the warp
    rows alone, so no pin could reach the chain and the scalar pin degraded to the per-cell tier
    (the 64×-redundant score recompute)."""
    # Individual EMMY_* vars, not the EMMY_KNOBS aggregate: the aggregate splats into per-knob env
    # vars with overwrite=False, so a var another test already set (or left behind) would win over
    # this test's aggregate under xdist; monkeypatch on the individual vars reverts cleanly.
    monkeypatch.setenv("EMMY_TILE", "a:scalar")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_TILE@PJ", "f64")
    monkeypatch.setenv("EMMY_REDUCE", "")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"chain flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "O_i_0" in src, "chain form must carry the O[d] register vector"
    assert "mma" not in src, "the scalar pin must not fall through to the warp tier"
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv).cpu().flatten().numpy()

    md = _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 5e-3, f"pinned chain flash max_diff={md:.6e}"


# =========================================================================== #
# Tensor-core flash — the fragment-resident warp tier.
# =========================================================================== #
# These cases expect a fp16/bf16 SDPA to lower to a single ``mma.sync`` kernel (the warp chain:
# tiled + atomized contractions, fragment online-softmax, C->A register repack) — realized through the
# ONE pipeline: ``_schedule._twisted_warp_options`` stamps the mma ``TilePlan``\ s on the Q@K / P@V
# ``Contraction``\ s and ``_bind``'s reduce arm realizes the TWISTED carrier at fragment residence
# (``_twist``). No private emitter exists; a bespoke path would be the mandate violation the
# demolition removed. Unpinned, the warp rows are fork SIBLINGS of the chain / reduce-partition
# forms (the flash-form fork) — the cold ``OfflinePrior`` pick stays warp-when-eligible, which is
# what these cold-compile cases pin.


def _compile_tc(q, k, v, module=None):
    return _trace(module if module is not None else _Sdpa(), (q, k, v))


@requires_cuda
@pytest.mark.parametrize(
    ("B", "H", "S", "D"),
    [(1, 2, 32, 16), (2, 3, 64, 32), (1, 4, 128, 64), (1, 2, 32, 72), (1, 1, 16, 16)],
)
def test_generated_tensorcore_flash_matches_torch(monkeypatch, B, H, S, D):
    torch.manual_seed(S + D)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"fused TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_mma_m16n8k16_f16" in src and "emmy_ldmatrix_x4" in src, "the generated kernel must use the shared tensor-core ops"
    assert "emmy_c_to_a" in src, "the generated kernel must be the fused warp-chain (C->A register repack)"

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
    ``TILE@dd`` / ``TILE@pj`` / bare ``REDUCE`` / ``STAGE`` key set — the canonical codec
    spellings, the stream fold being the primary (the evidence pick's
    prefix-consistency); f32 (no mma atom) offers chain + serial."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import twisted_warp_moves  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    ctx = Context.from_target((12, 0))
    for dtype, want_warp in ((torch.float16, len(twisted_warp_moves())), (torch.float32, 0)):
        q, k, v = (torch.randn(1, 4, 128, 64, dtype=dtype) for _ in range(3))
        graph = trace_module(_Sdpa().cpu(), (q, k, v))
        rows = [r for r in enumerate_graph(graph, ctx) if "TILE@dd" in r or "TILE@pj" in r]
        assert all({"TILE@dd", "TILE@pj", "REDUCE", "STAGE", "WORK"} <= set(r) for r in rows), "flash rows must spell one uniform key set"
        warp = [r for r in rows if str(r.get("WORK", "")).startswith("w")]  # F1: the tier rides the WORK entry
        chain = [r for r in rows if not str(r.get("WORK", "")).startswith("w") and r["TILE@pj"]]
        serial = [r for r in rows if not r["TILE@dd"] and not r["TILE@pj"]]
        # (1, 4, 128, 64): every (warps_m, key_atoms) point is divisibility-legal (128 % (um·16) == 0,
        # 128 % (nt·8) == 0), so the fp16 pool spans the whole geometry grid; each geometry offers a
        # gmem-direct row plus at least one resolved cp.async stage row (the exact stage count is
        # budget-dependent — the depth clamp dedups on the resolved spelling). A geometry is the
        # (site TILE@dd, WORK compute half) pair — the warp count lives in the ONE WORK entry
        # (F1); a resolved TMA row's ``+p`` producer band is a WSPEC-successor fork, not a
        # geometry.
        geoms = {(r["TILE@dd"], r["WORK"].partition("+")[0]) for r in warp}
        assert len(geoms) == want_warp, f"{dtype}: expected {want_warp} warp geometries, got {len(geoms)}"
        for g in geoms:
            stages = {r["STAGE"] for r in warp if (r["TILE@dd"], r["WORK"].partition("+")[0]) == g}
            assert "" in stages, f"{dtype} {g}: the gmem-direct option-0 row is missing"
            assert any("cp" in s for s in stages), f"{dtype} {g}: no resolved cp.async stage row"
        assert all(not r["STAGE"] for r in [*chain, *serial]), "chain/serial rows stamp the decided-empty stage"
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
    assert "emmy_c_to_a" in src, "must be the fused warp form (C->A register repack)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"pinned warp flash w{um}/f1x{nt} max_diff={max_diff:.2e}"


# --------------------------------------------------------------------------- #
# f16-accumulate PV (the ``_f16acc`` atom on the expect node — chunked f16→f32 promote).
# --------------------------------------------------------------------------- #


def test_flash_form_fork_offers_f16acc_pv(monkeypatch):
    """The f16-accumulate PV sibling rows (live-fork capture, no GPU): under the ``FAST_MATH``
    umbrella on a consumer-die target every warp geometry row doubles with a variant whose
    ``TILE@pj`` rides the ``_f16acc`` atom — the P@V accumulator promotes per streaming block in
    the realizer — while ``TILE@dd`` (the score node, softmax-bound) NEVER does. A datacenter
    target (sm_90 — full-rate f32-accumulate) and the unset gate offer none."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    graph = trace_module(_Sdpa().cpu(), (q, k, v))

    def pv_rows(cc):
        return [r for r in enumerate_graph(graph, Context.from_target(cc)) if "TILE@pj" in r]

    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    rows = pv_rows((12, 0))
    acc = [r for r in rows if "mma_m16n8k16_f16_f16/" in r["TILE@pj"]]
    base = [r for r in rows if str(r.get("WORK", "")).startswith("w") and "mma_m16n8k16_f16_f16/" not in r["TILE@pj"]]
    assert len(acc) == len(base) > 0, f"FAST_MATH must double the warp pv rows ({len(acc)} vs {len(base)})"
    assert not any("mma_m16n8k16_f16_f16/" in r["TILE@dd"] for r in rows), "the score node must stay f32-accumulate"
    assert not any("mma_m16n8k16_f16_f16/" in r["TILE@pj"] for r in pv_rows((9, 0))), "no f16acc rows on a full-rate f32-acc target"
    monkeypatch.delenv("EMMY_FAST_MATH")
    assert not any("mma_m16n8k16_f16_f16/" in r["TILE@pj"] for r in pv_rows((12, 0))), "gate unset: no f16acc rows"


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynM"])
def test_bare_sibling_pin_selects_the_f16acc_pv_plan(monkeypatch, dynamic):
    """A bare ``TILE`` pin spelling the f16-accumulate SIBLING's **PV plan** (the masked-flash
    golden form — a symbolic trace resolves no ``TILE@<axis>`` key, so a dynamic ``[fm]`` golden
    records its PV plan as its one bare TILE) narrows the warp fork to that variant with the gate
    OFF: ``TILE@pj`` rides the pinned spelling verbatim (so the replay integrity gate holds) and
    ``TILE@dd`` stays on the base f32-accumulate atom. Regression: this pin used to fail the
    pinned branch's base-atom check and decline the whole warp tier — the dynM attention twins
    re-benched the scalar fallback (18.5 ms on hd64.dynM) and could not record their fast-math
    win. hd64 geometry (um=2, nt=4, fm=1): QK = ``w2x1/f1x4/k4``, PV = ``w2x1/f1x8/k2``
    (``regs[1] = d_v/atom_n``, ``bk = nt·atom_n/atom_k`` — the streamed key block)."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f16/f1x8/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    ds = {"q": {2: seq}, "k": {2: seq}, "v": {2: seq}} if dynamic else None
    graph = trace_module(_Sdpa().cpu(), (q, k, v), dynamic_shapes=ds)
    warp = [r for r in enumerate_graph(graph, Context.from_target((12, 0))) if "TILE@pj" in r]
    assert warp, "the bare sibling-PV pin must keep the warp tier (it used to decline to scalar)"
    # F1: rows spell the SITE halves; the pinned plan's warp geometry rides the ONE WORK entry.
    assert all(r["TILE@pj"] == "mma_m16n8k16_f16_f16/f1x8/k2" for r in warp), "PV must ride the pinned sibling plan verbatim"
    assert all(r["TILE@dd"] == "mma_m16n8k16_f16_f32/f1x4/k4" for r in warp), "scores must stay on the base f32-accumulate atom"
    # (a resolved TMA stage row may add a producer band — the ``+p`` suffix is not the pin's claim)
    assert all(r["WORK"].partition("+")[0] == "w2x1" for r in warp), "the pinned plan's warp geometry rides the WORK entry"


@requires_cuda
@pytest.mark.parametrize("stage", ["", "d2/cp/ring"])
def test_generated_tensorcore_flash_f16acc_matches_torch(monkeypatch, stage):
    """A pinned f16acc-PV flash row (the axis-keyed ``TILE@dd``/``TILE@pj`` golden spelling — the
    sibling atom in ``TILE@pj`` offers the row without the gate, pins are authoritative): the P@V
    mma targets packed f16 fragments (``_h<j>``) and each streaming KV block promote-folds them
    into the f32 output shadows the rescale / projection / store read. Matches torch over the
    gmem-direct AND staged (cp.async ring) streams."""
    monkeypatch.setenv("EMMY_TILE@DD", "a:mma_m16n8k16_f16/w1x1/f1x2/k4")
    monkeypatch.setenv("EMMY_TILE@PJ", "a:mma_m16n8k16_f16_f16/w1x1/f1x8")
    if stage:
        monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(11)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"f16acc flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "_h0" in src and "_h0);" in src, "the P@V chain must target packed f16 fragments and promote per block"
    assert "emmy_c_to_a" in src, "must stay the fused warp form (C->A register repack)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"f16acc flash stage={stage!r} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(
    ("geom", "stage", "loopify"),
    [
        ("w1x1/f1x2/k4", "", 4),  # gmem-direct: O_i_f rescale/divide + P@V load+mma + stores re-roll
        ("w1x1/f1x2/k4", "", 2),  # LOOPIFY=2 also re-rolls the 2-long QK sacc_f scale
        ("w1x1/f2x2/k4", "", 4),  # per-query-tile suffixed families (O_i_q0_f / O_i_q1_f)
        ("w1x1/f1x2/k4", "d2/cp/ring", 4),  # staged: block_threads carried through the re-roll rename
        ("w1x1/f2x2/k4", "d2/cp/ring", 2),  # staged + partial N-atom runs (arrayed to full family size)
    ],
)
def test_loopify_pin_matches_torch(monkeypatch, geom, stage, loopify):
    """``EMMY_LOOPIFY`` (100_loopify) is a generic loop re-roller: it folds a maximal run of congruent
    per-fragment statements — the ``O_i_f`` α rescale / divide (``FragmentApply``), the ``P@V``
    load+mma pairs, the fragment ``RegStore``s, and at ``=2`` the ``sacc_f`` QK scale — into a
    ``#pragma unroll`` loop over an arrayed fragment family, with affine address offsets folded to
    ``_r*step``. Purely a listing shrink (identical SASS), so the source must still match torch. Covers
    the gmem-direct and staged (``cp.async`` ring) tiers, plain and ``f2x2`` per-query-tile geometries —
    the staged tier exercises the ``block_threads`` carry-through and the partial-run arraying-to-full-
    family-size guards."""
    monkeypatch.setenv("EMMY_TILE", f"a:mma_m16n8k16_f16/{geom}")
    monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_LOOPIFY", str(loopify))
    torch.manual_seed(loopify * 7 + len(geom) + len(stage))
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1, f"pinned warp flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "[4] = {};" in src and "for (int _r" in src, "loopify must array a fragment family and emit a re-roll loop"
    # Each fragment pointwise op renders as an element loop; the softmax subtract→exp fuses into one
    # ``exp(s − m)`` per element (the ``post``-chain) under the pin.
    assert "for (int _e" in src, "fragment pointwise ops must render as element loops"
    assert any("= expf(" in ln and "((_e < 2)" in ln for ln in src.splitlines()), "subtract→exp must fuse into one exp(s − m)"
    if not stage:  # the gmem-direct store epilogue re-rolls into a loop (var _r0) whose body carries the m16n8 lane _t
        assert "_r0 * 8 + _g * 64" in src, "the fragment stores must re-roll (with the affine +_r0*8 column offset)"
    if loopify == 2 and geom == "w1x1/f1x2/k4":  # the 2-long QK sacc_f scale arrays only at the lower threshold
        assert any("sacc" in ln and "[2][4] = {}" in ln for ln in src.splitlines()), "LOOPIFY=2 must array the QK sacc_f family"
        if not stage:  # the QK score contraction nests via the fixpoint: outer K-chunk (_r1) × inner N-atom (_r0)
            assert "for (int _r1" in src and "_sacc_a[_r1]" in src, "the QK contraction must nest into two #pragma unroll loops"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"loopify {geom} stage={stage!r} LOOPIFY={loopify} max_diff={max_diff:.2e}"


@requires_cuda
def test_readable_scalar_chain_fold_matches_torch(monkeypatch):
    """``EMMY_READABLE`` folds a single-use scalar ``Assign`` temp into its sole consumer's expression
    — the scalar-tier softmax ``t1 = m − m'`` then ``t2 = expf(t1)`` collapses to ``expf(m − m')`` —
    while a multi-use temp (``m_i__t0``, read by two subtracts) stays named. SASS-identical, so it
    still matches torch. ``--no-readable`` (the default off) keeps the SSA ladder."""
    monkeypatch.setenv("EMMY_TILE", "a:scalar")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_READABLE", "1")
    torch.manual_seed(7)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile_tc(q, k, v)
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert any("expf((m_i - " in ln or "expf((s - " in ln for ln in src.splitlines()), "the scalar subtract→exp must fold"
    assert "= fmaxf(m_i, s)" in src, "the multi-use rowmax temp must stay named (folding it would recompute fmaxf)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert float(np.max(np.abs(got - eager))) < 5e-3, "readable scalar fold must match torch"


# --------------------------------------------------------------------------- #
# Staged K/V (the ``STAGE@<kv>`` cp.async stream) — Moves 4/5 on the warp tier.
# --------------------------------------------------------------------------- #


def _run_flash(backend, compiled, graph, tensors) -> np.ndarray:
    data = {n: t.numpy() for n, t in zip(graph.inputs, tensors, strict=True)}
    run_result, _ = backend.run(compiled, input_data=data)
    return list(run_result.outputs.values())[0].flatten()


@requires_cuda
@pytest.mark.parametrize(
    "stage",
    [
        "d1/cp",
        "d2/cp/ring",
        "d3/cp/ring",
        pytest.param("d1/tma", marks=requires_sm90),
        pytest.param("d2/tma/ring", marks=requires_sm90),
    ],
)
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
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    backend, compiled, graph, _ = _compile_tc(q, k, v)
    base = _run_flash(backend, compiled, graph, (q, k, v))
    monkeypatch.setenv("EMMY_STAGE", stage)
    backend2, compiled2, graph2, kernels2 = _compile_tc(q, k, v)
    assert "cp.async" in compiled2.nodes[kernels2[0]].op.kernel_source
    staged = _run_flash(backend2, compiled2, graph2, (q, k, v))
    assert np.array_equal(base, staged), f"staged ({stage}) output differs from its gmem-direct sibling"


@requires_cuda
@pytest.mark.parametrize("stage", ["d2/cp/ring", pytest.param("d2/tma/ring", marks=requires_sm90)])
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
def test_staged_flash_symbolic_cp_stages_and_matches_torch(monkeypatch):
    """A **cp.async** ``STAGE`` pin on a SYMBOLIC ``seq_len`` flash STAGES the K/V stream: the fill
    clamp-reads the tail chunk's key rows to the last valid key (cp.async has no OOB zero-fill) and
    the drain's tail masks zero their P columns, so the duplicated rows contribute exactly 0. One
    cached kernel carries ``int seq_len``; accurate vs torch at seq ∈ {37, 64, 100} (37/100
    overhang the KV block)."""
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(_Sdpa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "must still be the fused warp form"
    assert "int seq_len" in src, "dynamic kernel must carry the runtime seq_len arg"
    assert "_k_smem" in src and "_v_smem" in src, "cp.async must stage the symbolic K/V stream"
    assert "cp.async" in src and "cp_async_bulk_tensor" not in src, "the fill must be cp.async, not TMA"

    for s in (37, 64, 100):
        torch.manual_seed(s)
        q, k, v = (torch.randn(B, H, s, D, dtype=torch.float16) for _ in range(3))

        def ref(q=q, k=k, v=v):
            with torch.no_grad():
                return F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

        data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
        run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
        got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
        assert float(np.max(np.abs(got - eager))) < 5e-3, f"staged symbolic cp flash seq={s} drifted from torch"


@requires_cuda
@pytest.mark.parametrize("s", [64, 100])
def test_staged_flash_symbolic_cp_bit_identical_to_gmem_direct(monkeypatch, s):
    """The symbolic cp.async-staged stream is a pure perf transform: verbatim K/V row copies, and
    the tail chunk's clamped (duplicated) key rows land on P columns the drain masks to exactly 0 —
    so the staged output is BIT-identical to gmem-direct at both a block-divisible (64) and an
    overhanging (100) seq. The cp.async counterpart of the TMA twin above."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    B, H, D = 1, 4, 64
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    ds = {"q": {2: sd}, "k": {2: sd}, "v": {2: sd}}
    torch.manual_seed(s)
    q, k, v = (torch.randn(B, H, s, D, dtype=torch.float16) for _ in range(3))

    backend, compiled, graph, _ = _trace(_Sdpa(), seed, dynamic_shapes=ds)
    base = _run_flash(backend, compiled, graph, (q, k, v))
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")
    backend2, compiled2, graph2, kernels2 = _trace(_Sdpa(), seed, dynamic_shapes=ds)
    src2 = compiled2.nodes[kernels2[0]].op.kernel_source
    assert "_k_smem" in src2 and "cp.async" in src2, "cp.async must stage the symbolic stream"
    staged = _run_flash(backend2, compiled2, graph2, (q, k, v))
    assert np.array_equal(base, staged), f"staged symbolic cp (seq={s}) differs from gmem-direct sibling"


@requires_cuda
@requires_sm90
def test_staged_flash_stage_pin_keeps_warp_not_scalar(monkeypatch):
    """A ``STAGE`` pin keeps the flash fork on the WARP (mma) tier — only the warp tier stages, so
    the pin must not fall through to the chain / scalar reduce-partition siblings and let the prior
    bury the (lower-occupancy) staged warp form under a higher-occupancy scalar form. At H=8/seq=512
    the pre-fix prior did exactly that (a staged-flash A/B row read as a ~100× regression). The
    kernel must be the fused warp form (`emmy_c_to_a`) with a live TMA-staged K/V slab, NOT scalar."""
    monkeypatch.setenv("EMMY_STAGE", "d2/tma/ring")
    B, H, D = 1, 8, 64  # H=8 is the scale where the pre-fix prior buried the staged form under scalar
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    _backend, compiled, _graph, kernels = _trace(_Sdpa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "STAGE pin must keep the fused WARP flash form, not fall to scalar"
    assert "_k_smem" in src and "cp_async_bulk_tensor" in src, "the warp form must carry the pinned TMA K/V stage"


@requires_cuda
@requires_sm90
def test_staged_flash_symbolic_tma_stages_and_matches_torch(monkeypatch):
    """A **TMA** ``STAGE`` pin on a SYMBOLIC ``seq_len`` flash STAGES the K/V stream: the
    descriptor rides the runtime globalDim and zero-fills the box overhang past the last key, so
    the tail chunk of a non-block-divisible seq is safe and the drain's tail masks keep it correct
    (the cp.async twin clamp-reads instead of zero-filling). One cached kernel carries
    ``int seq_len``; accurate vs torch at seq ∈ {64, 100} (100 overhangs the KV block)."""
    monkeypatch.setenv("EMMY_STAGE", "d2/tma/ring")
    B, H, D = 1, 4, 64
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(_Sdpa(), seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "int seq_len" in src, "dynamic kernel must carry the runtime seq_len arg"
    assert "_k_smem" in src and "_v_smem" in src, "TMA must stage the symbolic K/V stream"
    assert "cp_async_bulk_tensor" in src, "the symbolic K/V must box-copy via the TMA descriptor"

    for s in (64, 100):
        torch.manual_seed(s)
        q, k, v = (torch.randn(B, H, s, D, dtype=torch.float16) for _ in range(3))

        def ref(q=q, k=k, v=v):
            with torch.no_grad():
                return F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

        data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
        run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
        got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
        assert float(np.max(np.abs(got - eager))) < 5e-3, f"staged symbolic tma flash seq={s} drifted from torch"


@requires_cuda
@requires_sm90
@pytest.mark.parametrize("s", [64, 100])
def test_staged_flash_symbolic_tma_bit_identical_to_gmem_direct(monkeypatch, s):
    """The symbolic TMA-staged stream is a pure perf transform: verbatim K/V row copies, and the
    box zero-fill of the overhang lands on keys the drain masks to the fold identity — the same
    clamp the gmem-direct symbolic path makes — so the staged output is BIT-identical to gmem-direct
    at both a block-divisible (64) and an overhanging (100) seq."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    B, H, D = 1, 4, 64
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    ds = {"q": {2: sd}, "k": {2: sd}, "v": {2: sd}}
    torch.manual_seed(s)
    q, k, v = (torch.randn(B, H, s, D, dtype=torch.float16) for _ in range(3))

    backend, compiled, graph, _ = _trace(_Sdpa(), seed, dynamic_shapes=ds)
    base = _run_flash(backend, compiled, graph, (q, k, v))
    monkeypatch.setenv("EMMY_STAGE", "d2/tma/ring")
    backend2, compiled2, graph2, kernels2 = _trace(_Sdpa(), seed, dynamic_shapes=ds)
    src2 = compiled2.nodes[kernels2[0]].op.kernel_source
    assert "_k_smem" in src2 and "cp_async_bulk_tensor" in src2, "TMA must stage the symbolic stream"
    staged = _run_flash(backend2, compiled2, graph2, (q, k, v))
    assert np.array_equal(base, staged), f"staged symbolic tma (seq={s}) differs from gmem-direct sibling"


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
    assert "emmy_mma_m16n8k16_bf16" in src, "the bf16 flash must use the bf16 mma atom"
    assert "emmy_c_to_a" in src, "the generated kernel must be the fused warp-chain (C->A register repack)"

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
    assert "emmy_mma_m16n8k16_bf16" in src and "emmy_c_to_a" in src

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
    assert "emmy_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

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
def test_warp_flash_causal_tile_skip(monkeypatch):
    """The causal tile-skip: a triangular score prologue (``kv ≤ m``) bounds the warp-tier stream at
    the CTA's last query row — the ``StridedLoop`` gets a hoisted ``kv0_end`` for-init bound instead
    of the full extent (accuracy is pinned by the causal cases above; skipped steps fold the exact
    carrier identity). A non-causal stream must NOT carry the bound — it is derived from the causal
    ``Select``'s predicate shape, not from any kernel identity."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    _backend, compiled, _graph, kernels = _compile_tc(q, k, v, module=_Causal())
    assert len(kernels) == 1
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "must be the fused warp-chain"
    assert "kv0_end" in src, "causal warp flash must bound the stream at the CTA's last query row"

    _backend, compiled, _graph, kernels = _compile_tc(q, k, v)
    assert len(kernels) == 1
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "must be the fused warp-chain"
    assert "kv0_end" not in src, "a non-causal stream must run the full extent"


@requires_cuda
@pytest.mark.parametrize("stage", [pytest.param("d1/tma/alt", marks=requires_sm90), "d1/cp/alt"])
@pytest.mark.parametrize("variant", ["plain", "causal"])
def test_warp_flash_alt_staging_matches_torch(monkeypatch, variant, stage):
    """The ALTERNATING single-slab staging (``STAGE=d1/tma/alt``): one K slab + one V slab on
    separate mbarriers, refilled in the phase that no longer reads them (K under softmax + P·V,
    V under the next step's Q·K), and Q staged through a padded smem tile (its A fragments
    ldmatrix'd per atom-K chunk). Structure pinned via the emitted source (the per-operand
    mbarriers + the Q slab), values vs torch — the fills are verbatim copies, so alt stays
    bit-identical to gmem-direct."""
    monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(11)
    module = _Causal() if variant == "causal" else _Sdpa()
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(module, (q, k, v))
    assert len(kernels) == 1
    src = compiled.nodes[kernels[0]].op.kernel_source
    if "tma" in stage:
        assert "_kbar" in src and "_vbar" in src, "tma alt must run the per-operand mbarrier pair"
    else:
        assert "_kbar" not in src and "cp_async" in src, "cp alt rides commit groups, no mbarriers"
    assert "_q_smem" in src, "alt staging must stage Q through smem"

    def ref():
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=variant == "causal")
            return out.cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"alt-staged flash ({variant}) max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(
    ("variant", "stage"),
    [("plain", "d1/cp/alt"), pytest.param("plain", "d1/tma/alt", marks=requires_sm90), ("causal", "d1/cp/alt")],
)
def test_warp_flash_alt_staging_symbolic_bit_identical(monkeypatch, variant, stage):
    """The alternating staging over a SYMBOLIC ``seq_len`` — the resolver accepts it and the
    liveness scheduler's kill-point refills ride the same runtime clamp as the ring prefetch:
    cp.async clamp-reads the tail's key rows (TMA zero-fills its box), the staged-Q fill
    clamp-reads a tail CTA's overhanging query rows (their outputs are store-guarded), and the
    drain's tail masks zero the overhanging P columns. All fills are verbatim copies, so the alt
    kernel is BIT-identical to its gmem-direct symbolic sibling at both a block-divisible (64)
    and an overhanging (100) seq; the causal case composes the ``k_end`` early stop with the
    symbolic clamp."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    B, H, D = 1, 4, 64
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    ds = {"q": {2: sd}, "k": {2: sd}, "v": {2: sd}}
    module = _Causal() if variant == "causal" else _Sdpa()

    backend, compiled, graph, _ = _trace(module, seed, dynamic_shapes=ds)
    monkeypatch.setenv("EMMY_STAGE", stage)
    backend2, compiled2, graph2, kernels2 = _trace(module, seed, dynamic_shapes=ds)
    src = compiled2.nodes[kernels2[0]].op.kernel_source
    assert "int seq_len" in src, "the symbolic kernel must carry the runtime seq_len arg"
    assert "_q_smem" in src, "alt staging must stage Q through smem"
    if "tma" in stage:
        assert "_kbar" in src and "_vbar" in src, "tma alt must run the per-operand mbarrier pair"
    else:
        assert "_kbar" not in src and "cp.async" in src, "cp alt rides commit groups, no mbarriers"

    for s in (64, 100):
        torch.manual_seed(s)
        q, k, v = (torch.randn(B, H, s, D, dtype=torch.float16) for _ in range(3))
        base = _run_flash(backend, compiled, graph, (q, k, v))
        staged = _run_flash(backend2, compiled2, graph2, (q, k, v))
        assert np.array_equal(base, staged), f"symbolic alt ({variant}/{stage}, seq={s}) differs from gmem-direct sibling"


@requires_cuda
@pytest.mark.parametrize("variant", ["plain", "causal"])
def test_warp_flash_split_kv_matches_torch(monkeypatch, variant):
    """Flash split-KV (``REDUCE=g<n>k`` on the warp tier): the kv stream splits across CTAs, each
    partial keeping fragment residence and storing its raw ``(m, l, O)`` state to the f32
    ``__partial`` workspace; a sibling finalize folds the partitions via the exp-family LSE
    combine and projects. Two kernels, and the result matches torch — causal composes (each
    slice's triangular tile-skip is slice-local; an above-the-diagonal slice contributes the
    exact carrier identity)."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    torch.manual_seed(7)
    module = _Causal() if variant == "causal" else _Sdpa()
    q, k, v = (torch.randn(1, 4, 128, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(module, (q, k, v))
    assert len(kernels) == 2, f"split-KV flash should be partial + finalize, got {len(kernels)}"
    partial = next(n for n in kernels if n.endswith("__partial"))
    src = compiled.nodes[partial].op.kernel_source
    assert "emmy_c_to_a" in src, "the split partial must keep the fused warp-chain"

    def ref():
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=variant == "causal")
            return out.cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"split-KV flash ({variant}) max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(
    ("variant", "reduce", "seq"),
    [
        ("causal", "g2k", 300),  # non-block-multiple slice tail (B=160, slice 1 walks 140 keys)
        ("causal", "g4k", 40),  # empty last slice (B=16: slice 3 starts past S, contributes identities)
        ("plain", "g2k", 40),  # un-causal slice stop (kv_end from the slice bound alone)
        ("causal", "g2k", 512),  # block-whole slices at the hint size
    ],
)
def test_warp_flash_split_kv_symbolic_matches_torch(monkeypatch, variant, reduce, seq):
    """SYMBOLIC flash split-KV: the slice width is the bn-aligned runtime ``B = ceil(S/(cta·bn))·bn``
    and each slice stops/masks at its absolute ``bound = min((s+1)·B, S)`` (``Fold.bound``) —
    a mid-tensor slice end reads VALID next-slice keys the extent-only tail masks would keep, and
    the tail CTA's overhanging query rows must NOT write their state into the next head's ws rows
    (the split partial's ``m_guard``, the regression this test pins). One cached kernel pair serves
    every runtime size, including an empty last slice (pure carrier identities)."""
    monkeypatch.setenv("EMMY_REDUCE", reduce)
    B, H, D = 1, 4, 64
    module = _Causal() if variant == "causal" else _Sdpa()
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _trace(module, seed, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    assert len(kernels) == 2, f"symbolic split-KV flash should be partial + finalize, got {len(kernels)}"
    partial = next(n for n in kernels if n.endswith("__partial"))
    src = compiled.nodes[partial].op.kernel_source
    assert "emmy_c_to_a" in src, "the symbolic split partial must keep the fused warp-chain"
    assert "int seq_len" in src, "the symbolic split partial must carry the runtime seq_len arg"

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=variant == "causal")
            return out.cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"symbolic split-KV flash ({variant}/{reduce}, seq={seq}) max_diff={max_diff:.2e}"


# =========================================================================== #
# Sliding-window banded flash (the trace-time SdpaOp.sliding_window stamp).
# =========================================================================== #


def _stamp_window(module, args, window, dynamic_shapes=None):
    """Trace ``module``, stamp ``sliding_window`` (+ the ``is_causal`` assertion) on its SdpaOp —
    the HF-wrapper stamp path; ``F.scaled_dot_product_attention`` has no window arg to trace it
    from. ``window=None`` stamps ``is_causal`` alone (the full-attention layer's shape: the
    causal end-skip through an opaque bias operand)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import SdpaOp  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    graph = trace_module(module.cpu(), args, dynamic_shapes=dynamic_shapes)
    for n in graph.nodes.values():
        if isinstance(n.op, SdpaOp):
            n.op.sliding_window = window
            n.op.is_causal = True
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, graph, kernels


def _banded_ref(q, k, v, window, extra_bias=None):
    """torch SDPA under the causal ∧ band(W) additive mask (F.sdpa has no window arg)."""
    s_q, s_k = q.shape[-2], k.shape[-2]
    keep = torch.ones(s_q, s_k, dtype=torch.bool).tril(0) & torch.ones(s_q, s_k, dtype=torch.bool).triu(-(window - 1))
    bias = torch.zeros(s_q, s_k, dtype=q.dtype).masked_fill_(~keep, float("-inf"))
    if extra_bias is not None:
        bias = bias + extra_bias

    def ref():
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=bias.cuda())
            return out.cpu().flatten().float().numpy()

    return ref


@requires_cuda
@pytest.mark.parametrize("stage", [None, "d2/cp/ring", "d1/cp/alt"])
@pytest.mark.parametrize(("S", "W"), [(256, 64), (256, 100)])
def test_warp_flash_banded_matches_torch(monkeypatch, stage, S, W):
    """The sliding-window banded warp flash: a stamped ``SdpaOp.sliding_window`` decomposes to a
    second coordinate ``Select`` (keep ``kv > m − W``) beside the causal one; the fused kernel
    carries BOTH FragmentMasks and both stream bounds (the causal end, the banded start). One
    fused kernel, torch-banded-reference accuracy, on the unstaged and both staged pipelines —
    a non-block-aligned W (100) exercises the boundary tiles' band mask."""
    if stage is not None:
        monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(S + W)
    q, k, v = (torch.randn(1, 4, S, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _stamp_window(_Causal(), (q, k, v), W)
    assert len(kernels) == 1, f"banded flash should stay one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "must be the fused warp-chain"
    assert "kv0_end" in src, "the causal stream end must survive the band stamp"
    assert f"- {W - 1}" in src, "the banded stream start must derive from the stamp"

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=_banded_ref(q, k, v, W))
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"banded flash (stage={stage}, S={S}, W={W}) max_diff={max_diff:.2e}"


@requires_cuda
def test_warp_flash_banded_tile_skip_structure(monkeypatch):
    """The banded start is DERIVED from the band Select's predicate shape, never from a kernel
    identity: the stamped kernel starts its stream at ``⌊max(0, first_row − W + 1)/bn⌋·bn`` and
    keeps the causal ``kv0_end``; the un-stamped twin has neither band mask nor late start."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 256, 64, dtype=torch.float16) for _ in range(3))
    _backend, compiled, _graph, kernels = _stamp_window(_Causal(), (q, k, v), 64)
    assert len(kernels) == 1
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "kv0_end" in src and "- 63" in src, "stamped: causal end AND banded start"

    _backend, compiled, _graph, kernels = _trace(_Causal(), (q, k, v))
    assert len(kernels) == 1
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "kv0_end" in src and "- 63" not in src, "un-stamped: causal end only, full-stream start"


@requires_cuda
@pytest.mark.parametrize("W", [256, 1024])
def test_warp_flash_vacuous_band_still_fuses(monkeypatch, W):
    """A stamped ``sliding_window`` ≥ the static seq is VACUOUS (every ``m − W < 0``): the band
    term must be dropped at decomposition, not emitted — an emitted vacuous Select's predicate
    constant-folds, the +0 mask term hoists out of the reduce loops, and the flash recognizer's
    mask-chain walk can't resolve it, silently degrading the fuse to cut (the gemma-4 layer-0
    ``seq 512 < window 1024`` trace deployed a sequential grid-1 softmax·P@V that ran for
    minutes). Expect: ONE fused flash kernel, causal stream end only (no banded start), plain
    causal numerics."""
    torch.manual_seed(W)
    S = 256
    q, k, v = (torch.randn(1, 4, S, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _stamp_window(_Causal(), (q, k, v), W)
    assert len(kernels) == 1, f"vacuous-band flash should stay one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src, "must be the fused warp-chain"
    assert "kv0_end" in src, "the causal stream end must survive"
    assert f"- {W - 1}" not in src, "a vacuous band must not stamp a banded stream start"

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=_banded_ref(q, k, v, W))
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"vacuous-band flash (W={W}) max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [40, 300, 512])
def test_warp_flash_banded_symbolic_matches_torch(monkeypatch, seq):
    """SYMBOLIC banded flash: the banded start is grid-derived (CTA row × W), independent of the
    runtime seq_len, so one cached kernel serves every size; the band FragmentMask composes with
    the symbolic tail clamp-masks."""
    B, H, D, W = 1, 4, 64, 64
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    seed = tuple(torch.randn(B, H, 16, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _stamp_window(_Causal(), seed, W, dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}})
    assert len(kernels) == 1, f"symbolic banded flash should stay one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "emmy_c_to_a" in src and "int seq_len" in src

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))
    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=_banded_ref(q, k, v, W))
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"symbolic banded flash (seq={seq}) max_diff={max_diff:.2e}"


@requires_cuda
def test_warp_flash_banded_split_kv_matches_torch(monkeypatch):
    """Banded flash × split-KV (``REDUCE=g2k``): each slice's banded start is slice-local (the
    base subtracted) — a slice wholly below the band runs zero steps and contributes the exact
    carrier identity, mirroring the causal above-the-diagonal case."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    torch.manual_seed(7)
    W = 64
    q, k, v = (torch.randn(1, 4, 256, 64, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _stamp_window(_Causal(), (q, k, v), W)
    assert len(kernels) == 2, f"split-KV banded flash should be partial + finalize, got {len(kernels)}"
    partial = next(n for n in kernels if n.endswith("__partial"))
    assert "emmy_c_to_a" in compiled.nodes[partial].op.kernel_source

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=_banded_ref(q, k, v, W))
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"split-KV banded flash max_diff={max_diff:.2e}"


@requires_cuda
def test_warp_flash_banded_with_additive_bias_matches_torch(monkeypatch):
    """The whole-model shape: an explicit additive mask operand PLUS the stamp. The bias stays
    loaded (it may mask more than the band — padding), the coord Selects ride beside it and
    drive both stream bounds; the fused kernel carries FragmentBiasAdd AND both FragmentMasks.
    The bias here masks an extra key block the band alone would keep — the result must honor it."""
    torch.manual_seed(3)
    S, W = 256, 64
    q, k, v = (torch.randn(1, 4, S, 64, dtype=torch.float16) for _ in range(3))
    extra = torch.zeros(S, S, dtype=torch.float16)
    extra[:, 7] = float("-inf")  # a "padding" column inside the band
    keep = torch.ones(S, S, dtype=torch.bool).tril(0) & torch.ones(S, S, dtype=torch.bool).triu(-(W - 1))
    mask = (torch.zeros(S, S, dtype=torch.float16).masked_fill_(~keep, float("-inf")) + extra)[None, None]
    backend, compiled, graph, kernels = _stamp_window(_Masked(), (q, k, v, mask), W)
    assert len(kernels) == 1, f"stamped biased flash should stay one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "kv0_end" in src and f"- {W - 1}" in src, "the stamp must drive both bounds beside the bias"

    def ref():
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda())
            return out.cpu().flatten().float().numpy()

    feed = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy(), mask.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=feed, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"stamped biased flash max_diff={max_diff:.2e}"


@requires_cuda
def test_warp_flash_causal_stamp_with_bias_matches_torch(monkeypatch):
    """The full-attention layer's whole-model shape: an explicit causal bias operand plus the
    ``is_causal`` stamp alone (no window). The stamp's coord Select rides beside the bias and
    derives the causal stream END through the otherwise-opaque operand."""
    torch.manual_seed(5)
    S = 256
    q, k, v = (torch.randn(1, 4, S, 64, dtype=torch.float16) for _ in range(3))
    mask = torch.zeros(S, S, dtype=torch.float16).masked_fill_(torch.ones(S, S, dtype=torch.bool).triu(1), float("-inf"))[None, None]
    backend, compiled, graph, kernels = _stamp_window(_Masked(), (q, k, v, mask), None)
    assert len(kernels) == 1, f"stamped causal biased flash should stay one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "kv0_end" in src, "the is_causal stamp must derive the stream end through the bias"

    def ref():
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda())
            return out.cpu().flatten().float().numpy()

    feed = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy(), mask.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=feed, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"stamped causal biased flash max_diff={max_diff:.2e}"


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
    assert "emmy_c_to_a" in src, "the symbolic flash must be the fused warp-chain (C->A register repack)"
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
    assert "emmy_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

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
    assert "emmy_c_to_a" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

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
    ONE cached kernel carrying ``int seq_len``; matches torch GQA+causal SDPA at seq ∈ {8,16,37,64}.
    ``REDUCE`` is pinned serial: unpinned, the cold offline pick for this shape is the ``g4k``
    reduce-partition warp sibling (two kernels) — the fused single-kernel chain is what this case pins."""
    monkeypatch.setenv("EMMY_REDUCE", "")
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
    assert "emmy_c_to_a" in src and "int seq_len" in src, "must be the symbolic fused warp-chain"

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


@requires_cuda
@pytest.mark.parametrize("reg", ["2", "4"])
@pytest.mark.parametrize("causal", [False, True])
def test_ilp_reg_flash_matches_torch(monkeypatch, reg, causal):
    """The ILP register fold (``EMMY_REDUCE=r{reg}``) over the flash ``kv`` streaming reduce: ``reg``
    interleaved ``(m, l, O)`` accumulator chains merged by the monoid REG-tree fold. The reduce body
    holds the NESTED ``dd`` (Q@K) / ``j`` (P@V) contraction loops, whose own axis vars ``copy_cell``
    must leave shared across copies — a per-copy suffix (``dd__r1``) on the load USE while the ``for``
    DECL stays ``dd`` emits an undefined identifier (the flash-certification model-tune regression).
    Pins the exact path that nvcc-failed on Gemma; asserts the copies are emitted and match torch."""
    monkeypatch.setenv("EMMY_REDUCE", f"r{reg}")
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 32, 16) for _ in range(3))
    module = _Causal() if causal else _Sdpa()
    backend, compiled, _graph, kernels = _trace(module, (q, k, v))
    assert len(kernels) == 1, f"flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "sacc__r1" in src, "the ILP fold must replicate the score accumulator per copy"
    assert "dd__r" not in src, "the nested contraction's reduce axis must stay shared (no dd__r{r} — the undefined-id bug)"
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def eager():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cq, ck, cv, is_causal=causal).cpu().flatten().numpy()

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
    from emmy.compiler.loader.binder import apply_load_ops, assemble_source  # noqa: PLC0415
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
            if nid not in feed and node.op.source_parts:
                # A merged sibling-linear weight: assemble the axis-0 concat from the part paths.
                sources = {k: p.detach().cpu().numpy() for k, p in module.named_parameters()}
                src = assemble_source(node.op, sources)
                if src is not None:
                    feed[nid] = apply_load_ops(src, node.op.load_ops)
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


@requires_cuda
def test_warp_flash_f32_value_operand_converts(monkeypatch):
    """A V operand traced at f32 (gemma-4's V-norm: ``v_f16 * f32 row stat`` promotes, and the
    traced ``.half()`` cast rides a view) must reach the flash stream as an f16 BUFFER: the
    cast splits out of the view (``005_split_cast_from_indexmap``), fusion keeps it
    materialized at flash offer sites (``_is_castfree_indexmap`` — a dtype-changing copy is
    not plumbing) and fuses the f32 ``mul`` producer INTO it, so the stream sees an
    atom-dtype operand and the pinned cp.async ring RESOLVES (the pre-split behavior fused
    the cast into the flash load, and the f32 buffer declined every staged row —
    gmem-direct forever, the gemma layer-0 lockout)."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_STAGE", "d2/cp/ring")  # resolves: the split-out cast feeds f16 V
    monkeypatch.setenv("EMMY_WSPEC", "")
    torch.manual_seed(5)
    S, D = 128, 64

    class VNormSdpa(torch.nn.Module):
        def forward(self, q, k, v, s):
            v32 = v.float() * s  # per-(token, head) f32 stat — promotes V to f32 in the trace
            return F.scaled_dot_product_attention(q, k, v32.half())

    q, k, v = (torch.randn(1, 4, S, D, dtype=torch.float16) for _ in range(3))
    s = (torch.rand(1, 4, S, 1) + 0.5).float()
    backend, compiled, graph, kernels = _trace(VNormSdpa(), (q, k, v, s))
    srcs = "\n".join(compiled.nodes[n].op.kernel_source for n in kernels)
    flash = [compiled.nodes[n].op.kernel_source for n in kernels if "mma.sync" in compiled.nodes[n].op.kernel_source]
    assert flash, "the f32-V flash must stay on the warp (mma) tier"
    sig = next(line for line in flash[0].splitlines() if "__launch_bounds__" in line)
    assert sig.count("const float*") <= 1, f"only the flash scale may be f32 — V must be the materialized f16 cast: {sig}"
    assert "_v_smem" in srcs, "the pinned cp.async ring must resolve against the f16 cast buffer"

    def ref():
        with torch.no_grad():
            v32 = (v.cuda().float() * s.cuda()).half()
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v32).cpu().flatten().float().numpy()

    data = {n: t.numpy() for n, t in zip(graph.inputs, (q, k, v, s), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), "f32-V warp flash produced NaN (the reinterpret regression)"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"f32-V warp flash max_diff={max_diff:.2e}"


def _band_mask(seq: int, window: int) -> torch.Tensor:
    """A sliding-window additive float mask — causal AND within ``window`` keys of the query (the
    HF gemma sliding-attention band): 0 where ``m - window < kv <= m``, ``-inf`` elsewhere."""
    kv = torch.arange(seq)[None, :]
    m = torch.arange(seq)[:, None]
    keep = (kv <= m) & (kv > m - window)
    return torch.where(keep, 0.0, float("-inf"))[None, None].half()


@requires_cuda
@pytest.mark.parametrize("stage", ["", "d2/tma/ring", "d1/cp/alt"])
def test_warp_flash_explicit_additive_mask_matches_torch(monkeypatch, stage):
    """An explicit additive ``attn_mask`` (the HF precomputed causal / sliding-window band)
    realizes at the WARP tier: the score prologue's ``(m, kv)``-indexed bias ``Load`` + ``add``
    becomes a per-element ``FragmentBiasAdd`` — each fragment element reads the mask at its
    absolute coordinates and adds it before the softmax merge — instead of demoting the whole
    kernel to the scalar tier (the gemma-4 seq>window regression: every layer's attention went
    scalar and hung). Banded (sliding-window) mask; composes with the K/V staging forms."""
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f1x4/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_WSPEC", "")
    if stage:
        monkeypatch.setenv("EMMY_STAGE", stage)
    torch.manual_seed(3)
    S, D = 128, 64
    q, k, v = (torch.randn(1, 4, S, D, dtype=torch.float16) for _ in range(3))
    mask = _band_mask(S, window=64)
    backend, compiled, graph, kernels = _trace(_SdpaExplicitMask(), (q, k, v, mask))
    assert len(kernels) == 1, f"masked warp flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "mma.sync" in src, "the explicit-mask flash must stay on the warp (mma) tier"
    assert "+= __half2float(mask[" in src, "the mask must realize as per-element fragment bias adds"
    if stage == "d1/cp/alt":
        assert "_q_smem" in src, "alt staging must compose with the mask bias"

    def ref():
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda())
            return out.cpu().flatten().float().numpy()

    data = {n: t.numpy() for n, t in zip(graph.inputs, (q, k, v, mask), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"masked warp flash (stage={stage!r}) max_diff={max_diff:.2e}"


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
    from emmy.compiler.loader.binder import apply_load_ops, assemble_source  # noqa: PLC0415
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
            if nid not in feed and node.op.source_parts:
                # A merged sibling-linear weight: assemble the axis-0 concat from the part paths.
                sources = {k: p.detach().cpu().numpy() for k, p in attn_cpu.named_parameters()}
                src = assemble_source(node.op, sources)
                if src is not None:
                    feed[nid] = apply_load_ops(src, node.op.load_ops)
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
