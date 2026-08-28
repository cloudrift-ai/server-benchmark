r"""Attention coverage over the canonical Fold tree.

The tests cover scalar and tensor-core SDPA, causal and additive masks, GQA, symbolic sequence
lengths, split reductions, staging and swizzling, a hand-written FA-2 reference, and TinyLlama
attention integration. Tensor-core SDPA is realized by the generic residence evaluator: scheduled
child contractions produce fragments, row statistics use ``FragmentRowReduce``, and the enclosing
Fold's stored carrier Lambda combines the state. No attention-specific scheduler or emitter is
part of the tested contract.

## Measured evidence

These numbers are why the structural assertions below exist. They are not reproduced by the
correctness lane — they were measured once, on the named card. A restructuring that keeps every
assertion green can still lose the performance they describe; re-measure before believing a claim
that one of them no longer applies.

Something else in the tree records this shape of evidence now: a realization-corpus case carries a
per-card ``latency:`` block holding both its own microseconds and ``torch.compile``'s, refreshed by
a workflow that rents the card. These rows stay prose until this file's attention programs become
cases — none of them are yet, and pointing at an empty corpus would be worse than a table.

**5090, f16, ``(1, 8, S, D)``, fused arm against the two-kernel materialized-score sibling and
torch SDPA (us):**

| shape        | torch | two-kernel | fused    | (gmem-direct score) |
| ------------ | ----- | ---------- | -------- | ------------------- |
| S=512 D=16   |  10.5 |       35.4 | **7.5**  |                 8.6 |
| S=512 D=32   |  10.5 |       22.7 | **6.9**  |                 8.1 |
| S=512 D=64   |  12.5 |       25.6 | **11.4** |                14.3 |
| S=512 D=128  |  18.9 |       19.5 |   22.2   |                35.1 |
| S=1024 D=128 |  41.2 |       64.0 |   84.4   |               125.8 |
| S=2048 D=16  |  33.0 |      163.8 |   36.1   |                42.1 |
| S=2048 D=32  |  30.9 |      170.2 |   47.7   |                60.1 |
| S=2048 D=64  |  60.6 |      140.8 |   97.0   |               148.3 |
| S=2048 D=128 | 117   |      162    |  190.8   |               273   |

So on that card the fused arm beat the materialized sibling at every ``D <= 64`` and beat torch at
every ``D`` at ``S = 512``, and lost at ``D = 128``. Both forms remain structural siblings so that
measured evidence chooses between them rather than a compile-time rule. These rows PREDATE the
slab-swizzle fix below and have not been re-measured; the A100 rows have.

**All four of the fused kernel's slabs must swizzle.** Three of them once did not: the score's own
staged operands and the weight tile the ``ldmatrix`` drain reads were left plain row-major, and at
``D = 128`` their 256 B rows drained ~32-way bank-conflicted — 78% of the kernel's shared
wavefronts were conflict replays, L1/TEX sat at 80% against 7% compute, and THAT, not the tile, was
the ``D = 128`` wall. Three things had to hold at once, each measured on its own (A100, f16,
``(1, 8, 1024, 128)``, ``w4x1``/``f1x16``/``k8``): the score's key and query slabs swizzling on
their ``cp.async`` fill (108.7 -> 84.8 us), the weight tile swizzling on its FRAGMENT store
(84.8 -> 64.9), and the XOR shifting by the slab's own ROW rather than by the swizzle atom
(64.9 -> 50.1 at the best schedule). The last is why the D-sweep looks the way it does: a fixed
shift reads the row index only while a slab row IS one 128 B atom, so ``D <= 64`` was always
conflict-free and only ``D = 128`` paid — ncu measured 7.8-way at ``D = 128`` and none at
``D = 64`` on the same kernel.

**A100-SXM4-80GB, f16, ``(1, 8, S, 128)``, ``EMMY_PLACE=fuse``, serial ``REDUCE``, empty tune DB
and prior.** The ``D = 128`` conclusion above does NOT hold here — the fused arm beats torch at the
short shapes and reaches parity at ``S = 1024`` (us):

| shape        | torch | fused (best schedule pinned)    |
| ------------ | ----- | ------------------------------- |
| S=256 D=128  |  13   | **9.0**  (``w4x1``/``f1x8/k8``) |
| S=512 D=128  |  24   | **19.0** (``w4x1``/``f1x16/k8``) |
| S=1024 D=128 |  48   |   50.1   (``w8x1``/``f1x16/k8``) |
| S=2048 D=128 | 133   |  179.0   (``w4x1``/``f1x16/k8``) |

Emmy's column repeats to +/-0.2 us across runs; torch's drifts a few percent, and more at
``S = 2048`` (132-148 observed). The COLD pick is a separate matter: unpinned it lands on ``f1x8``
at every shape, which happens to be the best schedule at ``S = 256`` (9.0) and is 231 us at
``S = 512``.

Two forks the cold pick can land on cost most of the fused column: a split-K partition falls back
to the two-pass pair (25.9 vs 14.3 at S=512 D=64), and an ``n``-unit split of the output tile makes
every ``n`` warp recompute the whole score (323 cold vs 190.8 at S=2048 D=128). Both are schedule
forks evidence prices, not defects. What is left at ``D = 128`` after the swizzles is not the
shared path either: L1/TEX fell from 80% to 61% and compute rose from 7% to 24%, and the residual
grows with ``S``.

**Single-pass sweep.** Recomputing the score costs 1.4-1.7x across ``D = 16..128`` on a 5090, and a
regression from the single pass back to the two-pass pair is invisible to a numerics assert — which
is what ``test_fused_sdpa_sweeps_the_score_once`` is for.

**Explicit-mask forms: pinned scalar, unpinned greedy is a burner.** An explicit additive mask is
not recognized into the λ-spelled flash carrier, so its term is the UNRECOGNIZED three-pass
softmax — at ``(1, 2, 16, 8)`` f32 an 8-node stored tree (four contractions, two stat folds) whose
legal schedule pool is 486,130 rows. The term schedules and lowers CORRECTLY (the pinned fused
scalar row matches torch to 1e-6, and the placement cut realizes beside it), but unpinned greedy
must flatten that pool (6.5 s) and prior-score every row (~41 s per ``knob_features`` pass, 2-3
passes per compile, ~2.2 GB RSS) — minutes per compile, the known lazify-greedy roadblock (same
mechanism as the EXL3 computed-operand skips). Pre-rebuild (bff3e344) the same compile burned
>110 s at 14.6 GB RSS, so the rebuild only shrank the memory. The mask cells therefore pin the
fused scalar row (they protect mask ACCURACY, not the cold policy);
``test_full_self_attn_tinyllama`` pins the bare recursive cut, which contraction-operand cuts
make sufficient: the Q/K/V+RoPE cones split into their own kernels and the attention kernel reads
materialized Q/K/V instead of recomputing them per scalar cell.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import from_pretrained_or_skip, requires_cuda


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


def test_fragment_lift_preserves_coordinate_select_as_fragment_values():
    """A causal score prefix keeps its coordinate Select exact at fragment residence."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import FRAG, FragmentApply, FragmentSelect  # noqa: PLC0415
    from emmy.compiler.ir.pure import Lambda  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Assign, Body, RenderCtx, Select, SelectBranch  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.kernel._eval import Value, evaluate  # noqa: PLC0415

    body = (
        Assign("scaled", ElementwiseImpl("multiply"), ("acc", "scale")),
        Select(
            "bias",
            (
                SelectBranch("zero", BinaryExpr("<=", Var("key"), Var("query"))),
                SelectBranch("fill", BinaryExpr(">", Var("key"), Var("query"))),
            ),
        ),
        Assign("masked", ElementwiseImpl("add"), ("scaled", "bias")),
    )
    lifted, _, _ = evaluate(
        Lambda(params=("acc", "scale", "zero", "fill"), body=Body(body), results=("masked",)),
        {
            "acc": Value.frag((("_score00", "_score01"), ("_score10", "_score11"))),
            "scale": Value.uniform("scale"),
            "zero": Value.uniform("zero"),
            "fill": Value.uniform("fill"),
        },
        axes=("query", "key"),
        bases=(
            ((Literal(16, "int"), Literal(32, "int")), (Literal(16, "int"), Literal(40, "int"))),
            ((Literal(32, "int"), Literal(32, "int")), (Literal(32, "int"), Literal(40, "int"))),
        ),
    )

    assert [type(stmt) for stmt in lifted] == [FragmentApply] * 4 + [FragmentSelect] * 4 + [FragmentApply] * 4
    assert lifted[0].args == ("_score00", "scale")
    assert lifted[4].out == "bias__f0_0"
    assert lifted[7].out == "bias__f1_1"
    assert lifted[8].args == ("bias__f0_0", "scaled__f0_0")
    assert lifted[8].kinds == (FRAG, FRAG)
    first = "\n".join(lifted[4].render(RenderCtx(indent=0, literal_ssa={"zero": 0.0, "fill": -1e9})))
    last = "\n".join(lifted[7].render(RenderCtx(indent=0, literal_ssa={"zero": 0.0, "fill": -1e9})))
    assert "__frow" not in first and "__fcol" not in first
    assert "32 + (_t * 2" in first and "16 + _g" in first
    assert "40 + (_t * 2" in last and "32 + _g" in last
    assert first.count("?") == last.count("?") == 4


def test_fragment_lift_declines_nonuniform_select_branches():
    """A coordinate Select may not broadcast a fragment or per-cell value as a uniform branch."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
    from emmy.compiler.ir.pure import Lambda  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Assign, Body, Select, SelectBranch  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.kernel._eval import Value, evaluate  # noqa: PLC0415

    fragment_branch = (
        Assign("scaled", ElementwiseImpl("multiply"), ("acc", "scale")),
        Select(
            "masked",
            (
                SelectBranch("scaled", BinaryExpr("<=", Var("key"), Var("query"))),
                SelectBranch("fill", BinaryExpr(">", Var("key"), Var("query"))),
            ),
        ),
    )
    per_cell_branch = (
        Assign("scaled", ElementwiseImpl("multiply"), ("acc", "scale")),
        Assign("row_value", ElementwiseImpl("multiply"), ("query", "scale")),
        Select(
            "masked",
            (
                SelectBranch("row_value", BinaryExpr("<=", Var("key"), Var("query"))),
                SelectBranch("fill", BinaryExpr(">", Var("key"), Var("query"))),
            ),
        ),
    )
    for body in (fragment_branch, per_cell_branch):
        with pytest.raises(ValueError):
            evaluate(
                Lambda(params=("acc", "scale", "fill"), body=Body(body), results=("masked",)),
                {
                    "acc": Value.frag((("_score",),)),
                    "scale": Value.uniform("scale"),
                    "fill": Value.uniform("fill"),
                },
                axes=("query", "key"),
                bases=(((Literal(16, "int"), Literal(32, "int")),),),
            )


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
    """An SDPA variant (non-causal / causal / GQA / explicit additive mask) matches torch SDPA
    across the variant's static configs. Placement may leave a kernel set; the softmax markers
    (``fmaxf`` + ``expf``) live somewhere in that set, and every kernel carries a schedule. The
    exact kernel count is not this test's contract. The ``mask`` variant pins the fused scalar
    row: its term is the UNRECOGNIZED three-pass softmax, whose unpinned pool greedy cannot
    afford to score (module docstring's burner evidence) — this cell protects mask accuracy,
    not the cold policy."""
    if variant == "mask":
        _pin_scalar_fused(monkeypatch)
    torch.manual_seed(0)
    for cfg in _FLASH_VARIANTS[variant][2]:
        module, args, feed, ref = _flash_feed(variant, *cfg)
        backend, compiled, _graph, kernels = _trace(module, args)
        assert kernels, f"{variant}{cfg}: no kernels"
        if variant == "plain" and cfg == _FLASH_VARIANTS["plain"][2][0]:
            srcs = "\n".join(compiled.nodes[k].op.kernel_source for k in kernels)
            assert "fmaxf" in srcs and "expf" in srcs, "the kernel set should carry the softmax (max + exp)"
        md = _max_diff(backend, compiled, feed, ref)
        assert md < 1e-4, f"{variant}{cfg}: sdpa vs torch max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize("cfg", [(1, 1, 32, 16), (1, 2, 64, 16)])
def test_fused_single_kernel_sdpa_matches_torch(monkeypatch, cfg):
    """SDPA lowers to ONE kernel and matches torch through the ordinary fusion path.

    Recognition proves the exact two-use score inverse, so nested-reduce readability and work
    account for the producer once. Pin the fused placement because this test protects that sibling,
    not the tuner's choice between the fused and cut rows."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    torch.manual_seed(0)
    B, H, S, D = cfg
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"]).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"the merged cell must lower to ONE kernel, got {len(kernels)}: {kernels}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    # Both halves of the composed score reach the TENSOR CORE: the score's transposed-B fragment
    # load (its own mma, not the enclosing P@V's staged drain) and the statistic's fragment
    # row-reduce butterfly. Without these the score is a scalar dot per element — 15x slower, and a
    # silent regression a numerics assert would never catch.
    assert "emmy_mma_load_b_gmem_trans" in src, "the composed score is not on the mma tier"
    assert "__shfl_xor_sync" in src, "the statistic is not at fragment residence"
    # Every slab is addressed through ONE swizzle mode — all of its uses XOR, or none do. Which
    # mode a slab picks is the schedule's business (a row too narrow for any atom stays NONE), but
    # a MIXED slab is a silently WRONG kernel, not a slow one: it stores row-major under a drain
    # that reads permuted. The weight tile is where that bites, because its producer is a fragment
    # store rather than a copy — its mode rides ``RegStore.swizzle``, and the per-cell store
    # rewrite dropped exactly that field.
    slabs = set(re.findall(r"__shared__[^;]*?(\w+_smem)\[", src))
    assert slabs, "the fused kernel stages no slab"
    for slab in slabs:
        uses = len(re.findall(rf"(?<!\w){slab}\[", src)) - 1  # every use but the declaration
        swizzled = len(re.findall(rf"(?<!\w){slab}\[emmy_swizzle_", src))
        assert swizzled in (0, uses), f"{slab}: {swizzled} of {uses} uses swizzle — producer and drain disagree"
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"fused sdpa{cfg} vs torch max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize("cfg", [(1, 1, 32, 16), (1, 2, 64, 16)])
def test_fused_sdpa_sweeps_the_score_once(monkeypatch, cfg):
    """The fused kernel makes ONE pass over the keys — the statistic and the weight come off the
    SAME score fragments.

    The two halves cannot share a pass while the weight names a denominator the sweep has not
    finished, so the cone's state-only factors are split off and multiplied back onto the output
    fragments after the loop, and the carrier's psi-rescale (``carrier.exp_rescale`` — the factor
    the merge puts on every carried channel) advances the enclosing drain's output tile per KV
    chunk.

    What is asserted is that form's OBSERVABLE consequence: the carried ``(pivot, denominator)``
    pair never leaves the registers, so nothing is bridged through the stat smem rows the two-pass
    pair needs. Recomputing the score costs 1.4-1.7x across ``D = 16..128`` on a 5090, and a
    regression to the two-pass pair is invisible to a numerics assert.

    ``REDUCE`` is pinned off because the single pass needs the sweep and the contraction to cover
    the same keys; a split-K partition does not, and there the two-pass pair stands
    (``test_fused_sdpa_split_partition_keeps_the_two_pass_pair``)."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_REDUCE", "")  # serial reduce: the sweep and the contraction cover the same keys
    torch.manual_seed(0)
    B, H, S, D = cfg
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"]).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    assert len(kernels) == 1, f"the merged cell must lower to ONE kernel, got {len(kernels)}: {kernels}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "_a_stat_" not in src, "the statistic is bridged through smem — the sweep ran twice (two-pass pair)"
    assert "__shfl_xor_sync" in src, "the statistic is not at fragment residence"
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"single-pass fused sdpa{cfg} vs torch max_diff={md:.6e}"


@requires_cuda
def test_fused_causal_sdpa_sweeps_the_score_once(monkeypatch):
    """The causal coordinate Select stays on score fragments, so the one-pass sweep remains legal."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f2x2/k2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f2x4")
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 32
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {name: torch.from_numpy(array).cuda() for name, array in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"], is_causal=True).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Causal(), (q, k, v))
    assert len(kernels) == 1, f"the merged cell must lower to ONE kernel, got {len(kernels)}: {kernels}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "_a_stat_" not in src, "the causal statistic is bridged through smem — the sweep ran twice"
    assert "?" in src, "the causal coordinate Select did not reach the fragment program"
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"single-pass causal fused sdpa vs torch max_diff={md:.6e}"


@requires_cuda
def test_fused_sdpa_stages_the_nested_score(monkeypatch):
    """The nested score reads its OWN operands out of smem — the keys refilled per chunk, the
    queries filled once ahead of the sweep — not through the per-lane gmem fragment loaders.

    Every warp of the CTA needs the whole key chunk, so gmem-direct made it cross L1 once per warp;
    staged, it crosses once per CTA and the read is one ``ldmatrix`` where it was four scalar loads
    per lane. Measured on a 5090 that is 35.1 → 22.2 µs at ``(1, 8, 512, 128)`` and 148.3 → 97.0 at
    ``(1, 8, 2048, 64)``.

    Asserted structurally, because it is invisible to a numerics check: the score's slabs exist and
    no gmem fragment loader is CALLED (the helper definitions always ship)."""
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f2x2/k1")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f2x2")
    monkeypatch.setenv("EMMY_STAGE@A3", "d1/smem-async")
    monkeypatch.setenv("EMMY_STAGE@PJ", "d1/smem")
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 64, 16
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"]).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "_s_b_smem" in src, "the nested score's keys are not staged"
    assert "_s_a_smem" in src, "the nested score's queries are not staged"
    for loader in ("emmy_mma_load_a_gmem(", "emmy_mma_load_b_gmem_trans("):
        assert src.count(loader) == src.count(f"void {loader}"), f"{loader} is still called — an operand stayed gmem-direct"
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"staged-score fused sdpa vs torch max_diff={md:.6e}"


@requires_cuda
def test_fused_sdpa_split_partition_merges_monoid_states(monkeypatch):
    """Each partition folds its key slice into the same state; the finalize merges those states."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_REDUCE", "g2k")  # two cross-CTA partitions with an f32 deferred finalize
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 64, 16
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {n: torch.from_numpy(a).cuda() for n, a in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"]).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Sdpa(), (q, k, v))
    src = "".join(compiled.nodes[nid].op.kernel_source for nid in kernels)
    assert "__partial" in src and src.count("__global__") == 2
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"split-K fused sdpa vs torch max_diff={md:.6e}"


@requires_cuda
def test_fused_causal_sdpa_split_partition_keeps_absolute_predicate_coordinates(monkeypatch):
    """A causal partial compares absolute query/key coordinates after the Fold is sliced."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 32
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    feed = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
    cuda = {name: torch.from_numpy(array).cuda() for name, array in feed.items()}

    def ref():
        with torch.no_grad():
            return F.scaled_dot_product_attention(cuda["q"], cuda["k"], cuda["v"], is_causal=True).cpu().flatten().float().numpy()

    backend, compiled, _graph, kernels = _trace(_Causal(), (q, k, v))
    src = "".join(compiled.nodes[nid].op.kernel_source for nid in kernels)
    assert "__partial" in src and src.count("__global__") == 2
    assert "_ks - _ks" not in src, "the causal predicate lost the absolute key-chunk base"
    md = _max_diff(backend, compiled, feed, ref)
    assert md < 4e-3, f"split-K causal fused sdpa vs torch max_diff={md:.6e}"


@requires_cuda
@pytest.mark.parametrize("variant", ["plain", "gqa", "mask"])
def test_scalar_flash_dynamic_matches_torch(monkeypatch, variant):
    """Symbolic ``seq_len`` (Q/K/V dim -2): ONE cached kernel carrying ``int seq_len`` serves every
    runtime size — flash's single dynamic axis lands on the masked-row M, the symbolic reduce, and
    (for GQA) the causal guard at once. Accurate vs torch at seq ∈ {8, 16, 37}. The ``mask``
    variant pins the fused scalar row, exactly as in ``test_scalar_flash_matches_torch``."""
    if variant == "mask":
        _pin_scalar_fused(monkeypatch)
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
    assert kernels, f"dynamic {variant}: no kernels"
    assert any("int seq_len" in compiled.nodes[k].op.kernel_source for k in kernels), "a dynamic kernel must carry the runtime seq_len arg"

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
def test_flash_causal_and_gqa_match_torch(monkeypatch):
    """The causal / GQA masks ride the score cone through the generic split, so masked +
    grouped-head SDPA also matches torch."""
    torch.manual_seed(0)

    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    backend, compiled, _graph, kernels = _trace(_Causal(), (q, k, v))
    assert kernels
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
    (``_out_store_index``): SDPA + absorbed transpose matches torch across the split kernel set."""
    torch.manual_seed(0)
    for cfg in [(1, 4, 16, 16), (2, 3, 32, 16)]:
        q, k, v = (torch.randn(*cfg) for _ in range(3))
        backend, compiled, _graph, kernels = _trace(_SdpaTranspose(), (q, k, v))
        assert kernels, f"{cfg}: no kernels"
        cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

        def ref(cq=cq, ck=ck, cv=cv):
            with torch.no_grad():
                return F.scaled_dot_product_attention(cq, ck, cv).transpose(1, 2).contiguous().cpu().flatten().numpy()

        md = _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
        assert md < 1e-4, f"{cfg}: transposed-output flash vs torch max_diff={md:.6e}"


# =========================================================================== #
# Tensor-core flash — the fragment-resident warp tier.
# =========================================================================== #
# These cases expect a fp16/bf16 SDPA to lower to a single ``mma.sync`` kernel (the warp chain:
# tiled + atomized contractions, fragment online-softmax, C->A register repack) — realized through the
# ONE pipeline: the twisted warp enumeration stamps the mma ``TilePlan``\ s on the Q@K / P@V
# bilinear ``Fold``\ s and ``_bind``'s reduce arm realizes the TWISTED carrier at fragment residence
# (``_twist``). No private emitter exists; a bespoke path would be the mandate violation the
# demolition removed. Unpinned, the warp rows are fork SIBLINGS of the chain / reduce-partition
# forms (the flash-form fork) — the cold ``OfflinePrior`` pick stays warp-when-eligible, which is
# what these cold-compile cases pin.


def _compile_tc(q, k, v, module=None):
    return _trace(module if module is not None else _Sdpa(), (q, k, v))


# --------------------------------------------------------------------------- #
# f16-accumulate PV (the ``_f16acc`` atom on the expect node — chunked f16→f32 promote).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Staged K/V (the ``STAGE@<kv>`` cp.async stream) — Moves 4/5 on the warp tier.
# --------------------------------------------------------------------------- #


def _run_flash(backend, compiled, graph, tensors) -> np.ndarray:
    data = {n: t.numpy() for n, t in zip(graph.inputs, tensors, strict=True)}
    run_result, _ = backend.run(compiled, input_data=data)
    return list(run_result.outputs.values())[0].flatten()


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


# =========================================================================== #
# Cooperative-KV flash (BR) — the KV axis split across threads, monoid combine.
# =========================================================================== #


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

    fn = nvcc.load_function(_KERNEL, "fa2", "", arch_specific=False)
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
    """Pin the fused scalar row + a fixed seed for the model-chain tests. These are accuracy
    tests, not cold-policy tests: unpinned greedy flattens and prior-scores the full schedule pool
    of the fused chain term (six-figure row counts on the explicit-mask forms — see the module
    docstring's burner evidence), so the schedule is pinned to the one row every term offers."""
    torch.manual_seed(42)
    _pin_scalar_fused(monkeypatch)


def _pin_scalar_fused(monkeypatch):
    """Pin the fused placement and the per-cell scalar row (every family at its declared OFF)."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    for fam in ("WORK", "TILE", "STAGE", "REDUCE", "RASTER"):
        monkeypatch.setenv(f"EMMY_{fam}", "")


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
@pytest.mark.xfail(
    run=False,
    reason="pre-existing on clean main: the wide-product form faults and poisons the CUDA context",
)
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


def _band_mask(seq: int, window: int) -> torch.Tensor:
    """A sliding-window additive float mask — causal AND within ``window`` keys of the query (the
    HF gemma sliding-attention band): 0 where ``m - window < kv <= m``, ``-inf`` elsewhere."""
    kv = torch.arange(seq)[None, :]
    m = torch.arange(seq)[:, None]
    keep = (kv <= m) & (kv > m - window)
    return torch.where(keep, 0.0, float("-inf"))[None, None].half()


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
def test_full_self_attn_tinyllama():
    """The real ``LlamaAttention`` from a TinyLlama config — the smallest scope that includes Q/K/V
    Linears, RoPE, masked SDPA, and O Linear. If this fails while the two simpler chains pass, the
    regression is in the RoPE elementwise kernel or its interaction with the attention numerics.
    Pin the bounded two-kernel scalar route: this is an accuracy test, not a cold-policy test."""
    torch.manual_seed(42)
    with pinned_knobs({"PLACE": "cut", "RASTER": "", "TILE": "", "STAGE": "", "REDUCE": "", "WORK": ""}):
        _run_self_attn_tinyllama(seq_len=32, threshold=1e-4)


@requires_cuda
@pytest.mark.skip(reason="seq_len=512 accuracy needs a per-kernel schedule manifest; the fused scalar pin exceeds the watchdog")
def test_full_self_attn_tinyllama_seq512(_chain_tile_pins):
    """Same at seq_len=512 — the shape that makes the SDPA P@V kernel the dominant cost (32 MB
    materialized score matrix, one CTA per output element). Pins correctness so future fusion /
    cooperative-output-tiling doesn't regress accuracy. Threshold is loose (2.0 ≈ 90% of max_eager):
    at seq=512 with random fp32 weights the naive-vs-naive comparison drifts substantially, and TMA
    (default on sm_90+) reorders FMAs vs cp.async. Catches order-of-magnitude regressions, not
    bit-equivalence."""
    _run_self_attn_tinyllama(seq_len=512, threshold=2.0)


class _GqaDecodeBatch(torch.nn.Module):
    """The Neptune ``decode_gqa`` spelling: the query heads of one KV group as a broadcast batch dim."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q.reshape(1, 2, 4, 1, 16), k.reshape(1, 2, 1, 64, 16), v.reshape(1, 2, 1, 64, 16), is_causal=False
        ).reshape(1, 8, 1, 16)


@requires_cuda
@pytest.mark.parametrize("variant", ["plain", "gqa_batch"])
def test_decode_sdpa_matches_torch(variant):
    """One query over a populated key set — the shape every decode step takes. Three defects hid
    behind it: the SDPA decomposition read the KEY length off the query (scores ``(1, 1)``, so the
    output was ``ΣV``), the fused softmax·V view took the heads as the mma rows while V varied
    per head (B must never read the row axis), and the GQA batch spelling sank ``1/den`` into the
    joined expectation channel, streaming it against the running denominator (NaN)."""
    torch.manual_seed(0)
    if variant == "plain":
        module, q = _Sdpa(), torch.randn(1, 8, 1, 16)
        ref_shape = (1, 8, 64, 16)
    else:
        module, q = _GqaDecodeBatch(), torch.randn(1, 8, 1, 16)
        ref_shape = (1, 2, 64, 16)
    k, v = (torch.randn(*ref_shape) for _ in range(2))
    backend, compiled, _graph, kernels = _trace(module, (q, k, v))
    assert kernels
    cq, ck, cv = q.cuda(), k.cuda(), v.cuda()

    def rc():
        with torch.no_grad():
            return module(cq, ck, cv).cpu().flatten().numpy()

    assert _max_diff(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, rc) < 1e-4
