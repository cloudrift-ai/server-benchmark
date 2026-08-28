"""Cooperative-reduce coverage — what a realization case cannot state.

The cross-execution-unit reduction the monoid-combine work generalizes (lanes → warps → CTAs) is
**carrier-generic**: a plain reduction (``Accum``), online softmax / its stats (``Accum`` max+sum),
and flash attention (the ``(m, d, o)`` twisted ``Monoid``) all fold through the SAME combine,
differing only in carrier state. That matrix — op type × reduction variant × (static / symbolic)
reduce axis, the cross-CTA split and its atomic / deferred finalize, the transposed ``coop-t``
band, the 2D segmented row, and the fused residual-add producer — is replayed as data by
``tests/compiler/realization/cases/reduce``: one case per (program, authored schedule), asserting
the schedule is offered, realized (every authored family stamped on the kernel), built and correct.

Three things stay here because no case can say them:

- **the exp-family generator's algebra** — a pure IR unit test with no schedule to author.
- **the transposed band is OFFERED** at a swept extent 32 does not divide — an enumeration-only
  claim about a schedule nothing then realizes. Neither of these needs a GPU.
- **the two verdicts a row has no spelling for**: an emitted-source signature no knob pins (the
  online-softmax pairing), and a refusal message (the atomic finalize declining a projection that
  does not distribute over the add).
"""

from __future__ import annotations

import pytest
import torch

from emmy.compiler.ir.pure.carrier import exp_combine_states, exp_merge
from emmy.compiler.trace.torch import trace_module
from tests.compiler.helpers import requires_cuda

# --------------------------------------------------------------------------- #
# Cross-CTA split-reduce — the projection epilogue the atomic finalize refuses.
# --------------------------------------------------------------------------- #
# The ATOMIC finalize applies the projection to each CTA's partition before the ``atomicAdd``, so it
# is correct only when the projection DISTRIBUTES over the add (``Σ φ(xₛ) = φ(Σ xₛ)``): ``mean``'s
# ``×1/N`` (a constant scale) distributes; ``l2``'s ``sqrt`` does not. The three cells that DO
# realize are corpus cases (``split-reduce-mean-atomic``, ``split-reduce-mean-kernel``,
# ``split-reduce-l2-kernel``); the fourth stays here, because a schedule the compiler correctly
# refuses has no row to live in.
_L2_SPLIT_REDUCE = "torch.sqrt((lambda t: (t * t).sum(dim=1, keepdim=True))(torch.randn(4, 1024)))"


@requires_cuda
def test_split_reduce_projection_epilogue(monkeypatch):
    """``l2``'s ``sqrt`` epilogue does not distribute over the add, so the pinned ATOMIC cross-CTA
    finalize must raise and direct the caller to ``g<n>k`` — not silently finalize raw partial sums
    (the bug this guards silently dropped ``mean``'s ``×1/N`` that way). Refusal-only: the accurate
    cells of the old matrix are realization cases now."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_REDUCE", "g2a")
    with pytest.raises(ValueError, match="must distribute over the add"):
        CudaBackend().compile(graph_from_code(_L2_SPLIT_REDUCE)[0])


# --------------------------------------------------------------------------- #
# Online-softmax fusion — the two-pass → one-pass streaming rewrite.
# --------------------------------------------------------------------------- #
# The standalone two-pass softmax (row-max reduce + ``Σ exp(x − max)`` reduce + normalize) fuses
# into a single streaming online-softmax ``(m, d)`` ``Monoid`` pass (3 reads of ``x`` → 2). The Tile
# rewrite has its own unit tests (``passes/test_twisted_rewrite.py``); what is pinned here is that
# it survives all the way into the emitted kernel.


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


def test_exp_family_generator_builds_asymmetric_monoid() -> None:
    # state (m, d), partial (s); the asymmetric LSE monoid's streaming merge folds exactly the
    # injected score, and the cross-partition state⊕state combine is generated from the same spec.
    # The merge's external read set is read through the ordinary per-stmt ``deps()`` — a generated
    # program is a run of ordinary stmts, so nothing has to surface its reads specially.
    merge = exp_merge(("m", "d"), ("s", 1.0), key="m")
    reads = {r for st in merge for r in st.deps()} - {st.name for st in merge} - {"m", "d"}
    assert reads == {"s"}
    assert exp_combine_states(("m", "d"), ("m__o", "d__o")), "combine_states must be derived for the asymmetric LSE monoid"


@requires_cuda
@pytest.mark.parametrize("shape", [(4, 128), (8, 256), (2, 64), (2, 4, 128)])
def test_online_softmax_pairing_reaches_the_kernel(shape) -> None:
    """The pairing must have fired: ONE fused kernel streaming the twisted carrier, whose signature
    is the dissolved exp-family merge's rescale temps (``<state>__tN = expf(...)`` — ``exp_merge``
    namespaces them on the carried state, so they are stable across SSA renaming).

    This is the only assertion in the tree that the rewrite reaches CUDA. No knob names the fusion,
    so the corpus cases (``online-softmax-*``) author the post-fusion schedule whether or not it
    fires, and an unfused two-pass softmax computes the identical numbers — their ``correct`` stage
    provably cannot see the rewrite stop firing."""
    import re

    from emmy.compiler.backend.cuda.backend import CudaBackend

    torch.manual_seed(0)
    graph = trace_module(_Softmax().cpu(), (torch.randn(*shape),))
    compiled = CudaBackend().compile(graph)
    srcs = [getattr(node.op, "kernel_source", "") for node in compiled.nodes.values()]
    assert any(re.search(r"__t\d+ = expf\(", src) for src in srcs), "online-softmax pairing did not fire"


# --------------------------------------------------------------------------- #
# Transposed cooperative band (``coop-t``) — the k-major matvec sweep's enumeration.
# --------------------------------------------------------------------------- #

_COOPT_K = 256  # the contraction extent; kept apart from the swept extent


def _matvec_code(n_out: int) -> str:
    """A k-major matvec (``F.linear`` — weights ``(N, K)``): the shape the transposed band sweeps."""
    return f"torch.nn.functional.linear(torch.randn(1, {_COOPT_K}), torch.randn({n_out}, {_COOPT_K}))"


@pytest.mark.parametrize("n_out", [512, 500])
def test_transposed_coop_band_is_offered_on_a_non_divisible_sweep(n_out, monkeypatch):
    """The band's rows (bare and the ``g<n>k/`` split composites) are OFFERED at a swept extent 32
    does not divide — no GPU, enumeration only. The 32-divisibility rule used to drop every one of
    them, so no golden could record the band on such a shape."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph

    for var in ("EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE"):
        monkeypatch.delenv(var, raising=False)
    rows = enumerate_graph(graph_from_code(_matvec_code(n_out))[0], Context.from_target((12, 0))).rows
    offered = {str(v) for r in rows for k, v in r.items() if k.startswith("REDUCE")}
    assert any(s.endswith("coop-t") for s in offered), offered
