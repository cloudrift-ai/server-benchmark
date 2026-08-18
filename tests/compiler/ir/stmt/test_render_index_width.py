"""Address arithmetic widens past ``INT_MAX``.

A flat gmem address is a sum of ``index * stride`` terms, which C evaluates in ``int`` unless an
operand is wider. Past ``INT_MAX`` that sum wraps NEGATIVE and the access lands outside the
allocation — an illegal access that poisons the CUDA context for the rest of the process, not a
wrong answer. Measured before the fix: a 4x512-token Qwen3 trunk planned a 2^32-element activation
buffer, `k_sdpa_linear_reduce` wrote it through 32-bit addressing, and the fault took down every
later CUDA test sharing that xdist worker.
"""

from emmy.compiler.dim import DYNAMIC_DIM_MAX, Dim
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.base import RenderCtx, render_index

_SMALL = (Dim(4), Dim(512), Dim(1024))  # 2_097_152 elements
_BIG = (Dim(4), Dim(512), Dim(2048), Dim(1024))  # 2^32 elements — the buffer that faulted


def _render(shape):
    ctx = RenderCtx(shapes={"buf": shape})
    return render_index("buf", tuple(Var(f"a{i}") for i in range(len(shape))), ctx)


def test_small_buffer_keeps_32_bit_addressing():
    """Byte-identical to the pre-fix output — 64-bit index math is not free, so a buffer that
    cannot reach ``INT_MAX`` must not pay for it."""
    assert "long long" not in _render(_SMALL)


def test_buffer_past_int_max_widens_every_term():
    """The cast must sit on each TERM, not on the finished sum: ``(long long)(a*b + c*d)``
    computes the whole sum in ``int`` and widens an already-wrapped result. Verified on device —
    casting the result left the fault in place; casting per term cleared it."""
    src = _render(_BIG)
    assert src.count("(long long)") == len(_BIG), src
    assert "(long long)(a0)" in src.replace(" ", "") or "(longlong)(a0)" in src.replace(" ", ""), src


def test_symbolic_dim_is_bounded_by_the_exported_cap_not_its_hint():
    """A symbolic axis is taken at ``DYNAMIC_DIM_MAX`` — the bound every ``--dynamic`` axis is
    exported with — so the verdict covers every shape the program can legally resolve at. Using
    ``Dim.hint`` instead would be unsound: it is advisory and a run may exceed it."""
    # Product with the symbolic axis at its cap: 4 * 4096 * 2048 * 1024 > INT_MAX -> widen.
    assert "long long" in _render((Dim(4), Dim("seq_len"), Dim(2048), Dim(1024)))
    # Same axis, small companions: 4096 * 1024 fits, so a dynamic-shape kernel keeps 32-bit.
    assert "long long" not in _render((Dim("seq_len"), Dim(1024)))
    # The hint (512) is far below the cap; the cap is what decides.
    assert Dim("seq_len").hint < DYNAMIC_DIM_MAX
