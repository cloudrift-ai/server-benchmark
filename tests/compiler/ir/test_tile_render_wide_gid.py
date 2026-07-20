"""``Tile.render`` widens the flat thread id past INT32_MAX.

A per-cell coop-reduce grid can exceed 2^31 total threads (gemma-4-12B's M=4096
down-proj demote: 4096 * 3840 * 256 = 4.03e9); the 32-bit ``blockIdx.x * blockDim.x``
then wraps negative, every decoded index goes wild, and the first launch dies with
``cudaErrorIllegalAddress`` (the M=4096 prefill-chunk twin crash). The decode must
ride a 64-bit ``_gid`` exactly when the static iteration space needs it — and stay
32-bit otherwise (widening everywhere would re-key every kernel's cubin)."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Body, RenderCtx


def _render(axes, **kw) -> str:
    return "\n".join(Tile(axes=axes, body=Body(()), **kw).render(RenderCtx()))


def test_wide_gid_past_int32():
    src = _render((Axis("m", 4096), Axis("n", 3840), Axis("k_co", 256)), block_threads=256)
    assert "long long _gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;" in src
    # The per-axis decode still lands in 32-bit vars (each extent fits int).
    assert "int m = _gid / 983040;" in src


def test_int_gid_below_int32():
    src = _render((Axis("m", 64), Axis("n", 32)))
    assert "int _gid = blockIdx.x * blockDim.x + threadIdx.x;" in src
    assert "long long" not in src


def test_wide_gid_aux_band_casts_blockidx():
    # The warp-specialized decode (blockIdx * block_threads) needs the same widening.
    src = _render((Axis("m", 4096), Axis("n", 3840), Axis("k_co", 256)), block_threads=256, aux_threads=128)
    assert "long long _gid = (long long)blockIdx.x * 256 + threadIdx.x % 256;" in src
