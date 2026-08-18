"""Side-effect-free DeepSeek V4 serving boundaries compiled by Emmy.

Stateful sparse attention, cache mutation, and distributed collectives remain
owned by the serving runtime.  The modules here spell exact dense or pointwise
tensor contracts on either side of those boundaries.
"""

from __future__ import annotations

_ROW_RMS_NORM_ROPE_OP = None


def _qnorm_rope_reference(q, positions, cos_sin_cache, rope_dim: int, eps: float):
    import torch

    half_rope = rope_dim // 2
    values = q.float()
    rrms = torch.rsqrt((values * values).mean(dim=-1, keepdim=True) + eps)
    normalized = values * rrms
    rotary = cos_sin_cache[positions]
    cos = rotary[:, :half_rope].float().unsqueeze(1)
    sin = rotary[:, half_rope:].float().unsqueeze(1)
    nope = normalized[..., :-rope_dim]
    pairs = normalized[..., -rope_dim:].reshape(*q.shape[:-1], half_rope, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    roped = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((nope, roped.flatten(-2)), dim=-1).half()


def _validate_row_rms_norm_rope(q, positions, cos_sin_cache, rope_dim: int, eps: float) -> None:
    import math

    if str(q.dtype) != "torch.float16" or q.ndim != 3:
        raise TypeError(f"row_rms_norm_rope requires rank-3 float16 Q, got {q.dtype} {tuple(q.shape)}")
    if str(positions.dtype) != "torch.int64" or positions.ndim != 1 or positions.shape[0] != q.shape[0]:
        raise TypeError("row_rms_norm_rope requires int64 positions matching the Q row count")
    if str(cos_sin_cache.dtype) != "torch.float32" or cos_sin_cache.ndim != 2:
        raise TypeError("row_rms_norm_rope requires a rank-2 float32 cosine/sine cache")
    if not isinstance(rope_dim, int) or isinstance(rope_dim, bool) or rope_dim <= 0 or rope_dim % 2:
        raise ValueError(f"row_rms_norm_rope rope_dim must be a positive even integer, got {rope_dim}")
    if q.shape[-1] <= rope_dim or cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError(f"row_rms_norm_rope requires head_dim > rope_dim and cache width {rope_dim}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"row_rms_norm_rope eps must be finite and positive, got {eps}")


def _row_rms_norm_rope_op():
    """Return the lazy trace boundary for one fused Q normalization and RoPE kernel."""
    import torch

    global _ROW_RMS_NORM_ROPE_OP
    if _ROW_RMS_NORM_ROPE_OP is not None:
        return _ROW_RMS_NORM_ROPE_OP

    @torch.library.custom_op(
        "emmy::row_rms_norm_rope",
        mutates_args=(),
        schema="(Tensor q, Tensor positions, Tensor cos_sin_cache, int rope_dim, float eps) -> Tensor",
    )
    def op(q: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rope_dim: int, eps: float) -> torch.Tensor:
        _validate_row_rms_norm_rope(q, positions, cos_sin_cache, rope_dim, eps)
        return _qnorm_rope_reference(q, positions, cos_sin_cache, rope_dim, eps)

    @op.register_fake
    def fake(q: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rope_dim: int, eps: float) -> torch.Tensor:
        _validate_row_rms_norm_rope(q, positions, cos_sin_cache, rope_dim, eps)
        return q.new_empty(q.shape)

    _ROW_RMS_NORM_ROPE_OP = op
    return op


def _weighted_rms(x, weight, eps: float = 1e-6):
    import torch

    value = x.float()
    rrms = torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + eps)
    return (value * rrms * weight.float()).half()


class FusedQKvRmsNormModule:
    """Normalize the strided Q-rank/KV views of one fused projection output."""

    def __init__(self, q_size: int = 1024):
        self.q_size = int(q_size)

    def module(self):
        import torch

        q_size = self.q_size

        class Module(torch.nn.Module):
            def forward(self, fused_q_kv, q_weight, kv_weight):
                qr = fused_q_kv[:, :q_size]
                kv = fused_q_kv[:, q_size:]
                return _weighted_rms(qr, q_weight), _weighted_rms(kv, kv_weight)

        return Module()


class InverseRopeModule:
    """Undo GPT-J interleaved RoPE on the trailing dimensions of each head."""

    def __init__(self, rope_dim: int = 64):
        self.rope_dim = int(rope_dim)

    def module(self):
        import torch

        rope_dim = self.rope_dim
        half_rope = rope_dim // 2

        class Module(torch.nn.Module):
            def forward(self, x, positions, cos_sin_cache):
                rotary = cos_sin_cache[positions]
                cos = rotary[:, :half_rope].float().unsqueeze(1)
                sin = rotary[:, half_rope:].float().unsqueeze(1)
                nope = x[..., :-rope_dim]
                pairs = x[..., -rope_dim:].float().reshape(*x.shape[:-1], half_rope, 2)
                even = pairs[..., 0]
                odd = pairs[..., 1]
                unrotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
                return torch.cat((nope, unrotated.flatten(-2).half()), dim=-1)

        return Module()


class QNormRopeModule:
    """Apply per-head RMSNorm and GPT-J interleaved forward RoPE to Q."""

    def __init__(self, rope_dim: int = 64, eps: float = 1e-6):
        self.rope_dim = int(rope_dim)
        self.eps = float(eps)

    def module(self):
        import torch

        rope_dim = self.rope_dim
        eps = self.eps

        class Module(torch.nn.Module):
            def forward(self, q, positions, cos_sin_cache):
                return _row_rms_norm_rope_op()(q, positions, cos_sin_cache, rope_dim, eps)

        return Module()


def _dynamic_rows(*names: str):
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    return build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{name}:0" for name in names]))


def trace_fused_q_kv_rmsnorm(*, rows: int, q_size: int = 1024, kv_size: int = 512, dynamic: bool = False):
    import torch

    from emmy.compiler.trace.torch import trace_module

    return trace_module(
        FusedQKvRmsNormModule(q_size).module(),
        (
            torch.empty((rows, q_size + kv_size), dtype=torch.float16, device="meta"),
            torch.empty((q_size,), dtype=torch.float16, device="meta"),
            torch.empty((kv_size,), dtype=torch.float16, device="meta"),
        ),
        dynamic_shapes=_dynamic_rows("fused_q_kv") if dynamic else None,
    )


def trace_inverse_rope(
    *,
    rows: int,
    heads: int = 8,
    head_dim: int = 512,
    rope_dim: int = 64,
    context: int = 1_048_576,
    dynamic: bool = False,
):
    import torch

    from emmy.compiler.trace.torch import trace_module

    return trace_module(
        InverseRopeModule(rope_dim).module(),
        (
            torch.empty((rows, heads, head_dim), dtype=torch.float16, device="meta"),
            torch.empty((rows,), dtype=torch.int64, device="meta"),
            torch.empty((context, rope_dim), dtype=torch.float32, device="meta"),
        ),
        dynamic_shapes=_dynamic_rows("x", "positions") if dynamic else None,
    )


def trace_qnorm_rope(
    *,
    rows: int,
    heads: int = 8,
    head_dim: int = 512,
    rope_dim: int = 64,
    context: int = 1_048_576,
    eps: float = 1e-6,
    dynamic: bool = False,
):
    import torch

    from emmy.compiler.trace.torch import trace_module

    return trace_module(
        QNormRopeModule(rope_dim, eps).module(),
        (
            torch.empty((rows, heads, head_dim), dtype=torch.float16, device="meta"),
            torch.empty((rows,), dtype=torch.int64, device="meta"),
            torch.empty((context, rope_dim), dtype=torch.float32, device="meta"),
        ),
        dynamic_shapes=_dynamic_rows("q", "positions") if dynamic else None,
    )
