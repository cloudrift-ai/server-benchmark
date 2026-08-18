"""Side-effect-free DeepSeek V4 serving boundaries compiled by Emmy.

Stateful sparse attention, cache mutation, and distributed collectives remain
owned by the serving runtime.  The modules here spell exact dense or pointwise
tensor contracts on either side of those boundaries.
"""

from __future__ import annotations


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
