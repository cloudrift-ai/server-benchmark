"""Exact DeepSeek V4 multi-stream residual algebra for serving adapters.

The runtime-specific integration lives in :mod:`emmy.serving.onecat`; this
module is the side-effect-free tensor contract that Emmy traces and tests.
"""

from __future__ import annotations

_FIXED_SINKHORN_OP = None


def _sinkhorn_reference(logits, eps: float, iterations: int):
    import torch

    values = torch.softmax(logits, dim=-1) + eps
    values = values / (values.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        values = values / (values.sum(dim=-1, keepdim=True) + eps)
        values = values / (values.sum(dim=-2, keepdim=True) + eps)
    return values


def _validate_fixed_sinkhorn(logits, eps: float, iterations: int) -> None:
    import math

    from emmy.compiler.ir.tensor.ir import FixedSinkhornOp

    if str(logits.dtype) != "torch.float32":
        raise TypeError(f"fixed_sinkhorn requires float32 logits, got {logits.dtype}")
    if logits.ndim != 3:
        raise ValueError(f"fixed_sinkhorn requires rank-3 [M,N,N] logits, got shape {tuple(logits.shape)}")
    rows, cols = logits.shape[-2:]
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise ValueError("fixed_sinkhorn requires static matrix dimensions")
    if rows != cols:
        raise ValueError(f"fixed_sinkhorn requires square matrices, got {rows}x{cols}")
    if not 1 <= rows <= FixedSinkhornOp.MAX_SIZE:
        raise ValueError(f"fixed_sinkhorn matrix size must be in [1,{FixedSinkhornOp.MAX_SIZE}], got {rows}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"fixed_sinkhorn eps must be finite and positive, got {eps}")
    if not 1 <= iterations <= FixedSinkhornOp.MAX_ITERATIONS:
        raise ValueError(f"fixed_sinkhorn iterations must be in [1,{FixedSinkhornOp.MAX_ITERATIONS}], got {iterations}")


def _fixed_sinkhorn_op():
    """Return the lazily registered trace boundary for bounded Sinkhorn normalization."""
    import torch

    global _FIXED_SINKHORN_OP
    if _FIXED_SINKHORN_OP is not None:
        return _FIXED_SINKHORN_OP

    @torch.library.custom_op(
        "emmy::fixed_sinkhorn",
        mutates_args=(),
        schema="(Tensor logits, float eps, int iterations) -> Tensor",
    )
    def op(logits: torch.Tensor, eps: float, iterations: int) -> torch.Tensor:
        _validate_fixed_sinkhorn(logits, eps, iterations)
        return _sinkhorn_reference(logits, eps, iterations)

    @op.register_fake
    def fake(logits: torch.Tensor, eps: float, iterations: int) -> torch.Tensor:
        _validate_fixed_sinkhorn(logits, eps, iterations)
        return logits.new_empty(logits.shape)

    _FIXED_SINKHORN_OP = op
    return op


def fixed_sinkhorn(logits, eps: float = 1e-6, iterations: int = 20):
    """Stable row-softmax followed by bounded alternating column/row normalization."""
    return _fixed_sinkhorn_op()(logits, float(eps), int(iterations))


def _rms(x, eps: float = 1e-6):
    import torch

    x = x.float()
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)


def _weighted_rms(x, weight, eps: float = 1e-6):
    return _rms(x, eps).half() * weight


def _pointwise_stream_sum(values, weights):
    """Spell the fixed four-stream reduction as one wide pointwise expression."""
    values = values.float()
    weights = weights.float()
    terms = tuple(values[:, index] * weights[:, index].unsqueeze(-1) for index in range(4))
    return ((terms[0] + terms[1]) + terms[2]) + terms[3]


def _mix(
    residual,
    fn,
    scale,
    base,
    *,
    prenorm_residual=None,
    pointwise_streams: bool = False,
    eps: float = 1e-6,
    sinkhorn_iters: int = 20,
):
    import torch
    import torch.nn.functional as F

    tokens, streams, _ = residual.shape
    flat = _rms((prenorm_residual if prenorm_residual is not None else residual).flatten(1), eps)
    logits = F.linear(flat, fn.float())
    pre_logits, post_logits, comb_logits = logits.split((streams, streams, streams * streams), dim=-1)
    pre_base, post_base, comb_base = base.float().split((streams, streams, streams * streams), dim=-1)
    pre = torch.sigmoid(pre_logits * scale[0].float() + pre_base) + eps
    post = 2.0 * torch.sigmoid(post_logits * scale[1].float() + post_base)
    comb = comb_logits.view(tokens, streams, streams) * scale[2].float() + comb_base.view(streams, streams)
    comb = fixed_sinkhorn(comb, eps, sinkhorn_iters)
    collapsed = (_pointwise_stream_sum(residual, pre) if pointwise_streams else (pre.unsqueeze(-1) * residual.float()).sum(dim=1)).half()
    return post.unsqueeze(-1), comb, collapsed


def _post(x, residual, post, comb):
    return _post_float(x, residual, post, comb).half()


def _post_float(x, residual, post, comb, *, pointwise_streams: bool = False):
    import torch

    if pointwise_streams:
        mixed = tuple(_pointwise_stream_sum(residual, comb[:, :, index]) for index in range(4))
        return torch.stack(tuple(value + x.float() * post[:, index].float() for index, value in enumerate(mixed)), dim=1)
    return x.float().unsqueeze(1) * post.float() + torch.bmm(comb.float().transpose(1, 2), residual.float())


class MhcPreModule:
    """Multi-stream pre mapping with the following layer RMSNorm."""

    def __new__(cls, *, pointwise_streams: bool = False):
        import torch

        class Module(torch.nn.Module):
            def forward(self, residual, fn, scale, base, norm_weight):
                post, comb, collapsed = _mix(residual, fn, scale, base, pointwise_streams=pointwise_streams)
                return post, comb, _weighted_rms(collapsed, norm_weight)

        return Module()


class MhcBroadcastModule:
    """First-stage single-stream broadcast plus pre mapping and RMSNorm."""

    def __new__(cls, *, pointwise_streams: bool = False):
        import torch
        import torch.nn.functional as F

        class Module(torch.nn.Module):
            def forward(self, x, fn_broadcast, scale, base, norm_weight):
                tokens, hidden = x.shape
                streams = 4
                residual = x.unsqueeze(1).expand(tokens, streams, hidden).contiguous()
                logits = F.linear(_rms(x), fn_broadcast.float())
                pre_logits, post_logits, comb_logits = logits.split((streams, streams, streams * streams), dim=-1)
                pre_base, post_base, comb_base = base.float().split((streams, streams, streams * streams), dim=-1)
                pre = torch.sigmoid(pre_logits * scale[0].float() + pre_base) + 1e-6
                post = 2.0 * torch.sigmoid(post_logits * scale[1].float() + post_base)
                comb = comb_logits.view(tokens, streams, streams) * scale[2].float() + comb_base.view(streams, streams)
                comb = fixed_sinkhorn(comb)
                collapsed = (
                    _pointwise_stream_sum(residual, pre) if pointwise_streams else (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
                ).half()
                return residual, post.unsqueeze(-1), comb, _weighted_rms(collapsed, norm_weight)

        return Module()


class MhcPostModule:
    """Apply one sublayer output to the four residual streams."""

    def __new__(cls, *, pointwise_streams: bool = False):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, residual, post, comb):
                return _post_float(x, residual, post, comb, pointwise_streams=pointwise_streams).half()

        return Module()


class MhcFusedModule:
    """Post mapping followed by the next pre mapping and layer RMSNorm."""

    def __new__(cls, *, fp32_stage: bool = True, pointwise_streams: bool = False):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, residual, post, comb, fn, scale, base, norm_weight):
                residual_float = _post_float(x, residual, post, comb, pointwise_streams=pointwise_streams)
                residual = residual_float.half()
                next_post, next_comb, collapsed = _mix(
                    residual,
                    fn,
                    scale,
                    base,
                    prenorm_residual=residual_float if fp32_stage else residual,
                    pointwise_streams=pointwise_streams,
                )
                return residual, next_post, next_comb, _weighted_rms(collapsed, norm_weight)

        return Module()


class HcHeadModule:
    """Final four-stream collapse before the model's shared RMSNorm."""

    def __new__(cls, *, pointwise_streams: bool = False):
        import torch
        import torch.nn.functional as F

        class Module(torch.nn.Module):
            def forward(self, residual, fn, scale, base):
                normalized = _rms(residual.flatten(1))
                logits = F.linear(normalized, fn.float())
                mix = torch.sigmoid(logits * scale.float() + base.float()) + 1e-6
                collapsed = _pointwise_stream_sum(residual, mix) if pointwise_streams else (mix.unsqueeze(-1) * residual.float()).sum(dim=1)
                return collapsed.half()

        return Module()


def _trace(module, examples, dynamic_inputs: tuple[str, ...] = ()):
    from emmy.compiler.trace.torch import trace_module

    dynamic_shapes = None
    if dynamic_inputs:
        from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{name}:0" for name in dynamic_inputs]))
    return trace_module(module, examples, dynamic_shapes=dynamic_shapes)


def trace_mhc_fused(*, rows: int, hidden: int = 4096, streams: int = 4, dynamic: bool = False):
    """Trace the exact live post-to-pre transition with FP32 mixing parameters."""
    import torch

    return _trace(
        MhcFusedModule(fp32_stage=rows <= 16, pointwise_streams=rows > 16),
        (
            torch.empty((rows, hidden), dtype=torch.float16, device="meta"),
            torch.empty((rows, streams, hidden), dtype=torch.float16, device="meta"),
            torch.empty((rows, streams, 1), dtype=torch.float32, device="meta"),
            torch.empty((rows, streams, streams), dtype=torch.float32, device="meta"),
            torch.empty((streams * (streams + 2), streams * hidden), dtype=torch.float32, device="meta"),
            torch.empty((3,), dtype=torch.float32, device="meta"),
            torch.empty((streams * (streams + 2),), dtype=torch.float32, device="meta"),
            torch.empty((hidden,), dtype=torch.float16, device="meta"),
        ),
        ("x", "residual", "post", "comb") if dynamic else (),
    )


def trace_mhc_pre(*, rows: int, hidden: int = 4096, streams: int = 4, dynamic: bool = False):
    """Trace the exact live pre mapping with FP32 mixing parameters."""
    import torch

    return _trace(
        MhcPreModule(pointwise_streams=rows > 16),
        (
            torch.empty((rows, streams, hidden), dtype=torch.float16, device="meta"),
            torch.empty((streams * (streams + 2), streams * hidden), dtype=torch.float32, device="meta"),
            torch.empty((3,), dtype=torch.float32, device="meta"),
            torch.empty((streams * (streams + 2),), dtype=torch.float32, device="meta"),
            torch.empty((hidden,), dtype=torch.float16, device="meta"),
        ),
        ("residual",) if dynamic else (),
    )


def trace_mhc_broadcast(*, rows: int, hidden: int = 4096, streams: int = 4, dynamic: bool = False):
    """Trace the exact live first-stage broadcast and pre mapping."""
    import torch

    return _trace(
        MhcBroadcastModule(pointwise_streams=rows > 16),
        (
            torch.empty((rows, hidden), dtype=torch.float16, device="meta"),
            torch.empty((streams * (streams + 2), hidden), dtype=torch.float32, device="meta"),
            torch.empty((3,), dtype=torch.float32, device="meta"),
            torch.empty((streams * (streams + 2),), dtype=torch.float32, device="meta"),
            torch.empty((hidden,), dtype=torch.float16, device="meta"),
        ),
        ("x",) if dynamic else (),
    )


def trace_mhc_post(*, rows: int, hidden: int = 4096, streams: int = 4, dynamic: bool = False):
    """Trace the exact live post mapping."""
    import torch

    return _trace(
        MhcPostModule(pointwise_streams=rows > 16),
        (
            torch.empty((rows, hidden), dtype=torch.float16, device="meta"),
            torch.empty((rows, streams, hidden), dtype=torch.float16, device="meta"),
            torch.empty((rows, streams, 1), dtype=torch.float32, device="meta"),
            torch.empty((rows, streams, streams), dtype=torch.float32, device="meta"),
        ),
        ("x", "residual", "post", "comb") if dynamic else (),
    )


def trace_hc_head(*, rows: int, hidden: int = 4096, streams: int = 4, dynamic: bool = False):
    """Trace the final exact live multi-stream collapse."""
    import torch

    return _trace(
        HcHeadModule(pointwise_streams=rows > 16),
        (
            torch.empty((rows, streams, hidden), dtype=torch.float16, device="meta"),
            torch.empty((streams, streams * hidden), dtype=torch.float32, device="meta"),
            torch.empty((1,), dtype=torch.float32, device="meta"),
            torch.empty((streams,), dtype=torch.float32, device="meta"),
        ),
        ("residual",) if dynamic else (),
    )
