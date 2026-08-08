"""The torch surface of the EXL3 trellis decode — one custom op, so a coded matmul is
WRITABLE as a torch expression.

Every tuning / benching / reproduction path in the search machinery drives a shape from a
torch snippet (:func:`~emmy.compiler.pipeline.search.golden.matmul_snippet` →
``trace_inline_code`` → ``torch.export``). A trellis-coded matmul has no aten spelling — a
trellis walk is not composed of tensor ops — so without a torch-level name for it, a coded
shape can be neither tuned nor recorded as a golden. This module registers that name:

    ``torch.ops.emmy.trellis_decode(codes, cb, out_features, in_features) -> (out, in) f16``

The tracer (``trace/torch.py``) maps it 1:1 onto the frontend
:class:`~emmy.compiler.ir.frontend.ir.TrellisDecodeOp` in the HAT basis — exactly the node the
EXL3 speller (``loader/quant.py``) builds in a served model — so a snippet-traced graph and the
in-model graph reach the same kernel and stamp the same structural key.

The eager implementation is the numpy reference decode (:func:`..loader.exl3.decode_trellis`):
correct and bit-exact, but not fast (~0.7 s at 4096x11008, ~7 s at 22016²). It exists to give
``run --bench`` an accuracy reference. Time a coded shape with ``--bench-backends emmy`` so the
eager row never re-decodes once per iteration.
"""

from __future__ import annotations

OP_TARGET = "emmy.trellis_decode"  # ``str(fx_node.target)`` prefix the tracer matches on

_REGISTERED = False


def register() -> None:
    """Register ``emmy::trellis_decode`` with torch (idempotent, torch imported lazily).

    Called from ``trace_inline_code`` before a snippet is exec'd, so a snippet may name
    ``torch.ops.emmy.trellis_decode`` without importing anything itself."""
    global _REGISTERED
    if _REGISTERED:
        return

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from emmy.compiler.loader.exl3 import decode_trellis  # noqa: PLC0415

    # Explicit schema: this module carries ``from __future__ import annotations``, so torch's
    # signature inference would see unresolved string annotations.
    @torch.library.custom_op(
        "emmy::trellis_decode",
        mutates_args=(),
        schema="(Tensor codes, int cb, int out_features, int in_features) -> Tensor",
    )
    def trellis_decode(codes: torch.Tensor, cb: int, out_features: int, in_features: int) -> torch.Tensor:
        """``(k/16, n/16, 16*K)`` int16 codes → the ``(out_features, in_features)`` hat-basis
        weight, in the ``F.linear`` orientation (the decode's own ``(k, n)`` transposed)."""
        w = decode_trellis(codes.detach().cpu().numpy(), cb).T[:out_features, :in_features]
        return torch.from_numpy(np.ascontiguousarray(w)).to(codes.device)

    @trellis_decode.register_fake
    def _(codes, cb, out_features, in_features):
        return codes.new_empty((out_features, in_features), dtype=torch.float16)

    _REGISTERED = True
