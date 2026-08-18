"""Narrow Emmy kernel integration for the 1Cat serving runtime.

1Cat retains model loading, TP/PP, routing, attention/cache state, and quantized
expert execution. This module replaces only a side-effect-free leaf whose live
tensor contract is already generic dense algebra. Every unsupported or unverified
call falls back to the original runtime kernel.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class _RmsNormModule:
    """Weighted FP16 RMSNorm used at the exact DeepSeek-V4 decode boundary."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, weight):
                x_fp32 = x.float()
                variance = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
                normalized = (x_fp32 * torch.rsqrt(variance + 1e-6)).half()
                return normalized * weight

        return Module()


def _build_rms_norm_program():
    import torch

    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.external import build_external_program

    examples = (
        torch.zeros((1, 4096), dtype=torch.float16),
        torch.ones((4096,), dtype=torch.float16),
    )
    graph = trace_module(_RmsNormModule(), examples)
    # The deployment recipe records one realized winner. The compiler still
    # enumerates every legal schedule outside this bounded serving boundary.
    program, plan = build_external_program(graph, pins={"WORK": "t256", "REDUCE": "coop"})
    if len(plan.launches) != 1:
        raise RuntimeError(f"1Cat RMSNorm expected one Emmy compiler launch, got {len(plan.launches)}")
    return program, plan.inputs, plan.outputs[0]


class _RmsNormAdapter:
    def __init__(self, original):
        self.original = original
        self._program = None
        self._inputs: list[str] = []
        self._output = ""
        self._disabled = False
        self._verified = False
        self._lock = threading.Lock()

    @staticmethod
    def _supported(layer, x, residual) -> bool:
        import torch

        capability = torch.cuda.get_device_capability() if x.is_cuda else None
        weight = layer.weight.data
        return (
            residual is None
            and capability == (7, 0)
            and x.dtype == weight.dtype == torch.float16
            and x.is_contiguous()
            and weight.is_contiguous()
            and tuple(x.shape) == (1, 4096)
            and tuple(weight.shape) == (4096,)
            and layer.variance_epsilon == 1e-6
            and layer.variance_size_override is None
        )

    def _ensure_program(self) -> bool:
        import torch

        if self._disabled:
            return False
        if self._program is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        with self._lock:
            if self._program is not None:
                return True
            try:
                self._program, self._inputs, self._output = _build_rms_norm_program()
            except Exception:  # noqa: BLE001 — runtime compatibility gate falls back by contract
                self._disabled = True
                logger.exception("1Cat RMSNorm: Emmy build failed; retaining the original kernel")
                return False
        return True

    def __call__(self, layer, x, residual=None):
        if not self._supported(layer, x, residual) or not self._ensure_program():
            return self.original(layer, x, residual)

        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        output = torch.empty_like(x)
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            self._program.run_once_external(
                {
                    self._inputs[0]: cp.from_dlpack(x),
                    self._inputs[1]: cp.from_dlpack(layer.weight.data),
                    self._output: cp.from_dlpack(output),
                }
            )

        if not self._verified:
            reference = self.original(layer, x, residual)
            if not torch.equal(output, reference):
                self._disabled = True
                logger.error("1Cat RMSNorm: first-use bitwise parity failed; retaining the original kernel")
                return reference
            self._verified = True
            logger.info("1Cat RMSNorm: Emmy compiler kernel active after first-use bitwise parity")
        return output


def register_onecat_kernels() -> None:
    """Install the guarded leaf replacement when the compatible 1Cat symbol exists."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
    except ImportError:
        logger.warning("1Cat RMSNorm requested, but the compatible vLLM layer is unavailable")
        return

    original = RMSNorm.forward_native
    if getattr(original, "_emmy_onecat_rms_norm", False):
        return
    adapter = _RmsNormAdapter(original)

    def forward_emmy(layer, x, residual=None):
        return adapter(layer, x, residual)

    forward_emmy._emmy_onecat_rms_norm = True  # type: ignore[attr-defined]
    RMSNorm.forward_native = forward_emmy
    logger.info("1Cat RMSNorm: installed guarded Emmy compiler adapter")
