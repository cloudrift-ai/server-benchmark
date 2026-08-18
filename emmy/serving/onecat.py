"""Narrow Emmy kernel integration for the 1Cat serving runtime.

1Cat retains model loading, TP/PP, routing, attention/cache state, and quantized
expert execution. This module replaces only a side-effect-free leaf whose live
tensor contract is already generic dense algebra. Every unsupported or unverified
call falls back to the original runtime kernel.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ROWS = 4096
_HIDDEN = 4096


class _RmsNormModule:
    """Weighted FP16 RMSNorm used at the exact DeepSeek-V4 decode boundary."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, weight):
                x_fp32 = x.float()
                variance = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
                normalized = x_fp32 * torch.rsqrt(variance + 1e-6)
                return (normalized * weight.float()).half()

        return Module()


@dataclass
class _RmsNormProgram:
    runtime: Any
    inputs: tuple[str, ...]
    output: str
    verified_rows: set[int] = field(default_factory=set)


def _rms_norm_graph():
    import torch

    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    examples = (
        torch.zeros((_MAX_ROWS, _HIDDEN), dtype=torch.float16, device="meta"),
        torch.ones((_HIDDEN,), dtype=torch.float16, device="meta"),
    )
    dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(["num_tokens@x:0"]))
    return trace_module(_RmsNormModule(), examples, dynamic_shapes=dynamic_shapes)


def _build_rms_norm_program() -> _RmsNormProgram:
    from emmy import config
    from emmy.serving.external import build_external_program, load_external_program

    # The deployment recipe records one realized winner. The compiler still
    # enumerates every legal schedule outside this bounded serving boundary.
    # The broad opt-in is pack-only so no request can compile inside a worker
    # RPC. Preserve compile-on-miss for the standalone RMSNorm experiment.
    load = load_external_program if config.onecat_deepseek_v4() else build_external_program
    program, plan = load(
        _rms_norm_graph(),
        pins={"WORK": "t256", "REDUCE": "coop"},
        symbolic_values={"num_tokens": _MAX_ROWS},
    )
    if len(plan.launches) != 1:
        raise RuntimeError(f"1Cat RMSNorm expected one Emmy compiler launch, got {len(plan.launches)}")
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != ("x", "weight") or len(outputs) != 1:
        raise RuntimeError(f"1Cat RMSNorm expected two inputs and one output, got {inputs!r} -> {outputs!r}")
    return _RmsNormProgram(program, inputs, outputs[0])


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _run_rms_norm_program(program: _RmsNormProgram, layer: Any, x: Any, output: Any, rows: int) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(x.device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        program.runtime.set_sym_values({"num_tokens": rows})
        program.runtime.run_once_external(
            {
                program.inputs[0]: cp.from_dlpack(x),
                program.inputs[1]: cp.from_dlpack(layer.weight.data),
                program.output: cp.from_dlpack(output),
            }
        )


class _RmsNormAdapter:
    def __init__(
        self,
        original,
        *,
        build_program: Callable[[], _RmsNormProgram] = _build_rms_norm_program,
        run_program: Callable[[_RmsNormProgram, Any, Any, Any, int], None] = _run_rms_norm_program,
        is_capturing: Callable[[], bool] = _is_capturing,
    ):
        self.original = original
        self._build_program = build_program
        self._run_program = run_program
        self._is_capturing = is_capturing
        self._program: _RmsNormProgram | None = None
        self._disabled = False
        self._lock = threading.RLock()

    @staticmethod
    def _supported(layer, x, residual) -> bool:
        import torch

        rows = x.shape[0] if x.ndim == 2 else -1
        capability = torch.cuda.get_device_capability() if x.is_cuda else None
        weight = layer.weight.data
        return (
            residual is None
            and capability == (7, 0)
            and x.dtype == weight.dtype == torch.float16
            and x.is_contiguous()
            and weight.is_contiguous()
            and tuple(x.shape) == (rows, _HIDDEN)
            and tuple(weight.shape) == (_HIDDEN,)
            and 0 < rows <= _MAX_ROWS
            and layer.variance_epsilon == 1e-6
            and layer.variance_size_override is None
        )

    def _ensure_program(self, *, capturing: bool) -> _RmsNormProgram | None:
        if self._disabled:
            return None
        if self._program is not None:
            return self._program
        if capturing:
            return None
        with self._lock:
            if self._program is not None:
                return self._program
            try:
                self._program = self._build_program()
            except Exception:  # noqa: BLE001 — runtime compatibility gate falls back by contract
                self._disabled = True
                logger.exception("1Cat RMSNorm: Emmy build failed; retaining the original kernel")
                return None
        return self._program

    def __call__(self, layer, x, residual=None):
        if not self._supported(layer, x, residual):
            return self.original(layer, x, residual)

        import torch

        rows = x.shape[0]
        capturing = self._is_capturing()
        program = self._ensure_program(capturing=capturing)
        if program is None or (capturing and rows not in program.verified_rows):
            return self.original(layer, x, residual)

        output = torch.empty_like(x)
        with self._lock:
            try:
                self._run_program(program, layer, x, output, rows)
            except Exception:  # noqa: BLE001 — runtime compatibility gate falls back by contract
                self._disabled = True
                self._program = None
                logger.exception("1Cat RMSNorm: Emmy launch failed for M=%d; retaining the original kernel", rows)
                return self.original(layer, x, residual)

            if rows not in program.verified_rows:
                reference = self.original(layer, x, residual)
                if not torch.equal(output, reference):
                    self._disabled = True
                    self._program = None
                    logger.error("1Cat RMSNorm: first-use bitwise parity failed for M=%d; retaining the original kernel", rows)
                    return reference
                program.verified_rows.add(rows)
                logger.info("1Cat RMSNorm: Emmy compiler kernel active for M=%d after first-use bitwise parity", rows)
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
