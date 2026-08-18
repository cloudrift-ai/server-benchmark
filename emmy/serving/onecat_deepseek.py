"""Opt-in Emmy adapters for two pure DeepSeek V4 leaves in pinned 1Cat.

The serving runtime keeps ownership of projections, attention, cache mutation,
quantization, and output projection. This module only replaces the exact SM70
tensor-returning Q/KV RMSNorm and inverse-RoPE functions when the broader
DeepSeek V4 serving opt-in is enabled.
"""

from __future__ import annotations

import functools
import importlib
import logging
import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PREFILL_ROWS = 4096
_Q_SIZE = 1024
_KV_SIZE = 512
_FUSED_SIZE = _Q_SIZE + _KV_SIZE
_HEADS = 8
_HEAD_DIM = 512
_ROPE_DIM = 64
_CONTEXT = 1_048_576
_PARITY_TOL = 3e-3


def _symbolic_profile(rows: int) -> Hashable | None:
    """Return one bounded-capacity cache key for every serving width."""
    return _MAX_PREFILL_ROWS if 0 < rows <= _MAX_PREFILL_ROWS else None


@dataclass(frozen=True)
class _ExternalProgram:
    runtime: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    prepare_rows: Callable[[int], None] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)


@dataclass
class _ProgramEntry:
    program: _ExternalProgram
    verified_rows: set[int] = field(default_factory=set)


class _ProgramCache:
    """Lazy program cache whose key policy may share one capacity build."""

    def __init__(
        self,
        family: str,
        builder: Callable[[int], _ExternalProgram],
        profile: Callable[[int], Hashable | None],
    ) -> None:
        self.family = family
        self.builder = builder
        self.profile = profile
        self.entries: dict[Hashable, _ProgramEntry] = {}
        self.disabled: set[Hashable] = set()
        self.lock = threading.Lock()

    def get(self, rows: int, *, capturing: bool) -> _ProgramEntry | None:
        key = self.profile(rows)
        if key is None or key in self.disabled:
            return None
        entry = self.entries.get(key)
        if entry is not None:
            return entry
        if capturing:
            return None
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                return entry
            if key in self.disabled:
                return None
            try:
                entry = _ProgramEntry(self.builder(rows))
            except Exception:  # noqa: BLE001 -- compatibility adapter permanently falls back
                self.disabled.add(key)
                logger.exception("1Cat %s: Emmy build failed for M=%d; retaining the original kernel", self.family, rows)
                return None
            self.entries[key] = entry
            return entry

    def disable(self, rows: int) -> None:
        key = self.profile(rows)
        if key is not None:
            self.disabled.add(key)
            self.entries.pop(key, None)


def _build_qkv_program(_rows: int) -> _ExternalProgram:
    from emmy.serving.deepseek import trace_fused_q_kv_rmsnorm
    from emmy.serving.external import load_external_program

    runtime, plan = load_external_program(
        trace_fused_q_kv_rmsnorm(rows=_MAX_PREFILL_ROWS, dynamic=True),
        symbolic_values={"num_tokens": _MAX_PREFILL_ROWS},
    )
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != ("fused_q_kv", "q_weight", "kv_weight") or len(outputs) != 2:
        raise RuntimeError(f"1Cat Q/KV RMSNorm expected three inputs and two outputs, got {inputs!r} -> {outputs!r}")
    return _ExternalProgram(runtime, inputs, outputs, lambda rows: runtime.set_sym_values({"num_tokens": rows}))


def _build_inverse_rope_program(_rows: int) -> _ExternalProgram:
    from emmy.serving.deepseek import trace_inverse_rope
    from emmy.serving.external import load_external_program

    runtime, plan = load_external_program(
        trace_inverse_rope(rows=_MAX_PREFILL_ROWS, dynamic=True),
        symbolic_values={"num_tokens": _MAX_PREFILL_ROWS},
    )
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != ("x", "positions", "cos_sin_cache") or len(outputs) != 1:
        raise RuntimeError(f"1Cat inverse RoPE expected three inputs and one output, got {inputs!r} -> {outputs!r}")
    return _ExternalProgram(runtime, inputs, outputs, lambda rows: runtime.set_sym_values({"num_tokens": rows}))


def _run_external(program: _ExternalProgram, bindings: tuple[tuple[str, Any], ...], device: Any) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        arrays = {name: cp.from_dlpack(tensor) for name, tensor in bindings}
        program.runtime.run_once_external(arrays)


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _is_exact_sm70(tensor: Any) -> bool:
    import torch

    return bool(tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) == (7, 0))


def _fused_qkv_view(qr: Any, kv: Any, q_weight: Any, kv_weight: Any, eps: float) -> Any | None:
    import torch

    rows = qr.shape[0] if qr.ndim == 2 else -1
    if not (
        _is_exact_sm70(qr)
        and qr.dtype == kv.dtype == q_weight.dtype == kv_weight.dtype == torch.float16
        and tuple(qr.shape) == (rows, _Q_SIZE)
        and tuple(kv.shape) == (rows, _KV_SIZE)
        and tuple(q_weight.shape) == (_Q_SIZE,)
        and tuple(kv_weight.shape) == (_KV_SIZE,)
        and qr.device == kv.device == q_weight.device == kv_weight.device
        and qr.stride() == (_FUSED_SIZE, 1)
        and kv.stride() == (_FUSED_SIZE, 1)
        and q_weight.is_contiguous()
        and kv_weight.is_contiguous()
        and float(eps) == 1e-6
        and qr.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
        and kv.storage_offset() == qr.storage_offset() + _Q_SIZE
    ):
        return None
    try:
        fused = qr.as_strided(
            (rows, _FUSED_SIZE),
            (_FUSED_SIZE, 1),
            storage_offset=qr.storage_offset(),
        )
    except RuntimeError:
        return None
    return fused if fused.is_contiguous() else None


def _inverse_rope_supported(x: Any, positions: Any, cos_sin_cache: Any, rope_dim: int) -> bool:
    import torch

    rows = x.shape[0] if x.ndim == 3 else -1
    return bool(
        _is_exact_sm70(x)
        and x.dtype == torch.float16
        and positions.dtype == torch.int64
        and cos_sin_cache.dtype == torch.float32
        and tuple(x.shape) == (rows, _HEADS, _HEAD_DIM)
        and tuple(positions.shape) == (rows,)
        and tuple(cos_sin_cache.shape) == (_CONTEXT, _ROPE_DIM)
        and x.device == positions.device == cos_sin_cache.device
        and x.is_contiguous()
        and positions.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and int(rope_dim) == _ROPE_DIM
    )


def _outputs_close(actual: tuple[Any, ...], reference: tuple[Any, ...]) -> bool:
    import torch

    return len(actual) == len(reference) and all(
        torch.allclose(got, expected, rtol=_PARITY_TOL, atol=_PARITY_TOL) for got, expected in zip(actual, reference, strict=True)
    )


class _FusedQKvRmsNormAdapter:
    def __init__(
        self,
        original: Callable[..., tuple[Any, Any]],
        *,
        program_builder: Callable[[int], _ExternalProgram] = _build_qkv_program,
        profile: Callable[[int], Hashable | None] = _symbolic_profile,
        runner: Callable[[_ExternalProgram, tuple[tuple[str, Any], ...], Any], None] = _run_external,
    ) -> None:
        self.original = original
        self.cache = _ProgramCache("Q/KV RMSNorm", program_builder, profile)
        self.runner = runner

    def __call__(self, qr: Any, kv: Any, q_weight: Any, kv_weight: Any, eps: float) -> tuple[Any, Any]:
        fused = _fused_qkv_view(qr, kv, q_weight, kv_weight, eps)
        if fused is None:
            return self.original(qr, kv, q_weight, kv_weight, eps)

        rows = qr.shape[0]
        capturing = _is_capturing()
        entry = self.cache.get(rows, capturing=capturing)
        if entry is None or (capturing and rows not in entry.verified_rows):
            return self.original(qr, kv, q_weight, kv_weight, eps)

        q_out = qr.new_empty(qr.shape)
        kv_out = kv.new_empty(kv.shape)
        program = entry.program
        try:
            with program.lock:
                if program.prepare_rows is not None:
                    program.prepare_rows(rows)
                self.runner(
                    program,
                    (
                        (program.inputs[0], fused),
                        (program.inputs[1], q_weight),
                        (program.inputs[2], kv_weight),
                        (program.outputs[0], q_out),
                        (program.outputs[1], kv_out),
                    ),
                    qr.device,
                )
        except Exception:  # noqa: BLE001 -- compatibility adapter permanently falls back
            self.cache.disable(rows)
            logger.exception("1Cat Q/KV RMSNorm: Emmy launch failed for M=%d; retaining the original kernel", rows)
            return self.original(qr, kv, q_weight, kv_weight, eps)

        if rows not in entry.verified_rows:
            reference = self.original(qr, kv, q_weight, kv_weight, eps)
            if not _outputs_close((q_out, kv_out), reference):
                self.cache.disable(rows)
                logger.error("1Cat Q/KV RMSNorm: first-use parity failed for M=%d; retaining the original kernel", rows)
                return reference
            entry.verified_rows.add(rows)
            logger.info("1Cat Q/KV RMSNorm: Emmy compiler kernel active for M=%d", rows)
        return q_out, kv_out


class _InverseRopeAdapter:
    def __init__(
        self,
        original: Callable[..., Any],
        *,
        program_builder: Callable[[int], _ExternalProgram] = _build_inverse_rope_program,
        profile: Callable[[int], Hashable | None] = _symbolic_profile,
        runner: Callable[[_ExternalProgram, tuple[tuple[str, Any], ...], Any], None] = _run_external,
    ) -> None:
        self.original = original
        self.cache = _ProgramCache("inverse RoPE", program_builder, profile)
        self.runner = runner

    def __call__(self, x: Any, positions: Any, cos_sin_cache: Any, rope_dim: int) -> Any:
        if not _inverse_rope_supported(x, positions, cos_sin_cache, rope_dim):
            return self.original(x, positions, cos_sin_cache, rope_dim)

        rows = x.shape[0]
        capturing = _is_capturing()
        entry = self.cache.get(rows, capturing=capturing)
        if entry is None or (capturing and rows not in entry.verified_rows):
            return self.original(x, positions, cos_sin_cache, rope_dim)

        output = x.new_empty(x.shape)
        program = entry.program
        try:
            with program.lock:
                if program.prepare_rows is not None:
                    program.prepare_rows(rows)
                self.runner(
                    program,
                    (
                        (program.inputs[0], x),
                        (program.inputs[1], positions),
                        (program.inputs[2], cos_sin_cache),
                        (program.outputs[0], output),
                    ),
                    x.device,
                )
        except Exception:  # noqa: BLE001 -- compatibility adapter permanently falls back
            self.cache.disable(rows)
            logger.exception("1Cat inverse RoPE: Emmy launch failed for M=%d; retaining the original kernel", rows)
            return self.original(x, positions, cos_sin_cache, rope_dim)

        if rows not in entry.verified_rows:
            reference = self.original(x, positions, cos_sin_cache, rope_dim)
            if not _outputs_close((output,), (reference,)):
                self.cache.disable(rows)
                logger.error("1Cat inverse RoPE: first-use parity failed for M=%d; retaining the original kernel", rows)
                return reference
            entry.verified_rows.add(rows)
            logger.info("1Cat inverse RoPE: Emmy compiler kernel active for M=%d", rows)
        return output


def _qkv_wrapper(original: Callable[..., tuple[Any, Any]], adapter: _FusedQKvRmsNormAdapter) -> Callable[..., tuple[Any, Any]]:
    @functools.wraps(original)
    def fused_q_kv_rmsnorm_emmy(qr: Any, kv: Any, q_weight: Any, kv_weight: Any, eps: float) -> tuple[Any, Any]:
        return adapter(qr, kv, q_weight, kv_weight, eps)

    fused_q_kv_rmsnorm_emmy._emmy_onecat_deepseek_qkv = True  # type: ignore[attr-defined]
    fused_q_kv_rmsnorm_emmy._emmy_original = original  # type: ignore[attr-defined]
    fused_q_kv_rmsnorm_emmy._emmy_adapter = adapter  # type: ignore[attr-defined]
    return fused_q_kv_rmsnorm_emmy


def _inverse_wrapper(original: Callable[..., Any], adapter: _InverseRopeAdapter) -> Callable[..., Any]:
    @functools.wraps(original)
    def sm70_inverse_rope_emmy(x: Any, positions: Any, cos_sin_cache: Any, rope_dim: int) -> Any:
        return adapter(x, positions, cos_sin_cache, rope_dim)

    sm70_inverse_rope_emmy._emmy_onecat_deepseek_inverse_rope = True  # type: ignore[attr-defined]
    sm70_inverse_rope_emmy._emmy_original = original  # type: ignore[attr-defined]
    sm70_inverse_rope_emmy._emmy_adapter = adapter  # type: ignore[attr-defined]
    return sm70_inverse_rope_emmy


def register_onecat_deepseek_kernels(
    ops_module: Any | None = None,
    attention_module: Any | None = None,
    projection_module: Any | None = None,
    *,
    qkv_program_builder: Callable[[int], _ExternalProgram] = _build_qkv_program,
    inverse_program_builder: Callable[[int], _ExternalProgram] = _build_inverse_rope_program,
    profile: Callable[[int], Hashable | None] = _symbolic_profile,
    runner: Callable[[_ExternalProgram, tuple[tuple[str, Any], ...], Any], None] = _run_external,
) -> bool:
    """Install both exact pinned 1Cat adapters, returning true only for both.

    Modules may be supplied by tests or an embedding runtime.  Missing modules
    are imported by their pinned paths.  The operation validates every alias
    before mutating any module and is safe to repeat or to call after the
    attention module imported its function alias.
    """
    try:
        ops_module = ops_module or importlib.import_module("vllm.models.deepseek_v4.common.ops")
        attention_module = attention_module or importlib.import_module("vllm.models.deepseek_v4.attention")
        projection_module = projection_module or importlib.import_module("vllm.models.deepseek_v4.sm70.projection")
        ops_function = ops_module.fused_q_kv_rmsnorm
        attention_function = attention_module.fused_q_kv_rmsnorm
        inverse_function = projection_module.sm70_inverse_rope
    except (AttributeError, ImportError):
        logger.warning("1Cat DeepSeek dense kernels requested, but the compatible pinned modules are unavailable")
        return False

    qkv_wrapper = next(
        (fn for fn in (ops_function, attention_function) if getattr(fn, "_emmy_onecat_deepseek_qkv", False)),
        None,
    )
    if qkv_wrapper is None:
        if ops_function is not attention_function:
            logger.warning("1Cat DeepSeek Q/KV RMSNorm aliases disagree; retaining the original kernels")
            return False
        qkv_original = ops_function
        qkv_adapter = _FusedQKvRmsNormAdapter(
            qkv_original,
            program_builder=qkv_program_builder,
            profile=profile,
            runner=runner,
        )
        qkv_wrapper = _qkv_wrapper(qkv_original, qkv_adapter)
    else:
        qkv_original = qkv_wrapper._emmy_original  # type: ignore[attr-defined]
        if ops_function not in (qkv_original, qkv_wrapper) or attention_function not in (qkv_original, qkv_wrapper):
            logger.warning("1Cat DeepSeek Q/KV RMSNorm has an incompatible third-party patch")
            return False

    if getattr(inverse_function, "_emmy_onecat_deepseek_inverse_rope", False):
        inverse_wrapper = inverse_function
    else:
        inverse_adapter = _InverseRopeAdapter(
            inverse_function,
            program_builder=inverse_program_builder,
            profile=profile,
            runner=runner,
        )
        inverse_wrapper = _inverse_wrapper(inverse_function, inverse_adapter)

    ops_module.fused_q_kv_rmsnorm = qkv_wrapper
    attention_module.fused_q_kv_rmsnorm = qkv_wrapper
    projection_module.sm70_inverse_rope = inverse_wrapper
    logger.info("1Cat DeepSeek dense kernels: installed guarded Emmy Q/KV RMSNorm and SM70 inverse-RoPE adapters")
    return True
