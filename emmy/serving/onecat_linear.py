"""Guarded Emmy replacement for 1Cat's unquantized SM70 linear leaves.

Pinned 1Cat routes five DeepSeek V4 projections through one helper: two outer
compressors, two unquantized indexer projections, and the replicated router.
The installer patches only that helper and its direct consumer alias; every
other dense or quantized linear remains owned by 1Cat.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

PROFILE_ROWS = (1, 2, 4, 8, 16, 128, 1024, 4096)
_PROFILE_ROWS = frozenset(PROFILE_ROWS)
_MAX_PREFILL_ROWS = 4096
_K = 4096
_ROUTER_WIDTH = 256
_FP32_WIDTHS = frozenset((256, 512, 1024, 2048))
_FP16_WIDTHS = frozenset((64,))
_PARITY_TOL = 1e-2
_SIGNATURE = ("x", "weight", "output_dtype")


@dataclass(frozen=True)
class _LinearProfile:
    width: int
    output_fp32: bool
    rows: int = 1
    symbolic: bool = False

    @property
    def fp32_accumulation(self) -> bool:
        # GateLinear's M>1 stock path is FP16 F.linear followed by an FP32 cast.
        # Its M=1 helper and every compressor use FP32 accumulation instead.
        return self.output_fp32 and (self.width != _ROUTER_WIDTH or (self.rows == 1 and not self.symbolic))


@dataclass
class _ProgramEntry:
    runtime: Any
    inputs: tuple[str, ...]
    output: str
    profile: _LinearProfile
    verified: bool = False


def _linear_graph(profile: _LinearProfile):
    import torch

    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    output_fp32 = profile.output_fp32
    fp32_accumulation = profile.fp32_accumulation

    class Linear(torch.nn.Module):
        def forward(self, x, weight):
            if fp32_accumulation:
                return torch.mm(x, weight.transpose(0, 1), out_dtype=torch.float32)
            output = torch.nn.functional.linear(x, weight)
            return output.to(torch.float32) if output_fp32 else output

    examples = (
        torch.empty((profile.rows, _K), dtype=torch.float16, device="meta"),
        torch.empty((profile.width, _K), dtype=torch.float16, device="meta"),
    )
    dynamic_shapes = None
    if profile.symbolic:
        dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(["num_tokens@x:0"]))
    return trace_module(Linear(), examples, dynamic_shapes=dynamic_shapes)


def _build_program(profile: _LinearProfile) -> _ProgramEntry:
    from emmy.serving.external import build_external_program

    symbolic_values = {"num_tokens": _MAX_PREFILL_ROWS} if profile.symbolic else None
    runtime, plan = build_external_program(_linear_graph(profile), symbolic_values=symbolic_values)
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != ("x", "weight") or len(outputs) != 1:
        raise RuntimeError(f"1Cat linear expected x/weight inputs and one output, got {inputs!r} -> {outputs!r}")
    return _ProgramEntry(runtime, inputs, outputs[0], profile)


def _run_external(entry: _ProgramEntry, x: Any, weight: Any, output: Any) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(x.device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        if entry.profile.symbolic:
            entry.runtime.set_sym_values({"num_tokens": int(x.shape[0])})
        bindings = {
            entry.inputs[0]: cp.from_dlpack(x),
            entry.inputs[1]: cp.from_dlpack(weight),
            entry.output: cp.from_dlpack(output),
        }
        entry.runtime.run_once_external(bindings)


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _is_sm70(x: Any, weight: Any) -> bool:
    import torch

    return bool(x.is_cuda and weight.is_cuda and x.device == weight.device and torch.cuda.get_device_capability(x.device) == (7, 0))


class _LinearAdapter:
    def __init__(
        self,
        original: Callable,
        *,
        build_program: Callable[[_LinearProfile], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, Any, Any, Any], None] = _run_external,
        platform_supported: Callable[[Any, Any], bool] = _is_sm70,
        is_capturing: Callable[[], bool] = _is_capturing,
    ) -> None:
        self.original = original
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._programs: dict[_LinearProfile, _ProgramEntry] = {}
        self._disabled: set[_LinearProfile] = set()
        self._lock = threading.RLock()

    def _profile(self, x: Any, weight: Any, output_dtype: Any) -> _LinearProfile | None:
        import torch

        try:
            rows = int(x.shape[0]) if x.ndim == 2 else -1
            width = int(weight.shape[0]) if weight.ndim == 2 else -1
            output_fp32 = output_dtype == torch.float32
            eligible_width = width in (_FP32_WIDTHS if output_fp32 else _FP16_WIDTHS)
            supported_rows = 0 < rows <= _MAX_PREFILL_ROWS
            supported = (
                self._platform_supported(x, weight)
                and x.dtype == weight.dtype == torch.float16
                and output_dtype in (torch.float16, torch.float32)
                and tuple(x.shape) == (rows, _K)
                and tuple(weight.shape) == (width, _K)
                and x.device == weight.device
                and x.stride() == (_K, 1)
                and weight.stride() == (_K, 1)
                and eligible_width
                and supported_rows
            )
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None
        if not supported:
            return None
        return _LinearProfile(
            width,
            output_fp32,
            rows if rows in _PROFILE_ROWS else _MAX_PREFILL_ROWS,
            symbolic=rows not in _PROFILE_ROWS,
        )

    def _disable(self, profile: _LinearProfile, message: str, *, exc_info: bool = False) -> None:
        self._disabled.add(profile)
        self._programs.pop(profile, None)
        logger.error(
            "1Cat linear N=%d M=%s: %s; retaining the original operation",
            profile.width,
            "symbolic" if profile.symbolic else profile.rows,
            message,
            exc_info=exc_info,
        )

    def _ensure(self, profile: _LinearProfile) -> _ProgramEntry | None:
        entry = self._programs.get(profile)
        if entry is not None:
            return entry
        try:
            entry = self._build_program(profile)
        except Exception:  # noqa: BLE001 -- compatibility gate permanently falls back for this profile
            self._disable(profile, "Emmy build failed", exc_info=True)
            return None
        if entry.profile != profile or entry.inputs != ("x", "weight"):
            self._disable(profile, "compiler boundary did not match the requested profile")
            return None
        self._programs[profile] = entry
        return entry

    @staticmethod
    def _matches(actual: Any, reference: Any) -> bool:
        import torch

        return bool(
            isinstance(reference, torch.Tensor)
            and actual.shape == reference.shape
            and actual.dtype == reference.dtype
            and torch.allclose(actual, reference, rtol=_PARITY_TOL, atol=_PARITY_TOL)
        )

    @staticmethod
    def _fallback_output(profile: _LinearProfile, x: Any, weight: Any):
        import torch

        if profile.fp32_accumulation:
            return torch.mm(x, weight.transpose(0, 1), out_dtype=torch.float32)
        output = torch.nn.functional.linear(x, weight)
        return output.to(torch.float32) if profile.output_fp32 else output

    def _reference(self, profile: _LinearProfile, x: Any, weight: Any, output_dtype: Any):
        import torch

        reference = self.original(x, weight, output_dtype)
        return reference if isinstance(reference, torch.Tensor) else self._fallback_output(profile, x, weight)

    def _verified(self, profile: _LinearProfile) -> bool:
        entry = self._programs.get(profile)
        return bool(entry is not None and entry.verified)

    def _run(self, entry: _ProgramEntry, x: Any, weight: Any, output_dtype: Any):
        output = x.new_empty((x.shape[0], entry.profile.width), dtype=output_dtype)
        self._run_program(entry, x, weight, output)
        return output

    def _warm_symbolic(self, profile: _LinearProfile, x: Any, weight: Any, output_dtype: Any, reference: Any) -> None:
        if profile.symbolic or profile.rows <= 16:
            return
        symbolic_profile = _LinearProfile(profile.width, profile.output_fp32, _MAX_PREFILL_ROWS, symbolic=True)
        if symbolic_profile in self._disabled or self._verified(symbolic_profile):
            return
        symbolic = self._ensure(symbolic_profile)
        if symbolic is None:
            return
        try:
            output = self._run(symbolic, x, weight, output_dtype)
        except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back for this profile
            self._disable(symbolic_profile, "symbolic prefill launch failed", exc_info=True)
            return
        if not self._matches(output, reference):
            self._disable(symbolic_profile, "symbolic prefill first-use parity failed")
            return
        symbolic.verified = True

    def dispatch(self, x: Any, weight: Any, output_dtype: Any):
        profile = self._profile(x, weight, output_dtype)
        if profile is None or profile in self._disabled:
            return self.original(x, weight, output_dtype)

        entry = self._programs.get(profile)
        capturing = self._is_capturing()
        if capturing and (entry is None or not entry.verified):
            return self.original(x, weight, output_dtype)

        with self._lock:
            if profile in self._disabled:
                return self.original(x, weight, output_dtype)
            entry = self._programs.get(profile)
            if entry is None:
                if capturing:
                    return self.original(x, weight, output_dtype)
                entry = self._ensure(profile)
                if entry is None:
                    return self._reference(profile, x, weight, output_dtype)

            try:
                output = self._run(entry, x, weight, output_dtype)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back for this profile
                self._disable(profile, "Emmy launch failed", exc_info=True)
                return self._reference(profile, x, weight, output_dtype)

            if not entry.verified:
                if capturing:
                    return self.original(x, weight, output_dtype)
                reference = self._reference(profile, x, weight, output_dtype)
                if not self._matches(output, reference):
                    self._disable(profile, "first-use parity failed")
                    return reference
                entry.verified = True
                self._warm_symbolic(profile, x, weight, output_dtype, reference)
            return output


_CUSTOM_OP: Any | None = None
_ACTIVE_ADAPTER: _LinearAdapter | None = None


def _adapter() -> _LinearAdapter:
    if _ACTIVE_ADAPTER is None:
        raise RuntimeError("1Cat linear custom op called before its adapter was installed")
    return _ACTIVE_ADAPTER


def _custom_op():
    global _CUSTOM_OP
    if _CUSTOM_OP is not None:
        return _CUSTOM_OP
    import torch

    @torch.library.custom_op(
        "emmy::onecat_dsv4_linear",
        mutates_args=(),
        schema="(Tensor x, Tensor weight, bool output_fp32) -> Tensor",
    )
    def op(x, weight, output_fp32):
        output_dtype = torch.float32 if output_fp32 else torch.float16
        adapter = _adapter()
        output = adapter.dispatch(x, weight, output_dtype)
        if output is None:
            profile = adapter._profile(x, weight, output_dtype)
            if profile is None:
                raise RuntimeError("1Cat linear custom op received an unsupported profile")
            output = adapter._fallback_output(profile, x, weight)
        return output

    @op.register_fake
    def fake(x, weight, output_fp32):
        output_dtype = torch.float32 if output_fp32 else torch.float16
        return x.new_empty((x.shape[0], weight.shape[0]), dtype=output_dtype)

    _CUSTOM_OP = op
    return op


def _signature_matches(function: Callable) -> bool:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) == len(_SIGNATURE) and all(
        parameter.name == name
        and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is inspect.Parameter.empty
        for parameter, name in zip(parameters, _SIGNATURE, strict=True)
    )


def _wrapper(adapter: _LinearAdapter, op: Callable) -> Callable:
    original = adapter.original

    def maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype):
        profile = adapter._profile(x, weight, output_dtype)
        if profile is None or profile in adapter._disabled:
            return original(x, weight, output_dtype)
        if adapter._is_capturing() and not adapter._verified(profile):
            return original(x, weight, output_dtype)
        import torch

        return op(x, weight, output_dtype == torch.float32)

    maybe_sm70_dsv4_fp16_gemv._emmy_onecat_linear = True  # type: ignore[attr-defined]
    maybe_sm70_dsv4_fp16_gemv._emmy_onecat_linear_adapter = adapter  # type: ignore[attr-defined]
    maybe_sm70_dsv4_fp16_gemv._emmy_onecat_linear_original = original  # type: ignore[attr-defined]
    return maybe_sm70_dsv4_fp16_gemv


def register_onecat_linear_kernels(
    gemv_module: ModuleType | None = None,
    attention_module: ModuleType | None = None,
) -> bool:
    """Atomically install the pinned helper and direct consumer aliases."""
    global _ACTIVE_ADAPTER

    try:
        gemv_module = gemv_module or importlib.import_module("vllm.models.deepseek_v4.sm70.gemv")
        attention_module = attention_module or importlib.import_module("vllm.models.deepseek_v4.attention")
    except ImportError:
        logger.warning("1Cat linear requested, but the compatible DeepSeek V4 modules are unavailable")
        return False

    source = getattr(gemv_module, "maybe_sm70_dsv4_fp16_gemv", None)
    consumer = getattr(attention_module, "maybe_sm70_dsv4_fp16_gemv", None)
    eligible = getattr(gemv_module, "can_use_sm70_dsv4_fp16_gemv", None)
    installed = [bool(getattr(function, "_emmy_onecat_linear", False)) for function in (source, consumer)]
    if all(installed) and source is consumer:
        _ACTIVE_ADAPTER = source._emmy_onecat_linear_adapter  # type: ignore[attr-defined]
        return True
    if any(installed):
        logger.error("1Cat linear: partial prior installation detected; no aliases changed")
        return False
    if not callable(source) or not callable(consumer) or not callable(eligible):
        logger.error("1Cat linear: compatible helper symbols are unavailable; no aliases installed")
        return False
    if source is not consumer or not _signature_matches(source) or not _signature_matches(eligible):
        logger.error("1Cat linear: helper identity or signature changed; no aliases installed")
        return False

    adapter = _LinearAdapter(source)
    replacement = _wrapper(adapter, _custom_op())
    previous_active = _ACTIVE_ADAPTER
    try:
        attention_module.maybe_sm70_dsv4_fp16_gemv = replacement
        gemv_module.maybe_sm70_dsv4_fp16_gemv = replacement
        _ACTIVE_ADAPTER = adapter
    except Exception:  # noqa: BLE001 -- restore the all-or-none alias invariant
        attention_module.maybe_sm70_dsv4_fp16_gemv = consumer
        gemv_module.maybe_sm70_dsv4_fp16_gemv = source
        _ACTIVE_ADAPTER = previous_active
        logger.exception("1Cat linear: alias installation failed; restored both originals")
        return False
    logger.info("1Cat linear: installed guarded Emmy adapters for five unquantized projections through M=4096")
    return True
