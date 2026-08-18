"""Guarded serving of loader-born 1Cat physical projections."""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from emmy.compiler.loader import onecat_sm70 as physical_loader

logger = logging.getLogger(__name__)

_PARITY_TOL = 2e-2
_SIGNATURE = (
    ("self", inspect.Parameter.empty),
    ("layer", inspect.Parameter.empty),
    ("x", inspect.Parameter.empty),
    ("bias", None),
)


@dataclass
class _ProgramEntry:
    runtime: Any
    inputs: tuple[str, ...]
    output: str
    profile: physical_loader.ProjectionProfile
    verified: bool = False


def _build_program(profile: physical_loader.ProjectionProfile) -> _ProgramEntry:
    from emmy.serving.external import load_external_program

    symbolic_values = {"num_tokens": physical_loader.PROFILE_CAPACITY} if profile.symbolic else None
    runtime, plan = load_external_program(
        physical_loader.projection_graph(profile),
        pins=physical_loader.PROFILE_PINS,
        symbolic_values=symbolic_values,
    )
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != physical_loader.expected_inputs() or len(outputs) != 1 or len(plan.launches) != 1:
        raise RuntimeError(
            "1Cat physical projection expected the loader ABI, one output, and one launch; "
            f"got {inputs!r} -> {outputs!r} in {len(plan.launches)} launch(es)"
        )
    scratch = [buffer.name for buffer in plan.buffers if buffer.role == "scratch"]
    if scratch:
        raise RuntimeError(f"1Cat physical projection plan materialized scratch storage: {scratch!r}")
    return _ProgramEntry(runtime, inputs, outputs[0], profile)


def _run_external(entry: _ProgramEntry, binding: physical_loader.ProjectionBinding, output: Any) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream(binding.x.device)):
        if entry.profile.symbolic:
            entry.runtime.set_sym_values({"num_tokens": int(binding.x.shape[0])})
        tensors = ((entry.inputs[0], binding.x), *binding.carriers, (entry.output, output))
        entry.runtime.run_once_external({name: cp.from_dlpack(tensor) for name, tensor in tensors})


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _is_sm70(*tensors: Any) -> bool:
    import torch

    if not tensors or not all(tensor.is_cuda for tensor in tensors):
        return False
    device = tensors[0].device
    return bool(all(tensor.device == device for tensor in tensors) and torch.cuda.get_device_capability(device) == (7, 0))


class _Adapter:
    def __init__(
        self,
        original: Callable,
        *,
        build_program: Callable[[physical_loader.ProjectionProfile], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, physical_loader.ProjectionBinding, Any], None] = _run_external,
        platform_supported: Callable[..., bool] = _is_sm70,
        is_capturing: Callable[[], bool] = _is_capturing,
    ) -> None:
        self.original = original
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._programs: dict[physical_loader.ProjectionProfile, _ProgramEntry] = {}
        self._disabled: set[physical_loader.ProjectionProfile] = set()
        self._lock = threading.RLock()

    def _binding(self, layer: Any, x: Any, bias: Any) -> physical_loader.ProjectionBinding | None:
        return physical_loader.bind_projection(layer, x, bias, self._platform_supported)

    def _disable(self, profile: physical_loader.ProjectionProfile, message: str, *, exc_info: bool = False) -> None:
        self._disabled.add(profile)
        self._programs.pop(profile, None)
        logger.error(
            "1Cat physical projection %s M=%s: %s; retaining the original operation",
            profile.spec.name,
            "symbolic" if profile.symbolic else profile.rows,
            message,
            exc_info=exc_info,
        )

    def _ensure(self, profile: physical_loader.ProjectionProfile) -> _ProgramEntry | None:
        entry = self._programs.get(profile)
        if entry is not None:
            return entry
        try:
            entry = self._build_program(profile)
        except Exception:  # noqa: BLE001 -- a missing strict pack permanently falls back for this profile
            self._disable(profile, "strict Emmy pack load failed", exc_info=True)
            return None
        if entry.profile != profile or entry.inputs != physical_loader.expected_inputs():
            self._disable(profile, "compiler boundary did not match the requested profile")
            return None
        self._programs[profile] = entry
        return entry

    def _run(self, entry: _ProgramEntry, binding: physical_loader.ProjectionBinding) -> Any:
        output = binding.x.new_empty((binding.x.shape[0], binding.profile.spec.n))
        self._run_program(entry, binding, output)
        return output.reshape(binding.output_shape)

    @staticmethod
    def _matches(actual: Any, reference: Any) -> bool:
        import torch

        return bool(
            isinstance(reference, torch.Tensor)
            and actual.shape == reference.shape
            and actual.dtype == reference.dtype
            and torch.allclose(actual, reference, rtol=_PARITY_TOL, atol=_PARITY_TOL)
        )

    def dispatch(self, method: Any, layer: Any, x: Any, bias: Any = None) -> Any:
        binding = self._binding(layer, x, bias)
        if binding is None or binding.profile in self._disabled:
            return self.original(method, layer, x, bias)
        entry = self._programs.get(binding.profile)
        capturing = self._is_capturing()
        if capturing and (entry is None or not entry.verified):
            return self.original(method, layer, x, bias)

        with self._lock:
            entry = self._programs.get(binding.profile)
            if entry is None:
                if capturing:
                    return self.original(method, layer, x, bias)
                entry = self._ensure(binding.profile)
                if entry is None:
                    return self.original(method, layer, x, bias)
            try:
                output = self._run(entry, binding)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self._disable(binding.profile, "Emmy launch failed", exc_info=True)
                return self.original(method, layer, x, bias)
            if not entry.verified:
                reference = self.original(method, layer, x, bias)
                if not self._matches(output, reference):
                    self._disable(binding.profile, "first-use parity failed")
                    return reference
                entry.verified = True
            return output


def _signature_matches(function: Callable) -> bool:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) == len(_SIGNATURE) and all(
        parameter.name == name and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default == default
        for parameter, (name, default) in zip(parameters, _SIGNATURE, strict=True)
    )


def register_onecat_fp8_linear_kernels(module: ModuleType | None = None) -> bool:
    """Install the loader-provided pinned method seam atomically."""
    try:
        method_class = physical_loader.linear_method_class(module)
    except ImportError:
        logger.warning("1Cat physical projections requested, but the compatible runtime module is unavailable")
        return False
    original = getattr(method_class, "apply", None)
    if getattr(original, "_emmy_onecat_fp8_linear", False):
        return True
    if method_class is None or not callable(original) or not _signature_matches(original):
        logger.error("1Cat physical projection apply signature changed; no method installed")
        return False

    adapter = _Adapter(original)

    def apply(method, layer, x, bias=None):
        return adapter.dispatch(method, layer, x, bias)

    apply._emmy_onecat_fp8_linear = True  # type: ignore[attr-defined]
    apply._emmy_onecat_fp8_linear_adapter = adapter  # type: ignore[attr-defined]
    apply._emmy_onecat_fp8_linear_original = original  # type: ignore[attr-defined]
    try:
        method_class.apply = apply
    except Exception:  # noqa: BLE001 -- leave the original method intact on installation failure
        method_class.apply = original
        logger.exception("1Cat physical projection installation failed; restored the original method")
        return False
    logger.info("1Cat physical projections: installed guarded Emmy coverage through M=4096")
    return True


__all__ = ["register_onecat_fp8_linear_kernels"]
