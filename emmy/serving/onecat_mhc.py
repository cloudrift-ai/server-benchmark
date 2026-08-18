"""Guarded Emmy replacement for 1Cat's complete DeepSeek V4 mHC family.

The serving plugin calls the installer only under the broader DeepSeek V4
opt-in, before the DeepSeek NVIDIA model is traced. Every unsupported, cold,
or unverified call retains the original 1Cat implementation.
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
_SMALL_ROWS = frozenset(PROFILE_ROWS[:5])
_PROFILE_ROWS = frozenset(PROFILE_ROWS)
_MAX_PREFILL_ROWS = 4096
_HIDDEN = 4096
_STREAMS = 4
_MIX = 24

_SYMBOLS = {
    "broadcast": "mhc_pre_broadcast_tilelang",
    "pre": "mhc_pre_tilelang",
    "fused": "mhc_fused_post_pre_tilelang",
    "post": "mhc_post_tilelang",
    "head": "hc_head_fused_kernel_tilelang",
}

_SIGNATURES = {
    "broadcast": (
        ("residual", inspect.Parameter.empty),
        ("fn", inspect.Parameter.empty),
        ("hc_scale", inspect.Parameter.empty),
        ("hc_base", inspect.Parameter.empty),
        ("rms_eps", inspect.Parameter.empty),
        ("hc_pre_eps", inspect.Parameter.empty),
        ("hc_sinkhorn_eps", inspect.Parameter.empty),
        ("hc_post_mult_value", inspect.Parameter.empty),
        ("sinkhorn_repeat", inspect.Parameter.empty),
        ("n_splits", 1),
        ("norm_weight", None),
        ("norm_eps", 1e-6),
        ("fn_broadcast", None),
    ),
    "pre": (
        ("residual", inspect.Parameter.empty),
        ("fn", inspect.Parameter.empty),
        ("hc_scale", inspect.Parameter.empty),
        ("hc_base", inspect.Parameter.empty),
        ("rms_eps", inspect.Parameter.empty),
        ("hc_pre_eps", inspect.Parameter.empty),
        ("hc_sinkhorn_eps", inspect.Parameter.empty),
        ("hc_post_mult_value", inspect.Parameter.empty),
        ("sinkhorn_repeat", inspect.Parameter.empty),
        ("n_splits", 1),
        ("norm_weight", None),
        ("norm_eps", 1e-6),
    ),
    "fused": (
        ("x", inspect.Parameter.empty),
        ("residual", inspect.Parameter.empty),
        ("post_layer_mix", inspect.Parameter.empty),
        ("comb_res_mix", inspect.Parameter.empty),
        ("fn", inspect.Parameter.empty),
        ("hc_scale", inspect.Parameter.empty),
        ("hc_base", inspect.Parameter.empty),
        ("rms_eps", inspect.Parameter.empty),
        ("hc_pre_eps", inspect.Parameter.empty),
        ("hc_sinkhorn_eps", inspect.Parameter.empty),
        ("hc_post_mult_value", inspect.Parameter.empty),
        ("sinkhorn_repeat", inspect.Parameter.empty),
        ("n_splits", 1),
        ("tile_n", 1),
        ("norm_weight", None),
        ("norm_eps", 1e-6),
    ),
    "post": (
        ("x", inspect.Parameter.empty),
        ("residual", inspect.Parameter.empty),
        ("post_layer_mix", inspect.Parameter.empty),
        ("comb_res_mix", inspect.Parameter.empty),
    ),
    "head": (
        ("hs_flat", inspect.Parameter.empty),
        ("fn", inspect.Parameter.empty),
        ("hc_scale", inspect.Parameter.empty),
        ("hc_base", inspect.Parameter.empty),
        ("rms_eps", inspect.Parameter.empty),
        ("hc_eps", inspect.Parameter.empty),
    ),
}

_PLAN_INPUTS = {
    "broadcast": ("x", "fn_broadcast", "scale", "base", "norm_weight"),
    "pre": ("residual", "fn", "scale", "base", "norm_weight"),
    "fused": ("x", "residual", "post", "comb", "fn", "scale", "base", "norm_weight"),
    "post": ("x", "residual", "post", "comb"),
    "head": ("residual", "fn", "scale", "base"),
}


@dataclass(frozen=True)
class _ProgramProfile:
    kind: str
    rows: int
    symbolic: bool = False


@dataclass
class _ProgramEntry:
    program: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    profile: _ProgramProfile
    verified: bool = False


def _trace_symbolic_prefill(kind: str):
    """Trace one capacity-4096 program whose token axis remains symbolic."""
    import torch

    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.mhc import HcHeadModule, MhcBroadcastModule, MhcFusedModule, MhcPostModule, MhcPreModule

    rows = _MAX_PREFILL_ROWS
    f16 = torch.float16
    f32 = torch.float32
    meta = "meta"
    if kind == "broadcast":
        module = MhcBroadcastModule()
        examples = (
            torch.empty((rows, _HIDDEN), dtype=f16, device=meta),
            torch.empty((_MIX, _HIDDEN), dtype=f32, device=meta),
            torch.empty((3,), dtype=f32, device=meta),
            torch.empty((_MIX,), dtype=f32, device=meta),
            torch.empty((_HIDDEN,), dtype=f16, device=meta),
        )
        names = ("x",)
    elif kind == "pre":
        module = MhcPreModule()
        examples = (
            torch.empty((rows, _STREAMS, _HIDDEN), dtype=f16, device=meta),
            torch.empty((_MIX, _STREAMS * _HIDDEN), dtype=f32, device=meta),
            torch.empty((3,), dtype=f32, device=meta),
            torch.empty((_MIX,), dtype=f32, device=meta),
            torch.empty((_HIDDEN,), dtype=f16, device=meta),
        )
        names = ("residual",)
    elif kind == "fused":
        module = MhcFusedModule(fp32_stage=False)
        examples = (
            torch.empty((rows, _HIDDEN), dtype=f16, device=meta),
            torch.empty((rows, _STREAMS, _HIDDEN), dtype=f16, device=meta),
            torch.empty((rows, _STREAMS, 1), dtype=f32, device=meta),
            torch.empty((rows, _STREAMS, _STREAMS), dtype=f32, device=meta),
            torch.empty((_MIX, _STREAMS * _HIDDEN), dtype=f32, device=meta),
            torch.empty((3,), dtype=f32, device=meta),
            torch.empty((_MIX,), dtype=f32, device=meta),
            torch.empty((_HIDDEN,), dtype=f16, device=meta),
        )
        names = ("x", "residual", "post", "comb")
    elif kind == "post":
        module = MhcPostModule()
        examples = (
            torch.empty((rows, _HIDDEN), dtype=f16, device=meta),
            torch.empty((rows, _STREAMS, _HIDDEN), dtype=f16, device=meta),
            torch.empty((rows, _STREAMS, 1), dtype=f32, device=meta),
            torch.empty((rows, _STREAMS, _STREAMS), dtype=f32, device=meta),
        )
        names = ("x", "residual", "post", "comb")
    elif kind == "head":
        module = HcHeadModule()
        examples = (
            torch.empty((rows, _STREAMS, _HIDDEN), dtype=f16, device=meta),
            torch.empty((_STREAMS, _STREAMS * _HIDDEN), dtype=f32, device=meta),
            torch.empty((1,), dtype=f32, device=meta),
            torch.empty((_STREAMS,), dtype=f32, device=meta),
        )
        names = ("residual",)
    else:
        raise KeyError(kind)

    dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{name}:0" for name in names]))
    return trace_module(module, examples, dynamic_shapes=dynamic_shapes)


def _build_program(profile: _ProgramProfile) -> _ProgramEntry:
    from emmy.serving.external import build_external_program
    from emmy.serving.mhc import trace_hc_head, trace_mhc_broadcast, trace_mhc_fused, trace_mhc_post, trace_mhc_pre

    if profile.symbolic:
        graph = _trace_symbolic_prefill(profile.kind)
    else:
        builders = {
            "broadcast": trace_mhc_broadcast,
            "pre": trace_mhc_pre,
            "fused": trace_mhc_fused,
            "post": trace_mhc_post,
            "head": trace_hc_head,
        }
        graph = builders[profile.kind](rows=profile.rows)
    symbolic_values = {"num_tokens": _MAX_PREFILL_ROWS} if profile.symbolic else None
    program, plan = build_external_program(graph, symbolic_values=symbolic_values)
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != _PLAN_INPUTS[profile.kind]:
        raise RuntimeError(f"1Cat mHC {profile.kind} input ABI changed: {inputs}")
    expected_outputs = 4 if profile.kind in ("broadcast", "fused") else 3 if profile.kind == "pre" else 1
    if len(outputs) != expected_outputs:
        raise RuntimeError(f"1Cat mHC {profile.kind} expected {expected_outputs} outputs, got {len(outputs)}")
    return _ProgramEntry(program, inputs, outputs, profile)


def _sm70_platform(tensors: tuple[Any, ...]) -> bool:
    import torch

    return bool(tensors) and all(tensor.is_cuda for tensor in tensors) and torch.cuda.get_device_capability(tensors[0].device) == (7, 0)


def _run_external(entry: _ProgramEntry, tensors: tuple[Any, ...], outputs: tuple[Any, ...]) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
        bindings = {
            **{name: cp.from_dlpack(tensor) for name, tensor in zip(entry.inputs, tensors, strict=True)},
            **{name: cp.from_dlpack(output) for name, output in zip(entry.outputs, outputs, strict=True)},
        }
        if entry.profile.symbolic:
            entry.program.set_sym_values({"num_tokens": int(outputs[0].shape[0])})
        entry.program.run_once_external(bindings)


class _MhcFamilyAdapter:
    def __init__(
        self,
        originals: dict[str, Callable],
        *,
        build_program: Callable[[_ProgramProfile], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, tuple[Any, ...], tuple[Any, ...]], None] = _run_external,
        platform_supported: Callable[[tuple[Any, ...]], bool] = _sm70_platform,
        is_capturing: Callable[[], bool] | None = None,
    ) -> None:
        self.originals = originals
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._programs: dict[_ProgramProfile, _ProgramEntry] = {}
        self._disabled = False
        self._lock = threading.RLock()

    def _capturing(self) -> bool:
        if self._is_capturing is not None:
            return self._is_capturing()
        import torch

        return torch.cuda.is_current_stream_capturing()

    @staticmethod
    def _shape(tensor, shape, dtype) -> bool:
        import torch

        return isinstance(tensor, torch.Tensor) and tensor.dtype == dtype and tensor.is_contiguous() and tuple(tensor.shape) == shape

    def _supported(self, kind: str, args: tuple[Any, ...]) -> tuple[bool, int]:
        import torch

        f16, f32 = torch.float16, torch.float32
        if kind == "broadcast":
            residual, fn, scale, base, rms, pre_eps, sink_eps, alpha, repeat, splits, norm, norm_eps, fn_broadcast = args
            rows = residual.shape[0] if isinstance(residual, torch.Tensor) and residual.ndim == 2 else -1
            tensors = (residual, fn, scale, base, norm, fn_broadcast)
            valid = (
                self._shape(residual, (rows, _HIDDEN), f16)
                and self._shape(fn, (_MIX, _STREAMS * _HIDDEN), f32)
                and self._shape(fn_broadcast, (_MIX, _HIDDEN), f32)
                and self._shape(scale, (3,), f32)
                and self._shape(base, (_MIX,), f32)
                and self._shape(norm, (_HIDDEN,), f16)
                and (rms, pre_eps, sink_eps, alpha, int(repeat), int(splits), norm_eps) == (1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1e-6)
            )
        elif kind == "pre":
            residual, fn, scale, base, rms, pre_eps, sink_eps, alpha, repeat, splits, norm, norm_eps = args
            rows = residual.shape[0] if isinstance(residual, torch.Tensor) and residual.ndim == 3 else -1
            tensors = (residual, fn, scale, base, norm)
            valid = (
                self._shape(residual, (rows, _STREAMS, _HIDDEN), f16)
                and self._shape(fn, (_MIX, _STREAMS * _HIDDEN), f32)
                and self._shape(scale, (3,), f32)
                and self._shape(base, (_MIX,), f32)
                and self._shape(norm, (_HIDDEN,), f16)
                and (rms, pre_eps, sink_eps, alpha, int(repeat), int(splits), norm_eps) == (1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1e-6)
            )
        elif kind == "fused":
            x, residual, post, comb, fn, scale, base, rms, pre_eps, sink_eps, alpha, repeat, splits, tile_n, norm, norm_eps = args
            rows = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim == 2 else -1
            tensors = (x, residual, post, comb, fn, scale, base, norm)
            valid = (
                self._shape(x, (rows, _HIDDEN), f16)
                and self._shape(residual, (rows, _STREAMS, _HIDDEN), f16)
                and self._shape(post, (rows, _STREAMS, 1), f32)
                and self._shape(comb, (rows, _STREAMS, _STREAMS), f32)
                and self._shape(fn, (_MIX, _STREAMS * _HIDDEN), f32)
                and self._shape(scale, (3,), f32)
                and self._shape(base, (_MIX,), f32)
                and self._shape(norm, (_HIDDEN,), f16)
                and (rms, pre_eps, sink_eps, alpha, int(repeat), int(splits), int(tile_n), norm_eps)
                == (1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, 1e-6)
            )
        elif kind == "post":
            x, residual, post, comb = args
            rows = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim == 2 else -1
            tensors = args
            valid = (
                self._shape(x, (rows, _HIDDEN), f16)
                and self._shape(residual, (rows, _STREAMS, _HIDDEN), f16)
                and self._shape(post, (rows, _STREAMS, 1), f32)
                and self._shape(comb, (rows, _STREAMS, _STREAMS), f32)
            )
        elif kind == "head":
            residual, fn, scale, base, rms, hc_eps = args
            rows = residual.shape[0] if isinstance(residual, torch.Tensor) and residual.ndim == 3 else -1
            tensors = (residual, fn, scale, base)
            valid = (
                self._shape(residual, (rows, _STREAMS, _HIDDEN), f16)
                and self._shape(fn, (_STREAMS, _STREAMS * _HIDDEN), f32)
                and self._shape(scale, (1,), f32)
                and self._shape(base, (_STREAMS,), f32)
                and (rms, hc_eps) == (1e-6, 1e-6)
            )
        else:
            raise KeyError(kind)
        same_device = bool(tensors) and all(tensor.device == tensors[0].device for tensor in tensors if isinstance(tensor, torch.Tensor))
        supported_rows = rows in _SMALL_ROWS or 16 < rows <= _MAX_PREFILL_ROWS
        return bool(valid and same_device and supported_rows and self._platform_supported(tensors)), int(rows)

    @staticmethod
    def _profile(kind: str, rows: int) -> _ProgramProfile:
        return _ProgramProfile(kind, rows if rows in _PROFILE_ROWS else _MAX_PREFILL_ROWS, symbolic=rows not in _PROFILE_ROWS)

    @staticmethod
    def _program_inputs(kind: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if kind == "broadcast":
            return (args[0], args[12], args[2], args[3], args[10])
        if kind == "pre":
            return (args[0], args[1], args[2], args[3], args[10])
        if kind == "fused":
            return (*args[:7], args[14])
        if kind == "post":
            return args
        return args[:4]

    @staticmethod
    def _outputs(kind: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        import torch

        source = args[0]
        rows = source.shape[0]
        if kind == "broadcast":
            return (
                source.new_empty((rows, _STREAMS, _HIDDEN)),
                torch.empty((rows, _STREAMS, 1), dtype=torch.float32, device=source.device),
                torch.empty((rows, _STREAMS, _STREAMS), dtype=torch.float32, device=source.device),
                source.new_empty((rows, _HIDDEN)),
            )
        if kind == "pre":
            return (
                torch.empty((rows, _STREAMS, 1), dtype=torch.float32, device=source.device),
                torch.empty((rows, _STREAMS, _STREAMS), dtype=torch.float32, device=source.device),
                source.new_empty((rows, _HIDDEN)),
            )
        if kind == "fused":
            return (
                args[1].new_empty((rows, _STREAMS, _HIDDEN)),
                torch.empty((rows, _STREAMS, 1), dtype=torch.float32, device=source.device),
                torch.empty((rows, _STREAMS, _STREAMS), dtype=torch.float32, device=source.device),
                source.new_empty((rows, _HIDDEN)),
            )
        if kind == "post":
            return (args[1].new_empty((rows, _STREAMS, _HIDDEN)),)
        return (source.new_empty((rows, _HIDDEN)),)

    @staticmethod
    def _tuple(value: Any) -> tuple[Any, ...]:
        return value if isinstance(value, tuple) else (value,)

    @staticmethod
    def _close(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> bool:
        import torch

        return len(actual) == len(expected) and all(
            a.shape == e.shape and a.dtype == e.dtype and torch.allclose(a, e, rtol=1e-2, atol=1e-2)
            for a, e in zip(actual, expected, strict=True)
        )

    def _disable(self, message: str, *, exc_info: bool = False) -> None:
        self._disabled = True
        logger.error("1Cat mHC: %s; retaining the complete original family", message, exc_info=exc_info)

    def _ensure(self, profile: _ProgramProfile) -> _ProgramEntry | None:
        entry = self._programs.get(profile)
        if entry is not None:
            return entry
        try:
            entry = self._build_program(profile)
        except Exception:  # noqa: BLE001 -- compatibility gate permanently falls back
            self._disable(f"{profile.kind} profile build failed", exc_info=True)
            return None
        if entry.profile != profile:
            self._disable(f"{profile.kind} builder returned the wrong profile")
            return None
        self._programs[profile] = entry
        return entry

    def dispatch(self, kind: str, *args):
        original = self.originals[kind]
        if self._disabled:
            return original(*args)
        try:
            supported, rows = self._supported(kind, args)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return original(*args)
        if not supported:
            return original(*args)

        profile = self._profile(kind, rows)
        entry = self._programs.get(profile)
        capturing = self._capturing()
        if capturing and (entry is None or not entry.verified):
            return original(*args)

        with self._lock:
            if self._disabled:
                return original(*args)
            entry = self._programs.get(profile)
            if entry is None:
                if capturing:
                    return original(*args)
                entry = self._ensure(profile)
                if entry is None:
                    return original(*args)

            try:
                outputs = self._outputs(kind, args)
                self._run_program(entry, self._program_inputs(kind, args), outputs)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self._disable(f"{kind} profile launch failed", exc_info=True)
                return original(*args)

            if not entry.verified:
                if capturing:
                    return original(*args)
                reference = original(*args)
                if not self._close(outputs, self._tuple(reference)):
                    self._disable(f"{kind} profile first-use parity failed")
                    return reference
                entry.verified = True

                # A profiled prefill call also realizes and verifies the shared
                # capacity program before any arbitrary-width piecewise capture.
                if rows in _PROFILE_ROWS and rows > 16:
                    symbolic_profile = _ProgramProfile(kind, _MAX_PREFILL_ROWS, symbolic=True)
                    symbolic = self._ensure(symbolic_profile)
                    if symbolic is None:
                        return reference
                    try:
                        symbolic_outputs = self._outputs(kind, args)
                        self._run_program(symbolic, self._program_inputs(kind, args), symbolic_outputs)
                    except Exception:  # noqa: BLE001 -- compatibility gate permanently falls back
                        self._disable(f"{kind} symbolic prefill launch failed", exc_info=True)
                        return reference
                    if not self._close(symbolic_outputs, self._tuple(reference)):
                        self._disable(f"{kind} symbolic prefill first-use parity failed")
                        return reference
                    symbolic.verified = True
            return outputs if len(outputs) != 1 else outputs[0]


_CUSTOM_OPS: dict[str, Any] | None = None
_ACTIVE_ADAPTER: _MhcFamilyAdapter | None = None


def _adapter() -> _MhcFamilyAdapter:
    if _ACTIVE_ADAPTER is None:
        raise RuntimeError("1Cat mHC custom op called before its adapter was installed")
    return _ACTIVE_ADAPTER


def _custom_ops() -> dict[str, Any]:
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    import torch

    @torch.library.custom_op(
        "emmy::onecat_mhc_broadcast",
        mutates_args=(),
        schema=(
            "(Tensor residual, Tensor fn, Tensor hc_scale, Tensor hc_base, float rms_eps, float hc_pre_eps, "
            "float hc_sinkhorn_eps, float hc_post_mult_value, SymInt sinkhorn_repeat, SymInt n_splits=1, "
            "Tensor? norm_weight=None, float norm_eps=1e-6, Tensor? fn_broadcast=None) -> (Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def broadcast(*args):
        return _adapter().dispatch("broadcast", *args)

    @broadcast.register_fake
    def broadcast_fake(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=1e-6,
        fn_broadcast=None,
    ):
        rows, hidden = residual.shape
        return (
            residual.new_empty((rows, _STREAMS, hidden)),
            residual.new_empty((rows, _STREAMS, 1), dtype=torch.float32),
            residual.new_empty((rows, _STREAMS, _STREAMS), dtype=torch.float32),
            residual.new_empty((rows, hidden)),
        )

    @torch.library.custom_op(
        "emmy::onecat_mhc_pre",
        mutates_args=(),
        schema=(
            "(Tensor residual, Tensor fn, Tensor hc_scale, Tensor hc_base, float rms_eps, float hc_pre_eps, "
            "float hc_sinkhorn_eps, float hc_post_mult_value, SymInt sinkhorn_repeat, SymInt n_splits=1, "
            "Tensor? norm_weight=None, float norm_eps=1e-6) -> (Tensor, Tensor, Tensor)"
        ),
    )
    def pre(*args):
        return _adapter().dispatch("pre", *args)

    @pre.register_fake
    def pre_fake(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=1e-6,
    ):
        rows, _, hidden = residual.shape
        return (
            residual.new_empty((rows, _STREAMS, 1), dtype=torch.float32),
            residual.new_empty((rows, _STREAMS, _STREAMS), dtype=torch.float32),
            residual.new_empty((rows, hidden)),
        )

    @torch.library.custom_op(
        "emmy::onecat_mhc_fused",
        mutates_args=(),
        schema=(
            "(Tensor x, Tensor residual, Tensor post_layer_mix, Tensor comb_res_mix, Tensor fn, Tensor hc_scale, "
            "Tensor hc_base, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, "
            "SymInt sinkhorn_repeat, SymInt n_splits=1, SymInt tile_n=1, Tensor? norm_weight=None, "
            "float norm_eps=1e-6) -> (Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def fused(*args):
        return _adapter().dispatch("fused", *args)

    @fused.register_fake
    def fused_fake(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=1e-6,
    ):
        rows, hidden = x.shape
        return (
            residual.new_empty(residual.shape),
            x.new_empty((rows, _STREAMS, 1), dtype=torch.float32),
            x.new_empty((rows, _STREAMS, _STREAMS), dtype=torch.float32),
            x.new_empty((rows, hidden)),
        )

    @torch.library.custom_op(
        "emmy::onecat_mhc_post",
        mutates_args=(),
        schema="(Tensor x, Tensor residual, Tensor post_layer_mix, Tensor comb_res_mix) -> Tensor",
    )
    def post(*args):
        return _adapter().dispatch("post", *args)

    @post.register_fake
    def post_fake(x, residual, post_layer_mix, comb_res_mix):
        return residual.new_empty(residual.shape)

    @torch.library.custom_op(
        "emmy::onecat_hc_head",
        mutates_args=(),
        schema="(Tensor hs_flat, Tensor fn, Tensor hc_scale, Tensor hc_base, float rms_eps, float hc_eps) -> Tensor",
    )
    def head(*args):
        return _adapter().dispatch("head", *args)

    @head.register_fake
    def head_fake(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
        return hs_flat.new_empty((hs_flat.shape[0], hs_flat.shape[-1]))

    _CUSTOM_OPS = {"broadcast": broadcast, "pre": pre, "fused": fused, "post": post, "head": head}
    return _CUSTOM_OPS


def _signature_matches(function: Callable, expected: tuple[tuple[str, Any], ...]) -> bool:
    parameters = tuple(inspect.signature(function).parameters.values())
    return len(parameters) == len(expected) and all(
        parameter.name == name and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default == default
        for parameter, (name, default) in zip(parameters, expected, strict=True)
    )


def _wrappers(adapter: _MhcFamilyAdapter, ops: dict[str, Any]) -> dict[str, Callable]:
    def broadcast(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=1e-6,
        fn_broadcast=None,
    ):
        return ops["broadcast"](
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
            fn_broadcast,
        )

    def pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=1e-6,
    ):
        return ops["pre"](
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def fused(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=1e-6,
    ):
        return ops["fused"](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def post(x, residual, post_layer_mix, comb_res_mix):
        return ops["post"](x, residual, post_layer_mix, comb_res_mix)

    def head(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
        return ops["head"](hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps)

    result = {"broadcast": broadcast, "pre": pre, "fused": fused, "post": post, "head": head}
    for kind, wrapper in result.items():
        wrapper._emmy_onecat_mhc = True  # type: ignore[attr-defined]
        wrapper._emmy_onecat_mhc_adapter = adapter  # type: ignore[attr-defined]
        wrapper._emmy_onecat_mhc_original = adapter.originals[kind]  # type: ignore[attr-defined]
    return result


def register_onecat_mhc_kernels(model_module: ModuleType | None = None) -> bool:
    """Atomically install all five guarded mHC aliases before model tracing."""
    global _ACTIVE_ADAPTER

    if model_module is None:
        try:
            model_module = importlib.import_module("vllm.models.deepseek_v4.nvidia.model")
        except ImportError:
            logger.warning("1Cat mHC requested, but the compatible DeepSeek V4 NVIDIA model is unavailable")
            return False

    functions: dict[str, Callable] = {}
    for kind, symbol in _SYMBOLS.items():
        function = getattr(model_module, symbol, None)
        if not callable(function):
            logger.error("1Cat mHC: missing compatible symbol %s; no aliases installed", symbol)
            return False
        functions[kind] = function

    installed = [bool(getattr(function, "_emmy_onecat_mhc", False)) for function in functions.values()]
    if all(installed):
        _ACTIVE_ADAPTER = next(iter(functions.values()))._emmy_onecat_mhc_adapter  # type: ignore[attr-defined]
        return True
    if any(installed):
        logger.error("1Cat mHC: partial prior installation detected; no aliases changed")
        return False
    for kind, function in functions.items():
        if not _signature_matches(function, _SIGNATURES[kind]):
            logger.error("1Cat mHC: incompatible signature for %s; no aliases installed", _SYMBOLS[kind])
            return False

    adapter = _MhcFamilyAdapter(functions)
    replacements = _wrappers(adapter, _custom_ops())
    previous_active = _ACTIVE_ADAPTER
    changed: list[tuple[str, Callable]] = []
    try:
        for kind, symbol in _SYMBOLS.items():
            changed.append((symbol, functions[kind]))
            setattr(model_module, symbol, replacements[kind])
        _ACTIVE_ADAPTER = adapter
    except Exception:  # noqa: BLE001 -- preserve the all-or-none installation invariant
        for symbol, original in changed:
            setattr(model_module, symbol, original)
        _ACTIVE_ADAPTER = previous_active
        logger.exception("1Cat mHC: alias installation failed; restored every original")
        return False
    logger.info("1Cat mHC: installed guarded Emmy adapters for the complete five-operation family")
    return True
