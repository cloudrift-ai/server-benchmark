"""Guarded Emmy adapter for the pure DeepSeek V4 C4 indexer-Q transform."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_CAPACITY = 4096
_HEADS = 64
_HEAD_DIM = 128
_ROPE_DIM = 64
_CONTEXT = 1_048_576
_SOFTMAX_SCALE = _HEAD_DIM**-0.5
_HEAD_SCALE = _HEADS**-0.5
_PARITY_TOL = 3e-3
_EXPECTED_INPUTS = ("positions", "index_q", "cos_sin_cache", "index_weights")
_SIGNATURE = (
    ("positions", inspect.Parameter.empty),
    ("index_q", inspect.Parameter.empty),
    ("index_q_cos_sin_cache", inspect.Parameter.empty),
    ("index_weights", inspect.Parameter.empty),
    ("index_weights_softmax_scale", inspect.Parameter.empty),
    ("index_weights_head_scale", inspect.Parameter.empty),
    ("use_fp4", False),
)


def _indexer_q_graph():
    import torch

    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    half_rope = _ROPE_DIM // 2
    nope_dim = _HEAD_DIM - _ROPE_DIM

    class Module(torch.nn.Module):
        def forward(self, positions, index_q, cos_sin_cache, index_weights):
            rotary = cos_sin_cache[positions]
            cos = rotary[:, :half_rope].float().unsqueeze(1)
            sin = rotary[:, half_rope:].float().unsqueeze(1)
            nope = index_q[..., :nope_dim]
            pairs = index_q[..., -_ROPE_DIM:].float().reshape(index_q.shape[0], _HEADS, half_rope, 2)
            even = pairs[..., 0]
            odd = pairs[..., 1]
            roped = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).half()
            q_out = torch.cat((nope, roped), dim=-1)
            weights_out = index_weights.float() * _SOFTMAX_SCALE * _HEAD_SCALE
            return q_out, weights_out

    example = (
        torch.empty((_CAPACITY,), dtype=torch.int64, device="meta"),
        torch.empty((_CAPACITY, _HEADS, _HEAD_DIM), dtype=torch.float16, device="meta"),
        torch.empty((_CONTEXT, _ROPE_DIM), dtype=torch.float32, device="meta"),
        torch.empty((_CAPACITY, _HEADS), dtype=torch.float16, device="meta"),
    )
    dynamic_shapes = build_torch_dynamic_shapes(
        parse_position_specs(["num_tokens@positions:0", "num_tokens@index_q:0", "num_tokens@index_weights:0"])
    )
    return trace_module(Module(), example, dynamic_shapes=dynamic_shapes)


@dataclass
class _ProgramEntry:
    runtime: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    verified_rows: set[int] = field(default_factory=set)


def _build_program() -> _ProgramEntry:
    from emmy.serving.external import load_external_program

    runtime, plan = load_external_program(_indexer_q_graph(), symbolic_values={"num_tokens": _CAPACITY})
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != _EXPECTED_INPUTS or len(outputs) != 2:
        raise RuntimeError(f"1Cat indexer-Q expected {_EXPECTED_INPUTS!r} and two outputs, got {inputs!r} -> {outputs!r}")
    return _ProgramEntry(runtime, inputs, outputs)


def _run_external(entry: _ProgramEntry, tensors: tuple[Any, ...], outputs: tuple[Any, ...], rows: int) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(outputs[0].device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        entry.runtime.set_sym_values({"num_tokens": rows})
        bindings = {
            **{name: cp.from_dlpack(tensor) for name, tensor in zip(entry.inputs, tensors, strict=True)},
            **{name: cp.from_dlpack(tensor) for name, tensor in zip(entry.outputs, outputs, strict=True)},
        }
        entry.runtime.run_once_external(bindings)


def _is_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _is_sm70(tensors: tuple[Any, ...]) -> bool:
    import torch

    return bool(
        tensors
        and all(tensor.is_cuda and tensor.device == tensors[0].device for tensor in tensors)
        and torch.cuda.get_device_capability(tensors[0].device) == (7, 0)
    )


def _reference(tensors: tuple[Any, ...]) -> tuple[Any, Any]:
    import torch

    positions, index_q, cos_sin_cache, index_weights = tensors
    half_rope = _ROPE_DIM // 2
    nope_dim = _HEAD_DIM - _ROPE_DIM
    rotary = cos_sin_cache[positions]
    cos = rotary[:, :half_rope].float().unsqueeze(1)
    sin = rotary[:, half_rope:].float().unsqueeze(1)
    nope = index_q[..., :nope_dim]
    pairs = index_q[..., -_ROPE_DIM:].float().reshape(index_q.shape[0], _HEADS, half_rope, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    roped = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).half()
    return torch.cat((nope, roped), dim=-1), index_weights.float() * _SOFTMAX_SCALE * _HEAD_SCALE


class _IndexerAdapter:
    def __init__(
        self,
        *,
        build_program: Callable[[], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, tuple[Any, ...], tuple[Any, ...], int], None] = _run_external,
        platform_supported: Callable[[tuple[Any, ...]], bool] = _is_sm70,
        is_capturing: Callable[[], bool] = _is_capturing,
        oracle: Callable[[tuple[Any, ...]], tuple[Any, Any]] = _reference,
        fallback: Callable[[tuple[Any, ...]], tuple[Any, Any]] | None = None,
    ) -> None:
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._oracle = oracle
        self._fallback = fallback or oracle
        self._program: _ProgramEntry | None = None
        self._disabled = False
        self._lock = threading.RLock()

    def supported(self, tensors: tuple[Any, ...]) -> int | None:
        import torch

        try:
            positions, index_q, cos_sin_cache, index_weights = tensors
            rows = positions.shape[0] if positions.ndim == 1 else -1
            valid = (
                positions.dtype == torch.int64
                and index_q.dtype == index_weights.dtype == torch.float16
                and cos_sin_cache.dtype == torch.float32
                and tuple(positions.shape) == (rows,)
                and tuple(index_q.shape) == (rows, _HEADS, _HEAD_DIM)
                and tuple(cos_sin_cache.shape) == (_CONTEXT, _ROPE_DIM)
                and tuple(index_weights.shape) == (rows, _HEADS)
                and positions.stride() == (1,)
                and index_q.is_contiguous()
                and cos_sin_cache.is_contiguous()
                and index_weights.is_contiguous()
            )
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None
        return int(rows) if valid and 0 < rows <= _CAPACITY and self._platform_supported(tensors) else None

    def dispatch(self, *tensors):
        import torch

        rows = self.supported(tensors)
        if rows is None or self._disabled:
            return self._fallback(tensors)
        entry = self._program
        capturing = self._is_capturing()
        if capturing and (entry is None or rows not in entry.verified_rows):
            return self._fallback(tensors)

        with self._lock:
            if self._disabled:
                return self._fallback(tensors)
            entry = self._program
            if entry is None:
                if capturing:
                    return self._fallback(tensors)
                try:
                    entry = self._build_program()
                except Exception:  # noqa: BLE001 -- compatibility failure permanently falls back
                    self._disabled = True
                    logger.exception("1Cat indexer-Q: Emmy pack load failed; retaining the original operation")
                    return self._fallback(tensors)
                if entry.inputs != _EXPECTED_INPUTS or len(entry.outputs) != 2:
                    self._disabled = True
                    logger.error("1Cat indexer-Q: compiler boundary changed; retaining the original operation")
                    return self._fallback(tensors)
                self._program = entry

            q_out = tensors[1].new_empty((rows, _HEADS, _HEAD_DIM))
            weights_out = torch.empty((rows, _HEADS), dtype=torch.float32, device=tensors[1].device)
            outputs = (q_out, weights_out)
            try:
                self._run_program(entry, tensors, outputs, rows)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self._disabled = True
                self._program = None
                logger.exception("1Cat indexer-Q: Emmy launch failed; retaining the original operation")
                return self._fallback(tensors)
            if rows not in entry.verified_rows:
                if capturing:
                    return self._fallback(tensors)
                expected = self._oracle(tensors)
                if not all(
                    actual.shape == reference.shape
                    and actual.dtype == reference.dtype
                    and torch.allclose(actual, reference, rtol=_PARITY_TOL, atol=_PARITY_TOL)
                    for actual, reference in zip(outputs, expected, strict=True)
                ):
                    self._disabled = True
                    self._program = None
                    logger.error("1Cat indexer-Q: first-use parity failed; retaining the original operation")
                    return self._fallback(tensors)
                entry.verified_rows.add(rows)
            return outputs


_CUSTOM_OP: Any | None = None
_ACTIVE_ADAPTER: _IndexerAdapter | None = None


def _adapter() -> _IndexerAdapter:
    if _ACTIVE_ADAPTER is None:
        raise RuntimeError("1Cat indexer-Q custom op called before its adapter was installed")
    return _ACTIVE_ADAPTER


def _custom_op():
    global _CUSTOM_OP
    if _CUSTOM_OP is not None:
        return _CUSTOM_OP
    import torch

    @torch.library.custom_op(
        "emmy::onecat_indexer_q",
        mutates_args=(),
        schema="(Tensor positions, Tensor index_q, Tensor cos_sin_cache, Tensor index_weights) -> (Tensor, Tensor)",
    )
    def indexer_q(positions, index_q, cos_sin_cache, index_weights):
        return _adapter().dispatch(positions, index_q, cos_sin_cache, index_weights)

    @indexer_q.register_fake
    def indexer_q_fake(positions, index_q, cos_sin_cache, index_weights):  # noqa: ARG001
        return index_q.new_empty(index_q.shape), index_weights.new_empty(index_weights.shape, dtype=torch.float32)

    _CUSTOM_OP = indexer_q
    return _CUSTOM_OP


def _signature_matches(function: Callable) -> bool:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) == len(_SIGNATURE) and all(
        parameter.name == name and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default == default
        for parameter, (name, default) in zip(parameters, _SIGNATURE, strict=True)
    )


def register_onecat_indexer_kernels(indexer_module: ModuleType | None = None) -> bool:
    """Install the exact SM70 C4 indexer-Q pure transform in every live alias."""
    global _ACTIVE_ADAPTER

    try:
        indexer_module = indexer_module or importlib.import_module("vllm.models.deepseek_v4.common.ops.fused_indexer_q")
    except ImportError:
        logger.warning("1Cat indexer-Q requested, but the compatible DeepSeek V4 operation is unavailable")
        return False
    original = getattr(indexer_module, "fused_indexer_q_rope_quant", None)
    if getattr(original, "_emmy_onecat_indexer", False):
        _ACTIVE_ADAPTER = original._emmy_onecat_indexer_adapter  # type: ignore[attr-defined]
        return True
    if not callable(original) or not _signature_matches(original):
        logger.error("1Cat indexer-Q: compatible function signature is unavailable; no aliases changed")
        return False

    def original_exact(tensors):
        return original(*tensors, _SOFTMAX_SCALE, _HEAD_SCALE, False)

    adapter = _IndexerAdapter(oracle=original_exact, fallback=original_exact)
    op = _custom_op()

    def replacement(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=False,
    ):
        tensors = (positions, index_q, index_q_cos_sin_cache, index_weights)
        if (
            use_fp4
            or float(index_weights_softmax_scale) != _SOFTMAX_SCALE
            or float(index_weights_head_scale) != _HEAD_SCALE
            or adapter.supported(tensors) is None
        ):
            return original(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                use_fp4,
            )
        return op(positions, index_q, index_q_cos_sin_cache, index_weights)

    replacement._emmy_onecat_indexer = True  # type: ignore[attr-defined]
    replacement._emmy_onecat_indexer_adapter = adapter  # type: ignore[attr-defined]
    replacement._emmy_onecat_indexer_original = original  # type: ignore[attr-defined]

    aliases: list[tuple[ModuleType, str]] = [(indexer_module, "fused_indexer_q_rope_quant")]
    public_module = sys.modules.get("vllm.models.deepseek_v4.common.ops")
    attention_module = sys.modules.get("vllm.models.deepseek_v4.attention")
    for module in (public_module, attention_module):
        if module is not None and getattr(module, "fused_indexer_q_rope_quant", None) is original:
            aliases.append((module, "fused_indexer_q_rope_quant"))

    previous_active = _ACTIVE_ADAPTER
    changed: list[tuple[ModuleType, str]] = []
    try:
        for module, name in aliases:
            setattr(module, name, replacement)
            changed.append((module, name))
        _ACTIVE_ADAPTER = adapter
    except Exception:  # noqa: BLE001 -- preserve the all-or-none installation invariant
        for module, name in changed:
            setattr(module, name, original)
        _ACTIVE_ADAPTER = previous_active
        logger.exception("1Cat indexer-Q: installation failed; restored every original alias")
        return False
    logger.info("1Cat indexer-Q: installed guarded C4 RoPE and weight-scale adapter")
    return True
