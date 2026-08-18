"""Guarded Emmy adapters for pure 1Cat embedding, output, and activation leaves.

The adapters stop at TP-local tensor boundaries. 1Cat retains vocabulary
masking and reduction, logits gathering and compact top-1, both packed
shared-expert GEMMs, and the shared/routed output combination.
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

_CAPACITY = 4096
_HIDDEN = 4096
_VOCAB = 129280
_TP = 8
_LOCAL_VOCAB = _VOCAB // _TP
_SHARED_GATE_UP = 512
_SHARED_INTERMEDIATE = 256
_CLAMP = 10.0
_PARITY_TOL = 1e-2
_KINDS = ("embedding", "lm_head", "clamp_swiglu")
_EXPECTED_INPUTS = {
    "embedding": ("weight", "indices"),
    "lm_head": ("x", "weight"),
    "clamp_swiglu": ("x",),
}
_SIGNATURES = {
    "embedding": (("self", inspect.Parameter.empty), ("layer", inspect.Parameter.empty), ("input_", inspect.Parameter.empty)),
    "lm_head": (
        ("self", inspect.Parameter.empty),
        ("layer", inspect.Parameter.empty),
        ("x", inspect.Parameter.empty),
        ("bias", None),
    ),
    "clamp_swiglu": (("self", inspect.Parameter.empty), ("x", inspect.Parameter.empty)),
}


@dataclass
class _ProgramEntry:
    runtime: Any
    inputs: tuple[str, ...]
    output: str
    kind: str
    verified: bool = False


def _embedding_graph():
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16, I64
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import GatherOp

    rows = Dim("num_tokens", hint=_CAPACITY)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("weight", (_LOCAL_VOCAB, _HIDDEN), dtype=F16), node_id="weight")
    graph.add_node(InputOp(), [], Tensor("indices", (rows,), dtype=I64), node_id="indices")
    graph.add_node(GatherOp(axis=0), ["weight", "indices"], Tensor("output", (rows, _HIDDEN), dtype=F16), node_id="output")
    graph.inputs = ["weight", "indices"]
    graph.outputs = ["output"]
    return graph


def _lm_head_graph():
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import LinearOp

    rows = Dim("num_tokens", hint=_CAPACITY)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (rows, _HIDDEN), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("weight", (_LOCAL_VOCAB, _HIDDEN), dtype=F16), node_id="weight")
    graph.add_node(LinearOp(), ["x", "weight"], Tensor("output", (rows, _LOCAL_VOCAB), dtype=F16), node_id="output")
    graph.inputs = ["x", "weight"]
    graph.outputs = ["output"]
    return graph


def _clamp_swiglu_graph():
    import torch
    import torch.nn.functional as F

    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    class Module(torch.nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, dim=-1)
            gate = torch.clamp(gate, max=_CLAMP)
            up = torch.clamp(up, min=-_CLAMP, max=_CLAMP)
            return F.silu(gate) * up

    example = (torch.empty((_CAPACITY, _SHARED_GATE_UP), dtype=torch.float16, device="meta"),)
    dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(["num_tokens@x:0"]))
    return trace_module(Module(), example, dynamic_shapes=dynamic_shapes)


def _build_program(kind: str) -> _ProgramEntry:
    from emmy.serving.external import build_external_program

    graphs = {
        "embedding": _embedding_graph,
        "lm_head": _lm_head_graph,
        "clamp_swiglu": _clamp_swiglu_graph,
    }
    runtime, plan = build_external_program(graphs[kind](), symbolic_values={"num_tokens": _CAPACITY})
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != _EXPECTED_INPUTS[kind] or len(outputs) != 1:
        raise RuntimeError(f"1Cat {kind} expected {_EXPECTED_INPUTS[kind]!r} and one output, got {inputs!r} -> {outputs!r}")
    return _ProgramEntry(runtime, inputs, outputs[0], kind)


def _run_external(entry: _ProgramEntry, tensors: tuple[Any, ...], output: Any, rows: int) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(output.device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        entry.runtime.set_sym_values({"num_tokens": rows})
        bindings = {
            **{name: cp.from_dlpack(tensor) for name, tensor in zip(entry.inputs, tensors, strict=True)},
            entry.output: cp.from_dlpack(output),
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


def _reference(kind: str, tensors: tuple[Any, ...]):
    import torch
    import torch.nn.functional as F

    if kind == "embedding":
        return F.embedding(tensors[1], tensors[0])
    if kind == "lm_head":
        return F.linear(tensors[0], tensors[1])
    gate, up = tensors[0].chunk(2, dim=-1)
    gate = torch.clamp(gate, max=_CLAMP)
    up = torch.clamp(up, min=-_CLAMP, max=_CLAMP)
    return F.silu(gate) * up


class _OutputAdapter:
    def __init__(
        self,
        *,
        build_program: Callable[[str], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, tuple[Any, ...], Any, int], None] = _run_external,
        platform_supported: Callable[[tuple[Any, ...]], bool] = _is_sm70,
        is_capturing: Callable[[], bool] = _is_capturing,
        references: dict[str, Callable[[tuple[Any, ...]], Any]] | None = None,
    ) -> None:
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._references = references or {kind: lambda tensors, kind=kind: _reference(kind, tensors) for kind in _KINDS}
        self._programs: dict[str, _ProgramEntry] = {}
        self._disabled: set[str] = set()
        self._lock = threading.RLock()

    def _supported(self, kind: str, tensors: tuple[Any, ...]) -> int | None:
        import torch

        try:
            if kind == "embedding":
                weight, indices = tensors
                rows = indices.shape[0] if indices.ndim == 1 else -1
                valid = (
                    weight.dtype == torch.float16
                    and indices.dtype == torch.int64
                    and tuple(weight.shape) == (_LOCAL_VOCAB, _HIDDEN)
                    and tuple(indices.shape) == (rows,)
                    and weight.stride() == (_HIDDEN, 1)
                    and indices.stride() == (1,)
                )
            elif kind == "lm_head":
                x, weight = tensors
                rows = x.shape[0] if x.ndim == 2 else -1
                valid = (
                    x.dtype == weight.dtype == torch.float16
                    and tuple(x.shape) == (rows, _HIDDEN)
                    and tuple(weight.shape) == (_LOCAL_VOCAB, _HIDDEN)
                    and x.stride() == (_HIDDEN, 1)
                    and weight.stride() == (_HIDDEN, 1)
                )
            elif kind == "clamp_swiglu":
                (x,) = tensors
                rows = x.shape[0] if x.ndim == 2 else -1
                valid = x.dtype == torch.float16 and tuple(x.shape) == (rows, _SHARED_GATE_UP) and x.stride() == (_SHARED_GATE_UP, 1)
            else:
                raise KeyError(kind)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None
        return int(rows) if valid and 0 < rows <= _CAPACITY and self._platform_supported(tensors) else None

    @staticmethod
    def _allocate(kind: str, tensors: tuple[Any, ...], rows: int):
        import torch

        if kind == "embedding":
            return torch.empty((rows, _HIDDEN), dtype=torch.float16, device=tensors[1].device)
        if kind == "lm_head":
            return tensors[0].new_empty((rows, _LOCAL_VOCAB))
        return tensors[0].new_empty((rows, _SHARED_INTERMEDIATE))

    def _disable(self, kind: str, message: str, *, exc_info: bool = False) -> None:
        self._disabled.add(kind)
        self._programs.pop(kind, None)
        logger.error("1Cat %s: %s; retaining the original operation", kind, message, exc_info=exc_info)

    def _ensure(self, kind: str) -> _ProgramEntry | None:
        entry = self._programs.get(kind)
        if entry is not None:
            return entry
        try:
            entry = self._build_program(kind)
        except Exception:  # noqa: BLE001 -- compatibility failure permanently falls back for this family
            self._disable(kind, "Emmy build failed", exc_info=True)
            return None
        if entry.kind != kind or entry.inputs != _EXPECTED_INPUTS[kind]:
            self._disable(kind, "compiler boundary did not match the requested operation family")
            return None
        self._programs[kind] = entry
        return entry

    def dispatch(self, kind: str, *tensors):
        rows = self._supported(kind, tensors)
        reference = self._references[kind]
        if rows is None or kind in self._disabled:
            return reference(tensors)

        entry = self._programs.get(kind)
        capturing = self._is_capturing()
        if capturing and (entry is None or not entry.verified):
            return reference(tensors)

        with self._lock:
            if kind in self._disabled:
                return reference(tensors)
            entry = self._programs.get(kind)
            if entry is None:
                if capturing:
                    return reference(tensors)
                entry = self._ensure(kind)
                if entry is None:
                    return reference(tensors)

            output = self._allocate(kind, tensors, rows)
            try:
                self._run_program(entry, tensors, output, rows)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back for this family
                self._disable(kind, "Emmy launch failed", exc_info=True)
                return reference(tensors)

            if not entry.verified:
                if capturing:
                    return reference(tensors)
                expected = reference(tensors)
                exact = kind == "embedding"
                matches = torch_equal(output, expected) if exact else torch_close(output, expected)
                if not matches:
                    self._disable(kind, "first-use parity failed")
                    return expected
                entry.verified = True
            return output


def torch_equal(actual: Any, expected: Any) -> bool:
    import torch

    return bool(actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual, expected))


def torch_close(actual: Any, expected: Any) -> bool:
    import torch

    return bool(
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.allclose(actual, expected, rtol=_PARITY_TOL, atol=_PARITY_TOL)
    )


_CUSTOM_OPS: dict[str, Any] | None = None
_ACTIVE_ADAPTER: _OutputAdapter | None = None


def _adapter() -> _OutputAdapter:
    if _ACTIVE_ADAPTER is None:
        raise RuntimeError("1Cat output custom op called before its adapter was installed")
    return _ACTIVE_ADAPTER


def _custom_ops() -> dict[str, Any]:
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    import torch

    @torch.library.custom_op("emmy::onecat_vocab_embedding", mutates_args=(), schema="(Tensor weight, Tensor indices) -> Tensor")
    def embedding(weight, indices):
        return _adapter().dispatch("embedding", weight, indices)

    @embedding.register_fake
    def embedding_fake(weight, indices):
        return weight.new_empty((*indices.shape, weight.shape[1]))

    @torch.library.custom_op("emmy::onecat_lm_head", mutates_args=(), schema="(Tensor x, Tensor weight) -> Tensor")
    def lm_head(x, weight):
        return _adapter().dispatch("lm_head", x, weight)

    @lm_head.register_fake
    def lm_head_fake(x, weight):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    @torch.library.custom_op("emmy::onecat_clamp_swiglu", mutates_args=(), schema="(Tensor x) -> Tensor")
    def clamp_swiglu(x):
        return _adapter().dispatch("clamp_swiglu", x)

    @clamp_swiglu.register_fake
    def clamp_swiglu_fake(x):
        return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))

    _CUSTOM_OPS = {"embedding": embedding, "lm_head": lm_head, "clamp_swiglu": clamp_swiglu}
    return _CUSTOM_OPS


def _embedding_layer_supported(layer: Any) -> bool:
    return bool(
        type(layer).__name__ == "VocabParallelEmbedding"
        and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "embed_tokens"
        and getattr(layer, "tp_size", None) == _TP
        and getattr(layer, "num_embeddings", None) == _VOCAB
        and getattr(layer, "num_embeddings_per_partition", None) == _LOCAL_VOCAB
        and getattr(layer, "embedding_dim", None) == _HIDDEN
    )


def _lm_head_layer_supported(layer: Any, bias: Any) -> bool:
    return bool(
        bias is None
        and type(layer).__name__ == "ParallelLMHead"
        and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "lm_head"
        and getattr(layer, "tp_size", None) == _TP
        and getattr(layer, "num_embeddings", None) == _VOCAB
        and getattr(layer, "num_embeddings_per_partition", None) == _LOCAL_VOCAB
        and getattr(layer, "embedding_dim", None) == _HIDDEN
    )


def _clamp_swiglu_supported(operation: Any) -> bool:
    try:
        return type(operation).__name__ == "SiluAndMulWithClamp" and float(operation.swiglu_limit) == _CLAMP
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _signature_matches(function: Callable, expected: tuple[tuple[str, Any], ...]) -> bool:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) == len(expected) and all(
        parameter.name == name and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default == default
        for parameter, (name, default) in zip(parameters, expected, strict=True)
    )


def register_onecat_output_kernels(
    embedding_module: ModuleType | None = None,
    activation_module: ModuleType | None = None,
) -> bool:
    """Atomically install the three guarded pure-compute replacements."""
    global _ACTIVE_ADAPTER

    try:
        embedding_module = embedding_module or importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
        activation_module = activation_module or importlib.import_module("vllm.model_executor.layers.activation")
    except ImportError:
        logger.warning("1Cat output requested, but compatible vLLM layers are unavailable")
        return False

    method_cls = getattr(embedding_module, "UnquantizedEmbeddingMethod", None)
    activation_cls = getattr(activation_module, "SiluAndMulWithClamp", None)
    functions = {
        "embedding": getattr(method_cls, "embedding", None),
        "lm_head": getattr(method_cls, "apply", None),
        "clamp_swiglu": getattr(activation_cls, "forward_cuda", None),
    }
    installed = [bool(getattr(function, "_emmy_onecat_output", False)) for function in functions.values()]
    if all(installed):
        _ACTIVE_ADAPTER = functions["embedding"]._emmy_onecat_output_adapter  # type: ignore[attr-defined]
        return True
    if any(installed):
        logger.error("1Cat output: partial prior installation detected; no methods changed")
        return False
    if not all(callable(function) and _signature_matches(function, _SIGNATURES[kind]) for kind, function in functions.items()):
        logger.error("1Cat output: compatible method signatures are unavailable; no methods changed")
        return False

    adapter = _OutputAdapter()
    ops = _custom_ops()
    originals = dict(functions)

    def embedding(method, layer, input_):
        if _embedding_layer_supported(layer):
            return ops["embedding"](layer.weight, input_)
        return originals["embedding"](method, layer, input_)

    def lm_head(method, layer, x, bias=None):
        if _lm_head_layer_supported(layer, bias):
            return ops["lm_head"](x, layer.weight)
        return originals["lm_head"](method, layer, x, bias)

    def clamp_swiglu(operation, x):
        if _clamp_swiglu_supported(operation):
            return ops["clamp_swiglu"](x)
        return originals["clamp_swiglu"](operation, x)

    replacements = {"embedding": embedding, "lm_head": lm_head, "clamp_swiglu": clamp_swiglu}
    for kind, replacement in replacements.items():
        replacement._emmy_onecat_output = True  # type: ignore[attr-defined]
        replacement._emmy_onecat_output_adapter = adapter  # type: ignore[attr-defined]
        replacement._emmy_onecat_output_original = originals[kind]  # type: ignore[attr-defined]

    previous_active = _ACTIVE_ADAPTER
    try:
        method_cls.embedding = embedding
        method_cls.apply = lm_head
        activation_cls.forward_cuda = clamp_swiglu
        _ACTIVE_ADAPTER = adapter
    except Exception:  # noqa: BLE001 -- preserve the all-or-none installation invariant
        method_cls.embedding = originals["embedding"]
        method_cls.apply = originals["lm_head"]
        activation_cls.forward_cuda = originals["clamp_swiglu"]
        _ACTIVE_ADAPTER = previous_active
        logger.exception("1Cat output: installation failed; restored every original method")
        return False
    logger.info("1Cat output: installed guarded embedding, full-logit, and shared clamp-SwiGLU adapters")
    return True
