"""Guarded Emmy ownership around DeepSeek V4 tensor-parallel vocabulary collectives.

The all-reduce and all-gather remain 1Cat operations. Emmy owns the exact
TP-local embedding mask/gather/zero path and the compact LM-head top-1 work on
either side of the all-gather.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_CAPACITY = 4096
_HIDDEN = 4096
_VOCAB = 129280
_TP = 8
_LOCAL_VOCAB = _VOCAB // _TP
_PARITY_TOL = 1e-2
_EXPECTED_INPUTS = {
    "embedding": ("weight", "input_ids"),
    "local_top1": ("x", "weight"),
    "rank_top1": ("gathered",),
}
_SIGNATURES = {
    "embedding": (("self", inspect.Parameter.empty), ("input_", inspect.Parameter.empty)),
    "top1": (
        ("self", inspect.Parameter.empty),
        ("lm_head", inspect.Parameter.empty),
        ("hidden_states", inspect.Parameter.empty),
        ("embedding_bias", None),
    ),
}


class _UseOriginal(RuntimeError):
    """Signal that the outer adapter must resume the exact 1Cat path."""


def _dynamic_rows(*names: str):
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    return build_torch_dynamic_shapes(parse_position_specs([f"num_tokens@{name}:0" for name in names]))


def _tp_embedding_graph(rank: int):
    import torch
    import torch.nn.functional as F

    from emmy.compiler.trace.torch import trace_module

    start = rank * _LOCAL_VOCAB
    end = start + _LOCAL_VOCAB

    class Module(torch.nn.Module):
        def forward(self, weight, input_ids):
            valid = (input_ids >= start) & (input_ids < end)
            local_ids = torch.where(valid, input_ids - start, 0)
            embedded = F.embedding(local_ids, weight)
            return torch.where(valid.unsqueeze(-1), embedded, 0.0)

    example = (
        torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16, device="meta"),
        torch.empty((_CAPACITY,), dtype=torch.int64, device="meta"),
    )
    return trace_module(Module(), example, dynamic_shapes=_dynamic_rows("input_ids"))


def _local_top1_graph(rank: int):
    import torch
    import torch.nn.functional as F

    from emmy.compiler.trace.torch import trace_module

    vocab_start = rank * _LOCAL_VOCAB

    class Module(torch.nn.Module):
        def forward(self, x, weight):
            logits = F.linear(x, weight)
            local_ids = torch.arange(_LOCAL_VOCAB, dtype=torch.int64, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            max_values = logits.max(dim=-1).values
            matches = logits == max_values.unsqueeze(-1)
            # amax over negated indices gives torch.argmax's stable first tie.
            candidates = torch.where(matches, -local_ids, -_LOCAL_VOCAB)
            global_ids = -candidates.amax(dim=-1) + vocab_start
            return torch.stack((max_values.float(), global_ids.float()), dim=-1)

    example = (
        torch.empty((_CAPACITY, _HIDDEN), dtype=torch.float16, device="meta"),
        torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16, device="meta"),
    )
    return trace_module(Module(), example, dynamic_shapes=_dynamic_rows("x"))


def _rank_top1_graph():
    import torch

    from emmy.compiler.trace.torch import trace_module

    class Module(torch.nn.Module):
        def forward(self, gathered):
            values = gathered[:, :, 0]
            global_ids = gathered[:, :, 1]
            max_values = values.max(dim=-1).values
            ranks = torch.arange(_TP, dtype=torch.int64, device=gathered.device).unsqueeze(0).expand(gathered.shape[0], -1)
            matches = values == max_values.unsqueeze(-1)
            candidates = torch.where(matches, -ranks, -_TP)
            chosen_rank = -candidates.amax(dim=-1)
            chosen = ranks == chosen_rank.unsqueeze(-1)
            top = torch.where(chosen, global_ids, -1.0).amax(dim=-1)
            return top.to(torch.int64)

    example = (torch.empty((_CAPACITY, _TP, 2), dtype=torch.float32, device="meta"),)
    return trace_module(Module(), example, dynamic_shapes=_dynamic_rows("gathered"))


@dataclass
class _ProgramEntry:
    runtime: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    kind: str
    rank: int | None
    verified_rows: set[int] = field(default_factory=set)


def _build_program(kind: str, rank: int | None) -> _ProgramEntry:
    from emmy.serving.external import load_external_program

    if kind == "embedding":
        assert rank is not None
        graph = _tp_embedding_graph(rank)
    elif kind == "local_top1":
        assert rank is not None
        graph = _local_top1_graph(rank)
    elif kind == "rank_top1":
        assert rank is None
        graph = _rank_top1_graph()
    else:
        raise KeyError(kind)
    runtime, plan = load_external_program(graph, symbolic_values={"num_tokens": _CAPACITY})
    inputs = tuple(plan.inputs)
    outputs = tuple(plan.outputs)
    if inputs != _EXPECTED_INPUTS[kind] or len(outputs) != 1:
        raise RuntimeError(f"1Cat {kind} expected {_EXPECTED_INPUTS[kind]!r} and one output, got {inputs!r} -> {outputs!r}")
    return _ProgramEntry(runtime, inputs, outputs, kind, rank)


def _run_external(entry: _ProgramEntry, tensors: tuple[Any, ...], output: Any, rows: int) -> None:
    import cupy as cp
    import torch

    from emmy.compiler.backend.gpu_lock import gpu_lock

    stream = torch.cuda.current_stream(output.device)
    with gpu_lock(), cp.cuda.Stream.from_external(stream):
        entry.runtime.set_sym_values({"num_tokens": rows})
        bindings = {
            **{name: cp.from_dlpack(tensor) for name, tensor in zip(entry.inputs, tensors, strict=True)},
            entry.outputs[0]: cp.from_dlpack(output),
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


def _reference(kind: str, rank: int | None, tensors: tuple[Any, ...]):
    import torch
    import torch.nn.functional as F

    if kind == "embedding":
        weight, input_ids = tensors
        assert rank is not None
        start = rank * _LOCAL_VOCAB
        valid = (input_ids >= start) & (input_ids < start + _LOCAL_VOCAB)
        local_ids = torch.where(valid, input_ids - start, torch.zeros_like(input_ids))
        return torch.where(valid.unsqueeze(-1), F.embedding(local_ids, weight), torch.zeros((), dtype=weight.dtype, device=weight.device))
    if kind == "local_top1":
        x, weight = tensors
        assert rank is not None
        values, local_ids = F.linear(x, weight).max(dim=-1)
        return torch.stack((values.float(), (local_ids + rank * _LOCAL_VOCAB).float()), dim=-1)
    (gathered,) = tensors
    max_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=max_rank).squeeze(-1).to(torch.int64)


class _VocabCollectiveAdapter:
    def __init__(
        self,
        *,
        build_program: Callable[[str, int | None], _ProgramEntry] = _build_program,
        run_program: Callable[[_ProgramEntry, tuple[Any, ...], Any, int], None] = _run_external,
        platform_supported: Callable[[tuple[Any, ...]], bool] = _is_sm70,
        is_capturing: Callable[[], bool] = _is_capturing,
        reference: Callable[[str, int | None, tuple[Any, ...]], Any] = _reference,
    ) -> None:
        self._build_program = build_program
        self._run_program = run_program
        self._platform_supported = platform_supported
        self._is_capturing = is_capturing
        self._reference = reference
        self._programs: dict[tuple[str, int | None], _ProgramEntry] = {}
        self._disabled: set[tuple[str, int | None]] = set()
        self._lock = threading.RLock()

    def supported(self, kind: str, rank: int | None, tensors: tuple[Any, ...]) -> int | None:
        import torch

        try:
            if kind == "embedding":
                weight, input_ids = tensors
                rows = input_ids.shape[0] if input_ids.ndim == 1 else -1
                valid = (
                    rank is not None
                    and weight.dtype == torch.float16
                    and input_ids.dtype == torch.int64
                    and tuple(weight.shape) == (_LOCAL_VOCAB, _HIDDEN)
                    and tuple(input_ids.shape) == (rows,)
                    and weight.stride() == (_HIDDEN, 1)
                    and input_ids.stride() == (1,)
                )
            elif kind == "local_top1":
                x, weight = tensors
                rows = x.shape[0] if x.ndim == 2 else -1
                valid = (
                    rank is not None
                    and x.dtype == weight.dtype == torch.float16
                    and tuple(x.shape) == (rows, _HIDDEN)
                    and tuple(weight.shape) == (_LOCAL_VOCAB, _HIDDEN)
                    and x.stride() == (_HIDDEN, 1)
                    and weight.stride() == (_HIDDEN, 1)
                )
            elif kind == "rank_top1":
                (gathered,) = tensors
                rows = gathered.shape[0] if gathered.ndim == 3 else -1
                valid = gathered.dtype == torch.float32 and tuple(gathered.shape) == (rows, _TP, 2) and gathered.is_contiguous()
            else:
                raise KeyError(kind)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None
        return int(rows) if valid and 0 < rows <= _CAPACITY and self._platform_supported(tensors) else None

    @staticmethod
    def _allocate(kind: str, tensors: tuple[Any, ...], rows: int):
        import torch

        if kind == "embedding":
            return tensors[0].new_empty((rows, _HIDDEN))
        if kind == "local_top1":
            return torch.empty((rows, 2), dtype=torch.float32, device=tensors[0].device)
        return torch.empty((rows,), dtype=torch.int64, device=tensors[0].device)

    @staticmethod
    def _matches(kind: str, actual: Any, expected: Any) -> bool:
        import torch

        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False
        if kind in ("embedding", "rank_top1"):
            return bool(torch.equal(actual, expected))
        return bool(
            torch.allclose(actual[:, 0], expected[:, 0], rtol=_PARITY_TOL, atol=_PARITY_TOL) and torch.equal(actual[:, 1], expected[:, 1])
        )

    def _disable(self, key: tuple[str, int | None], message: str, *, exc_info: bool = False) -> None:
        self._disabled.add(key)
        self._programs.pop(key, None)
        logger.error("1Cat %s rank %s: %s; retaining the original operation", key[0], key[1], message, exc_info=exc_info)

    def prepare(self, kind: str, rank: int | None, rows: int) -> bool:
        """Load a prewarmed pack before any collective and gate capture by exact M."""
        key = (kind, rank)
        if key in self._disabled:
            return False
        entry = self._programs.get(key)
        capturing = self._is_capturing()
        if entry is not None:
            return not capturing or rows in entry.verified_rows
        if capturing:
            return False
        with self._lock:
            if key in self._disabled:
                return False
            entry = self._programs.get(key)
            if entry is None:
                try:
                    entry = self._build_program(kind, rank)
                except Exception:  # noqa: BLE001 -- compatibility failure permanently falls back
                    self._disable(key, "Emmy pack load failed", exc_info=True)
                    return False
                if entry.inputs != _EXPECTED_INPUTS[kind] or entry.kind != kind or entry.rank != rank:
                    self._disable(key, "compiler boundary did not match the requested operation")
                    return False
                self._programs[key] = entry
            return True

    def dispatch(self, kind: str, rank: int | None, *tensors):
        rows = self.supported(kind, rank, tensors)
        reference = partial(self._reference, kind, rank, tensors)
        key = (kind, rank)
        if rows is None or not self.prepare(kind, rank, rows):
            raise _UseOriginal

        with self._lock:
            entry = self._programs.get(key)
            if entry is None:
                raise _UseOriginal

            output = self._allocate(kind, tensors, rows)
            try:
                self._run_program(entry, tensors, output, rows)
            except Exception:  # noqa: BLE001 -- launch incompatibility permanently falls back
                self._disable(key, "Emmy launch failed", exc_info=True)
                raise _UseOriginal from None
            if rows not in entry.verified_rows:
                expected = reference()
                if not self._matches(kind, output, expected):
                    self._disable(key, "first-use parity failed")
                    raise _UseOriginal
                entry.verified_rows.add(rows)
            return output


_CUSTOM_OPS: dict[str, Any] | None = None
_ACTIVE_ADAPTER: _VocabCollectiveAdapter | None = None


def _adapter() -> _VocabCollectiveAdapter:
    if _ACTIVE_ADAPTER is None:
        raise RuntimeError("1Cat vocabulary custom op called before its adapter was installed")
    return _ACTIVE_ADAPTER


def _custom_ops() -> dict[str, Any]:
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    import torch

    @torch.library.custom_op("emmy::onecat_tp_embedding", mutates_args=(), schema="(Tensor weight, Tensor input_ids, int rank) -> Tensor")
    def embedding(weight, input_ids, rank: int):
        return _adapter().dispatch("embedding", rank, weight, input_ids)

    @embedding.register_fake
    def embedding_fake(weight, input_ids, rank: int):  # noqa: ARG001
        return weight.new_empty((*input_ids.shape, weight.shape[1]))

    @torch.library.custom_op("emmy::onecat_local_top1", mutates_args=(), schema="(Tensor x, Tensor weight, int rank) -> Tensor")
    def local_top1(x, weight, rank: int):
        return _adapter().dispatch("local_top1", rank, x, weight)

    @local_top1.register_fake
    def local_top1_fake(x, weight, rank: int):  # noqa: ARG001
        return x.new_empty((*x.shape[:-1], 2), dtype=torch.float32)

    @torch.library.custom_op("emmy::onecat_rank_top1", mutates_args=(), schema="(Tensor gathered) -> Tensor")
    def rank_top1(gathered):
        return _adapter().dispatch("rank_top1", None, gathered)

    @rank_top1.register_fake
    def rank_top1_fake(gathered):
        return gathered.new_empty(gathered.shape[0], dtype=torch.int64)

    _CUSTOM_OPS = {"embedding": embedding, "local_top1": local_top1, "rank_top1": rank_top1}
    return _CUSTOM_OPS


def _layer_rank(layer: Any, *, lm_head: bool) -> int | None:
    expected_type = "ParallelLMHead" if lm_head else "VocabParallelEmbedding"
    expected_leaf = "lm_head" if lm_head else "embed_tokens"
    try:
        shard = layer.shard_indices
        start = int(shard.org_vocab_start_index)
        rank = start // _LOCAL_VOCAB
        valid = (
            type(layer).__name__ == expected_type
            and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == expected_leaf
            and layer.tp_size == _TP
            and layer.num_embeddings == _VOCAB
            and layer.org_vocab_size == _VOCAB
            and layer.num_embeddings_padded == _VOCAB
            and layer.num_embeddings_per_partition == _LOCAL_VOCAB
            and layer.embedding_dim == _HIDDEN
            and 0 <= rank < _TP
            and start == rank * _LOCAL_VOCAB
            and shard.org_vocab_end_index == start + _LOCAL_VOCAB
            and shard.num_org_vocab_padding == 0
            and shard.added_vocab_start_index == _VOCAB
            and shard.added_vocab_end_index == _VOCAB
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return rank if valid else None


def _processor_supported(processor: Any) -> bool:
    try:
        return bool(
            type(processor).__name__ == "LogitsProcessor"
            and processor.scale == 1.0
            and processor.soft_cap is None
            and not processor.logits_as_input
            and processor.vocab_size == _VOCAB
            and processor.org_vocab_size == _VOCAB
        )
    except (AttributeError, TypeError, ValueError):
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


def register_onecat_vocab_kernels(
    embedding_module: ModuleType | None = None,
    logits_module: ModuleType | None = None,
) -> bool:
    """Atomically install pure compute around the two retained TP collectives."""
    global _ACTIVE_ADAPTER

    try:
        embedding_module = embedding_module or importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
        logits_module = logits_module or importlib.import_module("vllm.model_executor.layers.logits_processor")
    except ImportError:
        logger.warning("1Cat vocabulary adapters requested, but compatible vLLM layers are unavailable")
        return False

    embedding_cls = getattr(embedding_module, "VocabParallelEmbedding", None)
    processor_cls = getattr(logits_module, "LogitsProcessor", None)
    originals = {
        "embedding": getattr(embedding_cls, "forward", None),
        "top1": getattr(processor_cls, "get_top_tokens", None),
    }
    installed = [bool(getattr(function, "_emmy_onecat_vocab", False)) for function in originals.values()]
    if all(installed):
        adapters = [getattr(function, "_emmy_onecat_vocab_adapter", None) for function in originals.values()]
        if adapters[0] is None or any(adapter is not adapters[0] for adapter in adapters[1:]):
            logger.error("1Cat vocabulary: prior wrappers do not share one adapter; no methods changed")
            return False
        _ACTIVE_ADAPTER = adapters[0]
        return True
    if any(installed):
        logger.error("1Cat vocabulary: partial prior installation detected; no methods changed")
        return False
    if not all(callable(function) and _signature_matches(function, _SIGNATURES[kind]) for kind, function in originals.items()):
        logger.error("1Cat vocabulary: compatible method signatures are unavailable; no methods changed")
        return False

    adapter = _VocabCollectiveAdapter()
    ops = _custom_ops()

    def embedding(layer, input_):
        rank = _layer_rank(layer, lm_head=False)
        tensors = (layer.weight, input_) if rank is not None else ()
        rows = adapter.supported("embedding", rank, tensors) if rank is not None else None
        if rank is None or rows is None or not adapter.prepare("embedding", rank, rows):
            return originals["embedding"](layer, input_)
        try:
            output_parallel = ops["embedding"](layer.weight, input_, rank)
        except _UseOriginal:
            return originals["embedding"](layer, input_)
        return embedding_module.tensor_model_parallel_all_reduce(output_parallel)

    def top1(processor, lm_head, hidden_states, embedding_bias=None):
        rank = _layer_rank(lm_head, lm_head=True)
        local_tensors = (hidden_states, lm_head.weight) if rank is not None else ()
        rows = adapter.supported("local_top1", rank, local_tensors) if rank is not None else None
        if (
            embedding_bias is not None
            or rank is None
            or not _processor_supported(processor)
            or rows is None
            or not adapter.prepare("local_top1", rank, rows)
            or not adapter.prepare("rank_top1", None, rows)
        ):
            return originals["top1"](processor, lm_head, hidden_states, embedding_bias)
        try:
            local_pair = ops["local_top1"](hidden_states, lm_head.weight, rank)
        except _UseOriginal:
            return originals["top1"](processor, lm_head, hidden_states, embedding_bias)
        maybe_sync = getattr(logits_module, "_maybe_sync_top1_all_gather", None)
        if callable(maybe_sync):
            maybe_sync(processor, local_pair)
        gathered = logits_module.tensor_model_parallel_all_gather(local_pair, dim=-1).view(hidden_states.shape[0], _TP, 2)
        try:
            top_tokens = ops["rank_top1"](gathered)
        except _UseOriginal:
            # The collective already completed. Resume only the exact stock
            # post-all-gather selection so it is never issued twice.
            top_tokens = _reference("rank_top1", None, (gathered,))
        maybe_dump = getattr(processor, "_maybe_dump_top_token_margin", None)
        if callable(maybe_dump):
            maybe_dump(lm_head, hidden_states, embedding_bias, top_tokens)
        return top_tokens

    replacements = {"embedding": embedding, "top1": top1}
    for kind, replacement in replacements.items():
        replacement._emmy_onecat_vocab = True  # type: ignore[attr-defined]
        replacement._emmy_onecat_vocab_adapter = adapter  # type: ignore[attr-defined]
        replacement._emmy_onecat_vocab_original = originals[kind]  # type: ignore[attr-defined]

    previous_active = _ACTIVE_ADAPTER
    try:
        embedding_cls.forward = embedding
        processor_cls.get_top_tokens = top1
        _ACTIVE_ADAPTER = adapter
    except Exception:  # noqa: BLE001 -- preserve the all-or-none installation invariant
        embedding_cls.forward = originals["embedding"]
        processor_cls.get_top_tokens = originals["top1"]
        _ACTIVE_ADAPTER = previous_active
        logger.exception("1Cat vocabulary: installation failed; restored both original methods")
        return False
    logger.info("1Cat vocabulary: installed guarded local embedding and compact top-1 adapters around TP collectives")
    return True
