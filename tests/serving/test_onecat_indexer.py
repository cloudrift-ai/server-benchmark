import sys
from types import ModuleType

import torch

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.serving.onecat_indexer import (
    _CAPACITY,
    _CONTEXT,
    _HEAD_DIM,
    _HEAD_SCALE,
    _HEADS,
    _ROPE_DIM,
    _SOFTMAX_SCALE,
    _indexer_q_graph,
    _IndexerAdapter,
    _ProgramEntry,
    register_onecat_indexer_kernels,
)


def test_indexer_graph_is_one_symbolic_pure_two_output_contract():
    graph = _indexer_q_graph()
    assert graph.inputs == ["positions", "index_q", "cos_sin_cache", "index_weights"]
    assert len(graph.outputs) == 2
    q_out, weights_out = (graph.nodes[name].output for name in graph.outputs)
    assert q_out.shape == (Dim("num_tokens"), _HEADS, _HEAD_DIM)
    assert q_out.dtype == F16
    assert weights_out.shape == (Dim("num_tokens"), _HEADS)
    assert weights_out.dtype == F32


def _inputs(rows=2):
    return (
        torch.empty((rows,), dtype=torch.int64, device="meta"),
        torch.empty((rows, _HEADS, _HEAD_DIM), dtype=torch.float16, device="meta"),
        torch.empty((_CONTEXT, _ROPE_DIM), dtype=torch.float32, device="meta"),
        torch.empty((rows, _HEADS), dtype=torch.float16, device="meta"),
    )


def test_adapter_accepts_only_exact_contiguous_sm70_abi():
    adapter = _IndexerAdapter(platform_supported=lambda tensors: True)
    for rows in (1, 17, _CAPACITY):
        assert adapter.supported(_inputs(rows)) == rows
    assert adapter.supported(_inputs(_CAPACITY + 1)) is None
    tensors = list(_inputs())
    tensors[3] = torch.empty((2, _HEADS + 1), dtype=torch.float16, device="meta")
    assert adapter.supported(tuple(tensors)) is None


def test_adapter_parity_latches_each_exact_row_and_capture_does_not_warm():
    references = []
    entry = _ProgramEntry(object(), ("positions", "index_q", "cos_sin_cache", "index_weights"), ("q", "weights"))

    def reference(tensors):
        references.append(tensors)
        rows = tensors[0].shape[0]
        return torch.ones((rows, _HEADS, _HEAD_DIM), dtype=torch.float16), torch.ones((rows, _HEADS), dtype=torch.float32)

    def run(entry, tensors, outputs, rows):
        for output in outputs:
            output.fill_(1)

    adapter = _IndexerAdapter(
        build_program=lambda: entry,
        run_program=run,
        platform_supported=lambda tensors: True,
        is_capturing=lambda: False,
        oracle=reference,
    )
    tensors = (torch.empty(2), torch.empty(2, dtype=torch.float16), torch.empty(2), torch.empty(2))
    adapter.supported = lambda tensors: 2
    outputs = adapter.dispatch(*tensors)
    assert outputs[0].shape == (2, _HEADS, _HEAD_DIM)
    assert entry.verified_rows == {2}
    assert len(references) == 1
    adapter.dispatch(*tensors)
    assert len(references) == 1

    adapter._is_capturing = lambda: True
    adapter.supported = lambda tensors: 1
    adapter.dispatch(*tensors)
    assert len(references) == 2
    assert entry.verified_rows == {2}

    builds = []
    fallbacks = []
    cold = _IndexerAdapter(
        build_program=lambda: builds.append(True),
        platform_supported=lambda tensors: True,
        is_capturing=lambda: True,
        fallback=lambda tensors: fallbacks.append(tensors) or ("original_q", "original_weights"),
    )
    cold.supported = lambda tensors: 1
    assert cold.dispatch(*tensors) == ("original_q", "original_weights")
    assert builds == []
    assert fallbacks == [tensors]


def test_registration_updates_loaded_aliases_and_guards_scale_and_fp4(monkeypatch):
    source = ModuleType("test_indexer_source")
    public = ModuleType("vllm.models.deepseek_v4.common.ops")
    attention = ModuleType("vllm.models.deepseek_v4.attention")
    calls = []

    def fused_indexer_q_rope_quant(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=False,
    ):
        calls.append((index_weights_softmax_scale, index_weights_head_scale, use_fp4))
        return "original"

    source.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant
    public.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant
    attention.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.common.ops", public)
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.attention", attention)

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def supported(self, tensors):
            return tensors[0].shape[0]

    sentinel = (object(), object())
    monkeypatch.setattr("emmy.serving.onecat_indexer._IndexerAdapter", Adapter)
    monkeypatch.setattr("emmy.serving.onecat_indexer._custom_op", lambda: lambda *args: sentinel)
    assert register_onecat_indexer_kernels(source)
    assert public.fused_indexer_q_rope_quant is source.fused_indexer_q_rope_quant
    assert attention.fused_indexer_q_rope_quant is source.fused_indexer_q_rope_quant

    args = _inputs(1)
    assert source.fused_indexer_q_rope_quant(*args, _SOFTMAX_SCALE, _HEAD_SCALE) is sentinel
    assert source.fused_indexer_q_rope_quant(*args, _SOFTMAX_SCALE, _HEAD_SCALE, True) == "original"
    assert source.fused_indexer_q_rope_quant(*args, 1.0, _HEAD_SCALE) == "original"
    assert len(calls) == 2
