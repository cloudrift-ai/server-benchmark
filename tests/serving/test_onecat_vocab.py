from types import ModuleType, SimpleNamespace

import pytest
import torch

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, I64
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, IndexMapOp, ReduceOp
from emmy.compiler.pipeline import TENSOR_PASSES, Pipeline
from emmy.serving.onecat_vocab import (
    _CAPACITY,
    _HIDDEN,
    _LOCAL_VOCAB,
    _TP,
    _VOCAB,
    _layer_rank,
    _local_top1_graph,
    _ProgramEntry,
    _rank_top1_graph,
    _reference,
    _tp_embedding_graph,
    _UseOriginal,
    _VocabCollectiveAdapter,
    register_onecat_vocab_kernels,
)


def test_graphs_cover_every_symbolic_tp_local_operation():
    embedding = _tp_embedding_graph(3)
    assert embedding.inputs == ["weight", "input_ids"]
    assert embedding.nodes[embedding.outputs[0]].output.shape == (Dim("num_tokens"), _HIDDEN)
    assert embedding.nodes[embedding.outputs[0]].output.dtype == F16

    local = _local_top1_graph(3)
    assert local.inputs == ["x", "weight"]
    assert local.nodes[local.outputs[0]].output.shape == (Dim("num_tokens"), 2)
    assert local.nodes[local.outputs[0]].output.dtype == F32

    rank = _rank_top1_graph()
    assert rank.inputs == ["gathered"]
    assert rank.nodes[rank.outputs[0]].output.shape == (Dim("num_tokens"),)
    assert rank.nodes[rank.outputs[0]].output.dtype == I64

    allowed = (InputOp, ConstantOp, ElementwiseOp, IndexMapOp, GatherOp, ReduceOp)
    for graph in (embedding, local, rank):
        lowered = Pipeline.build(TENSOR_PASSES).run(graph)
        assert all(isinstance(node.op, allowed) for node in lowered.nodes.values())
        assert not {node.op.name for node in lowered.nodes.values() if isinstance(node.op, ElementwiseOp)}.intersection(
            {"zeros_like", "full_like"}
        )


def test_references_preserve_mask_zero_and_stable_global_ties():
    gathered = torch.tensor(
        [
            [[4.0, 7.0], [4.0, 100.0], [3.0, 200.0], [2.0, 300.0], [1.0, 400.0], [0.0, 500.0], [-1.0, 600.0], [-2.0, 700.0]],
            [[-2.0, 7.0], [-1.0, 100.0], [0.0, 200.0], [1.0, 300.0], [2.0, 400.0], [3.0, 500.0], [4.0, 600.0], [5.0, 700.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(_reference("rank_top1", None, (gathered,)), torch.tensor([7, 700], dtype=torch.int64))


def test_adapter_guards_exact_shapes_and_verifies_each_runtime_row():
    entries = []
    references = []

    def build(kind, rank):
        entry = _ProgramEntry(object(), ("gathered",), ("top",), kind, rank)
        entries.append(entry)
        return entry

    def run(entry, tensors, output, rows):
        output.copy_(torch.tensor([7, 700], dtype=torch.int64)[:rows])

    def reference(kind, rank, tensors):
        references.append((kind, rank, tensors))
        return torch.tensor([7, 700], dtype=torch.int64)[: tensors[0].shape[0]]

    adapter = _VocabCollectiveAdapter(
        build_program=build,
        run_program=run,
        platform_supported=lambda tensors: True,
        is_capturing=lambda: False,
        reference=reference,
    )
    gathered = torch.empty((2, _TP, 2), dtype=torch.float32)
    assert adapter.supported("rank_top1", None, (gathered,)) == 2
    assert adapter.supported("rank_top1", None, (gathered[:, :, :1],)) is None

    assert torch.equal(adapter.dispatch("rank_top1", None, gathered), torch.tensor([7, 700]))
    assert entries[0].verified_rows == {2}
    assert len(references) == 1
    adapter.dispatch("rank_top1", None, gathered)
    assert len(references) == 1


def test_capture_uses_only_the_program_verified_for_that_exact_row():
    references = []
    entry = _ProgramEntry(object(), ("gathered",), ("top",), "rank_top1", None, verified_rows={1})
    adapter = _VocabCollectiveAdapter(
        build_program=lambda kind, rank: entry,
        run_program=lambda entry, tensors, output, rows: output.fill_(5),
        platform_supported=lambda tensors: True,
        is_capturing=lambda: True,
        reference=lambda kind, rank, tensors: references.append(tensors) or torch.zeros(tensors[0].shape[0], dtype=torch.int64),
    )
    adapter._programs[("rank_top1", None)] = entry

    assert torch.equal(adapter.dispatch("rank_top1", None, torch.empty((1, _TP, 2), dtype=torch.float32)), torch.tensor([5]))
    with pytest.raises(_UseOriginal):
        adapter.dispatch("rank_top1", None, torch.empty((2, _TP, 2), dtype=torch.float32))
    assert references == []


def _layer(name, *, prefix, rank):
    layer = type(name, (), {})()
    layer.prefix = prefix
    layer.quant_method = type("UnquantizedEmbeddingMethod", (), {})()
    layer.tp_size = _TP
    layer.num_embeddings = _VOCAB
    layer.org_vocab_size = _VOCAB
    layer.num_embeddings_padded = _VOCAB
    layer.num_embeddings_per_partition = _LOCAL_VOCAB
    layer.embedding_dim = _HIDDEN
    layer.weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16, device="meta")
    start = rank * _LOCAL_VOCAB
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=start,
        org_vocab_end_index=start + _LOCAL_VOCAB,
        num_org_vocab_padding=0,
        added_vocab_start_index=_VOCAB,
        added_vocab_end_index=_VOCAB,
    )
    return layer


def test_layer_rank_accepts_only_the_exact_tp_shard_contract():
    embedding = _layer("VocabParallelEmbedding", prefix="model.embed_tokens", rank=5)
    lm_head = _layer("ParallelLMHead", prefix="lm_head", rank=5)
    assert _layer_rank(embedding, lm_head=False) == 5
    assert _layer_rank(lm_head, lm_head=True) == 5
    lm_head.shard_indices.org_vocab_end_index -= 1
    assert _layer_rank(lm_head, lm_head=True) is None


def test_installation_leaves_exactly_one_collective_between_compiler_leaves(monkeypatch):
    embedding_module = ModuleType("test_vocab")
    logits_module = ModuleType("test_logits")
    calls = []

    def embedding_forward(self, input_):
        calls.append("original_embedding")
        return input_

    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        calls.append("original_top1")
        return hidden_states

    embedding_cls = type("VocabParallelEmbedding", (), {"forward": embedding_forward})
    processor_cls = type("LogitsProcessor", (), {"get_top_tokens": get_top_tokens})
    embedding_module.VocabParallelEmbedding = embedding_cls
    logits_module.LogitsProcessor = processor_cls
    embedding_module.tensor_model_parallel_all_reduce = lambda tensor: calls.append("all_reduce") or tensor

    def all_gather(tensor, dim=-1):
        calls.append(("all_gather", dim))
        return tensor.repeat(1, _TP)

    logits_module.tensor_model_parallel_all_gather = all_gather

    class Adapter:
        def supported(self, kind, rank, tensors):
            return tensors[1].shape[0] if kind == "embedding" else tensors[0].shape[0]

        def prepare(self, kind, rank, rows):
            return True

    monkeypatch.setattr("emmy.serving.onecat_vocab._VocabCollectiveAdapter", Adapter)
    monkeypatch.setattr(
        "emmy.serving.onecat_vocab._custom_ops",
        lambda: {
            "embedding": lambda weight, input_ids, rank: torch.ones((input_ids.shape[0], _HIDDEN), dtype=torch.float16),
            "local_top1": lambda x, weight, rank: torch.tensor([[2.0, float(rank * _LOCAL_VOCAB + 7)]], dtype=torch.float32),
            "rank_top1": lambda gathered: gathered[:, 0, 1].to(torch.int64),
        },
    )
    assert register_onecat_vocab_kernels(embedding_module, logits_module)

    embedding = _layer("VocabParallelEmbedding", prefix="model.embed_tokens", rank=2)
    output = embedding_cls.forward(embedding, torch.tensor([2 * _LOCAL_VOCAB + 7]))
    assert output.shape == (1, _HIDDEN)

    lm_head = _layer("ParallelLMHead", prefix="lm_head", rank=2)
    processor = processor_cls()
    processor.scale = 1.0
    processor.soft_cap = None
    processor.logits_as_input = False
    processor.vocab_size = _VOCAB
    processor.org_vocab_size = _VOCAB
    assert torch.equal(
        processor_cls.get_top_tokens(processor, lm_head, torch.empty((1, _HIDDEN), dtype=torch.float16)), torch.tensor([32327])
    )
    assert calls == ["all_reduce", ("all_gather", -1)]


def test_rows_above_capacity_are_never_adopted():
    adapter = _VocabCollectiveAdapter(platform_supported=lambda tensors: True)
    gathered = torch.empty((_CAPACITY + 1, _TP, 2), dtype=torch.float32)
    assert adapter.supported("rank_top1", None, (gathered,)) is None


def test_registration_rejects_prior_wrappers_with_different_adapters():
    embedding_module = ModuleType("test_vocab_prior")
    logits_module = ModuleType("test_logits_prior")

    def embedding(*_args):
        return None

    def top1(*_args):
        return None

    embedding._emmy_onecat_vocab = True
    embedding._emmy_onecat_vocab_adapter = object()
    top1._emmy_onecat_vocab = True
    top1._emmy_onecat_vocab_adapter = object()
    embedding_module.VocabParallelEmbedding = type("VocabParallelEmbedding", (), {"forward": embedding})
    logits_module.LogitsProcessor = type("LogitsProcessor", (), {"get_top_tokens": top1})

    assert not register_onecat_vocab_kernels(embedding_module, logits_module)
