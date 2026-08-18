import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import torch

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, I64
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.tensor.ir import GatherOp
from emmy.serving.onecat_output import (
    _CAPACITY,
    _HIDDEN,
    _LOCAL_VOCAB,
    _SHARED_GATE_UP,
    _SHARED_INTERMEDIATE,
    _VOCAB,
    _clamp_swiglu_graph,
    _embedding_graph,
    _lm_head_graph,
    _OutputAdapter,
    _ProgramEntry,
    _reference,
    _run_external,
    register_onecat_output_kernels,
)


def _adapter(*, build_program=None, run_program=None, capturing=None, reference_value=1):
    references = {}
    reference_calls = []

    def reference(tensors):
        reference_calls.append(tensors)
        rows = tensors[0].shape[0]
        return tensors[0].new_full((rows, _SHARED_INTERMEDIATE), reference_value)

    references["clamp_swiglu"] = reference

    def default_builder(kind):
        return _ProgramEntry(object(), ("x",), "output", kind)

    def default_runner(entry, tensors, output, rows):
        output.fill_(reference_value)

    adapter = _OutputAdapter(
        build_program=build_program or default_builder,
        run_program=run_program or default_runner,
        platform_supported=lambda tensors: True,
        is_capturing=capturing or (lambda: False),
        references=references,
    )
    return adapter, reference_calls


def _modules(*, incompatible=False):
    embedding_module = ModuleType("test_vocab_parallel_embedding")
    activation_module = ModuleType("test_activation")
    calls = []

    class UnquantizedEmbeddingMethod:
        def embedding(self, layer, input_):
            calls.append(("embedding", self, layer, input_))
            return layer.weight.new_full((input_.shape[0], layer.embedding_dim), -1)

        if incompatible:

            def apply(self, layer, x):
                calls.append(("lm_head", self, layer, x, None))
                return x.new_full((x.shape[0], layer.num_embeddings_per_partition), -2)

        else:

            def apply(self, layer, x, bias=None):
                calls.append(("lm_head", self, layer, x, bias))
                return x.new_full((x.shape[0], layer.num_embeddings_per_partition), -2)

    class SiluAndMulWithClamp:
        def forward_cuda(self, x):
            calls.append(("clamp_swiglu", self, x))
            return x[..., : x.shape[-1] // 2].new_full((*x.shape[:-1], x.shape[-1] // 2), -3)

    embedding_module.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
    activation_module.SiluAndMulWithClamp = SiluAndMulWithClamp
    return embedding_module, activation_module, calls


def _layer(name, method, *, prefix):
    layer_cls = type(name, (), {})
    layer = layer_cls()
    layer.quant_method = method
    layer.prefix = prefix
    layer.tp_size = 8
    layer.num_embeddings = _VOCAB
    layer.num_embeddings_per_partition = _LOCAL_VOCAB
    layer.embedding_dim = _HIDDEN
    layer.weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16, device="meta")
    return layer


def _eligible_output_inputs(kind, weight, *, rows=2):
    if kind == "embedding":
        return (weight, torch.arange(rows, dtype=torch.int64))
    return (torch.empty((rows, _HIDDEN), dtype=torch.float16), weight)


def _family_adapter(kind, *, runner, capturing=lambda: False, reference_value=1):
    reference_calls = []
    expected_inputs = ("weight", "indices") if kind == "embedding" else ("x", "weight")

    def reference(tensors):
        reference_calls.append(tensors)
        rows = tensors[1].shape[0] if kind == "embedding" else tensors[0].shape[0]
        width = _HIDDEN if kind == "embedding" else _LOCAL_VOCAB
        return tensors[0].new_full((rows, width), reference_value)

    adapter = _OutputAdapter(
        build_program=lambda requested: _ProgramEntry(object(), expected_inputs, "output", requested),
        run_program=runner,
        platform_supported=lambda tensors: True,
        is_capturing=capturing,
        references={kind: reference},
    )
    return adapter, reference_calls, expected_inputs


def test_graphs_preserve_symbolic_tp_local_contracts_and_clamp_semantics():
    embedding = _embedding_graph()
    assert embedding.inputs == ["weight", "indices"]
    assert embedding.outputs == ["output"]
    assert embedding.nodes["weight"].output.shape == (_LOCAL_VOCAB, _HIDDEN)
    assert embedding.nodes["weight"].output.dtype == F16
    assert embedding.nodes["indices"].output.shape == (Dim("num_tokens"),)
    assert embedding.nodes["indices"].output.dtype == I64
    assert isinstance(embedding.nodes["output"].op, GatherOp)
    assert embedding.nodes["output"].output.shape == (Dim("num_tokens"), _HIDDEN)

    lm_head = _lm_head_graph()
    assert lm_head.inputs == ["x", "weight"]
    assert lm_head.outputs == ["output"]
    assert lm_head.nodes["x"].output.shape == (Dim("num_tokens"), _HIDDEN)
    assert lm_head.nodes["weight"].output.shape == (_LOCAL_VOCAB, _HIDDEN)
    assert isinstance(lm_head.nodes["output"].op, LinearOp)
    assert lm_head.nodes["output"].output.shape == (Dim("num_tokens"), _LOCAL_VOCAB)
    assert lm_head.nodes["output"].output.dtype == F16

    clamp = _clamp_swiglu_graph()
    assert clamp.inputs == ["x"]
    assert clamp.nodes["x"].output.shape == (Dim("num_tokens"), _SHARED_GATE_UP)
    assert clamp.nodes[clamp.outputs[0]].output.shape == (Dim("num_tokens"), _SHARED_INTERMEDIATE)
    assert clamp.nodes[clamp.outputs[0]].output.dtype == F16

    x = torch.tensor([[20.0, -20.0, 20.0, -20.0]], dtype=torch.float16)
    expected = torch.nn.functional.silu(torch.clamp(x[:, :2], max=10.0)) * torch.clamp(x[:, 2:], -10.0, 10.0)
    assert torch.equal(_reference("clamp_swiglu", (x,)), expected)


def test_adapter_accepts_only_exact_contiguous_sm70_abis():
    adapter, _ = _adapter()
    weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16, device="meta")

    for rows in (1, 17, _CAPACITY):
        indices = torch.empty((rows,), dtype=torch.int64, device="meta")
        assert adapter._supported("embedding", (weight, indices)) == rows

        x = torch.empty((rows, _HIDDEN), dtype=torch.float16, device="meta")
        assert adapter._supported("lm_head", (x, weight)) == rows

        shared = torch.empty((rows, _SHARED_GATE_UP), dtype=torch.float16, device="meta")
        assert adapter._supported("clamp_swiglu", (shared,)) == rows

    assert adapter._supported("embedding", (weight, torch.empty((_CAPACITY + 1,), dtype=torch.int64, device="meta"))) is None
    assert adapter._supported("embedding", (weight, torch.empty((34,), dtype=torch.int64, device="meta")[::2])) is None
    assert adapter._supported("lm_head", (torch.empty((1, _HIDDEN), dtype=torch.float32, device="meta"), weight)) is None
    assert adapter._supported("clamp_swiglu", (torch.empty((1, _SHARED_GATE_UP + 2), dtype=torch.float16, device="meta"),)) is None

    unsupported, _ = _adapter()
    unsupported._platform_supported = lambda tensors: False
    assert unsupported._supported("clamp_swiglu", (torch.empty((1, _SHARED_GATE_UP), dtype=torch.float16),)) is None


def test_first_use_parity_latches_and_reuses_caller_owned_output():
    runs = []

    def runner(entry, tensors, output, rows):
        runs.append((entry, tensors, output, rows))
        output.fill_(1)

    adapter, reference_calls = _adapter(run_program=runner)
    x = torch.empty((2, _SHARED_GATE_UP), dtype=torch.float16)

    output = adapter.dispatch("clamp_swiglu", x)
    assert output.shape == (2, _SHARED_INTERMEDIATE)
    assert output.dtype == torch.float16
    assert torch.count_nonzero(output != 1) == 0
    assert runs[0][2] is output
    assert runs[0][3] == 2
    assert len(reference_calls) == 1
    assert adapter._programs["clamp_swiglu"].verified

    adapter.dispatch("clamp_swiglu", x)
    assert len(runs) == 2
    assert len(reference_calls) == 1


def test_embedding_and_lm_head_first_use_parity_latch_and_reuse():
    weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16)
    for kind in ("embedding", "lm_head"):
        runs = []

        def runner(entry, tensors, output, rows, runs=runs):
            runs.append((entry, tensors, output, rows))
            output.fill_(1)

        adapter, reference_calls, _ = _family_adapter(kind, runner=runner)
        tensors = _eligible_output_inputs(kind, weight)
        assert adapter._supported(kind, tensors) == 2

        output = adapter.dispatch(kind, *tensors)
        assert output.shape == (2, _HIDDEN if kind == "embedding" else _LOCAL_VOCAB)
        assert runs[0][2] is output
        assert runs[0][3] == 2
        assert len(reference_calls) == 1
        assert adapter._programs[kind].verified

        adapter.dispatch(kind, *tensors)
        assert len(runs) == 2
        assert len(reference_calls) == 1


def test_embedding_and_lm_head_mismatch_permanently_fall_back():
    weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16)
    for kind in ("embedding", "lm_head"):
        runs = []

        def runner(entry, tensors, output, rows, runs=runs):
            runs.append(entry)
            output.zero_()

        adapter, reference_calls, _ = _family_adapter(kind, runner=runner)
        tensors = _eligible_output_inputs(kind, weight)

        first = adapter.dispatch(kind, *tensors)
        second = adapter.dispatch(kind, *tensors)

        assert torch.count_nonzero(first != 1) == 0
        assert torch.count_nonzero(second != 1) == 0
        assert len(runs) == 1
        assert len(reference_calls) == 2
        assert kind in adapter._disabled


def test_embedding_and_lm_head_capture_use_only_preverified_programs():
    weight = torch.empty((_LOCAL_VOCAB, _HIDDEN), dtype=torch.float16)
    for kind in ("embedding", "lm_head"):
        runs = []

        def runner(entry, tensors, output, rows, runs=runs):
            runs.append(entry)
            output.fill_(1)

        adapter, reference_calls, expected_inputs = _family_adapter(kind, runner=runner, capturing=lambda: True)
        tensors = _eligible_output_inputs(kind, weight)

        adapter.dispatch(kind, *tensors)
        assert runs == [] and len(reference_calls) == 1

        adapter._programs[kind] = _ProgramEntry(object(), expected_inputs, "output", kind)
        adapter.dispatch(kind, *tensors)
        assert runs == [] and len(reference_calls) == 2

        adapter._programs[kind].verified = True
        adapter.dispatch(kind, *tensors)
        assert len(runs) == 1 and len(reference_calls) == 2


def test_external_views_follow_symbol_update_inside_torch_stream_context(monkeypatch):
    events = []
    active = False
    torch_stream = object()

    class ExternalStream:
        def __enter__(self):
            nonlocal active
            active = True
            events.append("stream_enter")

        def __exit__(self, exc_type, exc, traceback):
            nonlocal active
            active = False
            events.append("stream_exit")

    class Stream:
        @staticmethod
        def from_external(stream):
            assert stream is torch_stream
            events.append("from_external")
            return ExternalStream()

    cupy = ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Stream=Stream)

    def from_dlpack(tensor):
        assert active
        events.append(f"dlpack:{tensor.name}")
        return tensor.name

    cupy.from_dlpack = from_dlpack
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: events.append("current_stream") or torch_stream)

    from emmy.compiler.backend import gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)

    class Runtime:
        def set_sym_values(self, values):
            assert active
            assert values == {"num_tokens": 17}
            events.append("set_sym_values")

        def run_once_external(self, bindings):
            assert active
            assert bindings == {"weight": "weight", "indices": "indices", "output": "output"}
            events.append("run")

    entry = _ProgramEntry(Runtime(), ("weight", "indices"), "output", "embedding")
    device = object()
    _run_external(
        entry,
        (SimpleNamespace(name="weight"), SimpleNamespace(name="indices")),
        SimpleNamespace(name="output", device=device),
        17,
    )

    assert events == [
        "current_stream",
        "from_external",
        "stream_enter",
        "set_sym_values",
        "dlpack:weight",
        "dlpack:indices",
        "dlpack:output",
        "run",
        "stream_exit",
    ]


def test_build_boundary_and_parity_failures_permanently_fall_back():
    build_calls = 0

    def wrong_builder(kind):
        nonlocal build_calls
        build_calls += 1
        return _ProgramEntry(object(), ("wrong",), "output", kind)

    adapter, reference_calls = _adapter(build_program=wrong_builder)
    x = torch.empty((1, _SHARED_GATE_UP), dtype=torch.float16)
    adapter.dispatch("clamp_swiglu", x)
    adapter.dispatch("clamp_swiglu", x)
    assert build_calls == 1
    assert len(reference_calls) == 2
    assert "clamp_swiglu" in adapter._disabled

    run_calls = 0

    def mismatching_runner(entry, tensors, output, rows):
        nonlocal run_calls
        run_calls += 1
        output.zero_()

    adapter, reference_calls = _adapter(run_program=mismatching_runner)
    adapter.dispatch("clamp_swiglu", x)
    adapter.dispatch("clamp_swiglu", x)
    assert run_calls == 1
    assert len(reference_calls) == 2
    assert "clamp_swiglu" in adapter._disabled


def test_capture_uses_only_a_preverified_program():
    built = []
    runs = []

    def builder(kind):
        built.append(kind)
        return _ProgramEntry(object(), ("x",), "output", kind)

    def runner(entry, tensors, output, rows):
        runs.append(entry)
        output.fill_(1)

    adapter, reference_calls = _adapter(build_program=builder, run_program=runner, capturing=lambda: True)
    x = torch.empty((1, _SHARED_GATE_UP), dtype=torch.float16)
    adapter.dispatch("clamp_swiglu", x)
    assert built == [] and runs == [] and len(reference_calls) == 1

    adapter._programs["clamp_swiglu"] = _ProgramEntry(object(), ("x",), "output", "clamp_swiglu")
    adapter.dispatch("clamp_swiglu", x)
    assert runs == [] and len(reference_calls) == 2

    adapter._programs["clamp_swiglu"].verified = True
    output = adapter.dispatch("clamp_swiglu", x)
    assert output.shape == (1, _SHARED_INTERMEDIATE)
    assert len(runs) == 1 and len(reference_calls) == 2


def test_registration_patches_all_three_aliases_and_preserves_bypasses():
    embedding_module, activation_module, calls = _modules()
    method_cls = embedding_module.UnquantizedEmbeddingMethod
    activation_cls = activation_module.SiluAndMulWithClamp
    originals = (method_cls.embedding, method_cls.apply, activation_cls.forward_cuda)
    compact_top1 = object()
    embedding_module.ParallelLMHead = SimpleNamespace(maybe_get_sm70_lm_head_top1=compact_top1)

    assert register_onecat_output_kernels(embedding_module, activation_module)
    replacements = (method_cls.embedding, method_cls.apply, activation_cls.forward_cuda)
    assert all(replacement is not original for replacement, original in zip(replacements, originals, strict=True))
    assert all(replacement._emmy_onecat_output_original is original for replacement, original in zip(replacements, originals, strict=True))
    assert embedding_module.ParallelLMHead.maybe_get_sm70_lm_head_top1 is compact_top1
    assert register_onecat_output_kernels(embedding_module, activation_module)
    assert (method_cls.embedding, method_cls.apply, activation_cls.forward_cuda) == replacements

    method = method_cls()
    embedding_layer = _layer("VocabParallelEmbedding", method, prefix="model.embed_tokens")
    embedding = method.embedding(embedding_layer, torch.empty((17,), dtype=torch.int64, device="meta"))
    assert embedding.shape == (17, _HIDDEN)
    assert embedding.dtype == torch.float16

    lm_head_layer = _layer("ParallelLMHead", method, prefix="lm_head")
    logits = method.apply(lm_head_layer, torch.empty((17, _HIDDEN), dtype=torch.float16, device="meta"))
    assert logits.shape == (17, _LOCAL_VOCAB)
    assert logits.dtype == torch.float16

    activation = activation_cls()
    activation.swiglu_limit = 10.0
    shared = activation.forward_cuda(torch.empty((17, _SHARED_GATE_UP), dtype=torch.float16, device="meta"))
    assert shared.shape == (17, _SHARED_INTERMEDIATE)
    assert shared.dtype == torch.float16
    assert calls == []

    embedding_layer.prefix = "other"
    fallback = method.embedding(embedding_layer, torch.empty((1,), dtype=torch.int64, device="meta"))
    assert fallback.shape == (1, _HIDDEN)
    activation.swiglu_limit = None
    fallback = activation.forward_cuda(torch.empty((1, _SHARED_GATE_UP), dtype=torch.float16, device="meta"))
    assert fallback.shape == (1, _SHARED_INTERMEDIATE)
    assert [call[0] for call in calls] == ["embedding", "clamp_swiglu"]


def test_incompatible_signature_leaves_every_alias_unchanged():
    embedding_module, activation_module, _ = _modules(incompatible=True)
    method_cls = embedding_module.UnquantizedEmbeddingMethod
    activation_cls = activation_module.SiluAndMulWithClamp
    originals = (method_cls.embedding, method_cls.apply, activation_cls.forward_cuda)

    assert not register_onecat_output_kernels(embedding_module, activation_module)
    assert (method_cls.embedding, method_cls.apply, activation_cls.forward_cuda) == originals
