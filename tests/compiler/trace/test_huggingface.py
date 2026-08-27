"""Tests for trace/huggingface.py wrapper builders.

Covers the synthetic Per-Layer-Embedding (PLE) support for Gemma-nano blocks
exposing ``hidden_size_per_layer_input``: ``build_layer_wrapper`` (dynamic mode)
registers a seeded synthetic ``per_layer_input`` buffer sliced in-graph to the
runtime seq_len (like cos/sin), and ``_trace_model``'s static single-layer path
passes the same buffer as a concrete kwarg. Every other architecture takes the
unchanged path (no ``ple`` buffer, no extra kwarg). The attention-split carve
(serving) instead rejects PLE blocks loudly — it has no seam for the multiply.
"""

import pytest

from emmy.compiler.trace.torch import has_torch

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch not available")

HIDDEN = 16
PLE_DIM = 4


def _fake_rotary(sample, full_pos):
    import torch

    n_pos = sample.shape[1]
    return torch.ones(1, n_pos, 8), torch.zeros(1, n_pos, 8)


def _ple_block():
    import torch.nn as nn

    class PleBlock(nn.Module):
        hidden_size_per_layer_input = PLE_DIM

        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, per_layer_input=None, position_embeddings=None):
            assert per_layer_input is not None
            self.seen.append(tuple(per_layer_input.shape))
            return x * per_layer_input.mean()

    return PleBlock()


def _plain_block():
    import torch.nn as nn

    class PlainBlock(nn.Module):
        # No ``per_layer_input`` in the signature: passing it would TypeError.
        def forward(self, x, position_embeddings=None):
            return x

    return PlainBlock()


def test_layer_wrapper_supplies_synthetic_ple():
    """A PLE block gets a registered ``ple`` buffer, sliced to each runtime seq_len."""
    import torch

    from emmy.compiler.trace.huggingface import build_layer_wrapper

    block = _ple_block()
    wrapper = build_layer_wrapper(block, _fake_rotary, HIDDEN, torch.float32)
    assert "ple" in dict(wrapper.named_buffers())
    for s in (8, 3):
        out = wrapper(torch.randn(1, s, HIDDEN))
        assert out.shape == (1, s, HIDDEN)
    assert block.seen == [(1, 8, PLE_DIM), (1, 3, PLE_DIM)]


def test_layer_wrapper_ple_buffer_deterministic_and_nonuniform():
    """Seeded buffer: independent builds agree (emmy and torch see the same values in
    the accuracy check), and it is non-uniform so the PLE mul can't fold to identity."""
    import torch

    from emmy.compiler.trace.huggingface import build_layer_wrapper

    w1 = build_layer_wrapper(_ple_block(), _fake_rotary, HIDDEN, torch.float32)
    w2 = build_layer_wrapper(_ple_block(), _fake_rotary, HIDDEN, torch.float32)
    ple1 = dict(w1.named_buffers())["ple"]
    torch.testing.assert_close(ple1, dict(w2.named_buffers())["ple"])
    assert ple1.std() > 0


def test_layer_wrapper_non_ple_block_unchanged():
    """A block without ``hidden_size_per_layer_input`` registers no ``ple`` buffer and
    is called without the kwarg (its forward signature would reject it)."""
    import torch

    from emmy.compiler.trace.huggingface import build_layer_wrapper

    wrapper = build_layer_wrapper(_plain_block(), _fake_rotary, HIDDEN, torch.float32)
    assert "ple" not in dict(wrapper.named_buffers())
    out = wrapper(torch.randn(1, 8, HIDDEN))
    assert out.shape == (1, 8, HIDDEN)


def _traceable_ple_block():
    """Export-traceable PLE block reproducing the gemma-nano crash shape: forward
    multiplies by ``per_layer_input`` unconditionally, so tracing without the kwarg
    dies on ``FakeTensor * None`` exactly like modeling_gemma4. ``ple_dim == HIDDEN``
    keeps the mul a plain broadcast-free op the FX→IR walker handles."""
    import torch.nn as nn

    class TraceablePleBlock(nn.Module):
        hidden_size_per_layer_input = HIDDEN

        def forward(self, x, per_layer_input=None, position_embeddings=None):
            return x * per_layer_input

    return TraceablePleBlock()


def _fake_causal_lm(block, rotary=_fake_rotary, *, config=None):
    """Minimal AutoModelForCausalLM stand-in for ``_trace_model``: a decoder module
    exposing ``layers`` / ``rotary_emb`` / ``config`` (what ``_find_text_decoder``
    matches) under ``.model``."""
    from types import SimpleNamespace

    import torch.nn as nn

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([block])
            self.config = config or SimpleNamespace(hidden_size=HIDDEN)
            self.rotary_emb = rotary

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    return FakeModel()


def test_static_layer_trace_supplies_synthetic_ple(monkeypatch):
    """``_trace_model``'s static (non-dynamic) single-layer path passes the seeded
    synthetic ``per_layer_input`` kwarg to a PLE block — without it the trace crashes
    on ``FakeTensor * None`` (the ``emmy run google/gemma-4-E2B --layer 0`` bug)."""
    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    block = _traceable_ple_block()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda model_id, **kw: _fake_causal_lm(block))

    seq_len = 8
    graph, (mod, args, kwargs) = _trace_model("fake/ple-model", 0, seq_len)
    assert graph.nodes
    ple = kwargs["per_layer_input"]
    assert tuple(ple.shape) == (1, seq_len, HIDDEN)
    assert ple.std() > 0  # non-uniform: the PLE mul can't fold to identity


def test_static_layer_trace_non_ple_block_unchanged(monkeypatch):
    """A non-PLE block traces through the static path without the extra kwarg."""
    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda model_id, **kw: _fake_causal_lm(_plain_block()))

    graph, (mod, args, kwargs) = _trace_model("fake/plain-model", 0, 8)
    assert graph.nodes
    assert "per_layer_input" not in kwargs


def test_static_layer_trace_builds_multi_rope_mapping(monkeypatch):
    """DeepSeek-style rotary modules expose table keys unrelated to attention labels;
    trace all declared tables and pass the mapping the decoder block expects."""
    import torch
    import torch.nn as nn

    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    class MultiRotary(nn.Module):
        layer_types = ("main", "compress")

        def forward(self, x, position_ids, layer_type=None):
            value = 1.0 if layer_type == "main" else 2.0
            shape = (x.shape[0], x.shape[1], x.shape[-1] // 2)
            return torch.full(shape, value, dtype=x.dtype), torch.zeros(shape, dtype=x.dtype)

    class MultiRopeBlock(nn.Module):
        def forward(self, x, position_embeddings=None):
            return x + position_embeddings["main"][0].mean() + position_embeddings["compress"][0].mean()

    block = MultiRopeBlock()
    rotary = MultiRotary()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kw: _fake_causal_lm(block, rotary),
    )

    graph, (_mod, _args, kwargs) = _trace_model("fake/multi-rope-model", 0, 8)
    assert graph.nodes
    assert set(kwargs["position_embeddings"]) == {"main", "compress"}


def test_static_layer_trace_selects_laguna_rope_tuple_from_config(monkeypatch):
    """Laguna's configured attention type selects one RoPE tuple, not the whole
    multi-RoPE mapping used by DeepSeek-style decoder blocks."""
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    class LagunaRotary(nn.Module):
        layer_types = ("full_attention", "sliding_attention")

        def forward(self, x, position_ids, layer_type=None):
            assert layer_type in self.layer_types
            value = 1.0 if layer_type == "full_attention" else 2.0
            shape = (x.shape[0], x.shape[1], x.shape[-1] // 2)
            return torch.full(shape, value, dtype=x.dtype), torch.zeros(shape, dtype=x.dtype)

    class LagunaBlock(nn.Module):
        def forward(self, x, position_embeddings=None):
            assert isinstance(position_embeddings, tuple)
            return x + position_embeddings[0].mean()

    config = SimpleNamespace(hidden_size=HIDDEN, layer_types=["sliding_attention"])
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kw: _fake_causal_lm(LagunaBlock(), LagunaRotary(), config=config),
    )

    graph, (_mod, _args, kwargs) = _trace_model("fake/laguna-model", 0, 8)
    assert graph.nodes
    assert isinstance(kwargs["position_embeddings"], tuple)
    assert kwargs["position_embeddings"][0].mean().item() == 2.0


def test_static_layer_trace_preserves_hyper_connection_lanes(monkeypatch):
    """DeepSeek V4 blocks consume ``[B, S, hc_mult, H]`` state, not the generic
    ``[B, S, H]`` layer input."""
    from types import SimpleNamespace

    import torch.nn as nn

    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    class HyperConnectionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_hc = SimpleNamespace(hc_mult=4)

        def forward(self, x, position_embeddings=None):
            return x

    block = HyperConnectionBlock()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kw: _fake_causal_lm(block),
    )

    _graph, (_mod, args, _kwargs) = _trace_model("fake/hyper-connection-model", 0, 8)
    assert tuple(args[0].shape) == (1, 8, 4, HIDDEN)


def test_static_layer_trace_supplies_declared_attention_inputs(monkeypatch):
    """Model wrappers normally provide position IDs and the optional causal mask.
    Preserve them for attention implementations that declare the parameters as
    required even when concrete rotary embeddings have already been computed."""
    import torch
    import torch.nn as nn

    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    class RequiredInputsAttention(nn.Module):
        def forward(self, x, position_embeddings, position_ids, attention_mask):
            assert attention_mask is None
            return x + position_ids.unsqueeze(-1).to(x.dtype) * 0

    class RequiredInputsBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = RequiredInputsAttention()

        def forward(self, x, input_ids=None, position_embeddings=None, **kwargs):
            assert input_ids is not None
            return self.self_attn(x, position_embeddings=position_embeddings, **kwargs)

    block = RequiredInputsBlock()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kw: _fake_causal_lm(block),
    )

    graph, (_mod, _args, kwargs) = _trace_model("fake/required-attention-inputs", 0, 8)
    assert graph.nodes
    torch.testing.assert_close(kwargs["position_ids"], torch.arange(8).unsqueeze(0))
    assert kwargs["attention_mask"] is None
    torch.testing.assert_close(kwargs["input_ids"], torch.zeros((1, 8), dtype=torch.long))


def test_static_layer_trace_omits_optional_none_attention_mask(monkeypatch):
    """An optional None is call policy, not a tensor input to compile/bind."""
    import torch.nn as nn

    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    class OptionalMaskAttention(nn.Module):
        def forward(self, x, position_embeddings, attention_mask=None):
            assert attention_mask is None
            return x

    class OptionalMaskBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = OptionalMaskAttention()

        def forward(self, x, position_embeddings=None, **kwargs):
            return self.self_attn(x, position_embeddings=position_embeddings, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda model_id, **kw: _fake_causal_lm(OptionalMaskBlock()))
    graph, (_mod, _args, kwargs) = _trace_model("fake/optional-attention-mask", 0, 8)
    assert graph.nodes
    assert "attention_mask" not in kwargs


def test_attention_split_rejects_ple_block():
    """The serving carve has no seam for the PLE multiply — it must reject loudly,
    not silently drop the ``per_layer_input`` term."""
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    with pytest.raises(NotImplementedError, match="hidden_size_per_layer_input"):
        build_attention_split_wrapper(_ple_block())


def test_trace_inventory_replaces_router_with_representative_expert():
    """MoE inventory keeps expert/shared compute without tracing top-k/sort routing."""
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from emmy.compiler.trace.huggingface import replace_moe_with_traceable_expert

    hidden, intermediate, experts_count = 4, 3, 2

    class Experts(nn.Module):
        is_transposed = False
        is_concatenated = True
        has_bias = False

        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(experts_count, 2 * intermediate, hidden))
            self.down_proj = nn.Parameter(torch.randn(experts_count, hidden, intermediate))
            self.act_fn = nn.SiLU()

    class Shared(nn.Module):
        def forward(self, x):
            return x * 0.25

    expert_module = Experts()
    shared = Shared()
    original_mlp = SimpleNamespace(
        gate=nn.Linear(hidden, experts_count),
        experts=expert_module,
        shared_experts=shared,
        routed_scaling_factor=2.5,
    )
    block = SimpleNamespace(mlp=original_mlp)
    x = torch.randn(2, hidden)
    gate, up = F.linear(x, expert_module.gate_up_proj[0]).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, expert_module.down_proj[0]) * 2.5 + shared(x)

    assert replace_moe_with_traceable_expert(block)
    torch.testing.assert_close(block.mlp(x, input_ids=torch.zeros(2, dtype=torch.long)), expected)
    assert block.mlp.w_gate_up.untyped_storage().data_ptr() == expert_module.gate_up_proj.untyped_storage().data_ptr()


# ===================================================================
# Quantized-twin state-dict adapters: encode-padding trim + per-expert packing
# ===================================================================


def test_quantized_architecture_loader_trusts_remote_code_only_when_required(monkeypatch):
    """Custom quantized architectures get one explicit retry, while the default call
    remains the only path for built-in architectures."""
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import _auto_config_from_pretrained, _auto_model_from_config

    config_calls = []
    model_calls = []
    config = object()
    model = object()

    def from_pretrained(path, **kwargs):
        config_calls.append((path, kwargs))
        if not kwargs.get("trust_remote_code"):
            raise ValueError("set trust_remote_code=True to load this configuration")
        return config

    def from_config(actual_config, **kwargs):
        model_calls.append((actual_config, kwargs))
        if not kwargs.get("trust_remote_code"):
            raise ValueError("set trust_remote_code=True to load this model")
        return model

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", from_config)

    assert _auto_config_from_pretrained("custom-model") is config
    assert _auto_model_from_config(config, dtype="float16") is model
    assert config_calls == [("custom-model", {}), ("custom-model", {"trust_remote_code": True})]
    assert model_calls == [
        (config, {"dtype": "float16"}),
        (config, {"trust_remote_code": True, "dtype": "float16"}),
    ]


def test_quantized_architecture_loader_does_not_trust_other_value_errors(monkeypatch):
    """Unrelated configuration failures retain their original exception and do not
    authorize repository code execution."""
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import _auto_config_from_pretrained

    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        raise ValueError("invalid architecture")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    with pytest.raises(ValueError, match="invalid architecture"):
        _auto_config_from_pretrained("broken-model")
    assert calls == [("broken-model", {})]


def test_quantized_twin_constructs_directly_in_trace_dtype(monkeypatch, tmp_path):
    """Large quantized twins must never allocate an fp32 model before converting it.

    DeepSeek V4's fp32 architecture twin is roughly a terabyte by itself.  Passing
    the dtype to ``from_config`` avoids a second full-model allocation during
    ``.to(dtype)`` and keeps the trace path within host memory.
    """
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.loader import quant
    from emmy.compiler.trace.huggingface import load_quantized_twin

    seen = {}

    class FakeModel:
        def state_dict(self):
            return {}

        def load_state_dict(self, state, *, strict):
            assert state == {}
            assert strict is False
            return [], []

        def tie_weights(self):
            return None

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            raise AssertionError("quantized twin performed a post-construction dtype copy")

    config = SimpleNamespace(quantization_config={"quant_method": "fp8"})
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _path: config)

    def from_config(actual_config, **kwargs):
        seen.update(kwargs)
        assert not hasattr(actual_config, "quantization_config")
        return FakeModel()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", from_config)
    monkeypatch.setattr(quant, "load_dequantized_state_dict", lambda _path: {})

    assert isinstance(load_quantized_twin(tmp_path, torch.float16), FakeModel)
    assert seen == {"dtype": torch.float16}


def test_quantized_trace_twin_materializes_only_requested_layer(tmp_path):
    """The inventory-only loader preserves the original layer index while keeping all
    untraced blocks on meta, so host memory is proportional to one layer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_trace_twin

    config = transformers.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        vocab_size=32,
    )
    config.quantization_config = {"quant_method": "fp8"}
    config.save_pretrained(tmp_path)

    model = load_quantized_trace_twin(tmp_path, torch.float16, 1)
    decoder = model.model
    assert next(decoder.layers[1].parameters()).device.type == "cpu"
    assert next(decoder.layers[1].parameters()).dtype == torch.float16
    assert next(decoder.layers[0].parameters()).device.type == "meta"
    assert next(decoder.layers[2].parameters()).device.type == "meta"
    assert next(decoder.rotary_emb.buffers()).device.type == "cpu"


def test_quantized_layer_twin_streams_only_requested_value_layer(monkeypatch, tmp_path):
    """The runnable layer lane must not decode the other 31 decoder blocks."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    import emmy.compiler.trace.huggingface as huggingface

    class Rotary(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.register_buffer("inv_freq", torch.ones(2))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
            self.decoder.config = SimpleNamespace()
            self.decoder.rotary_emb = Rotary(self.decoder.config)
            self.config = self.decoder.config

    model = Model()
    seen = {}

    def split(path, dtype, **kwargs):
        seen.update(path=path, dtype=dtype, **kwargs)
        return model, {}

    monkeypatch.setattr(huggingface, "load_quantized_split", split)
    got = huggingface.load_quantized_layer_twin(tmp_path, torch.float16, 1)
    assert got is model
    assert seen == {
        "path": tmp_path,
        "dtype": torch.float16,
        "layer_range": (1, 2),
        "include_embed": False,
        "include_norm": False,
    }
    assert next(model.decoder.rotary_emb.buffers()).device.type == "cpu"


def test_architecture_trace_twin_replaces_laguna_experts_before_materialization(tmp_path):
    """A selected sparse Laguna layer retains one expert, its routed scale, and
    shared-expert compute; the packed all-expert tensor never reaches CPU."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_architecture_trace_twin

    config = transformers.LagunaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        sliding_window=8,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["full_attention", "sliding_attention"],
        num_attention_heads_per_layer=[4, 4],
        mlp_layer_types=["dense", "sparse"],
        moe_routed_scaling_factor=2.5,
    )
    config.save_pretrained(tmp_path)

    model = load_architecture_trace_twin(tmp_path, torch.float16, 1)
    decoder = model.model
    mlp = decoder.layers[1].mlp
    assert mlp._emmy_traceable_expert
    assert mlp.routed_scaling_factor == 2.5
    assert tuple(mlp.w_gate_up.shape) == (12, 16)
    assert tuple(mlp.w_down.shape) == (16, 6)
    assert mlp.w_gate_up.device.type == "cpu"
    assert next(decoder.layers[0].parameters()).device.type == "meta"
    assert not hasattr(mlp, "experts")


def test_representative_deepseek_expert_preserves_clamped_swiglu_eager_and_trace():
    import torch
    import torch.nn as nn

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.compiler.trace.huggingface import replace_moe_with_traceable_expert
    from emmy.compiler.trace.torch import trace_module

    class Experts(nn.Module):
        is_transposed = False
        is_concatenated = True
        has_bias = False
        limit = 10.0

        def __init__(self):
            super().__init__()
            self.act_fn = nn.functional.silu
            self.gate_up_proj = nn.Parameter(torch.tensor([[[20.0, 0.0], [0.0, 20.0], [20.0, 0.0], [0.0, -20.0]]]))
            self.down_proj = nn.Parameter(torch.eye(2).unsqueeze(0))

    class Routed(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Identity()
            self.experts = Experts()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Routed()

    block = Block()
    assert replace_moe_with_traceable_expert(block)
    x = torch.ones(1, 2)
    want = nn.functional.silu(torch.full((1, 2), 10.0)) * torch.tensor([[10.0, -10.0]])
    torch.testing.assert_close(block.mlp(x), want)

    graph = trace_module(block.mlp, (x,))
    clamp_ops = [node.op.name for node in graph.nodes.values() if isinstance(node.op, ElementwiseOp)]
    assert clamp_ops.count("minimum") >= 2
    assert clamp_ops.count("maximum") >= 1


def test_deepseek_architecture_trace_fails_if_representative_expert_is_not_confirmed(monkeypatch, tmp_path):
    import torch
    import transformers

    import emmy.compiler.trace.huggingface as huggingface

    config = transformers.DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        o_groups=1,
        o_lora_rank=8,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        layer_types=["heavily_compressed_attention"],
        mlp_layer_types=["moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
    )
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(huggingface, "replace_moe_with_traceable_expert", lambda _block: False)
    with pytest.raises(NotImplementedError, match="confirmed representative routed-expert replacement"):
        huggingface.load_architecture_trace_twin(tmp_path, torch.float16, 0)


def test_deepseek_hca_and_csa_stamp_actual_attention_sliding_window():
    from types import SimpleNamespace

    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.frontend.ir import SdpaOp
    from emmy.compiler.trace.huggingface import stamp_sliding_windows

    for layer_type in ("heavily_compressed_attention", "compressed_sparse_attention"):
        graph = Graph()
        node_id = graph.add_node(op=SdpaOp(), inputs=[], output=Tensor("attention", (1, 2, 8, 8), "float16"))
        config = SimpleNamespace(model_type="deepseek_v4", sliding_window=999)
        stamp_sliding_windows(graph, config, layer_type=layer_type, sliding_window=128)
        assert graph.nodes[node_id].op.sliding_window == 128
        assert graph.nodes[node_id].op.is_causal


def test_deepseek_csa_audit_rejects_genuinely_selective_topk():
    from types import SimpleNamespace

    from emmy.compiler.trace.huggingface import specialize_deepseek_full_coverage_compressor

    block = SimpleNamespace(
        self_attn=SimpleNamespace(
            compressor=SimpleNamespace(
                compress_rate=4,
                indexer=SimpleNamespace(compress_rate=4, index_topk=1),
            )
        )
    )
    with pytest.raises(NotImplementedError, match="cannot replace a selective top-k indexer"):
        specialize_deepseek_full_coverage_compressor(block, seq_len=8)


def test_deepseek_csa_audit_rejects_mismatched_compressor_rates():
    from types import SimpleNamespace

    from emmy.compiler.trace.huggingface import specialize_deepseek_full_coverage_compressor

    block = SimpleNamespace(
        self_attn=SimpleNamespace(
            compressor=SimpleNamespace(
                compress_rate=4,
                indexer=SimpleNamespace(compress_rate=8, index_topk=8),
            )
        )
    )
    with pytest.raises(ValueError, match="compressor and indexer to enumerate the same entries"):
        specialize_deepseek_full_coverage_compressor(block, seq_len=8)


def test_deepseek_csa_selected_layer_requires_full_coverage_specialization(monkeypatch):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    import emmy.compiler.trace.huggingface as huggingface

    class Attention(nn.Module):
        sliding_window = 4

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])
            self.rotary_emb = nn.Identity()
            self.config = SimpleNamespace(
                model_type="deepseek_v4",
                hidden_size=8,
                layer_types=["compressed_sparse_attention"],
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

    monkeypatch.setattr(huggingface, "specialize_deepseek_full_coverage_compressor", lambda _block, _seq_len: False)
    with pytest.raises(NotImplementedError, match="requires the confirmed full-coverage compressor specialization"):
        huggingface.trace_selected_layer(Model(), 0, 8, torch.float16)


def test_architecture_only_trace_does_not_load_checkpoint_weights(monkeypatch, tmp_path):
    """Unquantized inventory tracing goes through AutoConfig/from_config and must
    never call ``from_pretrained``, which would download the source weights."""
    transformers = pytest.importorskip("transformers")

    from emmy.commands.compile import _trace_model

    config = transformers.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
    )
    config.save_pretrained(tmp_path)

    def reject_weight_load(*_args, **_kwargs):
        raise AssertionError("architecture-only trace attempted to load checkpoint weights")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", reject_weight_load)
    graph, _bundle = _trace_model(str(tmp_path), 0, 4, architecture_only=True)
    assert graph.nodes


class _FakeStateModel:
    """Duck-typed stand-in: the two adapters read only ``model.state_dict()``."""

    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


def test_trim_padded_weights_slices_exact_roundups_only():
    """A decoded value overhanging its parameter trims only when every overhang is the
    declared dim's exact roundup to 128 (EXL3's encode padding); anything else stays for
    ``load_state_dict`` to report."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import _trim_padded_weights

    model = _FakeStateModel({"a.weight": torch.empty(100, 64), "b.weight": torch.empty(100, 64)})
    padded = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64)
    state = {"a.weight": padded.clone(), "b.weight": torch.zeros(120, 64)}  # 120 is not roundup128(100)
    _trim_padded_weights(model, state)
    assert tuple(state["a.weight"].shape) == (100, 64)
    torch.testing.assert_close(state["a.weight"], padded[:100], rtol=0, atol=0)
    assert tuple(state["b.weight"].shape) == (120, 64)


def test_pack_expert_state_stacks_per_expert_weights():
    """Per-expert ``experts.E.{gate,up,down}_proj.weight`` values (the DeepSeek/GLM checkpoint
    lineage) pack into the v5 3-D params: gate/up concatenated along the out axis, experts
    stacked on axis 0; the per-expert entries are consumed."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import _pack_expert_state

    e_count, inter, hidden = 2, 3, 4
    model = _FakeStateModel(
        {
            "m.experts.gate_up_proj": torch.empty(e_count, 2 * inter, hidden),
            "m.experts.down_proj": torch.empty(e_count, hidden, inter),
        }
    )
    gates = [torch.randn(inter, hidden) for _ in range(e_count)]
    ups = [torch.randn(inter, hidden) for _ in range(e_count)]
    downs = [torch.randn(hidden, inter) for _ in range(e_count)]
    state = {}
    for e in range(e_count):
        state[f"m.experts.{e}.gate_proj.weight"] = gates[e]
        state[f"m.experts.{e}.up_proj.weight"] = ups[e]
        state[f"m.experts.{e}.down_proj.weight"] = downs[e]
    _pack_expert_state(model, state)
    assert set(state) == {"m.experts.gate_up_proj", "m.experts.down_proj"}
    torch.testing.assert_close(state["m.experts.gate_up_proj"][1, :inter], gates[1], rtol=0, atol=0)
    torch.testing.assert_close(state["m.experts.gate_up_proj"][0, inter:], ups[0], rtol=0, atol=0)
    torch.testing.assert_close(state["m.experts.down_proj"][1], downs[1], rtol=0, atol=0)


def test_pack_expert_state_leaves_partial_sets_alone():
    """An incomplete per-expert set (or an already-packed checkpoint) stays as-is for
    ``load_state_dict`` to report."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import _pack_expert_state

    model = _FakeStateModel({"m.experts.down_proj": torch.empty(2, 4, 3)})
    state = {"m.experts.0.down_proj.weight": torch.randn(4, 3)}  # expert 1 missing
    _pack_expert_state(model, state)
    assert set(state) == {"m.experts.0.down_proj.weight"}


def test_pack_expert_state_shape_mismatch_raises():
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import _pack_expert_state

    model = _FakeStateModel({"m.experts.down_proj": torch.empty(1, 4, 3)})
    state = {"m.experts.0.down_proj.weight": torch.randn(3, 4)}  # transposed vs expected
    with pytest.raises(ValueError, match="expert packing"):
        _pack_expert_state(model, state)


def test_expert_slot_reads_per_expert_fp8_modules_and_stacks_them():
    """DeepSeek / Laguna store routed experts one MODULE per expert — ``experts.<e>.<proj>.weight``
    with a block ``weight_scale_inv`` — not the transformers-v5 E-stacked 3-D tensors. They map
    to the same E-leading program inputs: ``w_gate_up`` is ``[gate | up]`` along the output axis
    (the de-interleaved convention the expert wrapper's ``chunk(2)`` reads), ``w_down`` as
    stored, block scales concatenated alike, fp8 bits on the ``uint8`` carrier."""
    import torch

    from emmy.compiler.trace.huggingface import _expert_slot, _stack_expert_modules

    assert _expert_slot("model.layers.12.mlp.experts.3.gate_proj.weight") == (12, "w_gate", 3)
    assert _expert_slot("model.layers.12.mlp.experts.3.up_proj.weight_scale_inv") == (12, "w_up_scale", 3)
    assert _expert_slot("model.layers.12.mlp.experts.0.down_proj.weight") == (12, "w_down", 0)
    assert _expert_slot("model.layers.12.mlp.experts.0.down_proj.bias") is None

    e, inter, hidden = 2, 4, 8
    by_name = {
        "w_gate": {i: torch.full((inter, hidden), 10 * i + 1, dtype=torch.uint8) for i in range(e)},
        "w_up": {i: torch.full((inter, hidden), 10 * i + 2, dtype=torch.uint8) for i in range(e)},
        "w_down": {i: torch.full((hidden, inter), 10 * i + 3, dtype=torch.uint8) for i in range(e)},
        "w_gate_scale": {i: torch.full((1, 1), float(i + 1)) for i in range(e)},
        "w_up_scale": {i: torch.full((1, 1), float(i + 5)) for i in range(e)},
        "w_down_scale": {i: torch.full((1, 1), float(i + 9)) for i in range(e)},
    }
    out = _stack_expert_modules(12, by_name, model=None)
    assert set(out) == {"w_gate_up", "w_down", "w_gate_up_scale", "w_down_scale"}
    assert out["w_gate_up"].shape == (e, 2 * inter, hidden) and out["w_gate_up"].dtype == torch.uint8
    assert out["w_gate_up"][1, 0, 0] == 11 and out["w_gate_up"][1, inter, 0] == 12, "[gate | up] halves per expert"
    assert out["w_down"].shape == (e, hidden, inter)
    assert out["w_gate_up_scale"].shape == (e, 2, 1) and out["w_gate_up_scale"][1].flatten().tolist() == [2.0, 6.0]
    assert out["w_down_scale"][0].item() == 9.0


# --- Qwen3.5 linear-attention split: the traced/lowered half ----------------------------------
# The eager numerics live in ``tests/serving/test_linear_attention_split.py``; what matters here is
# that both halves of the carve survive ``torch.export`` and reach Loop IR, since that is the whole
# point of carving them out of a recurrence torch keeps.

_QWEN3_5_TINY = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    max_position_embeddings=64,
    layer_types=["linear_attention", "full_attention"],
)


def _qwen3_5_linear_block():
    import pytest
    import torch

    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**_QWEN3_5_TINY)).eval()
    return model.layers[0]


def test_linear_attention_split_pre_and_post_trace_and_lower():
    """Both carve halves export and lower: ``pre`` is the four input projections, ``post`` the
    output projection plus the layer's norm/MLP tail. Structure only — the lowered loops are not
    executed here."""
    import torch

    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper
    from emmy.compiler.trace.torch import trace_module

    block = _qwen3_5_linear_block()
    mixer = block.linear_attn
    pre, post = build_linear_attention_split_wrapper(block)

    t, h = 6, _QWEN3_5_TINY["hidden_size"]
    pre_graph = trace_module(pre, (torch.randn(t, h),))
    post_graph = trace_module(post, (torch.randn(t, mixer.value_dim), torch.randn(t, h)))

    # Four projections in, one out — the packed weights this carve exists to compile.
    assert len(pre_graph.outputs) == 4
    assert len(post_graph.outputs) == 1

    for graph in (pre_graph, post_graph):
        lowered = Pipeline.build(LOOP_PASSES).run(graph)
        assert any(type(node.op).__name__ == "LoopOp" for node in lowered.nodes.values())


def test_linear_attention_split_pre_traces_a_dynamic_token_count():
    """Serving packs a variable number of tokens into the flat axis, so the carve must export with
    that axis symbolic — the same dynamic-shapes argument the layer wrapper takes."""
    import torch

    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper
    from emmy.compiler.trace.torch import trace_module

    block = _qwen3_5_linear_block()
    pre, _ = build_linear_attention_split_wrapper(block)
    graph = trace_module(
        pre,
        (torch.randn(6, _QWEN3_5_TINY["hidden_size"]),),
        dynamic_shapes={"hidden": {0: torch.export.Dim("num_tokens", min=2, max=1024)}},
    )
    dims = {str(d) for out in graph.outputs for d in graph.nodes[out].output.shape}
    assert any(not str(d).isdigit() for d in dims), f"no symbolic token axis survived: {dims}"


# --- checkpoint keys vs twin parameter names ---------------------------------------------------
# A checkpoint may store its tensors under names the config-built twin does not have, and the
# mismatch is silent — the parameters simply stay on the meta device. Transformers registers the
# per-family translation and applies it inside ``from_pretrained``; emmy's shard-streamed loader
# does not take that path, so it reads the same table.


def _qwen3_5_multimodal_config():
    import pytest

    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(text_config=_QWEN3_5_TINY)


def _graph_with_input(name: str, *, consumed: bool):
    """A one-input graph, with that input either read by a node or left dangling."""
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("hidden", (4,), "f16"), node_id="hidden")
    g.add_node(InputOp(), [], Tensor(name, (4,), "f32"), node_id=name)
    g.inputs = ["hidden", name]
    src = name if consumed else "hidden"
    g.add_node(ElementwiseOp(op="copy"), [src], Tensor("out", (4,), "f16"), node_id="out")
    g.outputs = ["out"]
    return g


def test_a_none_kwarg_placeholder_stops_being_a_graph_input():
    """``torch.export`` records a ``None`` keyword argument as a scalar placeholder input, while the
    eager flattener drops the ``None`` — so the graph declares one more input than any caller
    supplies, and positional binding fails on the arity. Qwen3.5-family layers hit this because
    their blocks REQUIRE ``attention_mask``, so it cannot simply be omitted at the call."""
    from emmy.compiler.trace.huggingface import _drop_none_kwarg_inputs

    g = _graph_with_input("attention_mask", consumed=False)
    _drop_none_kwarg_inputs(g, {"attention_mask": None})
    assert g.inputs == ["hidden"], "the inert placeholder must not stay an input"
    assert "attention_mask" not in g.nodes


def test_a_mask_the_block_actually_reads_is_never_dropped():
    """The prune keys on having no consumers, not on the name — so a real mask survives."""
    from emmy.compiler.trace.huggingface import _drop_none_kwarg_inputs

    g = _graph_with_input("attention_mask", consumed=True)
    _drop_none_kwarg_inputs(g, {"attention_mask": None})
    assert "attention_mask" in g.inputs


def test_a_supplied_kwarg_is_never_dropped():
    """Only ``None`` values name a placeholder; a real tensor argument is left alone."""
    from emmy.compiler.trace.huggingface import _drop_none_kwarg_inputs

    g = _graph_with_input("attention_mask", consumed=False)
    _drop_none_kwarg_inputs(g, {"attention_mask": object()})
    assert "attention_mask" in g.inputs


def test_checkpoint_key_renamer_bridges_a_multimodal_prefix():
    """The Qwen3.5 releases are vision-language wrappers, so the text decoder's weights sit one
    module deeper in the checkpoint than in a text-only twin. The renamer closes that gap, and the
    name it produces is one the twin actually has."""
    import torch

    from emmy.compiler.trace.huggingface import _auto_model_from_config, _checkpoint_key_renamer

    with torch.device("meta"):
        twin = _auto_model_from_config(_qwen3_5_multimodal_config())
    rename = _checkpoint_key_renamer(twin)
    assert rename is not None, "the registered qwen3_5_text mapping must be found"

    owned = dict(twin.named_parameters())
    checkpoint_key = "model.language_model.layers.0.input_layernorm.weight"
    assert checkpoint_key not in owned, "the fixture must reproduce the mismatch, not hide it"
    assert rename(checkpoint_key) in owned
    # Already-canonical keys and model-level ones pass through untouched.
    assert rename("model.layers.0.input_layernorm.weight") == "model.layers.0.input_layernorm.weight"
    assert rename("lm_head.weight") == "lm_head.weight"


def test_checkpoint_key_renamer_leaves_a_plain_text_model_alone():
    """A family whose checkpoint names already are its parameter names gets no translation, so
    every existing loader path is byte-identical."""
    import torch
    import transformers

    from emmy.compiler.trace.huggingface import _checkpoint_key_renamer

    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
    )  # fmt: skip
    with torch.device("meta"):
        twin = transformers.Qwen3ForCausalLM(config)
    rename = _checkpoint_key_renamer(twin)
    key = "model.layers.0.self_attn.q_proj.weight"
    assert rename is None or rename(key) == key


def test_checkpoint_to_model_key_applies_the_laguna_literals():
    from emmy.compiler.trace.huggingface import _checkpoint_to_model_key

    laguna = "model.layers.0.mlp.shared_expert.gate_proj.weight"
    assert _checkpoint_to_model_key(laguna) == "model.layers.0.mlp.shared_experts.gate_proj.weight"
    assert _checkpoint_to_model_key("a.weight") == "a.weight"


def test_the_split_load_runs_both_key_translations(tmp_path):
    """Two independent renamers stand between a checkpoint key and the twin's parameter name, and a
    checkpoint can need both: the native one undoes an architecture published in its own flat
    namespace, the family one places the result where THIS twin holds its parameters.

    Running only one is silent. An unmatched name raises nothing — the parameter simply stays on the
    meta device and the twin comes back looking complete. This fixture is a wrapper-prefixed family,
    so the family renamer is the one that must fire; the native renamer is the identity here and
    must not displace it."""
    import json

    import torch

    from emmy.compiler.trace.huggingface import _auto_model_from_config, load_quantized_split

    torch.manual_seed(0)
    twin = _auto_model_from_config(_qwen3_5_multimodal_config()).eval()
    leaf = "model.layers.0.mlp.gate_proj.weight"
    _wrapper_prefixed_nvfp4_checkpoint(tmp_path / "ck", twin, leaf)
    # The helper's stub config only has to name a quantization scheme; a split load also BUILDS the
    # twin from it, so give it the real family config the mismatch belongs to.
    config = _qwen3_5_multimodal_config().to_dict()
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": []}
    (tmp_path / "ck" / "config.json").write_text(json.dumps(config))

    loaded, _experts = load_quantized_split(str(tmp_path / "ck"), torch.float16)
    weight = dict(loaded.named_parameters())[leaf]
    assert not weight.is_meta, "the wrapper-prefixed key never reached the twin's parameter name"


# --- the serving lane's reverse direction ------------------------------------------------------
# Retargeting re-addresses a traced constant from its wrapper-relative path to the key its weight
# came from, and only the checkpoint knows that name. For a family whose checkpoint names differ
# from its parameter names the two are not the same string, so the map's values must be checkpoint
# keys — otherwise the birth-time spellers resolve nothing and quietly leave the weight unspelled.


def _wrapper_prefixed_nvfp4_checkpoint(dirpath, twin, leaf: str):
    """Write ``leaf``'s weight as an NVFP4 trio under the checkpoint's OWN wrapper-prefixed name,
    every other parameter as a plain tensor. Returns that checkpoint key."""
    import json

    import numpy as np
    import torch
    from safetensors.torch import save_file

    from emmy.compiler.trace.huggingface import _checkpoint_key_renamer

    to_checkpoint = _checkpoint_key_renamer(twin, reverse=True)
    key = to_checkpoint(leaf)
    assert key != leaf, "the fixture must reproduce the naming mismatch, not hide it"
    out, k = dict(twin.named_parameters())[leaf].shape
    rng = np.random.default_rng(5)
    tensors = {
        key: torch.from_numpy(rng.integers(0, 256, (out, k // 2)).astype(np.uint8)),
        key + "_scale": torch.from_numpy(rng.integers(0, 0x7F, (out, k // 16)).astype(np.uint8)).view(torch.float8_e4m3fn),
        key + "_scale_2": torch.tensor(0.25, dtype=torch.float32),
    }
    dirpath.mkdir(exist_ok=True)
    save_file(tensors, str(dirpath / "model.safetensors"))
    (dirpath / "config.json").write_text(
        json.dumps({"model_type": "test", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": []}})
    )
    return key


def test_serving_retargeting_lands_on_checkpoint_keys_so_the_speller_fires(tmp_path):
    """The reverse direction, end to end: a constant carrying the wrapper-relative path a split
    trace stamps is retargeted, and the NVFP4 speller then finds its trio and rewrites the
    constant into the decode cone. Before the map carried checkpoint keys this resolved nothing —
    quietly, since the speller's miss is a bare ``continue``."""
    import torch

    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.quant import spell_quantized_constants
    from emmy.compiler.trace.huggingface import (
        _auto_model_from_config,
        build_linear_attention_split_wrapper,
        retarget_constants_to_model,
    )

    torch.manual_seed(0)
    twin = _auto_model_from_config(_qwen3_5_multimodal_config()).eval()
    leaf = "model.layers.0.mlp.gate_proj.weight"
    ckpt_key = _wrapper_prefixed_nvfp4_checkpoint(tmp_path / "ck", twin, leaf)

    # The split wrapper holds the block's own submodules, so tensor identity bridges its
    # wrapper-relative spelling to the twin's path — the mechanism retargeting relies on.
    _pre, post = build_linear_attention_split_wrapper(twin.model.layers[0])
    wrapper_path = next(p for p, t in post.named_parameters() if t is dict(twin.named_parameters())[leaf])

    shape = tuple(dict(twin.named_parameters())[leaf].shape)
    g = Graph()
    op = ConstantOp(name="w", source_path=wrapper_path, source_shape=shape, source_dtype="f32")
    g.add_node(op=op, inputs=[], output=Tensor("w", shape, "f32"), node_id="w")
    g.inputs, g.outputs = [], ["w"]

    retarget_constants_to_model(g, post, twin)
    assert g.nodes["w"].op.source_path == ckpt_key, "retargeting must land on the checkpoint's own key"
    assert spell_quantized_constants(g, str(tmp_path / "ck")) == 1, "the NVFP4 speller must fire through the serving lane's path"
