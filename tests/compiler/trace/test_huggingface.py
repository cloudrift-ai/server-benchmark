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
