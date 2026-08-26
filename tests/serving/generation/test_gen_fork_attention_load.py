"""``EmmyGenModel.load_weights`` — the DeepSeek V4 attention ownership table (no GPU).

The published checkpoint spells its attention weights three ways, one per layer type: sliding
layers carry the base projections, heavily-compressed layers add a compressor, compressed-sparse
layers add the compressor AND a lightning indexer (with its own inner compressor). ``load_weights``
must route every one of those keys into the fork's attention sublayer — through the fp8 ``.scale``
sibling rename, the compressor's ``mla_attn`` placement, and the two fused-projection stackings —
and fail loudly both ways: an attention key mapping to no parameter, and a parameter the stream
did not fully load. This suite drives ``load_weights`` unbound against a light double streaming
the EXACT published key sets, so the table is pinned without a GPU or a vLLM engine.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")

# The checkpoint's attention keys (relative to ``layers.N.attn.``), copied from the published
# model's safetensors index — one list per layer type.
SLIDING_KEYS = [
    "attn_sink",
    "kv_norm.weight",
    "q_norm.weight",
    "wkv.scale",
    "wkv.weight",
    "wo_a.scale",
    "wo_a.weight",
    "wo_b.scale",
    "wo_b.weight",
    "wq_a.scale",
    "wq_a.weight",
    "wq_b.scale",
    "wq_b.weight",
]
HCA_KEYS = SLIDING_KEYS + [
    "compressor.ape",
    "compressor.norm.weight",
    "compressor.wgate.weight",
    "compressor.wkv.weight",
]
CSA_KEYS = HCA_KEYS + [
    "indexer.compressor.ape",
    "indexer.compressor.norm.weight",
    "indexer.compressor.wgate.weight",
    "indexer.compressor.wkv.weight",
    "indexer.weights_proj.weight",
    "indexer.wq_b.scale",
    "indexer.wq_b.weight",
]

# Where each key must land in the fork's attention module (relative parameter name, fused shard).
# Hand-written on purpose — an independent statement of the ownership table, not derived from the
# implementation's own mapping.
EXPECTED_DEST = {
    "attn_sink": ("attn_sink", None),
    "kv_norm.weight": ("kv_norm.weight", None),
    "q_norm.weight": ("q_norm.weight", None),
    "wq_a.weight": ("fused_wqa_wkv.weight", 0),
    "wq_a.scale": ("fused_wqa_wkv.weight_scale_inv", 0),
    "wkv.weight": ("fused_wqa_wkv.weight", 1),
    "wkv.scale": ("fused_wqa_wkv.weight_scale_inv", 1),
    "wq_b.weight": ("wq_b.weight", None),
    "wq_b.scale": ("wq_b.weight_scale_inv", None),
    "wo_a.weight": ("wo_a.weight", None),
    "wo_a.scale": ("wo_a.weight_scale_inv", None),
    "wo_b.weight": ("wo_b.weight", None),
    "wo_b.scale": ("wo_b.weight_scale_inv", None),
    "compressor.ape": ("mla_attn.compressor.ape", None),
    "compressor.norm.weight": ("mla_attn.compressor.norm.weight", None),
    "compressor.wkv.weight": ("mla_attn.compressor.fused_wkv_wgate.weight", 0),
    "compressor.wgate.weight": ("mla_attn.compressor.fused_wkv_wgate.weight", 1),
    "indexer.compressor.ape": ("indexer.compressor.ape", None),
    "indexer.compressor.norm.weight": ("indexer.compressor.norm.weight", None),
    "indexer.compressor.wkv.weight": ("indexer.compressor.fused_wkv_wgate.weight", 0),
    "indexer.compressor.wgate.weight": ("indexer.compressor.fused_wkv_wgate.weight", 1),
    "indexer.weights_proj.weight": ("indexer.weights_proj.weight", None),
    "indexer.wq_b.weight": ("indexer.wq_b.weight", None),
    "indexer.wq_b.scale": ("indexer.wq_b.weight_scale_inv", None),
}

N_HEADS = 4


class _FakeAttention(torch.nn.Module):
    """A stand-in with the fork attention's parameter NAMES; every parameter records its loads."""

    def __init__(self, param_names):
        super().__init__()
        for dotted in param_names:
            parent = self
            *path, leaf = dotted.split(".")
            for part in path:
                child = getattr(parent, part, None)
                if child is None:
                    child = torch.nn.Module()
                    parent.add_module(part, child)
                parent = child
            size = N_HEADS if leaf == "attn_sink" else 2
            param = torch.nn.Parameter(torch.zeros(size), requires_grad=False)
            param.loads = []
            param.weight_loader = lambda p, w, shard=None, _param=param: _param.loads.append(shard)
            parent.register_parameter(leaf, param)


def _fake_layers():
    """One fake per layer type, parameter names taken from the expected destinations."""
    layers = []
    for keys in (SLIDING_KEYS, CSA_KEYS, HCA_KEYS):
        layers.append(_FakeAttention(sorted({EXPECTED_DEST[k][0] for k in keys})))
    return layers


def _model(fork_attn, vocab=8, hidden=4):
    head = torch.nn.Parameter(torch.full((vocab, hidden), float("nan")), requires_grad=False)
    model = types.SimpleNamespace(
        lm_head=types.SimpleNamespace(weight=head),
        config=types.SimpleNamespace(tie_word_embeddings=False, num_attention_heads=N_HEADS),
        sinks=None,
        runner=types.SimpleNamespace(_embed_scale=1.0, adopt_embed_table=lambda *a, **k: None),
        _model_id="does-not-exist/nowhere",
        _is_last_rank=True,
        fork_attn=fork_attn,
        start_layer=0,
        end_layer=len(fork_attn),
        reclaim_device_memory=lambda: None,
    )
    return model, head


def _stream(vocab=8, hidden=4):
    """The full checkpoint stream: three attention layers, the head, and families no one claims
    here — the runner-owned trunk (loaded at construction) and the MTP head (serves no twin)."""
    items = [("head.weight", torch.zeros(vocab, hidden))]
    for layer, keys in ((0, SLIDING_KEYS), (1, CSA_KEYS), (2, HCA_KEYS)):
        items += [(f"layers.{layer}.attn.{key}", torch.arange(N_HEADS, dtype=torch.float32)) for key in keys]
        items += [
            (f"layers.{layer}.attn_norm.weight", torch.zeros(2)),
            (f"layers.{layer}.ffn_norm.weight", torch.zeros(2)),
            (f"layers.{layer}.ffn.gate.weight", torch.zeros(2)),
            (f"layers.{layer}.ffn.experts.0.w1.weight", torch.zeros(2)),
            (f"layers.{layer}.hc_attn_scale", torch.zeros(2)),
        ]
    items += [
        ("embed.weight", torch.zeros(2)),
        ("norm.weight", torch.zeros(2)),
        ("mtp.0.attn.wq_a.weight", torch.zeros(2)),
    ]
    return items


@pytest.fixture(autouse=True)
def _single_rank_tp(monkeypatch):
    # The ``attn_sink`` lane reads the tensor-parallel group; this suite runs without one.
    monkeypatch.setattr(vllm_model_gen, "get_tp_group", lambda: types.SimpleNamespace(world_size=1, rank_in_group=0))


def test_every_published_attention_key_lands_where_the_fork_expects_it():
    layers = _fake_layers()
    model, head = _model(layers)

    loaded = vllm_model_gen.EmmyGenModel.load_weights(model, iter(_stream()))

    for layer, keys in ((0, SLIDING_KEYS), (1, CSA_KEYS), (2, HCA_KEYS)):
        params = dict(layers[layer].named_parameters())
        for key in keys:
            dest, shard = EXPECTED_DEST[key]
            assert f"fork_attn.{layer}.{dest}" in loaded, f"{key} was not claimed for layer {layer}"
            if dest == "attn_sink":
                assert torch.equal(params[dest].data, torch.arange(N_HEADS, dtype=torch.float32))
            else:
                assert shard in params[dest].loads, f"{key} did not reach {dest} (shard {shard})"
        for dest, param in params.items():
            expected = [0, 1] if "fused_" in dest else [] if dest == "attn_sink" else [None]
            assert sorted(param.loads, key=str) == expected, f"layer {layer} {dest}: loads {param.loads}"
    # ``head.weight`` is the published spelling of the head; the trunk and MTP families pass through.
    assert "lm_head.weight" in loaded
    assert not torch.isnan(head.data).any()


def test_an_unmapped_attention_key_is_a_loud_failure():
    model, _head = _model(_fake_layers())
    stream = _stream() + [("layers.0.attn.wq_c.weight", torch.zeros(2))]
    with pytest.raises(ValueError, match="wq_c"):
        vllm_model_gen.EmmyGenModel.load_weights(model, iter(stream))


def test_a_missing_attention_weight_is_a_loud_failure():
    """vLLM's own strict check waives fp8-quantized parameters, so a fork weight the checkpoint
    never sourced must be caught HERE — otherwise that layer serves construction garbage."""
    model, _head = _model(_fake_layers())
    stream = [(name, w) for name, w in _stream() if not name.endswith("2.attn.compressor.wgate.weight")]
    with pytest.raises(ValueError, match="fused_wkv_wgate"):
        vllm_model_gen.EmmyGenModel.load_weights(model, iter(stream))


def test_another_pipeline_ranks_attention_is_not_claimed():
    """Keys outside this rank's layer interval belong to another stage: skipped, not errors."""
    layers = _fake_layers()[:1]
    model, _head = _model(layers)
    model.start_layer, model.end_layer = 1, 2
    stream = [("head.weight", torch.zeros(8, 4))]
    stream += [(f"layers.1.attn.{key}", torch.arange(N_HEADS, dtype=torch.float32)) for key in SLIDING_KEYS]
    stream += [(f"layers.0.attn.{key}", torch.zeros(2)) for key in CSA_KEYS]  # first rank's layer

    loaded = vllm_model_gen.EmmyGenModel.load_weights(model, iter(stream))

    assert {name for name in loaded if name.startswith("fork_attn.")} == {f"fork_attn.0.{EXPECTED_DEST[key][0]}" for key in SLIDING_KEYS}
