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


def _fake_causal_lm(block):
    """Minimal AutoModelForCausalLM stand-in for ``_trace_model``: a decoder module
    exposing ``layers`` / ``rotary_emb`` / ``config`` (what ``_find_text_decoder``
    matches) under ``.model``."""
    from types import SimpleNamespace

    import torch.nn as nn

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([block])
            self.config = SimpleNamespace(hidden_size=HIDDEN)
            self.rotary_emb = _fake_rotary

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


def test_attention_split_rejects_ple_block():
    """The serving carve has no seam for the PLE multiply — it must reject loudly,
    not silently drop the ``per_layer_input`` term."""
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    with pytest.raises(NotImplementedError, match="hidden_size_per_layer_input"):
        build_attention_split_wrapper(_ple_block())
