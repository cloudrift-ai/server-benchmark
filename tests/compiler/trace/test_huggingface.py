"""Tests for trace/huggingface.py wrapper builders.

Covers ``build_layer_wrapper``'s synthetic Per-Layer-Embedding (PLE) support: a
Gemma-nano block exposing ``hidden_size_per_layer_input`` receives a seeded
synthetic ``per_layer_input`` buffer sliced in-graph to the runtime seq_len
(like cos/sin); every other architecture takes the unchanged path (no ``ple``
buffer, no extra kwarg).
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
