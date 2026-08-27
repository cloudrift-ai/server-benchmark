"""Conv1d and einsum reach the tensor dialect with torch's numbers.

Both ops used to stop the tracer dead — ``aten.conv1d`` had no mapping at all, and
``aten.einsum`` fell into the elementwise fallback and failed to broadcast. They are the
two operations Qwen3.8's Gated DeltaNet layers open with, so the interesting cases here
are that model's: a depthwise convolution with a causal left pad, and a batched
contraction.

The weights are graph inputs rather than module parameters so the comparison feeds them
directly instead of going through constant rebinding, which is a different mechanism and
already has its own tests.
"""

from __future__ import annotations

import numpy as np
import pytest


def _decompose(module, inputs):
    """Trace ``module`` and run the frontend decomposition, returning the tensor-dialect graph."""
    from emmy.compiler.pipeline import TENSOR_PASSES, Pipeline
    from emmy.compiler.trace.torch import trace_module

    return Pipeline.build(TENSOR_PASSES).run(trace_module(module, inputs))


def _assert_matches_eager(run_graph, module, tensors, *, tol=2e-6):
    """The decomposed graph and eager torch agree on the same inputs."""
    import torch

    graph = _decompose(module, tensors)
    with torch.no_grad():
        expected = module(*tensors).numpy()
    feed = {name: tensor.numpy() for name, tensor in zip(graph.inputs, tensors, strict=True)}
    outputs = run_graph(graph, feed)
    got = np.asarray(next(iter(outputs.values()))).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


@pytest.fixture
def conv_module():
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class Conv(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, x, w):
            return F.conv1d(x, w, **self.kwargs)

    return Conv


def test_depthwise_conv1d_matches_eager(run_graph, conv_module) -> None:
    """Qwen3.8's linear-attention convolution: depthwise, kernel 4, causal left pad."""
    import torch

    torch.manual_seed(0)
    # Batch 2 on purpose: a batch of 1 hides an index map that wrongly reads the batch
    # coordinate when indexing the per-channel weight.
    x, w = torch.randn(2, 8, 16), torch.randn(8, 1, 4)
    _assert_matches_eager(run_graph, conv_module(groups=8, padding=3), (x, w))


def test_causal_conv1d_with_zero_width_chunk_pad_matches_eager(run_graph) -> None:
    """The Qwen3.8 causal Conv1d target retains its empty chunk-alignment pad."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class CausalConv(nn.Module):
        def forward(self, x, w):
            return F.pad(F.conv1d(x, w, padding=3, groups=8), (0, 0))

    torch.manual_seed(0)
    x, w = torch.randn(2, 8, 16), torch.randn(8, 1, 4)
    _assert_matches_eager(run_graph, CausalConv(), (x, w))


def test_causal_conv1d_rejects_nonzero_generic_pad() -> None:
    """A coordinate-changing pad fails before the attribute-free elementwise fallback."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class CausalConv(nn.Module):
        def forward(self, x, w):
            return F.pad(F.conv1d(x, w, padding=3, groups=8), (1, 0))

    x, w = torch.randn(1, 8, 16), torch.randn(8, 1, 4)
    with pytest.raises(NotImplementedError, match="only explicit zero-width padding"):
        _decompose(CausalConv(), (x, w))


def test_dense_conv1d_im2col_matches_eager(run_graph, conv_module) -> None:
    """The im2col form, with the stride and dilation that make the window map non-trivial."""
    import torch

    torch.manual_seed(0)
    x, w = torch.randn(2, 5, 16), torch.randn(7, 5, 3)
    _assert_matches_eager(run_graph, conv_module(stride=2, padding=1, dilation=2), (x, w))


def test_conv1d_rejects_a_grouped_convolution(conv_module) -> None:
    """Neither form covers 1 < groups < C_in, and the tracer says so instead of guessing."""
    import torch

    x, w = torch.randn(1, 8, 16), torch.randn(8, 2, 4)
    with pytest.raises(NotImplementedError, match="groups=4"):
        _decompose(conv_module(groups=4), (x, w))


@pytest.fixture
def einsum_module():
    import torch
    import torch.nn as nn

    class Einsum(nn.Module):
        def __init__(self, equation):
            super().__init__()
            self.equation = equation

        def forward(self, a, b):
            return torch.einsum(self.equation, a, b)

    return Einsum


@pytest.mark.parametrize(
    "equation",
    [
        "bij,bjk->bik",  # the plain batched contraction
        "bij,bjk->bki",  # output labels permuted away from the product order
        "bhld,bhdm->bhlm",  # two batch labels, the delta-rule shape
    ],
)
def test_einsum_matches_eager(run_graph, einsum_module, equation) -> None:
    import torch

    torch.manual_seed(0)
    sizes = {"b": 2, "h": 3, "i": 4, "j": 5, "k": 6, "l": 4, "d": 5, "m": 6}
    a_labels, rest = equation.split(",")
    b_labels = rest.split("->")[0]
    a = torch.randn(*[sizes[label] for label in a_labels])
    b = torch.randn(*[sizes[label] for label in b_labels])
    _assert_matches_eager(run_graph, einsum_module(equation), (a, b))


@pytest.mark.parametrize(
    ("equation", "a_shape", "b_shape", "message"),
    [
        ("bii,bij->bj", (2, 5, 5), (2, 5, 6), "repeats a label"),
        ("...ij,...jk->...ik", (2, 4, 5), (2, 5, 6), "ellipsis"),
        ("bij,bjk->b", (2, 4, 5), (2, 5, 6), "free label"),
        ("bij,bjk->bijk", (2, 4, 5), (2, 5, 6), "one contracted"),
    ],
)
def test_einsum_rejects_forms_it_cannot_lower(einsum_module, equation, a_shape, b_shape, message) -> None:
    """A diagonal, an ellipsis, a reduction, or anything needing a reshape is refused by name."""
    import torch

    torch.manual_seed(0)
    a, b = torch.randn(*a_shape), torch.randn(*b_shape)
    with pytest.raises(NotImplementedError, match=message):
        _decompose(einsum_module(equation), (a, b))
