"""Network-free coverage for the Diffusers DiT block trace adapter."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.trace.torch import has_torch

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch not available")


def _tiny_transformer():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.Transformer2DModel(
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=4,
        out_channels=8,
        num_layers=1,
        dropout=0.0,
        norm_num_groups=1,
        sample_size=4,
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        attention_bias=True,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
    ).eval()


def test_dit_adapter_traces_tiny_block_and_binds_weights():
    """AdaLN-Zero, chunk routing, SDPA, GELU MLP, residuals, and every live
    parameter/buffer all survive the tiny offline trace."""
    import torch

    from emmy.commands.run import _bind_inputs
    from emmy.compiler.ir.frontend.ir import SdpaOp, SliceOp
    from emmy.compiler.trace.dit import DIT_CLASS_LABEL, DIT_TIMESTEP, trace_dit_transformer

    graph, (block, args, kwargs) = trace_dit_transformer(_tiny_transformer(), 0, hidden_shape=(1, 4, 8))

    assert tuple(args[0].shape) == (1, 4, 8)
    assert args[0].dtype == torch.float16
    assert kwargs["timestep"].item() == DIT_TIMESTEP
    assert kwargs["class_labels"].item() == DIT_CLASS_LABEL
    assert len([node for node in graph.nodes.values() if isinstance(node.op, SliceOp)]) >= 6
    assert len([node for node in graph.nodes.values() if isinstance(node.op, SdpaOp)]) == 1
    assert type(block.attn1.processor).__name__ == "AttnProcessor2_0"

    bound = _bind_inputs(graph, block, args, kwargs)
    assert all(node_id in bound for node_id, _op in graph.loadable_constants())


def test_dit_adapter_tiny_block_matches_loop_backend():
    """The fully decomposed/fused CPU path agrees with the same normalized DiT block."""
    from emmy.commands.run import _bind_inputs
    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.trace.dit import trace_dit_transformer

    graph, (block, args, kwargs) = trace_dit_transformer(_tiny_transformer(), 0, hidden_shape=(1, 4, 8))
    backend = LoopBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data=_bind_inputs(compiled, block, args, kwargs))

    expected = block(*args, **kwargs).detach().numpy()
    actual = next(iter(result.outputs.values())).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_dit_timestep_normalization_preserves_block_output():
    """Materializing static frequencies changes traceability, not DiT numerics."""
    import copy

    import torch
    from diffusers.models.attention_processor import AttnProcessor2_0

    from emmy.compiler.trace.dit import trace_dit_transformer

    transformer = _tiny_transformer()
    reference = copy.deepcopy(transformer.transformer_blocks[0]).to(dtype=torch.float16).eval()
    for module in reference.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(AttnProcessor2_0())

    _, (normalized, args, kwargs) = trace_dit_transformer(transformer, 0, hidden_shape=(1, 4, 8))
    torch.testing.assert_close(normalized(*args, **kwargs), reference(*args, **kwargs), rtol=0, atol=0)


def test_dit_loader_uses_only_transformer_subfolder(monkeypatch):
    """The adapter loads the transformer component without constructing a pipeline,
    VAE, or scheduler."""
    import torch

    import emmy.compiler.trace.dit as dit

    diffusers = pytest.importorskip("diffusers")
    sentinel = object()
    seen = {}

    def fake_from_pretrained(model_id, **kwargs):
        seen.update(model_id=model_id, **kwargs)
        return sentinel

    monkeypatch.setattr(diffusers.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(dit, "trace_dit_transformer", lambda transformer, layer: ("graph", (transformer, layer)))

    assert dit.trace_dit_model("facebook/DiT-XL-2-256", 3) == ("graph", (sentinel, 3))
    assert seen == {
        "model_id": "facebook/DiT-XL-2-256",
        "subfolder": "transformer",
        "torch_dtype": torch.float16,
    }


def test_dit_inputs_are_deterministic():
    """Independent adapter builds use seed 0 for identical hidden states."""
    from emmy.compiler.trace.dit import trace_dit_transformer

    _, (_, args_a, _) = trace_dit_transformer(_tiny_transformer(), 0, hidden_shape=(1, 4, 8))
    _, (_, args_b, _) = trace_dit_transformer(_tiny_transformer(), 0, hidden_shape=(1, 4, 8))
    np.testing.assert_array_equal(args_a[0].numpy(), args_b[0].numpy())
