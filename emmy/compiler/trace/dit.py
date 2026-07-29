"""Fixed-shape Diffusers DiT transformer-block trace adapter."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

DIT_HIDDEN_SHAPE = (1, 256, 1152)
DIT_NUM_LAYERS = 28
DIT_TIMESTEP = 500
DIT_CLASS_LABEL = 207


def trace_dit_model(model_id: str, layer: int):
    """Load only a checkpoint's transformer component and trace one DiT block."""
    import torch
    from diffusers import AutoModel

    logger.info("Pulling %s transformer...", model_id)
    transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    return trace_dit_transformer(transformer, layer)


def trace_dit_transformer(transformer, layer: int, *, hidden_shape: tuple[int, int, int] = DIT_HIDDEN_SHAPE):
    """Trace ``transformer.transformer_blocks[layer]`` for the fixed v1 workload.

    ``hidden_shape`` is injectable so the network-free test can exercise a tiny
    randomly initialized Diffusers block; the public adapter always uses
    :data:`DIT_HIDDEN_SHAPE`.
    """
    import torch
    from diffusers.models.attention_processor import AttnProcessor2_0

    from emmy.compiler.trace.torch import trace_module

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError(f"{type(transformer).__name__} has no transformer_blocks")
    if not 0 <= layer < len(blocks):
        raise ValueError(f"DiT layer {layer} not found (transformer has {len(blocks)} blocks)")

    block = blocks[layer].to(dtype=torch.float16).eval()
    expected_hidden = _block_hidden_size(block)
    if expected_hidden is not None and hidden_shape[-1] != expected_hidden:
        raise ValueError(f"DiT hidden width mismatch: v1 input has {hidden_shape[-1]} features but layer {layer} expects {expected_hidden}")

    for module in block.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(AttnProcessor2_0())
    _materialize_timestep_frequencies(block)

    generator = torch.Generator(device="cpu").manual_seed(0)
    hidden_states = torch.randn(hidden_shape, dtype=torch.float16, generator=generator)
    kwargs = {
        "timestep": torch.tensor([DIT_TIMESTEP], dtype=torch.long),
        "class_labels": torch.tensor([DIT_CLASS_LABEL], dtype=torch.long),
    }
    logger.info("Tracing DiT layer %d at hidden shape %s...", layer, hidden_shape)
    graph = trace_module(block, (hidden_states,), kwargs=kwargs)
    return graph, (block, (hidden_states,), kwargs)


def _block_hidden_size(block) -> int | None:
    """Return the block's self-attention input width when available."""
    projection = getattr(getattr(block, "attn1", None), "to_q", None)
    width = getattr(projection, "in_features", None)
    return int(width) if width is not None else None


def _materialize_timestep_frequencies(block) -> None:
    """Replace Diffusers' per-forward ``arange`` with an equivalent buffer.

    Diffusers constructs the sinusoidal frequency vector on every
    ``Timesteps.forward`` call. The vector is static model metadata, while
    Emmy's frontend intentionally has no tensor-creation ``arange`` op.
    Materializing it as a registered float32 buffer preserves the exact
    timestep embedding while leaving the runtime timestep tensor, sin/cos,
    class embedding, and AdaLayerNorm-Zero projections in the traced graph.
    """
    import torch
    import torch.nn as nn

    embedding = getattr(getattr(block, "norm1", None), "emb", None)
    projection = getattr(embedding, "time_proj", None)
    required = ("num_channels", "flip_sin_to_cos", "downscale_freq_shift", "scale")
    if projection is None or any(not hasattr(projection, name) for name in required):
        raise ValueError("DiT AdaLayerNorm-Zero block has no supported timestep projection")

    num_channels = int(projection.num_channels)
    half_dim = num_channels // 2
    flip_sin_to_cos = bool(projection.flip_sin_to_cos)
    scale = float(projection.scale)
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32)
    frequencies = torch.exp(exponent / (half_dim - float(projection.downscale_freq_shift)))

    class _MaterializedTimesteps(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("frequencies", frequencies)

        def forward(self, timesteps):
            values = timesteps[:, None].float() * self.frequencies[None, :]
            values = scale * values
            values = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
            if flip_sin_to_cos:
                values = torch.cat([values[:, half_dim:], values[:, :half_dim]], dim=-1)
            if num_channels % 2:
                values = torch.nn.functional.pad(values, (0, 1, 0, 0))
            return values

    embedding.time_proj = _MaterializedTimesteps()
