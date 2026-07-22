"""Weight-free serving-twin capture — the graphs serving compiles, without the weights.

``EmmyGenRunner`` compiles two programs per decoder layer (the ``pre`` and ``post``
attention halves) at three widths (static decode bucket, static prefill chunk, symbolic).
Capturing those graphs normally means downloading the full checkpoint
(``scripts/capture_gen_twins.py``), but nothing in a TRACE reads a weight value — only
config-derived shapes and dtypes matter. So this module builds a random-init skeleton from
the model's ``config.json`` alone (a few-KB fetch, or a checked-in fixture directory) and
traces the twins through the exact ``build_attention_split_wrapper`` / ``trace_split``
path serving uses.

The skeleton is trimmed before construction: ``layer_types`` collapses to one local layer
plus (when the model has any) one ``full_attention`` layer — per-layer structure is what's
traced, not depth — and the vocab shrinks to a stub since the twins never touch the
embedding or lm_head. Gemma-4's global layers use a different head_dim, so the ``-global``
twins genuinely differ from the local ones.

Consumed by ``emmy eval golden --in-model`` and the golden drift CI gate. Note the traced
graph tracks the installed ``transformers`` modeling code: a transformers bump that
changes the model's forward changes these twins — exactly as it would change serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

#: Serving's default static widths (``EmmyGenRunner`` / ``capture_gen_twins`` conventions).
DECODE_BUCKET = 32
PREFILL_BUCKET = 256


def capture_twin_graphs(
    model: str,
    *,
    decode_bucket: int = DECODE_BUCKET,
    prefill_bucket: int = PREFILL_BUCKET,
    symbolic: bool = True,
    dtype: str = "float16",
) -> dict[str, Graph]:
    """Trace every serving twin of ``model`` from its config alone (no weights).

    ``model`` is an HF hub id or a local directory holding ``config.json``. Returns
    ``{"pre32": Graph, "post32": …, "pre256": …, "pre-sym": …}`` plus ``-global``
    variants of each when the model has ``full_attention`` layers — the same names
    ``scripts/capture_gen_twins.py`` writes."""
    import torch  # noqa: PLC0415
    from transformers import AutoConfig, AutoModel  # noqa: PLC0415

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper  # noqa: PLC0415
    from emmy.serving.gen_runner import trace_split  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(model)
    text = getattr(cfg, "text_config", cfg)
    types = list(getattr(text, "layer_types", None) or [])
    local = next((t for t in types if "full" not in t), None)
    glob = next((t for t in types if "full" in t), None)
    if types:
        text.layer_types = [t for t in (local, glob) if t]
        text.num_hidden_layers = len(text.layer_types)
    else:
        text.num_hidden_layers = 1  # homogeneous model (no layer_types): one layer suffices
    text.vocab_size = 32  # the twins are decoder halves — the embedding/lm_head are never traced
    td = getattr(torch, dtype)
    with torch.device("cpu"):
        trunk = AutoModel.from_config(text).eval().to(td)
    hidden = text.hidden_size

    buckets: list[tuple[str, int | None]] = [(str(decode_bucket), decode_bucket)]
    if prefill_bucket:
        buckets.append((str(prefill_bucket), prefill_bucket))
    if symbolic:
        buckets.append(("-sym", None))

    layers = [(trunk.layers[0], "")]
    if glob is not None and len(trunk.layers) > 1:
        layers.append((trunk.layers[1], "-global"))

    graphs: dict[str, Graph] = {}
    for block, suffix in layers:
        pre_w, post_w = build_attention_split_wrapper(block)
        attn_width = block.self_attn.q_proj.out_features  # this layer's num_heads * head_dim
        for name, m in buckets:
            # The symbolic program traces at the example width serving uses (8) and ties
            # axis-0 to a ``num_tokens`` Dim; a static twin traces at its bucket, no Dim.
            rows = 8 if m is None else m
            halves = [
                ("pre", pre_w, [torch.zeros(rows, hidden, dtype=td)], ["hidden"] if m is None else None),
                (
                    "post",
                    post_w,
                    [torch.zeros(rows, attn_width, dtype=td), torch.zeros(rows, hidden, dtype=td)],
                    ["attn_out", "residual"] if m is None else None,
                ),
            ]
            for half, wrapper, example_args, argnames in halves:
                with torch.device("cpu"):
                    graphs[f"{half}{name}{suffix}"] = trace_split(wrapper, example_args, argnames)
    return graphs
