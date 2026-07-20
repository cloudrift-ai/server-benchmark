#!/usr/bin/env python3
"""Capture the serving `pre`/`post` twin graphs as tunable JSON IR.

`EmmyGenRunner` compiles two programs per decoder layer — the `pre` half (input norm ->
q/k/v projections -> per-head q/k norms) and the `post` half (o_proj -> residual ->
post-attention norm -> gated MLP) — at three widths: a static decode bucket, a static
prefill chunk, and one symbolic (any-width) program. Tuning those exact graphs is the only
evidence path that carries serving (`emmy tune <twin>.json` writes the `perf` rows the
deploy pick reads); an isolated golden snippet does not, because fusion in the real block
produces a different graph.

This writes one JSON per twin using `gen_runner.trace_split`, i.e. the same trace call
`_compile_split` makes, so the captured graph is the one serving compiles. Gemma-4 is not
homogeneous — its `full_attention` layers use a larger head_dim than the sliding ones — so
a global layer is captured alongside the sliding one as the `-global` variants.

Emits, under --out: {pre,post}<bucket>.json, {pre,post}<chunk>.json, {pre,post}-sym.json,
plus a `-global` twin of each when the model has global layers.

Usage:
    python scripts/capture_gen_twins.py --model google/gemma-4-12B --out _tune/twins
    python scripts/capture_gen_twins.py --model google/gemma-4-12B --out _tune/twins \
        --decode-bucket 32 --prefill-bucket 256

Then tune them (writes the box-local evidence serving reads):
    EMMY_TUNE_DB=_tune/twins/twins.db EMMY_ONLINE_FILE=_tune/twins/twins-online.json \
        ./venv/bin/emmy tune _tune/twins/pre32.json -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _global_layer(model) -> int | None:
    """Index of the first `full_attention` layer, or None for a homogeneous model."""
    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    text_config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(text_config, "layer_types", None) or []
    for i, kind in enumerate(layer_types):
        if "full" in kind and i < len(trunk.layers):
            return i
    return None


def _capture_layer(block, hidden, dtype, suffix, out_dir, buckets):
    """Write every twin of one decoder layer. `buckets` is a list of (name, M) where
    M=None traces the symbolic (any-width) program."""
    import torch

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper
    from emmy.serving.gen_runner import trace_split

    attn = block.self_attn
    attn_width = attn.q_proj.out_features  # this layer's num_heads * head_dim
    pre_w, post_w = build_attention_split_wrapper(block)

    written = []
    for name, m in buckets:
        # The symbolic program traces at the same example width `from_model` uses (8) and
        # ties axis-0 to a `num_tokens` Dim; a static twin traces at its bucket with no Dim.
        rows = 8 if m is None else m
        halves = [
            ("pre", pre_w, [torch.zeros(rows, hidden, dtype=dtype)], ["hidden"] if m is None else None),
            (
                "post",
                post_w,
                [torch.zeros(rows, attn_width, dtype=dtype), torch.zeros(rows, hidden, dtype=dtype)],
                ["attn_out", "residual"] if m is None else None,
            ),
        ]
        for half, wrapper, example_args, argnames in halves:
            with torch.device("cpu"):
                graph = trace_split(wrapper, example_args, argnames)
            path = out_dir / f"{half}{name}{suffix}.json"
            with open(path, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            logger.info("  wrote %s (%d nodes)", path, len(graph.nodes))
            written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Capture EmmyGenRunner pre/post twin graphs as JSON IR")
    parser.add_argument("--model", default="google/gemma-4-12B", help="HuggingFace model ID (default: google/gemma-4-12B)")
    parser.add_argument("--out", required=True, help="Output directory for the twin JSONs")
    parser.add_argument("--layer", type=int, default=0, help="Sliding/local decoder layer to capture (default: 0)")
    parser.add_argument(
        "--global-layer",
        type=int,
        default=None,
        help="Global (full_attention) layer to capture; default: the model's first, or none if homogeneous",
    )
    parser.add_argument("--decode-bucket", type=int, default=32, help="Static decode-bucket M (default: 32)")
    parser.add_argument("--prefill-bucket", type=int, default=256, help="Static prefill-chunk M, 0 to skip (default: 256)")
    parser.add_argument("--no-symbolic", action="store_true", help="Skip the symbolic (any-width) twins")
    parser.add_argument("--dtype", default="float16", help="Trace dtype (default: float16)")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch + transformers required: pip install -e '.[compile,serving]'")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    buckets = [(str(args.decode_bucket), args.decode_bucket)]
    if args.prefill_bucket:
        buckets.append((str(args.prefill_bucket), args.prefill_bucket))
    if not args.no_symbolic:
        buckets.append(("-sym", None))

    logger.info("Loading %s (%s, CPU trace)...", args.model, args.dtype)
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    trunk = getattr(model, "model", model)
    trunk = getattr(trunk, "language_model", trunk)
    text_config = getattr(model.config, "text_config", model.config)
    hidden = text_config.hidden_size

    targets = [(args.layer, "")]
    gl = args.global_layer if args.global_layer is not None else _global_layer(model)
    if gl is not None:
        targets.append((gl, "-global"))
    else:
        logger.info("No full_attention layer found — capturing local twins only")

    written = []
    for layer, suffix in targets:
        logger.info("Capturing layer %d%s (hidden=%d)...", layer, f" [{suffix.lstrip('-')}]" if suffix else "", hidden)
        written += _capture_layer(trunk.layers[layer], hidden, dtype, suffix, out_dir, buckets)

    logger.info("\nCaptured %d twins in %s", len(written), out_dir)
    logger.info(
        "Tune them with: EMMY_TUNE_DB=%s/twins.db EMMY_ONLINE_FILE=%s/twins-online.json ./venv/bin/emmy tune <twin>.json -v",
        out_dir,
        out_dir,
    )


if __name__ == "__main__":
    main()
