#!/usr/bin/env python3
"""Capture the serving `pre`/`post` twin graphs as tunable JSON IR.

`EmmyGenRunner` compiles two programs per decoder layer — the `pre` half (input norm ->
q/k/v projections -> per-head q/k norms) and the `post` half (o_proj -> residual ->
post-attention norm -> gated MLP) — at three widths: a static decode bucket, a static
prefill chunk, and one symbolic (any-width) program. Tuning those exact graphs is the only
evidence path that carries serving (`emmy tune <twin>.json` writes the `perf` rows the
deploy pick reads); an isolated golden snippet does not, because fusion in the real block
produces a different graph.

The graphs come from `emmy.serving.twins.capture_twin_graphs` — the same capture the release
audit (`emmy eval golden --serving-config`) runs, so one lane decides which layers represent
the trunk and what program each coded format spells. This file is its JSON writer, and the
only reason it exists separately: `emmy tune` reads a graph per file.

Emits one JSON per twin under --out, named exactly as the capture names it: {pre,post}<bucket>,
{pre,post}<chunk>, {pre,post}-sym, a `-global` (or per-structure) variant of each where the
architecture is not homogeneous, and a format suffix where the checkpoint's weights are coded —
`pre32@nvfp4.json` for an NVFP4 trunk, `pre32@b4.json` for an EXL3 one. A coded twin carries the
program serving compiles, packed weights and all; an f16 twin over dequantized weights would tune
kernels serving never runs.

Usage:
    python scripts/capture_gen_twins.py --model google/gemma-4-12B --out _tune/twins
    python scripts/capture_gen_twins.py --model nvidia/Qwen3-8B-NVFP4 --out _tune/twins \
        --decode-bucket 32 --prefill-bucket 256

Then tune them (writes the box-local evidence serving reads):
    EMMY_TUNE_DB=_tune/twins/twins.db EMMY_ONLINE_FILE=_tune/twins/twins-online.json \
        ./venv/bin/emmy tune _tune/twins/pre32@nvfp4.json -v
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


def main():
    parser = argparse.ArgumentParser(description="Capture EmmyGenRunner pre/post twin graphs as JSON IR")
    parser.add_argument("--model", default="google/gemma-4-12B", help="HuggingFace model ID (default: google/gemma-4-12B)")
    parser.add_argument("--out", required=True, help="Output directory for the twin JSONs")
    parser.add_argument("--decode-bucket", type=int, default=32, help="Static decode-bucket M (default: 32)")
    parser.add_argument("--prefill-bucket", type=int, default=256, help="Static prefill-chunk M, 0 to skip (default: 256)")
    parser.add_argument("--no-symbolic", action="store_true", help="Skip the symbolic (any-width) twins")
    parser.add_argument("--dtype", default="float16", help="Trace dtype (default: float16)")
    args = parser.parse_args()

    from emmy.serving.twins import capture_twin_graphs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Capturing %s twins (%s, CPU trace)...", args.model, args.dtype)
    try:
        graphs = capture_twin_graphs(
            args.model,
            decode_bucket=args.decode_bucket,
            prefill_bucket=args.prefill_bucket,
            symbolic=not args.no_symbolic,
            dtype=args.dtype,
        )
    except ImportError:
        logger.error("torch + transformers required: pip install -e '.[compile,serving]'")
        sys.exit(1)

    for name, graph in sorted(graphs.items()):
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        logger.info("  wrote %s (%d nodes)", path, len(graph.nodes))

    logger.info("\nCaptured %d twins in %s", len(graphs), out_dir)
    logger.info(
        "Tune them with: EMMY_TUNE_DB=%s/twins.db EMMY_ONLINE_FILE=%s/twins-online.json ./venv/bin/emmy tune <twin>.json -v",
        out_dir,
        out_dir,
    )


if __name__ == "__main__":
    main()
