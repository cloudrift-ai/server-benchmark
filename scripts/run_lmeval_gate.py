#!/usr/bin/env python3
"""Output-correctness gate for cross-engine serving benchmarks.

Scores a fixed GSM8K subset against an OpenAI-compatible endpoint via the lm-eval harness and emits one
JSON object (label, model, task, limit, seed, metrics, lm_eval_version). Run it once per serving
configuration (stock vLLM, vLLM+emmy, vLLM+emmy FAST_MATH); the labeled outputs are compared for
agreement within noise and against the model's published score.

Requires the `eval` extra (`pip install -e '.[eval]'`) and a serving endpoint that answers
`<base-url>/completions`. Uses lm-eval's Python API (`lm_eval.simple_evaluate`) rather than the CLI:
it returns the results dict directly — no output-directory discovery and re-parsing of the CLI's
timestamped results file — and lets us pin all three harness seeds in one call. lm-eval is imported
lazily inside main() so this module imports (for tests) without lm-eval installed.

Metric keys follow lm-eval 0.4.x (verified against 0.4.12): per-task rows keyed "<metric>,<filter>"
with stderr as "<metric>_stderr,<filter>", e.g. "exact_match,strict-match".
"""

import argparse
import json
import logging
from importlib.metadata import version as package_version
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

NUM_CONCURRENT = 8


def completions_url(base_url):
    """Full completions endpoint for an OpenAI-compatible API root (lm-eval wants the path included)."""
    url = base_url.rstrip("/")
    return url if url.endswith("/completions") else url + "/completions"


def build_model_args(base_url, model):
    """Constructor args for lm-eval's `local-completions` model (untokenized requests, fixed concurrency)."""
    return {
        "base_url": completions_url(base_url),
        "model": model,
        "num_concurrent": NUM_CONCURRENT,
        "tokenized_requests": False,
    }


def extract_metrics(results, task):
    """The task's metric row from a `simple_evaluate` results dict, minus the presentation-only alias."""
    row = results["results"][task]
    return {key: value for key, value in row.items() if key != "alias"}


def build_report(label, model, task, limit, seed, metrics, lm_eval_version):
    """The gate's output record. Deliberately timestamp-free so identical runs produce identical JSON."""
    return {
        "label": label,
        "model": model,
        "task": task,
        "limit": limit,
        "seed": seed,
        "metrics": metrics,
        "lm_eval_version": lm_eval_version,
    }


def main():
    setup_cli_logging()
    parser = argparse.ArgumentParser(description="Score a fixed eval subset against an OpenAI-compatible endpoint via lm-eval.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API root (default: %(default)s)")
    parser.add_argument("--model", default="google/gemma-4-12B-it", help="Served model name (default: %(default)s)")
    parser.add_argument("--task", default="gsm8k", help="lm-eval task (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=200, help="Number of examples (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for all lm-eval RNGs (default: %(default)s)")
    parser.add_argument("--num-fewshot", type=int, default=None, help="Few-shot count (default: task default)")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path (default: stdout)")
    parser.add_argument("--label", default="unlabeled", help="Config label recorded in the output, e.g. emmy-fastmath")
    args = parser.parse_args()

    import lm_eval  # heavy (pulls datasets/evaluate); deferred so the module imports without it

    logger.info("Evaluating %s (%s) on %s [limit=%d] via %s", args.model, args.label, args.task, args.limit, args.base_url)
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=build_model_args(args.base_url, args.model),
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    report = build_report(
        args.label,
        args.model,
        args.task,
        args.limit,
        args.seed,
        extract_metrics(results, args.task),
        package_version("lm_eval"),
    )

    text = json.dumps(report, indent=2)
    if args.out is None:
        print(text)  # the final JSON is the script's product; everything else goes through the logger
    else:
        args.out.write_text(text + "\n")
        logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
