"""Dry-run + expansion tests for the gemma-4-12B article experiments.

``experiments/gemma-4-12B/`` consolidates the blog article's benchmark infra: the
three-lane serving A/B (stock / emmy / emmy-fastmath) pinned to the article's five
workload points, the llama.cpp lane as a command recipe, the per-kernel golden-set
runs for both cards, and the accumulation-error sweep. These tests pin the variant
counts and the lane-distinguishing config so a recipe edit can't silently change
what the article's repro commands run.
"""

import os

from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe.recipe import load_recipe

EXP = "experiments/gemma-4-12B"


def _exp(project_root, name):
    return os.path.join(project_root, EXP, name)


def test_serving_ab_expands_to_18_lane_points(project_root):
    """6 stock + (5 + c64) emmy + (5 + c64) fastmath = 18 variants on the article grid
    (the 4K curve + the 8192/256 RAG point + the 256/256 API point + the c=1 diagnostic)."""
    tasks = enumerate_tasks([_exp(project_root, "serving_rtx5090")])
    assert len(tasks) == 18

    for t in tasks:
        b = t.recipe.benchmark
        assert b.seed == 0 and b.temperature == 0 and b.ignore_eos is True
        assert t.recipe.engine.llm.context_length == 8448

    # The article's six workload points, per lane.
    points = {(t.recipe.benchmark.random_input_len, t.recipe.benchmark.max_concurrency) for t in tasks}
    assert points == {(256, 1), (256, 64), (4096, 1), (4096, 4), (4096, 8), (8192, 4)}

    # Single-stream points carry repeats=3 (mean/stddev); batched points run once.
    for t in tasks:
        want = 3 if t.recipe.benchmark.max_concurrency == 1 else 1
        assert t.recipe.benchmark.repeats == want

    # Lane split: 6 stock variants, 12 emmy variants (6 + 6 fastmath). Equal-tuning
    # protocol: EVERY lane at util 0.96; the emmy cells carry the two documented
    # per-workload knobs — the decode bucket matched to the concurrency, and the 2048
    # chunk quantum (EMMY_GEN_PREFILL_BUCKET=2048 + mnbt 2056) on the mixed 4K/4K
    # c=4/c=8 cells.
    stock = [t for t in tasks if "vllm-openai" in t.recipe.engine.llm.vllm.image]
    emmy = [t for t in tasks if "vllm-emmy" in t.recipe.engine.llm.vllm.image]
    assert len(stock) == 6 and len(emmy) == 12
    for t in tasks:
        assert t.recipe.engine.llm.gpu_memory_utilization == 0.96
    for t in emmy:
        env = t.recipe.engine.llm.vllm.extra_env or ""
        conc = t.recipe.benchmark.max_concurrency
        if conc == 64:
            assert "EMMY_GEN_DECODE_BUCKET=64" in env
        elif conc == 1:
            assert "EMMY_GEN_DECODE_BUCKET=32" in env
        else:
            assert "EMMY_GEN_DECODE_BUCKET=8" in env
        args = t.recipe.engine.llm.vllm.extra_args or ""
        if t.recipe.benchmark.random_input_len == 4096 and conc in (4, 8):
            assert "EMMY_GEN_PREFILL_BUCKET=2048" in env
            assert "--max-num-batched-tokens 2056" in args
        else:
            assert "EMMY_GEN_PREFILL_BUCKET" not in env
    fm = [t for t in emmy if "EMMY_FAST_MATH=1" in (t.recipe.engine.llm.vllm.extra_env or "")]
    assert len(fm) == 6


def test_command_experiments_load(project_root):
    """The command experiments (llama.cpp lane, per-kernel runs, accum sweep) parse as
    command recipes with one variant per target card."""
    for name, gpu in [
        ("serving_llamacpp_rtx5090", "NVIDIA GeForce RTX 5090"),
        ("kernels_rtx5090", "NVIDIA GeForce RTX 5090"),
        ("kernels_rtx4090", "NVIDIA GeForce RTX 4090"),
        ("accum_error", "NVIDIA GeForce RTX 5090"),
    ]:
        r = load_recipe(_exp(project_root, name))
        assert r.kind == "command", name
        tasks = enumerate_tasks([_exp(project_root, name)])
        assert len(tasks) == 1, name
        assert tasks[0].recipe.deploy.gpu == gpu, name
