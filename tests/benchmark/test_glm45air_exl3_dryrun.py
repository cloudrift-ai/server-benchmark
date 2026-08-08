"""Expansion + consistency tests for the GLM-4.5-Air EXL3 recipe and its benchmark lanes.

The serving recipe and the experiment's emmy lane describe the same server twice — one as a
deployment config, one as a benchmarked lane — so the thing worth pinning is that they agree.
Everything they agree ON is still an estimate at this point (the release workflow's headroom
sweep has not run), which is exactly why a drift between the two files must not pass silently.

The rest guards the checkpoint pin, which is load-bearing in a way a reader cannot see: the
2.25 bpw rung is a BRANCH, and without ``--revision`` the same repo id serves the 2.00 rung,
a measurably worse model.
"""

import os

import pytest

from emmy.benchmark.command_workload import build_substitution_map, render_command
from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe.recipe import load_recipe

RECIPE = "GLM-4.5-Air-EXL3"
EXP = "experiments/GLM-4.5-Air-EXL3"

# The pinned rung: turboderp/GLM-4.5-Air-exl3 branch 2.25bpw. See the recipe header and
# plans/vq-phase0-findings.md for the quality numbers that chose it over 2.00.
REVISION = "6a309ed6d606fc0154e6e1aeb0912cd3c25534fe"

# The workload grid every lane shares: the Phase 0 concurrency/prompt-count pattern at two
# input lengths, so the short-input cells compare directly against the re-measured baseline.
POINTS = {(inp, conc) for inp in (512, 2048) for conc in (1, 4, 8, 16)}
PROMPTS = {1: 8, 4: 24, 8: 48, 16: 64}


def _exp(project_root, name):
    return os.path.join(project_root, EXP, name)


def test_recipe_is_a_single_serving_variant(recipes_dir):
    """One variant, no sweep — a recipe answers "how do I serve this", not "which is better"."""
    tasks = enumerate_tasks([os.path.join(recipes_dir, RECIPE)])
    assert len(tasks) == 1, f"serving recipe expanded to {len(tasks)} variants"

    raw = open(os.path.join(recipes_dir, RECIPE, "recipe.yaml")).read()
    body = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))
    assert "benchmark:" not in body and "benchmark." not in body, "serving recipe grew a workload grid"


def test_recipe_pins_the_rung_and_the_exl3_overrides(recipes_dir):
    """Three things vLLM cannot infer and one it actively refuses.

    The revision selects the rung (a branch, not `main`). `quantization_config: null` is what
    lets the boot happen at all — vLLM has no EXL3 quantization method and aborts on the
    checkpoint's own config. The capture ladder is capped at 1 because MoE decode capture is
    fixed-slot: wider decode steps host-sync and cannot be graph-captured.
    """
    llm = enumerate_tasks([os.path.join(recipes_dir, RECIPE)])[0].recipe.engine.llm
    args = llm.vllm.extra_args
    assert f"--revision {REVISION}" in args
    assert '"quantization_config":null' in args.replace(" ", "")
    assert '"architectures":["EmmyGenModel"]' in args.replace(" ", "")
    assert '"cudagraph_capture_sizes":[1]' in args.replace(" ", "")
    assert "--kv-cache-dtype fp8_e4m3" in args, "fp8 KV is not optional here — fp16 halves an already tiny pool"


@pytest.mark.parametrize(
    ("name", "variants"),
    [("serving_rtx5090", 9), ("serving_exllamav3_rtx5090", 16), ("serving_llamacpp_rtx5090", 8)],
)
def test_lanes_are_command_recipes_with_the_expected_shape(project_root, name, variants):
    """8 workload points per lane; the emmy grid plus the single stock load-failure record is 9,
    and the exllamav3 lane runs the grid twice, once per cache precision."""
    recipe = load_recipe(_exp(project_root, name))
    assert recipe.kind == "command", name
    tasks = enumerate_tasks([_exp(project_root, name)])
    assert len(tasks) == variants, name
    for t in tasks:
        assert t.recipe.deploy.gpu == "NVIDIA GeForce RTX 5090"
        b = t.recipe.benchmark
        assert (b.random_input_len, b.max_concurrency) in POINTS
        assert b.num_prompts == PROMPTS[b.max_concurrency]
        assert b.random_output_len == 128


@pytest.mark.parametrize("name", ["serving_rtx5090", "serving_exllamav3_rtx5090", "serving_llamacpp_rtx5090"])
def test_every_variant_renders_its_command(project_root, name):
    """`render_command` raises on a template variable no matrix key supplies — the failure mode
    of a command recipe, and one that otherwise surfaces only after a VM has been provisioned."""
    for task in enumerate_tasks([_exp(project_root, name)]):
        subs = build_substitution_map(task.variant, [0], "/repo", "/task")
        rendered = render_command(task.recipe.command.run, subs)
        assert "scripts/bench_serve_sweep.py" in rendered or "stock" in str(task.variant)


def test_emmy_lane_serves_the_shape_the_recipe_deploys(project_root, recipes_dir):
    """The experiment measures the deployment config, so the two must not drift. Every value
    checked here is a `TODO(Phase 5)` estimate in the recipe; when the headroom sweep pins them,
    this test is what forces both files to move together."""
    llm = enumerate_tasks([os.path.join(recipes_dir, RECIPE)])[0].recipe.engine.llm
    lane = open(os.path.join(_exp(project_root, "serving_rtx5090"), "recipe.yaml")).read()
    body = "\n".join(ln for ln in lane.splitlines() if not ln.lstrip().startswith("#"))

    assert f"--max-model-len {llm.context_length}" in body
    assert f"--gpu-memory-utilization {llm.gpu_memory_utilization}" in body
    for flag in ("--max-num-batched-tokens", "--kv-cache-dtype"):
        value = llm.vllm.extra_args.split(flag, 1)[1].split()[0]
        assert f"{flag} {value}" in body, f"{flag} differs between the recipe and the emmy lane"
    for key, value in llm.vllm.extra_env.items():
        assert f"-e {key}={value}" in body, f"{key} differs between the recipe and the emmy lane"
