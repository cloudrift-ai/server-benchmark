"""The serving-image release naming schema, its golden coverage gate, and the invocation
`serve.sh` renders.

Three invariants worth pinning. The **slug** names both the published image and the pinned
config file, so a slug that changes shape silently repoints a release at a different config
— the cache-key parity failure the whole warm/bake contract exists to prevent. The **golden
gate** decides whether a release is allowed to warm at all: no goldens for the (model, card)
pair means the warm bakes cold-greedy fork picks, which on unseeded projection shapes are
catastrophically slow rather than merely suboptimal. And the **rendered invocation** is the
cache key's other half: warm and release run the same `serve.sh` under the same config, so a
change to how that script spells its arguments moves every shipped image's kernel set.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVE_DIR = PROJECT_ROOT / "docker" / "vllm-emmy-serve"
SLUG_SCRIPT = SERVE_DIR / "model_slug.sh"
SERVE_SCRIPT = SERVE_DIR / "serve.sh"

_spec = importlib.util.spec_from_file_location("check_serving_goldens", PROJECT_ROOT / "scripts" / "check_serving_goldens.py")
csg = importlib.util.module_from_spec(_spec)
sys.modules["check_serving_goldens"] = csg
_spec.loader.exec_module(csg)


def slug(model: str) -> str:
    out = subprocess.run([str(SLUG_SCRIPT), model], capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.parametrize(
    "model, want",
    [
        ("google/gemma-4-12B-it", "gemma-4-12b-it"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen3-embedding-0.6b"),
        ("meta-llama/Llama-3.1-8B", "llama-3.1-8b"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b-chat-v1.0"),
        ("gemma-4-12B-it", "gemma-4-12b-it"),  # no org
        ("org/Weird Model!!Name", "weird-model-name"),  # runs of junk collapse to one dash
        ("org/--trim--", "trim"),  # docker rejects leading/trailing separators
    ],
)
def test_slug_schema(model, want):
    assert slug(model) == want


def test_slug_drops_org_so_forks_share_an_image_name():
    """Two orgs' copies of one checkpoint warm to the same kernels — deliberately one slug."""
    assert slug("google/gemma-4-12B-it") == slug("unsloth/gemma-4-12B-it")


def test_slug_rejects_empty_and_missing_args():
    """An id sanitizing to nothing would build the malformed tag `vllm-emmy-:0.23.0-sha`."""
    assert subprocess.run([str(SLUG_SCRIPT), "org/!!!"], capture_output=True).returncode == 1
    assert subprocess.run([str(SLUG_SCRIPT)], capture_output=True).returncode == 2


def test_python_slug_matches_the_shell_schema():
    """The gate shells out to the one implementation — assert it really agrees."""
    for model in ("google/gemma-4-12B-it", "Qwen/Qwen3-Embedding-0.6B", "org/Weird Model!!Name"):
        assert csg.model_slug(model) == slug(model)


@pytest.mark.parametrize(
    "golden_model, target, want",
    [
        ("google/gemma-4-12B", "gemma-4-12b-it", True),  # base checkpoint covers its fine-tune
        ("google/gemma-4-12B-it", "gemma-4-12b-it", True),  # exact
        ("unsloth/gemma-4-12B", "gemma-4-12b-it", True),  # org-blind, same shapes
        ("google/gemma-4-27B", "gemma-4-12b-it", False),  # different geometry
        ("google/gemma-4-12B-FP8", "gemma-4-12b-it", False),  # quant variant is not the base
        (None, "gemma-4-12b-it", False),  # untagged goldens never imply coverage
        ("", "gemma-4-12b-it", False),
    ],
)
def test_golden_model_coverage_rule(golden_model, target, want):
    assert csg.covers(golden_model, target) is want


def test_prefix_rule_respects_dash_boundaries():
    """`gemma-4-1` must not cover `gemma-4-12b` — a substring match would alias two models."""
    assert not csg.covers("google/gemma-4-1", "gemma-4-12b")


REQUIRED_KEYS = {
    "SERVE_MODEL",
    "SERVE_MAX_MODEL_LEN",
    "SERVE_MAX_NUM_BATCHED_TOKENS",
    "SERVE_GPU_MEM_UTIL",
    "SERVE_DECODE_BUCKET",
    "SERVE_GPU",
}
# Optional because every one of them defaults to the behaviour the pipeline had before it
# could express them: no revision pin, no quantization arm, the power-of-two capture ladder,
# no extra flags. A model that needs one and omits it is not a config error here — it is
# whatever that omission causes downstream (a wrong-rung bake, a rejected boot).
OPTIONAL_KEYS = {"SERVE_WARM_SHAPES", "SERVE_REVISION", "SERVE_QUANT", "SERVE_CAPTURE_SIZES", "SERVE_EXTRA_ARGS"}


def config_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#") and "=" in line
    )


def test_every_model_config_is_complete_and_named_by_its_slug():
    """models/<slug>.env is resolved from MODEL by the Makefile and the shell scripts, so a
    file whose name disagrees with its SERVE_MODEL is unreachable from `make serve-* MODEL=`."""
    configs = sorted(SERVE_DIR.glob("models/*.env"))
    assert configs, "no pinned release configs found"
    for path in configs:
        values = config_values(path)
        missing = REQUIRED_KEYS - values.keys()
        assert not missing, f"{path.name} is missing {sorted(missing)}"
        assert slug(values["SERVE_MODEL"]) == path.stem, f"{path.name} holds SERVE_MODEL={values['SERVE_MODEL']}"


def test_model_configs_hold_no_unknown_keys():
    """A key nothing reads is silent: `SERVE_REVISON=<sha>` would leave the release unpinned
    and warm the repo default, which is exactly the failure the pin exists to stop."""
    known = REQUIRED_KEYS | OPTIONAL_KEYS
    for path in sorted(SERVE_DIR.glob("models/*.env")):
        unknown = config_values(path).keys() - known
        assert not unknown, f"{path.name} sets {sorted(unknown)}, which no reader consumes"


def test_model_config_is_both_bash_sourceable_and_make_includable():
    """The config is read TWO ways — `include`d by the Makefile and `source`d by
    warm.sh/verify.sh — so a value with spaces must be quoted or bash runs its second word
    as a command (`GeForce: command not found`, which is how this was found). Assert both
    readers agree on the one value that has spaces."""
    for path in sorted(SERVE_DIR.glob("models/*.env")):
        sourced = subprocess.run(
            ["bash", "-c", f'set -euo pipefail; source "{path}"; printf "%s" "$SERVE_GPU"'],
            capture_output=True,
            text=True,
        )
        assert sourced.returncode == 0, f"{path.name} is not bash-sourceable: {sourced.stderr}"
        assert sourced.stdout, f"{path.name} sourced to an empty SERVE_GPU"

        # Strip the parent make's jobserver env: this suite is itself usually run under
        # `make test`, and a child make that inherits MAKEFLAGS can block waiting on a
        # jobserver FD that pytest-xdist never passed down.
        env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS")}
        made = subprocess.run(
            ["make", "--no-print-directory", "serve-config", f"MODEL={path.stem}"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=120,
        )
        assert made.returncode == 0, made.stderr
        gpu_line = next(ln for ln in made.stdout.splitlines() if ln.startswith("target GPU"))
        from_make = gpu_line.split("=", 1)[1].strip()
        assert from_make == sourced.stdout, f"{path.name}: make says {from_make!r}, bash says {sourced.stdout!r}"
        assert '"' not in from_make, "make kept the bash quotes — they would word-split into argv"


def render_serve_sh(tmp_path: Path, env: dict[str, str]) -> list[str]:
    """The argv `serve.sh` would exec, captured by shadowing `python3` with a stub that
    prints its arguments. The script ends in `exec python3 …`, so this is the real
    invocation, not a re-derivation of it."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "python3"
    stub.write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    stub.chmod(0o755)
    result = subprocess.run(
        ["sh", str(SERVE_SCRIPT)],
        capture_output=True,
        text=True,
        env={**env, "PATH": f"{stub_dir}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_serve_sh_renders_the_shipped_invocation_unchanged(tmp_path):
    """The pinned configs set none of the per-checkpoint knobs, so each must render exactly
    the invocation the shipped images were warmed with. Every argument here is a cubin
    cache-key input: a spelling that drifts by one character invalidates a released image."""
    for path in sorted(SERVE_DIR.glob("models/*.env")):
        values = config_values(path)
        if values.keys() & {"SERVE_REVISION", "SERVE_QUANT", "SERVE_CAPTURE_SIZES", "SERVE_EXTRA_ARGS"}:
            continue  # a config that opts in is covered by the EXL3/MoE case below
        assert render_serve_sh(tmp_path, values) == [
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            values["SERVE_MODEL"],
            "--runner",
            "generate",
            "--dtype",
            "float16",
            "--max-model-len",
            values["SERVE_MAX_MODEL_LEN"],
            "--max-num-batched-tokens",
            values["SERVE_MAX_NUM_BATCHED_TOKENS"],
            "--gpu-memory-utilization",
            values["SERVE_GPU_MEM_UTIL"],
            "--no-enable-prefix-caching",
            "--hf-overrides",
            '{"architectures": ["EmmyGenModel"]}',
            "--compilation-config",
            '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '
            '[1, 2, 4, 8, 16, 32, 64, 128, 256], "custom_ops": ["+rotary_embedding"]}',
        ], f"{path.name} no longer renders the invocation its image was warmed with"


def test_serve_sh_renders_the_quantized_moe_invocation(tmp_path):
    """A trellis-coded MoE checkpoint on a branch: the three arms that a dense, unquantized,
    default-branch model does not exercise. Without them the warm bakes the wrong rung
    (silently), or dies at config parsing, or captures a ladder the runner rejects."""
    argv = render_serve_sh(
        tmp_path,
        {
            "SERVE_MODEL": "turboderp/GLM-4.5-Air-exl3",
            "SERVE_REVISION": "6a309ed6d606fc0154e6e1aeb0912cd3c25534fe",
            "SERVE_MAX_MODEL_LEN": "8192",
            "SERVE_MAX_NUM_BATCHED_TOKENS": "2064",
            "SERVE_GPU_MEM_UTIL": "0.97",
            "SERVE_QUANT": "exl3",
            "SERVE_CAPTURE_SIZES": "[1]",
            "SERVE_EXTRA_ARGS": "--kv-cache-dtype fp8_e4m3",
        },
    )
    assert argv[argv.index("--model") + 1 : argv.index("--model") + 4] == [
        "turboderp/GLM-4.5-Air-exl3",
        "--revision",
        "6a309ed6d606fc0154e6e1aeb0912cd3c25534fe",
    ]
    # One spelling of the override, shared with emmy/commands/serve.py's json.dumps: vLLM has
    # no EXL3 quantization method and refuses the boot, yet nothing in the engine needs one.
    assert argv[argv.index("--hf-overrides") + 1] == json.dumps({"architectures": ["EmmyGenModel"], "quantization_config": None})
    assert '"cudagraph_capture_sizes": [1],' in argv[argv.index("--compilation-config") + 1]
    assert argv[-2:] == ["--kv-cache-dtype", "fp8_e4m3"], "SERVE_EXTRA_ARGS must word-split into flags"


def test_release_scripts_are_syntactically_valid():
    """`serve.sh` runs under the image's /bin/sh; warm.sh and verify.sh use bash arrays. A
    syntax error surfaces mid-release otherwise — warm.sh's first failure mode is hours in."""
    for script, shell in ((SERVE_SCRIPT, "sh"), (SERVE_DIR / "warm.sh", "bash"), (SERVE_DIR / "verify.sh", "bash")):
        assert subprocess.run([shell, "-n", str(script)], capture_output=True).returncode == 0, script.name


def test_gate_reports_a_card_with_no_goldens():
    result = subprocess.run(
        [sys.executable, "scripts/check_serving_goldens.py", "--model", "google/gemma-4-12B-it", "--gpu", "NVIDIA Made Up 9000"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1
    assert "no goldens recorded" in result.stdout


def test_gate_reports_a_tuned_card_with_no_goldens_for_this_model():
    """The distinction that matters to the operator: the card is tuned, just not for this."""
    result = subprocess.run(
        [sys.executable, "scripts/check_serving_goldens.py", "--model", "meta-llama/Llama-3.1-8B", "--gpu", "NVIDIA GeForce RTX 5090"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1
    assert "none recorded for" in result.stdout
    assert "models this card IS tuned for" in result.stdout


def test_gate_passes_for_the_shipped_gemma4_config():
    """The one model with a pinned config must pass its own gate on its pinned card."""
    result = subprocess.run(
        [sys.executable, "scripts/check_serving_goldens.py", "--model", "google/gemma-4-12B-it", "--gpu", "NVIDIA GeForce RTX 5090"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stdout
    assert "google/gemma-4-12B" in result.stdout  # provenance is reported, not just a count
