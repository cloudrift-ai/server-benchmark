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
from types import SimpleNamespace

import pytest

from emmy.publish import model_slug as library_model_slug

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
    """The library owns the rule; the gate and shell adapters must agree with it."""
    for model in ("google/gemma-4-12B-it", "Qwen/Qwen3-Embedding-0.6B", "org/Weird Model!!Name"):
        assert library_model_slug(model) == slug(model)
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


@pytest.mark.parametrize(
    "model, want",
    [
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", ("turboderp/GLM-4.5-Air-exl3", "2.25bpw")),
        ("google/gemma-4-12B-it", ("google/gemma-4-12B-it", None)),
        ("org/model@", ("org/model", None)),  # an empty tag is no claim, not an empty revision
    ],
)
def test_revision_splits_off_the_provenance_tag(model, want):
    assert csg.split_revision(model) == want


@pytest.mark.parametrize(
    "golden_model, revision, want",
    [
        # The defect: a `<repo>@<revision>` tag must still match its own repo. Run through
        # `model_slug` whole it sanitizes to `glm-4.5-air-exl3-2.25bpw`, which matches nothing —
        # so the release gate reported ZERO coverage for a card that had some.
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", "2.25bpw", True),
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", "2.0bpw", False),  # another rung is another kernel set
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", None, False),  # unevaluable is not coverage
        # An UNTAGGED golden makes no revision claim, so it covers every revision of its repo —
        # this is what keeps every pre-existing (untagged) golden file behaving exactly as before.
        ("turboderp/GLM-4.5-Air-exl3", "2.25bpw", True),
        ("turboderp/GLM-4.5-Air-exl3", None, True),
        # The repo half still binds, tagged or not.
        ("turboderp/GLM-4.5-Air-exl3-FP8@2.25bpw", "2.25bpw", False),
    ],
)
def test_revision_tagged_goldens_match_their_own_revision_only(golden_model, revision, want):
    assert csg.covers(golden_model, "glm-4.5-air-exl3", revision) is want


@pytest.mark.parametrize(
    "golden_rev, target_rev, want",
    [
        ("2.25bpw", "2.25bpw", True),
        ("abc1234def5678", "abc1234", True),  # `git rev-parse --short` names the same commit
        ("abc1234", "abc1234def5678", True),
        ("abc12", "abc1234def5678", False),  # too short to be a sha abbreviation
        ("main", "main-2", False),  # branch names never prefix-match
        ("2.25bpw", "a1b2c3d4e5f6", False),  # a branch and a sha are not resolvable to each other
    ],
)
def test_revision_comparison(golden_rev, target_rev, want):
    assert csg.revision_matches(golden_rev, target_rev) is want


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
OPTIONAL_KEYS = {
    "SERVE_WARM_SHAPES",
    "SERVE_REVISION",
    "SERVE_QUANT",
    "SERVE_CAPTURE_SIZES",
    "SERVE_EXTRA_ARGS",
    "SERVE_EMBED_HOST",
    "SERVE_PREFILL_CAPACITY",
    "SERVE_PREFILL_BUCKET",
    "SERVE_M1_TIER",
    "SERVE_STATIC_ONLY",
}


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


def test_serve_config_guard_rejects_unfilled_release_placeholders(tmp_path):
    config = tmp_path / "draft.env"
    config.write_text("SERVE_MODEL=org/model\nSERVE_REVISION=__FILL_FINAL_HF_REVISION_40_SHA__\n")
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS")}
    result = subprocess.run(
        ["make", "--no-print-directory", "serve-config-guard", f"SERVE_CONFIG={config}"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0
    assert "still contains __FILL_FINAL_" in result.stdout + result.stderr


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


def render_warm_shape_env(spec: str, env: dict[str, str]) -> list[str]:
    """Run only warm.sh's pure env renderer, without touching Docker or a GPU."""
    body = (SERVE_DIR / "warm.sh").read_text()
    functions = "runner_env() {" + body.split("runner_env() {", 1)[1].split("\n\nmkdir -p", 1)[0]
    result = subprocess.run(
        ["bash", "-c", f'{functions}\nshape_env "$SPEC"'],
        capture_output=True,
        text=True,
        env={**env, "SPEC": spec},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def render_serve_runner_env(tmp_path: Path, env: dict[str, str]) -> list[str]:
    stub_dir = tmp_path / "env-stub"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "python3"
    stub.write_text(
        "#!/bin/sh\n"
        "for name in EMMY_GEN_EMBED_HOST EMMY_GEN_PREFILL_CAPACITY EMMY_GEN_PREFILL_BUCKET EMMY_GEN_M1_TIER; do\n"
        '  value=$(printenv "$name") && printf "%s=%s\\n" "$name" "$value" || printf "%s=UNSET\\n" "$name"\n'
        "done\n"
    )
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


def test_warm_shape_env_keeps_runner_memory_lane_and_prefill_override_wins():
    env = {
        "SERVE_DECODE_BUCKET": "32",
        "SERVE_MAX_NUM_BATCHED_TOKENS": "64",
        "SERVE_EMBED_HOST": "1",
        "SERVE_PREFILL_CAPACITY": "64",
        "SERVE_PREFILL_BUCKET": "0",
        "SERVE_M1_TIER": "1",
    }
    pinned = render_warm_shape_env("", env)
    override = render_warm_shape_env("8:2048:2056", env)
    assert pinned == [
        "-e",
        "EMMY_GEN_EMBED_HOST=1",
        "-e",
        "EMMY_GEN_PREFILL_CAPACITY=64",
        "-e",
        "EMMY_GEN_M1_TIER=1",
        "-e",
        "EMMY_GEN_DECODE_BUCKET=32",
        "-e",
        "EMMY_GEN_PREFILL_BUCKET=0",
        "-e",
        "SERVE_MAX_NUM_BATCHED_TOKENS=64",
    ]
    assert "EMMY_GEN_PREFILL_BUCKET=0" not in override
    assert "EMMY_GEN_PREFILL_BUCKET=2048" in override
    assert "EMMY_GEN_DECODE_BUCKET=8" in override
    assert "SERVE_MAX_NUM_BATCHED_TOKENS=2056" in override


def test_serve_sh_unsets_empty_optional_runner_envs(tmp_path):
    env = {
        "SERVE_MODEL": "org/model",
        "SERVE_MAX_MODEL_LEN": "128",
        "SERVE_MAX_NUM_BATCHED_TOKENS": "64",
        "SERVE_GPU_MEM_UTIL": "0.9",
        "EMMY_GEN_EMBED_HOST": "",
        "EMMY_GEN_PREFILL_CAPACITY": "",
        "EMMY_GEN_PREFILL_BUCKET": "",
        "EMMY_GEN_M1_TIER": "",
    }
    assert render_serve_runner_env(tmp_path, env) == [
        "EMMY_GEN_EMBED_HOST=UNSET",
        "EMMY_GEN_PREFILL_CAPACITY=UNSET",
        "EMMY_GEN_PREFILL_BUCKET=UNSET",
        "EMMY_GEN_M1_TIER=UNSET",
    ]


def test_runner_memory_config_is_warm_bake_verify_cache_parity():
    make = (PROJECT_ROOT / "Makefile").read_text()
    warm = (SERVE_DIR / "warm.sh").read_text()
    dockerfile = (SERVE_DIR / "Dockerfile").read_text()
    verify = (SERVE_DIR / "verify.sh").read_text()
    mappings = {
        "SERVE_EMBED_HOST": ("EMBED_HOST", "EMMY_GEN_EMBED_HOST"),
        "SERVE_PREFILL_CAPACITY": ("PREFILL_CAPACITY", "EMMY_GEN_PREFILL_CAPACITY"),
        "SERVE_PREFILL_BUCKET": ("PREFILL_BUCKET", "EMMY_GEN_PREFILL_BUCKET"),
        "SERVE_M1_TIER": ("M1_TIER", "EMMY_GEN_M1_TIER"),
    }
    for serve, (build_arg, emmy) in mappings.items():
        assert serve in OPTIONAL_KEYS
        assert serve in warm and emmy in warm
        assert f"ARG {build_arg}=" in dockerfile and f'{emmy}="${{{build_arg}}}"' in dockerfile
        assert f"--build-arg {build_arg}=$({serve})" in make
        assert serve in verify and emmy in verify


def test_serving_images_carry_canonical_publication_labels():
    make = (PROJECT_ROOT / "Makefile").read_text()
    emmy_dockerfile = (SERVE_DIR / "Dockerfile").read_text()
    onecat_dockerfile = (PROJECT_ROOT / "docker" / "1cat-vllm-sm70" / "Dockerfile.triton-cache").read_text()
    for build_arg in ("PUBLISH_FAMILY", "PUBLISH_VERSION", "PUBLISH_REVISION", "MODEL"):
        assert f"ARG {build_arg}" in emmy_dockerfile
        assert f"ARG {build_arg}" in onecat_dockerfile
    for label in (
        "ai.emmy.publish.family",
        "ai.emmy.model.id",
        "ai.emmy.model.revision",
        "ai.emmy.target.gpu",
        "org.opencontainers.image.version",
        "org.opencontainers.image.revision",
    ):
        assert label in emmy_dockerfile
        assert label in onecat_dockerfile
    assert "--build-arg PUBLISH_FAMILY=vllm-emmy" in make
    assert "--build-arg PUBLISH_VERSION=" in make
    assert "--build-arg PUBLISH_REVISION=" in make
    push_body = make.split("serve-push:", 1)[1].split("\nbench:", 1)[0]
    assert "emmy publish <recipe>" in push_body
    assert "docker push" not in push_body


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
    result = _gate("google/gemma-4-12B-it")
    assert result.returncode == 0, result.stdout
    assert "google/gemma-4-12B" in result.stdout  # provenance is reported, not just a count


def _gate(model: str, *args: str, gpu: str = "NVIDIA GeForce RTX 5090"):
    cmd = [sys.executable, "scripts/check_serving_goldens.py", "--model", model, "--gpu", gpu, *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)


# The `@revision`-tagged golden file that made these cases real. It is tagged by sha
# rather than by the `2.25bpw` branch it was cut from, because the release pipeline pins
# `SERVE_REVISION` to a sha and the gate compares revisions exactly. Before the revision half of
# the matcher existed, every one of the three checks below reported "none recorded for this model".
_TAGGED_MODEL = "turboderp/GLM-4.5-Air-exl3"
_TAGGED_REV = "6a309ed6d606fc0154e6e1aeb0912cd3c25534fe"


def _tagged_gate(monkeypatch, capsys, model: str, *args: str):
    """Run the CLI against one synthetic tagged record, independent of shipped models."""
    from emmy.compiler.pipeline.search import golden

    record = SimpleNamespace(
        gpu_name="NVIDIA GeForce RTX 5090",
        compute_cap=(12, 0),
        model=f"{_TAGGED_MODEL}@{_TAGGED_REV}",
        origin_ops=("matmul",),
    )
    monkeypatch.setattr(golden, "GOLDEN_RECORDS", [record])
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_serving_goldens.py", "--model", model, "--gpu", record.gpu_name, *args],
    )
    return SimpleNamespace(returncode=csg.main(), stdout=capsys.readouterr().out)


def test_gate_matches_a_revision_tagged_golden_set(monkeypatch, capsys):
    result = _tagged_gate(monkeypatch, capsys, _TAGGED_MODEL, "--revision", _TAGGED_REV)
    assert result.returncode == 0, result.stdout
    assert f"{_TAGGED_MODEL}@{_TAGGED_REV}" in result.stdout
    assert _TAGGED_REV in result.stdout


def test_gate_rejects_a_different_revision_of_the_same_repo(monkeypatch, capsys):
    """An EXL3 rung differs in exactly the per-tensor bit allocation the shape keys carry, so
    another rung's goldens are not coverage — and the message must say THAT, not "no goldens"."""
    result = _tagged_gate(monkeypatch, capsys, _TAGGED_MODEL, "--revision", "2.0bpw")
    assert result.returncode == 1
    assert f"recorded against {_TAGGED_REV}, not '2.0bpw'" in result.stdout
    assert "none recorded for" not in result.stdout


def test_gate_fails_loudly_when_the_revision_cannot_be_evaluated(monkeypatch, capsys):
    """Tagged goldens + an unnamed release revision is UNEVALUABLE. The gate must not report it
    as zero coverage (the pre-fix behaviour) and must not report it as OK either."""
    result = _tagged_gate(monkeypatch, capsys, _TAGGED_MODEL)
    assert result.returncode == 1
    assert "CANNOT be evaluated" in result.stdout
    assert "--revision" in result.stdout


def test_gate_accepts_the_tagged_spelling_on_the_model_flag(monkeypatch, capsys):
    """`<repo>@<rev>` is the golden file's spelling; typing it at the CLI must not mangle the slug."""
    assert _tagged_gate(monkeypatch, capsys, f"{_TAGGED_MODEL}@{_TAGGED_REV}").returncode == 0


def test_untagged_goldens_cover_a_revision_pinned_release():
    """gemma-4's goldens carry no revision tag, so pinning SERVE_REVISION cannot un-cover them —
    the compatibility half of the rule, and what keeps every shipped golden file working."""
    result = _gate("google/gemma-4-12B-it", "--revision", "a1b2c3d4e5f6a1b2")
    assert result.returncode == 0, result.stdout


def test_makefile_forwards_the_pinned_revision_to_the_gate():
    """The gate can only apply the revision rule if the release pipeline tells it the revision.
    `make serve-goldens` is the only caller in the workflow."""
    body = (PROJECT_ROOT / "Makefile").read_text().split("serve-goldens:", 1)[1].split("\nserve-warm:", 1)[0]
    assert "check_serving_goldens.py" in body
    assert "--revision" in body and "SERVE_REVISION" in body
    assert "--strict-major-gaps" in body and "--release-config" in body
    assert "--checkpoint" in body and "CHECKPOINT" in body
    assert "--static-only-release" in body and "SERVE_STATIC_ONLY" in body


def test_release_config_widths_include_pinned_and_warm_decode_prefill(tmp_path):
    config = tmp_path / "model.env"
    config.write_text(
        "SERVE_DECODE_BUCKET=32\nSERVE_PREFILL_CAPACITY=96\nSERVE_PREFILL_BUCKET=0\n"
        'SERVE_WARM_SHAPES="8:2048:2056 64::4096 32:512:544:fm"\n'
    )
    assert csg.release_widths(config) == (8, 32, 64, 512, 2048)


def test_release_config_uses_capacity_as_default_prefill_bucket(tmp_path):
    config = tmp_path / "model.env"
    config.write_text("SERVE_DECODE_BUCKET=32\nSERVE_PREFILL_CAPACITY=96\n")
    assert csg.release_widths(config) == (32, 96)


def test_release_config_rejects_zero_prefill_capacity(tmp_path):
    config = tmp_path / "model.env"
    config.write_text("SERVE_DECODE_BUCKET=32\nSERVE_PREFILL_CAPACITY=0\n")
    with pytest.raises(ValueError, match="SERVE_PREFILL_CAPACITY must be positive"):
        csg.release_widths(config)


def test_static_only_gate_rejects_release_config_without_m1_proof(tmp_path):
    config = tmp_path / "model.env"
    config.write_text(
        "SERVE_STATIC_ONLY=1\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=1\n"
        "SERVE_DECODE_BUCKET=1\n"
        "SERVE_PREFILL_CAPACITY=1\n"
        "SERVE_PREFILL_BUCKET=0\n"
        "SERVE_M1_TIER=0\n"
        "SERVE_CAPTURE_SIZES=[1]\n"
    )
    result = _gate(
        "google/gemma-4-12B-it",
        "--strict-major-gaps",
        "--static-only-release",
        "--release-config",
        str(config),
    )
    assert result.returncode == 2
    assert "static-only release scope is unsafe" in result.stdout
    assert "SERVE_M1_TIER=1" in result.stdout


def test_local_checkpoint_override_preserves_release_provenance():
    source, provenance = csg.release_capture_source("cloudriftai/model-exl3", "0123456789abcdef", "/local/exact")
    assert source == "/local/exact"
    assert provenance == "cloudriftai/model-exl3@0123456789abcdef"


def test_strict_release_twin_audit_fails_major_gaps(monkeypatch):
    import emmy.compiler.pipeline.search.audit as audit
    import emmy.serving.twins as twins

    class MajorKey:
        is_warp = True
        reduce_max = 128

        def __str__(self):
            return "major-key"

    key = MajorKey()
    seen = {}

    def fake_capture(model, **kwargs):
        seen.update(model=model, **kwargs)
        return {"pre1": object()}

    monkeypatch.setattr(twins, "capture_in_model_graphs", fake_capture)
    monkeypatch.setattr(
        audit,
        "audit_card",
        lambda _graphs, _gpu, _cap: {
            "pre1": [{"verdict": "GAP", "key": key}],
        },
    )

    assert not csg.audit_release_twins(
        "/local/exact-checkpoint",
        "NVIDIA GeForce RTX 5090",
        [(12, 0)],
        (32, 512),
        provenance="org/model@revision",
    )
    assert seen == {"model": "/local/exact-checkpoint", "extra_widths": (32, 512)}


def test_static_only_release_twin_audit_captures_only_m1(monkeypatch):
    import emmy.compiler.pipeline.search.audit as audit
    import emmy.serving.twins as twins

    seen = {}

    def fake_capture(model, **kwargs):
        seen.update(model=model, **kwargs)
        return {"pre1": object()}

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    monkeypatch.setattr(audit, "audit_card", lambda _graphs, _gpu, _cap: {"pre1": [{"verdict": "MATCH", "key": None}]})

    assert csg.audit_release_twins(
        "/local/exact-checkpoint",
        "NVIDIA GeForce RTX 5090",
        [(12, 0)],
        (),
        provenance="org/model@revision",
        static_only=True,
    )
    assert seen == {
        "model": "/local/exact-checkpoint",
        "decode_bucket": 1,
        "prefill_bucket": 0,
        "extra_widths": (),
        "static_only": True,
    }


def test_strict_release_twin_audit_fails_when_provider_rejects_widths(monkeypatch, capsys):
    import emmy.serving.twins as twins

    monkeypatch.setattr(
        twins,
        "capture_in_model_graphs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("fixed architecture width")),
    )
    assert not csg.audit_release_twins("org/deepseek", "Tesla V100-SXM3-32GB", [(7, 0)], (64,))
    assert "FAIL: strict in-model coverage cannot represent this release: fixed architecture width" in capsys.readouterr().out


def test_preflight_enumeration_applies_the_same_revision_rule():
    """The preflight renders + nvcc-compiles the model's golden set, so enumerating another
    rung's shapes would gate a rental on the wrong kernels. Same matcher, revision included."""
    body = (PROJECT_ROOT / "scripts" / "preflight_serving_kernels.sh").read_text()
    assert "select_goldens" in body and "REVISION" in body
