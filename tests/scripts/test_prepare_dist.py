"""The distribution staging step that `make wheel` and the publish workflow run.

Two things it must get right. Recipes live outside the `emmy` package, so a wheel only
carries them because this script copies in runnable recipe files and canonical model goldens,
not local benchmark output or ``RESULTS.md``. The README's repo-relative links
resolve on GitHub but 404 on PyPI, which renders it detached from the repo.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("prepare_dist", PROJECT_ROOT / "scripts" / "prepare_dist.py")
prepare_dist = importlib.util.module_from_spec(_spec)
sys.modules["prepare_dist"] = prepare_dist
_spec.loader.exec_module(prepare_dist)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A miniature repo root, with the script pointed at it."""
    monkeypatch.setattr(prepare_dist, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prepare_dist, "BUNDLED_RECIPES", tmp_path / "emmy" / "recipes")
    return tmp_path


def test_stages_recipe_files_without_experiment_artifacts(fake_repo):
    """Raw results, records, and reports must not reach the wheel."""
    model = fake_repo / "recipes" / "gemma-4-12B-it"
    model.mkdir(parents=True)
    (model / "recipe.yaml").write_text("model:\n  name: google/gemma-4-12b-it\n")
    results_dir = model / "results"
    results_dir.mkdir()
    (results_dir / "row.benchmark.log").write_text("raw output")
    (model / "row.experiment.yaml").write_text("schema_version: 1\n")
    (model / "RESULTS.md").write_text("# Results\n")

    assert prepare_dist.stage_recipes() == 1

    staged = fake_repo / "emmy" / "recipes"
    assert [p.name for p in staged.iterdir()] == ["gemma-4-12B-it"]
    assert (staged / "gemma-4-12B-it" / "recipe.yaml").is_file()
    staged_model = staged / "gemma-4-12B-it"
    assert not (staged_model / "results").exists()
    assert not (staged_model / "row.experiment.yaml").exists()
    assert not (staged_model / "RESULTS.md").exists()


def test_staging_is_idempotent(fake_repo):
    """Re-running must not accumulate recipes deleted since the last build."""
    model = fake_repo / "recipes" / "kept"
    model.mkdir(parents=True)
    (model / "recipe.yaml").write_text("model: {}\n")
    prepare_dist.stage_recipes()
    stale = fake_repo / "emmy" / "recipes" / "removed"
    stale.mkdir(parents=True)
    (stale / "recipe.yaml").write_text("model: {}\n")

    prepare_dist.stage_recipes()

    assert [p.name for p in (fake_repo / "emmy" / "recipes").iterdir()] == ["kept"]


def test_stages_only_runnable_recipes(fake_repo):
    for name, tags in (
        ("maintained", ["maintained"]),
        ("best-effort", ["best-effort"]),
        ("obsolete", ["obsolete"]),
        ("onboarding", ["onboarding", "untested"]),
    ):
        model = fake_repo / "recipes" / name
        model.mkdir(parents=True)
        (model / "recipe.yaml").write_text(f"tags: {tags!r}\nmodel:\n  huggingface: org/{name}\n")
        golden = model / "golden" / "rtx5090_sm120.yaml"
        golden.parent.mkdir()
        golden.write_text("gpu_name: NVIDIA GeForce RTX 5090\n")

    assert prepare_dist.stage_recipes() == 2
    staged = fake_repo / "emmy" / "recipes"
    assert sorted(p.name for p in staged.iterdir()) == ["best-effort", "maintained", "obsolete", "onboarding"]
    assert (staged / "maintained" / "recipe.yaml").is_file()
    assert (staged / "best-effort" / "recipe.yaml").is_file()
    assert not (staged / "obsolete" / "recipe.yaml").exists()
    assert not (staged / "onboarding" / "recipe.yaml").exists()
    assert all(
        (staged / name / "golden" / "rtx5090_sm120.yaml").is_file() for name in ("maintained", "best-effort", "obsolete", "onboarding")
    )


def test_refuses_to_build_a_recipe_less_package(fake_repo):
    (fake_repo / "recipes").mkdir()

    with pytest.raises(SystemExit):
        prepare_dist.stage_recipes()


def test_rewrites_relative_links_and_leaves_the_rest_alone(fake_repo):
    (fake_repo / "emmy").mkdir()
    (fake_repo / "STYLE.md").write_text("style")
    (fake_repo / "README.md").write_text(
        "See [the package](emmy/) and [STYLE.md](STYLE.md).\nAbsolute [docs](https://example.com/x) and anchor [top](#overview) stay put.\n"
    )

    assert prepare_dist.absolutize_readme() == 2

    text = (fake_repo / "README.md").read_text()
    assert "[the package](https://github.com/cloudrift-ai/emmy/tree/main/emmy)" in text
    assert "[STYLE.md](https://github.com/cloudrift-ai/emmy/blob/main/STYLE.md)" in text
    assert "[docs](https://example.com/x)" in text
    assert "[top](#overview)" in text


def test_a_link_to_a_missing_path_fails_the_build(fake_repo):
    (fake_repo / "README.md").write_text("[gone](does/not/exist.md)\n")

    with pytest.raises(SystemExit, match="does/not/exist.md"):
        prepare_dist.absolutize_readme()


def test_every_readme_link_in_this_repo_resolves():
    """The real README — a moved file would otherwise only surface during a release."""
    text = (PROJECT_ROOT / "README.md").read_text()

    missing = [t for t in prepare_dist.RELATIVE_LINK.findall(text) if not (PROJECT_ROOT / t).exists()]

    assert missing == []
