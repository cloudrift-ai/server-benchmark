"""Benchmark task enumeration lifecycle behavior."""

from emmy.benchmark.tasks import enumerate_tasks


def test_obsolete_recipe_is_skipped(tmp_path, caplog):
    recipe = tmp_path / "obsolete"
    recipe.mkdir()
    (recipe / "recipe.yaml").write_text("tags:\n  - obsolete\nmodel:\n  huggingface: org/old\nmatrices:\n  deploy.gpu: NVIDIA H200 141GB\n")

    assert enumerate_tasks([str(recipe)]) == []
    assert "is obsolete, skipping" in caplog.text


def test_best_effort_recipe_is_enumerated(tmp_path):
    recipe = tmp_path / "best-effort"
    recipe.mkdir()
    (recipe / "recipe.yaml").write_text(
        "tags:\n  - best-effort\nmodel:\n  huggingface: org/useful\nmatrices:\n  deploy.gpu: NVIDIA H200 141GB\n"
    )

    tasks = enumerate_tasks([str(recipe)])

    assert len(tasks) == 1
    assert tasks[0].recipe.lifecycle == "best-effort"
