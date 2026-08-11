"""Benchmark task enumeration lifecycle behavior."""

from emmy.benchmark.tasks import enumerate_tasks


def test_obsolete_recipe_is_skipped(tmp_path, caplog):
    recipe = tmp_path / "obsolete"
    recipe.mkdir()
    (recipe / "recipe.yaml").write_text("tags:\n  - obsolete\nmodel:\n  huggingface: org/old\nmatrices:\n  deploy.gpu: NVIDIA H200 141GB\n")

    assert enumerate_tasks([str(recipe)]) == []
    assert "is obsolete, skipping" in caplog.text
