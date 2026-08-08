"""Resolution of the recipes shipped inside an installed wheel.

`emmy/recipes/` only exists after `scripts/prepare_dist.py --recipes` stages it, so a dev
checkout may or may not have it. Every test here fakes the packaged directory rather than
reading the real one, so the suite behaves the same either way.
"""

import pytest

from emmy.recipe import bundled
from emmy.recipe.bundled import resolve_recipe_dir


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """Stand in for the packaged emmy/recipes/, returning its path."""
    packaged = tmp_path / "packaged"
    (packaged / "Qwen3-Embedding-0.6B").mkdir(parents=True)
    (packaged / "Qwen3-Embedding-0.6B" / "recipe.yaml").write_text("model:\n  name: Qwen/Qwen3-Embedding-0.6B\n")
    monkeypatch.setattr(bundled, "files", lambda _package: packaged)
    return packaged


def test_existing_directory_is_used_as_given(tmp_path, monkeypatch):
    """A real path wins outright — no bundled lookup, no copying."""
    local = tmp_path / "my-recipe"
    local.mkdir()
    monkeypatch.chdir(tmp_path)

    assert resolve_recipe_dir(str(local)) == str(local)


def test_bundled_name_materializes_a_local_copy(tmp_path, monkeypatch, fake_package):
    """deploy writes its compose file into the recipe dir, so site-packages will not do."""
    monkeypatch.chdir(tmp_path)

    resolved = resolve_recipe_dir("Qwen3-Embedding-0.6B")

    assert resolved == "Qwen3-Embedding-0.6B"
    copied = tmp_path / "Qwen3-Embedding-0.6B" / "recipe.yaml"
    assert copied.is_file()
    assert "Qwen3-Embedding-0.6B" in copied.read_text()


def test_local_directory_shadows_a_bundled_name(tmp_path, monkeypatch, fake_package):
    """An edited working copy must not be silently overwritten by the shipped one."""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "Qwen3-Embedding-0.6B"
    local.mkdir()
    (local / "recipe.yaml").write_text("model:\n  name: edited\n")

    resolve_recipe_dir("Qwen3-Embedding-0.6B")

    assert "edited" in (local / "recipe.yaml").read_text()


def test_unknown_name_lists_what_is_available(tmp_path, monkeypatch, fake_package):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Qwen3-Embedding-0.6B"):
        resolve_recipe_dir("no-such-recipe")


def test_no_bundled_recipes_when_the_package_ships_none(monkeypatch):
    """A source checkout has no emmy.recipes; that is not an error, just an empty list."""

    def _missing(_package):
        raise ModuleNotFoundError("emmy.recipes")

    monkeypatch.setattr(bundled, "files", _missing)

    assert bundled.bundled_names() == []
