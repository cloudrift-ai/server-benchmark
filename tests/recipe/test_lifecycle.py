"""Recipe lifecycle tag validation."""

import pytest

from emmy.recipe.lifecycle import recipe_is_runnable, recipe_lifecycle, validate_recipe_tags


def test_legacy_and_maintained_recipes_are_runnable():
    assert recipe_is_runnable({})
    assert recipe_is_runnable({"tags": ["maintained"]})
    assert recipe_lifecycle({"tags": ["maintained"]}) == "maintained"


@pytest.mark.parametrize("tags", [["obsolete"], ["onboarding", "untested"]])
def test_obsolete_and_onboarding_recipes_are_disabled(tags):
    assert not recipe_is_runnable({"tags": tags})


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ("maintained", "list of strings"),
        (["Maintained"], "lowercase kebab-case"),
        (["maintained", "maintained"], "unique"),
        (["maintained", "obsolete"], "at most one lifecycle"),
        (["onboarding"], "must appear together"),
        (["untested"], "must appear together"),
    ],
)
def test_rejects_invalid_recipe_tags(tags, message):
    with pytest.raises(ValueError, match=message):
        validate_recipe_tags(tags)
