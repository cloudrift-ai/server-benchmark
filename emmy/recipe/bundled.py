"""Recipes shipped inside the installed package.

`pip install emmy-ml` has no repo checkout, so the wheel bundles every
`recipes/<model>/recipe.yaml` under `emmy/recipes/` (staged at build time by
`scripts/prepare_dist.py`). A bundled recipe is read-only — it lives in
site-packages, while `deploy` writes its compose file into the recipe directory
and `bench` creates run directories there — so referring to one by name
materializes a working copy in the current directory first.
"""

import shutil
from importlib.resources import files
from pathlib import Path


def bundled_names() -> list[str]:
    """Names of the recipes shipped with the installed package."""
    try:
        root = files("emmy.recipes")
    except ModuleNotFoundError:
        return []
    return sorted(entry.name for entry in root.iterdir() if (entry / "recipe.yaml").is_file())


def resolve_recipe_dir(name_or_path: str) -> str:
    """Return a usable recipe directory for a CLI `--recipe` value.

    An existing directory is used as given. Otherwise the value is looked up
    among the bundled recipes and copied into the current directory, because
    both `deploy` and `bench` write alongside the recipe they run.
    """
    if Path(name_or_path).is_dir():
        return name_or_path

    if name_or_path in bundled_names():
        target = Path(name_or_path)
        target.mkdir(parents=True)
        source = files("emmy.recipes") / name_or_path / "recipe.yaml"
        shutil.copyfile(str(source), target / "recipe.yaml")
        return str(target)

    available = bundled_names()
    hint = f" Bundled recipes: {', '.join(available)}." if available else ""
    raise FileNotFoundError(f"No recipe directory {name_or_path!r}.{hint}")
