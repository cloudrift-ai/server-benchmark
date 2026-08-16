"""Resolve recipes from an editable checkout or installed package.

`pip install emmy-ml` has no repo checkout, so the wheel bundles every
`recipes/<model>/recipe.yaml` under `emmy/recipes/` (staged at build time by
`scripts/prepare_dist.py`). A bundled recipe is read-only — it lives in
site-packages, while `deploy` writes its compose file into the recipe directory
and `bench` creates a timestamped run directory — so referring to one by name
materializes a working copy in the current directory first.
"""

import shutil
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


def editable_recipe_root() -> Path | None:
    """Return the source recipe root when Emmy is imported from a checkout."""
    checkout = Path(__file__).resolve().parents[2]
    recipes = checkout / "recipes"
    if (checkout / "pyproject.toml").is_file() and recipes.is_dir():
        return recipes
    return None


@contextmanager
def bundled_root():
    """Yield a filesystem root for the recipes shipped with the package."""
    try:
        resource = files("emmy.recipes")
    except ModuleNotFoundError:
        yield None
        return
    with as_file(resource) as root:
        yield root


@contextmanager
def default_recipe_root():
    """Yield live checkout recipes for editable installs, otherwise the wheel bundle."""
    editable = editable_recipe_root()
    if editable is not None:
        yield editable
        return
    with bundled_root() as root:
        yield root


def bundled_names() -> list[str]:
    """Names of the recipes shipped with the installed package."""
    with bundled_root() as root:
        if root is None:
            return []
        return _recipe_names(root)


def _recipe_names(root: Path) -> list[str]:
    return sorted(entry.name for entry in root.iterdir() if (entry / "recipe.yaml").is_file())


def resolve_recipe_dir(name_or_path: str) -> str:
    """Return a usable recipe directory for a CLI `--recipe` value.

    An existing directory is used as given. Otherwise the value is looked up
    in the installation's default recipe catalog and copied into the current
    directory, because both `deploy` and `bench` write alongside the recipe
    they run.
    """
    if Path(name_or_path).is_dir():
        return name_or_path

    with default_recipe_root() as source:
        available = _recipe_names(source) if source is not None else []
        if name_or_path in available:
            assert source is not None
            target = Path(name_or_path)
            target.mkdir(parents=True)
            shutil.copyfile(source / name_or_path / "recipe.yaml", target / "recipe.yaml")
            return str(target)

    hint = f" Available recipes: {', '.join(available)}." if available else ""
    raise FileNotFoundError(f"No recipe directory {name_or_path!r}.{hint}")
