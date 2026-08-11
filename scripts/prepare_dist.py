#!/usr/bin/env python3
"""Stage the source tree for a distribution build.

Two independent steps, each behind its own flag, both run from the repo root:

``--recipes``
    Copy every ``recipes/<model>/recipe.yaml`` into ``emmy/recipes/`` so the wheel
    ships the recommended serving configs (``recipes/`` sits outside the ``emmy``
    package, so setuptools cannot pick it up in place). Only the recipe files are
    copied — local benchmark output and ``RESULTS.md`` are not.

``--readme``
    Rewrite README.md's repo-relative links to absolute GitHub URLs, in place.
    PyPI renders the README detached from the repo, so ``](emmy/compiler/)`` 404s
    there. Only the publish workflow runs this: it mutates a tracked file, and a
    local build has no reason to.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_RECIPES = REPO_ROOT / "emmy" / "recipes"
GITHUB_TREE = "https://github.com/cloudrift-ai/emmy/tree/main"
GITHUB_BLOB = "https://github.com/cloudrift-ai/emmy/blob/main"

# ](target) where target is neither an absolute URL nor an in-page anchor.
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def stage_recipes() -> int:
    """Copy recipes/<model>/recipe.yaml into the package. Returns the count."""
    if BUNDLED_RECIPES.exists():
        shutil.rmtree(BUNDLED_RECIPES)

    staged = 0
    for recipe in sorted((REPO_ROOT / "recipes").glob("*/recipe.yaml")):
        target = BUNDLED_RECIPES / recipe.parent.name / "recipe.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe, target)
        staged += 1

    if not staged:
        sys.exit("No recipes found under recipes/*/recipe.yaml — refusing to build a recipe-less package.")
    print(f"staged {staged} recipes into {BUNDLED_RECIPES.relative_to(REPO_ROOT)}/")
    return staged


def absolutize_readme() -> int:
    """Rewrite README relative links to absolute GitHub URLs. Returns the count."""
    readme = REPO_ROOT / "README.md"
    text = readme.read_text()
    rewritten = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewritten
        target = match.group(1)
        if not (REPO_ROOT / target).exists():
            sys.exit(f"README links to a missing path: {target}")
        base = GITHUB_TREE if target.endswith("/") else GITHUB_BLOB
        rewritten += 1
        return f"]({base}/{target.rstrip('/')})"

    readme.write_text(RELATIVE_LINK.sub(replace, text))
    print(f"rewrote {rewritten} README links to absolute GitHub URLs")
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipes", action="store_true", help="Stage recipes/*/recipe.yaml into emmy/recipes/")
    parser.add_argument("--readme", action="store_true", help="Rewrite README links to absolute GitHub URLs (mutates README.md)")
    args = parser.parse_args()

    if not (args.recipes or args.readme):
        parser.error("nothing to do: pass --recipes and/or --readme")
    if args.recipes:
        stage_recipes()
    if args.readme:
        absolutize_readme()


if __name__ == "__main__":
    main()
