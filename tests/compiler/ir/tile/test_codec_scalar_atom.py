"""The scalar tile codec has no alias vocabulary."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import Tile


@pytest.mark.parametrize("alias", ["a:scalar", "a:none", "a:scalar/f4x8"])
def test_scalar_atom_aliases_are_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match="TILE"):
        Tile.parse(alias, None)


@pytest.mark.parametrize("alias", ["f04", "f4x01", "/f4", "f4/"])
def test_noncanonical_scalar_spellings_are_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match="TILE"):
        Tile.parse(alias, None)
