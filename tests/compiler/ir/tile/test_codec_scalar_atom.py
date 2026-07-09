"""The pin-only scalar-atom aliases ``a:scalar`` / ``a:none``.

These give a pin an *explicit* spelling of the scalar output tile — symmetric with the warp form's
``a:<mma-atom>`` — instead of the invisible empty string. They are vocabulary only: a producer never
emits them, so :meth:`TilePlan.spell` must always round-trip them back to the canonical scalar form
(``""`` / ``n../f..``), and :func:`is_warp_codec` must classify them as scalar despite the ``a:``
prefix (the one exception to "any ``a:`` token ⇒ warp").
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import TilePlan, has_scalar_atom_alias, is_warp_codec


@pytest.mark.parametrize("alias", ["a:scalar", "a:none"])
def test_alias_is_scalar_not_warp(alias: str) -> None:
    plan = TilePlan.parse(alias)
    assert not is_warp_codec(alias)  # ``a:`` prefix but names the scalar atom
    assert not plan.is_warp
    assert (plan.units, plan.regs) == ((1, 1), (1, 1))  # bare alias = the per-cell tier


@pytest.mark.parametrize(
    ("alias", "body_units", "body_regs"),
    [("a:scalar/f4x8", (1, 1), (4, 8)), ("a:none/n8x16", (8, 16), (1, 1)), ("a:scalar/n4x4/f2x2", (4, 4), (2, 2))],
)
def test_alias_composes_with_scalar_body(alias: str, body_units: tuple, body_regs: tuple) -> None:
    plan = TilePlan.parse(alias)
    assert not plan.is_warp
    assert (plan.units, plan.regs) == (body_units, body_regs)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("a:scalar", ""), ("a:none", ""), ("a:scalar/f4x8", "f4x8"), ("a:none/n8x16", "n8x16")],
)
def test_alias_never_survives_to_spell(alias: str, canonical: str) -> None:
    # The alias is pin-only: it must normalize to the canonical scalar spelling so it never rides a
    # stored knob dict / prior key / golden YAML.
    assert TilePlan.parse(alias).spell() == canonical
    assert not has_scalar_atom_alias(canonical)


def test_alias_detection() -> None:
    assert has_scalar_atom_alias("a:scalar")
    assert has_scalar_atom_alias("a:none/f2x2")
    assert not has_scalar_atom_alias("n8x16")
    assert not has_scalar_atom_alias("a:mma_m16n8k16_f16/w2x4/f2x2/k4")
    assert not has_scalar_atom_alias("")
