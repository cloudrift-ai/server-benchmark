"""The pin-only scalar-atom aliases ``a:scalar`` / ``a:none``.

These give a pin an *explicit* spelling of the scalar output tile — symmetric with the warp form's
bare leading atom name — instead of the invisible empty string. They are vocabulary only: a producer
never emits them, so :meth:`TilePlan.parse` must strip them and :meth:`TilePlan.spell` re-emit the
canonical scalar form (``""`` / ``f..``), and a stripped alias must never read as the warp tier
despite naming an "atom".
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import TilePlan, Workers, has_scalar_atom_alias

_THREADS = Workers.parse("t8x16")


@pytest.mark.parametrize("alias", ["a:scalar", "a:none"])
def test_alias_is_scalar_not_warp(alias: str) -> None:
    plan = TilePlan.parse(alias, None)
    assert not plan.is_warp  # names the scalar atom, never a tensor-core one
    assert (plan.units, plan.regs) == ((1, 1), (1, 1))  # bare alias = the per-cell tier


@pytest.mark.parametrize(
    ("alias", "work", "body_units", "body_regs"),
    [("a:scalar/f4x8", None, (1, 1), (4, 8)), ("a:none", _THREADS, (1, 1), (1, 1)), ("a:scalar/f2x2", _THREADS, (8, 16), (2, 2))],
)
def test_alias_composes_with_scalar_body(alias: str, work, body_units: tuple, body_regs: tuple) -> None:
    plan = TilePlan.parse(alias, work)
    assert not plan.is_warp
    assert (plan.units, plan.regs) == (body_units, body_regs)


@pytest.mark.parametrize(("alias", "canonical"), [("a:scalar", ""), ("a:none", ""), ("a:scalar/f4x8", "f4x8")])
def test_alias_never_survives_to_spell(alias: str, canonical: str) -> None:
    # The alias is pin-only: it must normalize to the canonical scalar spelling so it never rides a
    # stored knob dict / prior key / golden YAML.
    assert TilePlan.parse(alias, None).spell() == canonical
    assert not has_scalar_atom_alias(canonical)


def test_alias_detection() -> None:
    assert has_scalar_atom_alias("a:scalar")
    assert has_scalar_atom_alias("a:none/f2x2")
    assert not has_scalar_atom_alias("f8x16")
    assert not has_scalar_atom_alias("mma_m16n8k16_f16_f32/f2x2/k4")
    assert not has_scalar_atom_alias("")
