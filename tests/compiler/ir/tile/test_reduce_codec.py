"""The ``REDUCE`` codec round-trips — including the cross-CTA GRID finalize letter.

``ReducePlan.parse`` / ``.spell`` are the schedule's single reduce-partition codec (site-local:
the cooperative WIDTH lives in the kernel's ``WORK`` inventory, so the value spells a bare
``coop`` and parses against a :class:`Workers`). The ``g<n>[a|k]`` finalize letter (atomic vs
deferred-kernel cross-CTA split) must survive the round-trip so ``030_split_reduce`` can read
``ReducePlan.finalize`` — it was historically parsed then dropped (``spell`` never re-emitted it),
making ``g2a`` and ``g2k`` indistinguishable.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import ReducePlan, Workers

_T32 = Workers.parse("t32")


@pytest.mark.parametrize("spec", ["", "coop", "r4", "g2a", "g2k", "g4a/coop", "g2k/coop/r4"])
def test_reduce_codec_round_trips(spec: str) -> None:
    assert ReducePlan.parse(spec, _T32).spell() == spec


def test_grid_finalize_letter_decodes() -> None:
    assert ReducePlan.parse("g2a", None).finalize == "atomic"
    assert ReducePlan.parse("g2k", None).finalize == "kernel"
    assert ReducePlan.parse("g2", None).finalize == "kernel"  # default when the letter is omitted
    assert ReducePlan.parse("coop", _T32).finalize == "kernel"  # no GRID stage → the default value


def test_needs_split_only_for_grid() -> None:
    assert ReducePlan.parse("g2k", None).needs_split
    assert not ReducePlan.parse("coop/r4", _T32).needs_split
    assert not ReducePlan.parse("", None).needs_split
