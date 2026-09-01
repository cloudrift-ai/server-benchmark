"""The ``REDUCE`` codec's cross-CTA GRID finalize letter.

``Reduce.parse`` / ``.spell`` are the schedule's single reduce-partition codec (site-local:
the cooperative WIDTH lives in the kernel's ``WORK`` inventory, so the value spells a bare
``coop`` and parses against a :class:`Work`). The ``g<n>[a|k]`` finalize letter (atomic vs
deferred-kernel cross-CTA split) must survive the round-trip so ``030_cut`` can read
``Reduce.finalize`` — it was historically parsed then dropped (``spell`` never re-emitted it),
making ``g2a`` and ``g2k`` indistinguishable. The round-trip cases themselves live once, in
``test_codec_roundtrip.py``, over the wider corpus this file's list was a subset of; what is
unique here is refusing an omitted letter, which no round-trip can state.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import Reduce, Work

_T32 = Work.parse("t32")


def test_grid_finalize_letter_decodes() -> None:
    assert Reduce.parse("g2a", None).finalize == "atomic"
    assert Reduce.parse("g2k", None).finalize == "kernel"
    with pytest.raises(ValueError, match="not canonical"):
        Reduce.parse("g2", None)  # the finalize letter is never implicit on the wire
    assert Reduce.parse("coop", _T32).finalize == "kernel"  # no GRID stage → the default value


def test_needs_split_only_for_grid() -> None:
    assert Reduce.parse("g2k", None).needs_split
    assert not Reduce.parse("coop/r4", _T32).needs_split
    assert not Reduce.parse("", None).needs_split
