"""Every recorded STATIC attention golden's pins must still bind — resolve the golden's own
snippet under its recorded knobs and assert the stamped ``TILE@*`` spellings match the record.

The findings-5 F2 guard: a moveset / pin-routing change that silently stops a recorded config
from binding turns every golden A/B for that shape into a lie (the "golden" row benches some
other config). CPU-only — the pins bind (or not) at fork-build time, no kernel is compiled.
Dynamic (``.dynM``) attention goldens record a bare ``TILE`` whose binding is exercised by the
masked-flash e2e tests; the static axis-keyed pins are the spelling this guard pins down.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS  # noqa: E402

_STATIC_ATTENTION = [g for g in GOLDEN_RECORDS if "torch.sdpa" in g.origin_ops and not g.dynamic]


@pytest.mark.parametrize("golden", _STATIC_ATTENTION, ids=lambda g: f"{g.name}@{g.gpu_name.split()[-1]}")
def test_static_attention_golden_pins_bind(golden, monkeypatch):
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import Fork
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    g = golden.target_program.copy()

    def decide(fp):
        o = fp.options[0]
        while isinstance(o, Fork) and not o.is_leaf:
            o = o.expand()[0]
        return o

    ctx = Context(compute_capability=tuple(golden.compute_cap))
    with pinned_knobs({**golden.pin_map, **golden.knobs}):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(g, decide)
    tile_pins = {k: v for k, v in golden.knobs.items() if k.startswith("TILE@")}
    assert tile_pins, f"{golden.name}: static attention golden should record axis-keyed TILE pins"
    stamped = next((k for _, n in out.nodes.items() if (k := getattr(n.op, "knobs", None)) and set(tile_pins) <= set(k)), None)
    assert stamped is not None, f"{golden.name}: no node stamps the recorded TILE keys {sorted(tile_pins)}"
    # The site value grammar: each recorded ``TILE@<axis>`` realizes verbatim (modulo the codec's
    # normal form), and the entry's worker geometry — recorded once, in ``WORK`` — realizes as the
    # kernel's ONE inventory.
    from emmy.compiler.pipeline.knob import canon_family_value, values_equal  # noqa: PLC0415

    for key, want in tile_pins.items():
        assert stamped[key] == canon_family_value(key, want), f"{golden.name}: recorded {key}={want!r} resolved to {stamped[key]!r}"
    if "WORK" in golden.knobs:  # the inventory is pinned directly — it must realize verbatim
        assert values_equal("WORK", golden.knobs["WORK"], stamped.get("WORK", "")), (
            f"{golden.name}: recorded WORK={golden.knobs['WORK']!r} resolved to {stamped.get('WORK')!r}"
        )
