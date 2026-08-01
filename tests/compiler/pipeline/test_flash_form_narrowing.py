"""Per-node ``TILE`` pins must select the flash fork row whose stamped spelling matches.

The unit tests over the old ``_schedule._narrow_flash_forms`` helper went with the scheduler; what
survives is the end-to-end pin contract, which is a property of ANY scheduler: a static attention
golden pins ``TILE@dd`` / ``TILE@pj`` AND ``STAGE``, and the resolved row must stamp exactly the
pinned pair (a stage-pinned fast path must not bypass the tile narrowing — the findings-5 F2
regression). It is scheduler-shaped, so it rides the xfail registry until enumeration returns.
"""

from __future__ import annotations


def test_stage_pin_does_not_bypass_keyed_tile_pins():
    import pytest

    torch = pytest.importorskip("torch")  # noqa: F841

    from emmy.commands.run import _pinned_knobs
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import Fork
    from emmy.compiler.pipeline.pipeline import Run

    d = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,128,64,dtype=torch.float16), "
        "torch.randn(1,2,128,64,dtype=torch.float16), torch.randn(1,2,128,64,dtype=torch.float16), is_causal=False)"
    )
    g = d["graph"] if isinstance(d, dict) else d
    pins = {
        "TILE@dd": "mma_m16n8k16_f16_f32/f1x8/k4",
        "TILE@pj": "mma_m16n8k16_f16_f32/f1x8/k4",
        "WORK": "w2x1",
        "STAGE": "d2/cp/ring",
    }

    def decide(fp):
        o = fp.options[0]
        while isinstance(o, Fork) and not o.is_leaf:
            o = o.expand()[0]
        return o

    with _pinned_knobs(pins):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context(compute_capability=(8, 9))).resolve(g, decide)
    stamped = next(k for _, n in out.nodes.items() if (k := getattr(n.op, "knobs", None)) and "TILE@dd" in k)
    # F1: the stamp is the SITE spelling; the pin's worker half lands in the ONE WORK entry.
    assert stamped["TILE@dd"] == "mma_m16n8k16_f16_f32/f1x8/k4", stamped
    assert stamped["TILE@pj"] == "mma_m16n8k16_f16_f32/f1x8/k4", stamped
    assert stamped["WORK"] == "w2x1", stamped
