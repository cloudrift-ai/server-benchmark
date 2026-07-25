"""The golden drift CI gate — do the model-tagged goldens still deploy in the serving twins?

The gemma-4 sessions proved the isolated golden check is not enough: pinned snippets reproduced
68/68 while the in-model deploys drifted (the cast-splice class), because fusion in the real
block produces a different graph than any snippet. This gate closes that hole in CI:

- The serving twin graphs are re-traced **fresh every run**, weight-free, from the checked-in
  ``fixtures/gemma4_12b/config.json`` (``emmy.serving.twins`` builds a trimmed random-init
  skeleton — no network, no checkpoint, no GPU; ``HF_HUB_OFFLINE`` is forced to prove it).
  Re-tracing (rather than checking in traced graphs) means tracer-side drift is covered too:
  a transformers bump that changes the model's forward changes these twins exactly as it
  changes serving, and the gate is SUPPOSED to go red then.
- Each card's audit (``search/audit.audit_card``) compiles with the golden tier as the only
  evidence, targeting the golden file's own card — identical verdicts on a GPU-less box.

Gate policy: DRIFT and COMPILE_FAIL are always failures. Coverage is gated as a ratchet over
**every** GAP key — contractions, reductions/norms, and pointwise forks alike must be covered
by a golden once a card's baseline empties (goldens cover ALL kernel forks in the model; the
only kernels outside the gate are fork-free deterministic lowerings — rope/embedding gathers —
which never consult the golden tier). The per-card baseline must match exactly: a NEW gap
fails (an uncovered kernel appeared — record a golden or deliberately extend the baseline in
review); a CLOSED one also fails, asking for its line to be deleted so the ratchet only
tightens. Warp-contraction gaps (the misdeploy/hang hazard class) are the ones to close first.

Interactive twin: ``emmy eval golden --in-model``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card, gap_keys, summarize
from emmy.compiler.pipeline.search.data.shape import ShapeKey

_FIXTURE = Path(__file__).parent / "fixtures" / "gemma4_12b"


# Known-uncovered kernel forks per card — ALL kinds (contractions, rms_norm/reduce sweeps,
# pointwise), not just the warp-contraction hazards. This list may only change deliberately:
# remove a line when its golden gets recorded (the test fails until you do); add one only
# when review accepts a new uncovered kernel. BOTH CARDS ARE EMPTY as of the 2026-07-22
# merged-sibling reseed (5090 seeded locally, 4090 on a rented card — see the WS1 sections in
# both gemma4 golden YAMLs): full model coverage is ENFORCED again.
# 2026-07-24: the audit twin set widened to EVERY deployed width (32/64/256/4096 + sym —
# the m64/m4096 coverage regressions were invisible before), which surfaced the aux keys
# below at the new widths: pointwise/RoPE/cast glue (reduce_max=0), the per-head qk-norm
# rms rows, and two small scalar o_proj-side reduces. All are greedy-near-optimal classes;
# burn them down by recording aux rows opportunistically (manual --ab), majors stay zero.
EXPECTED_GAPS = {
    "NVIDIA GeForce RTX 5090": {
        ShapeKey(free_prod=15728640, reduce_max=0, is_warp=False, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=245760, reduce_max=0, is_warp=False, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=33554432, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=35651584, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=4096, reduce_max=3840, is_warp=False, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=557056, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=62914560, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=15360),
        ShapeKey(free_prod=64, reduce_max=3840, is_warp=False, is_dyn=False, kind="", free_max=64),
        ShapeKey(free_prod=8388608, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=983040, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=15360),
        # 2026-07-24b: the m1 (gemv-tier) width joined the audit — its aux keys (m1
        # pointwise/rope glue, per-head rms at M=1, the small o_proj/lm_head-side forms;
        # on the 4090 also the fused m1 keys, never seeded there — no m1 serving on 24 GB).
        ShapeKey(free_prod=1, reduce_max=3840, is_warp=False, is_dyn=False, kind="", free_max=0),
        ShapeKey(free_prod=2048, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=3840, reduce_max=0, is_warp=False, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=4096, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=512, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8192, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=8192, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8704, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
    },
    "NVIDIA GeForce RTX 4090": {
        ShapeKey(free_prod=15728640, reduce_max=0, is_warp=False, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=245760, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=262144, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=32768, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=33554432, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=35651584, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=4096, reduce_max=3840, is_warp=False, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=524288, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=524288, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=557056, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=62914560, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=15360),
        ShapeKey(free_prod=8388608, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=983040, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=15360),
        # 2026-07-24b: the m1 (gemv-tier) width joined the audit — its aux keys (m1
        # pointwise/rope glue, per-head rms at M=1, the small o_proj/lm_head-side forms;
        # on the 4090 also the fused m1 keys, never seeded there — no m1 serving on 24 GB).
        ShapeKey(free_prod=2048, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=30720, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=30720),
        ShapeKey(free_prod=3840, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=3840, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=3840, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=4096, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=512, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8192, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=8192, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
        ShapeKey(free_prod=8192, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8704, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=8704, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8704),
    },
}

# A wholesale re-key of the twins (tracer/classifier change) turns MATCHes into GAPs without
# a single DRIFT — the floor catches that failure mode; it is NOT a coverage target (the
# exact count churns benignly whenever a golden YAML gains or loses entries).
# Rebased 2026-07-22: the sibling-linear concat REMOVED half the per-projection fork sites
# from the twins (that is WS1's point), so the pre-merge 101-match count is unreachable —
# both cards audit at 74 after the merged-key reseed.
MIN_MATCH = {"NVIDIA GeForce RTX 5090": 70, "NVIDIA GeForce RTX 4090": 70}

CARDS = [
    pytest.param("NVIDIA GeForce RTX 5090", (12, 0), id="rtx5090"),
    pytest.param("NVIDIA GeForce RTX 4090", (8, 9), id="rtx4090"),
]


@pytest.fixture(scope="module")
def twins():
    """The gemma-4 serving twins, traced weight-free from the fixture config — with HF hub
    access hard-disabled to prove the gate never touches the network."""
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        from emmy.serving.twins import capture_twin_graphs

        return capture_twin_graphs(str(_FIXTURE))
    finally:
        mp.undo()


@pytest.mark.parametrize(("gpu_name", "cap"), CARDS)
def test_gemma4_goldens_deploy_in_serving_twins(twins, gpu_name, cap):
    res = audit_card(twins, gpu_name, cap)
    counts = summarize(res)

    fails = [(name, r) for name, records in res.items() for r in records if r["verdict"] == COMPILE_FAIL]
    assert not fails, f"serving twins failed to compile: {[(n, r['error']) for n, r in fails]}"

    drifts = [(name, r) for name, records in res.items() for r in records if r["verdict"] == "DRIFT"]
    assert not drifts, (
        f"{len(drifts)} golden(s) no longer deploy in the serving twins of {gpu_name}:\n  "
        + "\n  ".join(f"{name} {r['node']}: {r['golden']}  [{r['key']}]" for name, r in drifts)
        + "\nEither a compiler graph/enumeration change re-keyed the fork (fix the drift or re-record the golden), "
        "or a transformers bump changed the traced model — the twins track the installed modeling code, exactly "
        "as serving does."
    )

    gaps = gap_keys(res)
    new = gaps - EXPECTED_GAPS[gpu_name]
    assert not new, (
        f"NEW uncovered kernel fork(s) on {gpu_name}: {sorted(new, key=str)}\n"
        "Every kernel fork in the model must be golden-covered. Record a golden for each (manual "
        "pinned sweep on the card, `run --bench --ab`), or deliberately extend EXPECTED_GAPS in review."
    )
    closed = EXPECTED_GAPS[gpu_name] - gaps
    assert not closed, (
        f"gap(s) on {gpu_name} are now covered — delete their EXPECTED_GAPS lines so the ratchet tightens: {sorted(closed, key=str)}"
    )

    assert counts["MATCH"] >= MIN_MATCH[gpu_name], (
        f"only {counts['MATCH']} golden matches on {gpu_name} (floor {MIN_MATCH[gpu_name]}) with no DRIFT — "
        "the twins likely re-keyed wholesale (tracer or ShapeKey classifier change), turning matches into GAPs."
    )
