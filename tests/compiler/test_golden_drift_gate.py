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

Gate policy: DRIFT and COMPILE_FAIL are always failures. GAP is coverage, gated as a ratchet:
the **major** gaps (uncovered warp-contraction forks — the misdeploy/hang hazard class) must
equal the checked-in baseline exactly. A NEW major gap fails (an uncovered contraction kernel
appeared — record a golden for it or deliberately extend the baseline in review); a CLOSED one
also fails, asking for the baseline line to be deleted so the ratchet only tightens.

Interactive twin: ``emmy eval golden --in-model``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card, major_gap_keys, summarize
from emmy.compiler.pipeline.search.data.shape import ShapeKey

_FIXTURE = Path(__file__).parent / "fixtures" / "gemma4_12b"

# Known-uncovered warp-contraction forks per card (see `major_gap_keys`). This list may only
# change deliberately: remove a line when its golden gets recorded (the test fails until you
# do); add one only when review accepts a new uncovered contraction kernel.
EXPECTED_MAJOR_GAPS = {
    "NVIDIA GeForce RTX 5090": {
        # The global-layer prefill-256 projections (goldens exist only at s2048) and the
        # dynamic fused gate⊗up cone — the coverage holes this gate first surfaced.
        ShapeKey(free_prod=131072, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=512),
        ShapeKey(free_prod=15360, reduce_max=3840, is_warp=True, is_dyn=True, kind="fused", free_max=0),
        ShapeKey(free_prod=2097152, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=8192),
    },
    # The 4090 gemma file is much sparser (7 in-model matches) — most static projection /
    # fused shapes were never recorded for it.
    "NVIDIA GeForce RTX 4090": {
        ShapeKey(free_prod=1048576, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=122880, reduce_max=15360, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=122880, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=122880, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=131072, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=4096),
        ShapeKey(free_prod=131072, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=512),
        ShapeKey(free_prod=15360, reduce_max=3840, is_warp=True, is_dyn=True, kind="fused", free_max=0),
        ShapeKey(free_prod=16384, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=512),
        ShapeKey(free_prod=2097152, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=262144, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=3932160, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=15360),
        ShapeKey(free_prod=491520, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=15360),
        ShapeKey(free_prod=524288, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=2048),
        ShapeKey(free_prod=65536, reduce_max=3840, is_warp=True, is_dyn=False, kind="", free_max=2048),
        ShapeKey(free_prod=983040, reduce_max=15360, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=983040, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=983040, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
    },
}

# A wholesale re-key of the twins (tracer/classifier change) turns MATCHes into GAPs without
# a single DRIFT — the floor catches that failure mode; it is NOT a coverage target (the
# exact count churns benignly whenever a golden YAML gains or loses entries).
MIN_MATCH = {"NVIDIA GeForce RTX 5090": 50, "NVIDIA GeForce RTX 4090": 5}

CARDS = [
    pytest.param("NVIDIA GeForce RTX 5090", (12, 0), id="rtx5090"),
    pytest.param(
        "NVIDIA GeForce RTX 4090",
        (8, 9),
        id="rtx4090",
        marks=pytest.mark.xfail(
            reason="9 pre-existing DRIFTs: every dynM projection golden (q/kv/o/mlp_down + global twins) records a "
            "staged d2/cp config the symbolic-M enumeration no longer offers on sm_89 — needs a 4090 re-record or "
            "prune session; drop this mark once the file is healed"
        ),
    ),
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

    majors = major_gap_keys(res)
    new = majors - EXPECTED_MAJOR_GAPS[gpu_name]
    assert not new, (
        f"NEW uncovered warp-contraction fork(s) on {gpu_name}: {sorted(new, key=str)}\n"
        "Record a golden for each (tune on the card, `run --bench --ab`), or deliberately extend "
        "EXPECTED_MAJOR_GAPS in review."
    )
    closed = EXPECTED_MAJOR_GAPS[gpu_name] - majors
    assert not closed, (
        f"major gap(s) on {gpu_name} are now covered — delete their EXPECTED_MAJOR_GAPS lines so the "
        f"ratchet tightens: {sorted(closed, key=str)}"
    )

    assert counts["MATCH"] >= MIN_MATCH[gpu_name], (
        f"only {counts['MATCH']} golden matches on {gpu_name} (floor {MIN_MATCH[gpu_name]}) with no DRIFT — "
        "the twins likely re-keyed wholesale (tracer or ShapeKey classifier change), turning matches into GAPs."
    )
