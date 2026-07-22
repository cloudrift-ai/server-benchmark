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


def _k(free_prod, reduce_max, kind, free_max=0, dyn=False):
    return ShapeKey(free_prod=free_prod, reduce_max=reduce_max, is_warp=True, is_dyn=dyn, kind=kind, free_max=free_max)


# Known-uncovered kernel forks per card — ALL kinds (contractions, rms_norm/reduce sweeps,
# pointwise), not just the warp-contraction hazards. This list may only change deliberately:
# remove a line when its golden gets recorded (the test fails until you do); add one only
# when review accepts a new uncovered kernel.
#
# 2026-07-22 (merged sibling-linear WS1): the concat re-keyed every merged edge — fused
# norm→q||k(||v) (N=8192 sliding / 8704 global), fused norm→gate_up (N=30720), the down-proj
# GeGLU-combine cone, and the global pre twin's merged-projection copy. The 5090 keys were
# seeded that session (manual pinned --ab, `goldens/rtx5090_sm120_gemma4.yaml`) and its set is
# EMPTY again; the 4090's await a 4090 measurement session and are baselined below (the same
# key list the 5090 seeding covered, plus the sym down fork whose unrealizable dynM rows were
# deleted). Re-seed on a rented 4090 and empty this set.
_MERGED_EDGE_KEYS_4090 = {
    _k(983040, 3840, "fused", 30720),  # gate_up m32 (post32/post32-global)
    _k(7864320, 3840, "fused", 30720),  # gate_up m256
    _k(30720, 3840, "fused", dyn=True),  # gate_up dynM
    _k(2097152, 3840, "fused", 8192),  # qkv sliding m256
    _k(8192, 3840, "fused", dyn=True),  # qkv sliding dynM
    _k(278528, 3840, "fused", 8704),  # qk global m32
    _k(2228224, 3840, "fused", 8704),  # qk global m256
    _k(8704, 3840, "fused", dyn=True),  # qk global dynM
    _k(122880, 15360, "fused", 3840),  # down cone m32
    _k(983040, 15360, "fused", 3840),  # down cone m256
    _k(3840, 15360, "", dyn=True),  # down cone dynM (kind="" at sym; stale rows deleted)
    _k(278528, 0, "", 8704),  # dup__view m32 (global pre)
    _k(2228224, 0, "", 8704),  # dup__view m256
    _k(8704, 0, "", dyn=True),  # dup__view dynM
}
EXPECTED_GAPS = {
    "NVIDIA GeForce RTX 5090": set(),
    "NVIDIA GeForce RTX 4090": set(_MERGED_EDGE_KEYS_4090),
}

# A wholesale re-key of the twins (tracer/classifier change) turns MATCHes into GAPs without
# a single DRIFT — the floor catches that failure mode; it is NOT a coverage target (the
# exact count churns benignly whenever a golden YAML gains or loses entries).
# Rebased 2026-07-22: the sibling-linear concat REMOVED half the per-projection fork sites
# from the twins (that is WS1's point), so the pre-merge 101-match count is unreachable —
# the 5090 audits at 74 post-seed, the 4090 at 52 with its merged keys still un-seeded.
MIN_MATCH = {"NVIDIA GeForce RTX 5090": 70, "NVIDIA GeForce RTX 4090": 45}

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
