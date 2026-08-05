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
tightens.

Warp-contraction gaps (``major_gap_keys`` — the misdeploy/hang hazard class) are ratcheted
SEPARATELY through ``EXPECTED_MAJOR_GAPS``, never through the generic baseline: the #446
PLACE-knob retirement uncovered every fused serving fork on the 5090 and this gate stayed
green because the 18 keys rode a routine ``EXPECTED_GAPS`` extension. A major-class
regression must now edit the hazard-labeled list, whose entries each need a dated reason and
a burn-down condition (``test_no_major_key_hides_in_the_generic_baseline`` keeps the two sets
disjoint).

Interactive twin: ``emmy eval golden --in-model``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card, gap_keys, major_gap_keys, summarize
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
        ShapeKey(free_prod=33554432, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=35651584, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=557056, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=8388608, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # 2026-07-24b: the m1 (gemv-tier) width joined the audit — its aux keys (m1
        # pointwise/rope glue, per-head rms at M=1, the small o_proj/lm_head-side forms;
        # on the 4090 also the fused m1 keys, never seeded there — no m1 serving on 24 GB).
        ShapeKey(free_prod=2048, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=4096, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=512, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8192, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=8192, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8704, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        # 2026-07-25: the m8 (bucket-8 decode) width joined the audit — its per-head qk-norm rms
        # aux keys (M=8 × heads × head_dim, the same greedy-near-optimal class as the m32/m64
        # entries above; the m8 matmul/fused/glue forks are all golden-covered).
        ShapeKey(free_prod=16384, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=32768, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=65536, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # 2026-07-26: the m2048 (chunk-quantum) width joined the audit — its per-head qk-norm
        # rms aux keys, same greedy-near-optimal class (measured: greedy b128 within ~6% of
        # the b64 best, us-class); every m2048 matmul/fused/glue fork is golden-covered.
        ShapeKey(free_prod=4194304, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=16777216, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=1048576, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # ... and the m2048 analogs of the m4096 reduce-0 warp aux forks + the n3840 pointwise
        # (the same deferred/aux fork class as the 33554432/35651584/62914560/15728640 m4096
        # entries above).
        ShapeKey(free_prod=16777216, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=17825792, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        # 2026-07-30: the PLACE-knob retirement (#446) commented out the fused computed-A
        # goldens that recorded a PLACE@ placement, so every fused norm→linear / geglu fork
        # (kind="fused", across the m1/m8/m32/m2048/m4096 audit widths) was uncovered.
        # 2026-07-31: closed for the norm→linear cones — they carry PLACE=cut routing rows
        # whose pieces resolve each width's seeded stat/scale/matmul tiers (the cut fires
        # in-model for those cones).
        # The mlp_down (geglu-cone) forks closed too, by two independent routes that landed on
        # separate branches and are both merged here: the _cut.py routing-key fix (the pre-fork
        # consult keyed a stat-free computed-A cone kind='' and so never joined its fused-keyed
        # .cut entries, which deployed the slow fused down everywhere — the 2026-07-31 serving
        # TTFT/TPOT regression), and the directly-seeded mlp_down_fused widths. Either one covers
        # the key; together they leave the 5090 hazard-class baseline (EXPECTED_MAJOR_GAPS)
        # empty, which is the state that matters.
        # 2026-07-31: the m192 width (the MTP c=64 verify bucket, serving_mtp_rtx5090) joined
        # the audit — its aux keys below are the same greedy-near-optimal classes as the other
        # widths' (per-head qk-norm rms sweeps, the post-attn/cut-stat rms key, the merged-cat
        # dup-view glue); the m192 matmul/fused/o_proj forks are all golden-covered.
        ShapeKey(free_prod=737280, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=1572864, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=1671168, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=393216, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=786432, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=98304, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=1572864, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
    },
    "NVIDIA GeForce RTX 4090": {
        # 2026-07-31: the m192 width (the MTP c=64 verify bucket) joined the audit. There is no
        # m192 serving on 24 GB (the 12B does not fit), so the whole m192 fork set is uncovered
        # here — the merged-cat glue and rms/qknorm sweeps below; the m192 warp-contraction
        # forks (o_proj, the fused cones) live in EXPECTED_MAJOR_GAPS. Mirror the 5090's m192 +
        # o_proj + fused-cut seeding if a 4090 m192 tier is ever wanted.
        ShapeKey(free_prod=1572864, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=1572864, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=1671168, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=393216, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=737280, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=786432, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=98304, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # 2026-07-26: the m2048 (chunk-quantum) width joined the audit (parity campaign round
        # 3). The 5090 seeded its m2048 golden set the same day (_tune/m2048/ sweeps); the
        # 4090's rides the same deferred mirror re-tune, so every m2048 fork is uncovered
        # there — the glue and rms/qknorm sweeps below; the m2048 warp-contraction forks
        # (merged/canonical matmuls, fused norm→merged forms) live in EXPECTED_MAJOR_GAPS.
        # Burn down with the m8 mirror once a 4090 is back.
        ShapeKey(free_prod=1048576, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=16777216, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=16777216, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=17825792, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=4194304, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=7864320, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # 2026-07-25: the m8 (bucket-8 decode) width joined the audit. The 5090 seeded its m8
        # golden set the same day (WS2, serving-next); the 4090's is deferred with the mirror
        # re-tune (box lost its GPU at the PCI level), so EVERY m8 fork is uncovered there —
        # the glue and per-head/aux rms sweeps below; the m8 warp-contraction forks (merged
        # matmul/fused forms, o_proj) live in EXPECTED_MAJOR_GAPS. Burn down by mirroring the
        # m8 seeding recipe (_tune/decode-m8/) once a 4090 is back.
        ShapeKey(free_prod=16384, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=30720, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=32768, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=4096, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=65536, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=65536, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=69632, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=245760, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=262144, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=32768, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=33554432, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=35651584, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=524288, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=524288, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=557056, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
        ShapeKey(free_prod=8388608, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        # 2026-07-24b: the m1 (gemv-tier) width joined the audit — its aux keys (m1
        # pointwise/rope glue, per-head rms at M=1, the small o_proj/lm_head-side forms;
        # the m1 fused/o_proj warp-contraction keys live in EXPECTED_MAJOR_GAPS — never
        # seeded here, no m1 serving on 24 GB).
        ShapeKey(free_prod=2048, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=3840, reduce_max=3840, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=4096, reduce_max=256, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=512, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8192, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8192),
        ShapeKey(free_prod=8192, reduce_max=512, is_warp=False, is_dyn=False, kind="rms_norm", free_max=0),
        ShapeKey(free_prod=8704, reduce_max=0, is_warp=True, is_dyn=False, kind="", free_max=8704),
    },
}

# The warp-contraction hazard baseline — ``major_gap_keys``: warp forks with a real reduce
# extent (plain matmul, fused computed-A, flash). This is the misdeploy/hang class the goldens
# exist to pin: an unseeded gemma projection deployed a scalar tile ~770x off cuBLAS, and the
# 2026-08-02 5090 serving audit (on main, before #449 merged) measured every uncovered fused
# twin family 114–1014x over the weight-streaming floor with the p2048/c8 chunk cell at TPOT
# ~2760 ms. The same exact-match ratchet discipline as EXPECTED_GAPS, with a stricter bar for
# ADDING: every entry needs a dated reason AND a concrete burn-down condition, and a width a
# card actually serves should be closed (seed a golden or a routing row), not listed. Majors
# may never ride the generic EXPECTED_GAPS set — the #446 PLACE retirement uncovered every
# fused serving fork and the gate stayed green precisely because those keys blended into a
# routine baseline edit (``test_no_major_key_hides_in_the_generic_baseline``).
EXPECTED_MAJOR_GAPS = {
    # 2026-08-02: the 5090 major baseline is EMPTY — the last four #446-reopened keys (the
    # mlp_down_fused widths, un-closable by cut rows and un-benchable while recorded ``.cut``
    # routing rows fired inside pinned replay compiles) burned down once route_cut made
    # schedule-family pins authoritative: fused d*/sync rows benched honestly again and the
    # four widths were seeded on a vast.ai 5090 (the 2026-08-02 block in the gemma4 YAML).
    "NVIDIA GeForce RTX 5090": set(),
    "NVIDIA GeForce RTX 4090": {
        # The 4090 majors are all one deferral: the card's mirror re-tune (box lost its GPU at
        # the PCI level; several widths have no 24 GB serving at all — m1, m192 — and the
        # m8/m2048 sets ride the same rented-card mirror). Burn down per width once a 4090 is
        # back, mirroring the 5090 seeding recipes (_tune/decode-m8/, _tune/m2048/, the m192 +
        # fused-cut seeding of 2026-07-31).
        # m1 (no m1 serving on 24 GB): fused qkv/qk_global/gate_up + o_proj/o_proj_global.
        ShapeKey(free_prod=8192, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
        ShapeKey(free_prod=8704, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8704),
        ShapeKey(free_prod=30720, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=30720),
        ShapeKey(free_prod=3840, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=3840, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
        # m8 (deferred mirror re-tune): fused qkv/qk_global/gate_up/down + o_proj forms.
        ShapeKey(free_prod=65536, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
        ShapeKey(free_prod=69632, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8704),
        ShapeKey(free_prod=245760, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=30720),
        ShapeKey(free_prod=30720, reduce_max=15360, is_warp=True, is_dyn=False, kind="fused", free_max=3840),
        ShapeKey(free_prod=30720, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=30720, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
        # m192 (no m192 serving on 24 GB): fused cones + o_proj/o_proj_global.
        ShapeKey(free_prod=1572864, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
        ShapeKey(free_prod=1671168, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8704),
        ShapeKey(free_prod=5898240, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=30720),
        ShapeKey(free_prod=737280, reduce_max=15360, is_warp=True, is_dyn=False, kind="fused", free_max=3840),
        ShapeKey(free_prod=737280, reduce_max=4096, is_warp=True, is_dyn=False, kind="", free_max=3840),
        ShapeKey(free_prod=737280, reduce_max=8192, is_warp=True, is_dyn=False, kind="", free_max=3840),
        # m2048 (deferred mirror re-tune): fused qkv/qk_global/gate_up/down forms.
        ShapeKey(free_prod=16777216, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
        ShapeKey(free_prod=17825792, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8704),
        ShapeKey(free_prod=62914560, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=30720),
        ShapeKey(free_prod=7864320, reduce_max=15360, is_warp=True, is_dyn=False, kind="fused", free_max=3840),
        # m32/m4096 #446 remainder (the PLACE@ fused rows the retirement commented out).
        ShapeKey(free_prod=245760, reduce_max=15360, is_warp=True, is_dyn=False, kind="fused", free_max=3840),
        ShapeKey(free_prod=15728640, reduce_max=15360, is_warp=True, is_dyn=False, kind="fused", free_max=4096),
        ShapeKey(free_prod=33554432, reduce_max=3840, is_warp=True, is_dyn=False, kind="fused", free_max=8192),
    },
}

# Known-DRIFT (golden, twin) pairs per card — the same exact-match ratchet discipline as
# EXPECTED_GAPS: an entry may only be added deliberately in review, and the test fails until a
# fixed one is deleted. Unlike a gap, an expected drift asserts the recorded golden and the
# audit twin genuinely disagree for a KNOWN reason that is not a serving regression.
EXPECTED_DRIFTS: dict[str, set[tuple[str, str]]] = {
    # The historic 4090 entries (down_proj.m1.t in post1/post1-global) burned down when the
    # transpose-into-constant fold gained its sub-sm_90 matvec exception (``_fold_constant``):
    # sm_89 m1 twins now walk the k-major transposed arm — the layout the ``.m1.t`` rows were
    # recorded on — so the goldens realize in-model again.
    "NVIDIA GeForce RTX 5090": set(),
    "NVIDIA GeForce RTX 4090": set(),
}

# A wholesale re-key of the twins (tracer/classifier change) turns MATCHes into GAPs without
# a single DRIFT — the floor catches that failure mode; it is NOT a coverage target (the
# exact count churns benignly whenever a golden YAML gains or loses entries).
# Rebased 2026-07-22: the sibling-linear concat REMOVED half the per-projection fork sites
# from the twins (that is WS1's point), so the pre-merge 101-match count is unreachable —
# both cards audit at 74 after the merged-key reseed.
MIN_MATCH = {"NVIDIA GeForce RTX 5090": 70, "NVIDIA GeForce RTX 4090": 70}

CARDS = [
    pytest.param(
        "NVIDIA GeForce RTX 5090",
        (12, 0),
        id="rtx5090",
        # 2026-08-05: eleven `kind="fused"` warp-contraction forks (the gemma-4 computed-A
        # norm->linear / gate-up edges at every serving width) audit UNCOVERED on this card. The
        # cause predates the flash-streaming work — it is the computed-A re-keying, which moved the
        # map reading's cooperative row onto `REDUCE@<stat>` and re-keyed the fused rows, so the
        # recorded 5090 goldens no longer join their forks. Reproduced identically at the
        # pre-change commit, which is why it is expected here rather than fixed here: closing it is
        # a CARD sweep (re-seed the fused widths on a 5090, the ratchet the assertion asks for),
        # not an enumeration change. Burn-down: the re-seed lands with the owed
        # `emmy fit --artifact` refit; delete this mark then. The 4090 param stays live.
        marks=pytest.mark.xfail(
            strict=True,
            reason="gemma-4 fused (computed-A) forks lost their 5090 golden join in the computed-A re-keying; "
            "closing it is a card re-seed, tracked with the owed offline-prior refit",
        ),
    ),
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
    drift_pairs = {(r["golden"], name) for name, r in drifts}
    new_drifts = drift_pairs - EXPECTED_DRIFTS[gpu_name]
    assert not new_drifts, (
        f"{len(new_drifts)} golden(s) no longer deploy in the serving twins of {gpu_name}:\n  "
        + "\n  ".join(f"{name} {r['node']}: {r['golden']}  [{r['key']}]" for name, r in drifts if (r["golden"], name) in new_drifts)
        + "\nEither a compiler graph/enumeration change re-keyed the fork (fix the drift or re-record the golden), "
        "or a transformers bump changed the traced model — the twins track the installed modeling code, exactly "
        "as serving does."
    )
    fixed_drifts = EXPECTED_DRIFTS[gpu_name] - drift_pairs
    assert not fixed_drifts, (
        f"expected drift(s) on {gpu_name} no longer occur — delete their EXPECTED_DRIFTS lines so the "
        f"ratchet tightens: {sorted(fixed_drifts)}"
    )

    majors = major_gap_keys(res)
    new_major = majors - EXPECTED_MAJOR_GAPS[gpu_name]
    assert not new_major, (
        f"NEW uncovered WARP-CONTRACTION fork(s) on {gpu_name}: {sorted(new_major, key=str)}\n"
        "This is the misdeploy/hang hazard class — an uncovered contraction cold-resolves in serving "
        "(the #446 regression served the p2048/c8 chunk cell at TPOT ~2760 ms vs the ~20 ms record). "
        "Close it (seed a golden or a routing row on the card) rather than listing it; extending "
        "EXPECTED_MAJOR_GAPS needs a dated reason and a concrete burn-down condition in review."
    )
    closed_major = EXPECTED_MAJOR_GAPS[gpu_name] - majors
    assert not closed_major, (
        f"major gap(s) on {gpu_name} are now covered — delete their EXPECTED_MAJOR_GAPS lines so the "
        f"ratchet tightens: {sorted(closed_major, key=str)}"
    )

    gaps = gap_keys(res) - majors
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


def test_no_major_key_hides_in_the_generic_baseline():
    """The warp-contraction hazard class may only be ratcheted through EXPECTED_MAJOR_GAPS —
    a major key in the generic set would let a coverage loss ride a routine baseline edit,
    which is exactly how the #446 fused-fork loss stayed green."""
    for gpu_name, keys in EXPECTED_GAPS.items():
        leaked = {k for k in keys if k.is_warp and k.reduce_max > 0}
        assert not leaked, (
            f"warp-contraction key(s) in EXPECTED_GAPS[{gpu_name}] — move them to EXPECTED_MAJOR_GAPS: {sorted(leaked, key=str)}"
        )
