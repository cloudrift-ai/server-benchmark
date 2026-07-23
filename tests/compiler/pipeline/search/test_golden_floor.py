"""The golden FLOOR on prior-less resolves (``policy/greedy.greedy_decide`` with
``prior=None``): the card's recorded goldens are tier-1 evidence independent of any
prior, so a fork whose golden still realizes deploys the golden — never option-0.

The prior-loaded path already implements the floor by tier order (the golden tier
decides before evidence and the model — ``test_golden_evidence_deploy``); these tests
pin the two paths that resolve WITHOUT a prior and previously skipped the tier
entirely: a failed ``load_prior`` (corrupt/unreadable online checkpoint), and
``Pipeline.run``'s last-resort emission-order resolve after the validity-retry
budget exhausts (where one over-budget node used to cost every OTHER kernel its
verified golden)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from emmy import config
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig
from emmy.compiler.pipeline.search.policy.greedy import _fork_shape_key, greedy_decide, tile_identity

CARD = "NVIDIA GeForce RTX 4090"
CAP = (8, 9)

# q_proj-shaped fp16 matmul: M512 x N4096, K3840 — static stamps (S_dtype_f32 absent -> warp tier).
_SIG = {"S_ext_free_prod": 2097152.0, "S_ext_free_max": 4096.0, "S_ext_reduce_max": 3840.0, "S_ext_reduce_prod": 3840.0}

_STD_TILE = "a:mma_m16n8k16_f16_f32/w2x2/f4x4/k2"
_W1X1_TILE = "a:mma_m16n8k16_f16_f32/w1x1/f1x1"

# Candidate rows spell the axis-stamped realization (``TILE@a2``) — the golden's bare
# spelling matches through ``pin_key_matches`` + ``values_equal``.
_ROW_W1X1 = {"TILE@a2": _W1X1_TILE, "STAGE@a2": "d2/cp/ring", "REDUCE@a2": ""}
_ROW_GOLD = {"TILE@a2": _STD_TILE, "STAGE@a2": "d2/cp/ring/p2", "REDUCE@a2": ""}


def _golden(us=78.0):
    return MatmulGoldenConfig(
        name="gemma4_12b.q_proj",
        M=512,
        N=4096,
        K=3840,
        dtype="fp16",
        knobs={"TILE": _STD_TILE, "STAGE": "d2/cp/ring/p2", "REDUCE": "", "WSPEC": "", "RASTER": ""},
        emmy_us=us,
        gpu_name=CARD,
        compute_cap=CAP,
    )


class _FakeFP(SimpleNamespace):
    score = None


def _decide_no_prior(monkeypatch, goldens, leaves, blocked=None):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", list(goldens))
    with config.nvcc_flags_override(""):  # the deployable -O3 regime (H_opt=3) golden consultation is gated on
        ctx = Context.from_target(CAP, gpu_name=CARD)
    fp = _FakeFP(ctx=ctx, options=list(leaves), root_op=SimpleNamespace(knobs=dict(_SIG)), node_id="n0", match=None)
    return greedy_decide(blocked, prior=None)(fp), fp


def test_no_prior_resolve_deploys_realizable_golden(monkeypatch, caplog):
    """A resolve with no prior at all must still land on the recorded golden when it
    realizes — previously this path fell straight to option-0 (the degenerate w1x1
    leaf here), silently missing the verified entry. The override is logged loud."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves)
    assert picked is leaves[1]
    assert fp.score == 78.0
    assert any("golden floor" in r.message for r in caplog.records)


def test_no_prior_drift_warns_and_falls_to_option0(monkeypatch, caplog):
    """A golden keyed to the fork's shape that NO offered candidate realizes is
    enumeration drift — the loud drift warning fires on this path too (it was silent
    here before the floor: the tier was never consulted), then emission order."""
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves)
    assert picked is leaves[0]
    assert fp.score is None
    assert any("no longer realize" in r.message for r in caplog.records)


def test_no_prior_without_goldens_is_pure_emission_order(monkeypatch, caplog):
    """No golden for the live card ⇒ the old behavior bit-for-bit: option-0, no
    warnings (parity with ``test_resolve.test_option0_decide_matches_no_prior_greedy``)."""
    other_card = MatmulGoldenConfig(
        name="gemma4_12b.q_proj",
        M=512,
        N=4096,
        K=3840,
        dtype="fp16",
        knobs={"TILE": _STD_TILE},
        emmy_us=78.0,
        gpu_name="NVIDIA GeForce RTX 5090",
        compute_cap=(12, 0),
    )
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [other_card], leaves)
    assert picked is leaves[0]
    assert fp.score is None
    assert not caplog.records


def test_floor_respects_blocklist(monkeypatch, caplog):
    """``Pipeline.run``'s last-resort resolve threads ``blocked`` through, so the
    floor can never re-pick a tile that already failed ``validate(ctx)`` — with the
    golden's realization blocklisted the shape reads as drift and option-0 stands."""
    blocked = {"n0": {tile_identity(dict(_ROW_GOLD))}}
    leaves = [SimpleNamespace(knobs=dict(_ROW_W1X1)), SimpleNamespace(knobs=dict(_ROW_GOLD))]
    with caplog.at_level(logging.WARNING):
        picked, fp = _decide_no_prior(monkeypatch, [_golden()], leaves, blocked=blocked)
    assert picked is leaves[0]
    assert fp.score is None
    assert any("no longer realize" in r.message for r in caplog.records)


def test_fork_shape_key_flash_sniff_scans_every_row():
    """The flash OFFER signal (the ``TILE@dd`` + ``TILE@pj`` pair) marks the fork when
    ANY row carries it — row 0 is whichever leaf the planner emitted first and need
    not be a warp leaf. A row-0-only sniff mis-keyed such a fork ``kind=""`` (GAP), and
    the attention goldens were silently skipped in favor of the model. The sig is the
    RESTRUCTURED twisted op's: re-derived extents only, no histogram — so the stamp
    classifier cannot fire and the key depends on the offer sniff alone."""
    flash_sig = {
        "S_ext_free_prod": 2097152.0,
        "S_ext_reduce_max": 512.0,
        "S_ext_n_free_axis": 3.0,
        "S_ext_n_reduce_axis": 3.0,
        "S_ext_n_symbolic_axis": 0.0,
        "H_opt": 3.0,
    }
    scalar_row = {**flash_sig, "TILE": "b32x8/f1x1"}  # a bare-TILE non-warp sibling emitted first
    warp_row = {
        **flash_sig,
        "PLACE@fold": "fuse",
        "TILE@dd": "a:mma_m16n8k16_f16_f32/w4x1/f1x2/k16",
        "TILE@pj": "a:mma_m16n8k16_f16_f32/w4x1/f1x32",
        "REDUCE@kv": "",
        "STAGE@kv": "d2/cp/ring",
    }
    assert _fork_shape_key([warp_row, scalar_row]).kind == "flash"
    assert _fork_shape_key([scalar_row, warp_row]).kind == "flash", "the sniff must not depend on row order"
