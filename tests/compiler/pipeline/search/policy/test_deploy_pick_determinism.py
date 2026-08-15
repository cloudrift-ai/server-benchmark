"""Deploy-pick determinism at every evidence tier.

A deploy pick must be a function of the candidates' CONTENT, never of their
enumeration order: the offline prior can score many same-featurized siblings
identically (8 exact ties at the gemma-4 m16 mlp_down / o_proj forks, where the
recorded goldens don't realize against the serving trace), and an order-broken tie
flips the deployed kernel whenever leaf order shifts across processes — the 2026-07
RTX 5090 gemma-4 image's bimodal boot-time cubin set. Every pick tier (model argmin,
measured-evidence argmin, golden realization) therefore selects the same content under
any permutation of the candidate rows. The backend source-determinism test separately
pins rendered bytes across fresh interpreters.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.policy import greedy as greedy_mod
from emmy.compiler.pipeline.search.prior.base import Prior


class _ConstPrior(Prior):
    """Every candidate scores identically — the pick must fall to the canonical
    content tiebreak, so any order dependence shows immediately."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def fitted(self) -> bool:
        return True

    def fit(self) -> None:  # pragma: no cover — never trained
        pass

    def mean_score(self, knobs: dict) -> float:
        return 1.0


def _rows() -> list[dict]:
    base = {"H_opt": 3.0, "S_ext_k": "512"}
    # SITE-LOCAL ``TILE`` values (the worker widths ride the row's one ``WORK`` entry) — the same
    # spelling a stamped row carries, so the tiebreak is exercised over live content.
    tiles = [
        "mma_m16n8k16_f16_f32/f2x4",
        "mma_m16n8k16_f16_f32/f4x2",
        "mma_m16n8k16_f16_f32/f2x4/k2",
        "mma_m16n8k16_f16_f32/f4x2/k2",
    ]
    return [{**base, "TILE@k": t, "REDUCE@k": r, "STAGE@k": "", "RASTER": "", "WORK": "w1x1"} for t in tiles for r in ("", "g2k")]


def _selected(pick: tuple[int, float] | None, rows: list[dict]):
    assert pick is not None
    return {k: v for k, v in rows[pick[0]].items() if not k.startswith(("S_", "H_"))}


def test_model_argmin_is_order_invariant():
    prior = _ConstPrior()
    rows = _rows()
    want = _selected(prior.pick(rows), rows)
    for perm in (rows[::-1], rows[3:] + rows[:3]):
        assert _selected(prior.pick(perm), perm) == want


def test_evidence_pick_tie_is_order_invariant():
    prior = _ConstPrior()
    rows = _rows()
    # One measured row that matches EVERY candidate (records no tunable knob) — all
    # candidates tie at its µs, so only the canonical tiebreak can decide.
    prior.add_rows([({"H_opt": 3.0, "S_ext_k": "512"}, 12.5)])
    want = _selected(prior.evidence_pick(rows), rows)
    assert _selected(prior.evidence_pick(rows[::-1]), rows[::-1]) == want


def test_db_measured_pick_tie_is_order_invariant():
    rows = _rows()
    sig = frozenset({("S_ext_k", "512")})
    index = {sig: [({}, 33.0, True)]}  # matches every candidate at one µs — a full tie
    rows_fwd, rows_rev = rows, rows[::-1]
    a = _selected(greedy_mod._db_measured_pick(index, rows_fwd), rows_fwd)
    b = _selected(greedy_mod._db_measured_pick(index, rows_rev), rows_rev)
    assert a == b
