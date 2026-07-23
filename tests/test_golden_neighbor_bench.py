"""scripts/golden_neighbor_bench.py — the pure selection/resume logic (no GPU, no emmy imports).

Covers the component-aware knob distance, the order-stable point spec/key, the
remaining-proportional randomized batch sampling, the slice machinery (assignment,
share-weighted slice draw, the stable tail subsample, share normalization), and the
ledger's resume semantics (terminal statuses stick, non-terminal ones retry up to the
attempt cap).
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import golden_neighbor_bench as gnb  # noqa: E402


class TestKnobDistance:
    def test_identical_rows_are_distance_zero(self):
        row = {"TILE@d": "a:mma/w2x2/f4x4/k2", "REDUCE@a1": "g2k"}
        assert gnb.knob_distance(row, dict(row)) == 0

    def test_plain_value_diff_counts_one(self):
        assert gnb.knob_distance({"REDUCE@a1": "g2k"}, {"REDUCE@a1": "g8k"}) == 1

    def test_codec_counts_differing_segments_not_whole_knob(self):
        a = {"TILE@d": "a:mma/w2x2/f4x4/k2"}
        assert gnb.knob_distance(a, {"TILE@d": "a:mma/w2x2/f2x4/k2"}) == 1
        assert gnb.knob_distance(a, {"TILE@d": "a:mma/w2x4/f2x4/k2"}) == 2

    def test_missing_key_compares_as_empty(self):
        assert gnb.knob_distance({}, {"STAGE@a1": "d2"}) == 1
        # A codec against an absent key counts each present segment.
        assert gnb.knob_distance({}, {"TILE@d": "a:mma/w2x2/f4x4"}) == 3

    def test_distance_sums_across_knobs(self):
        a = {"TILE@d": "a:mma/w2x2/f4x4/k2", "REDUCE@a1": "g2k"}
        b = {"TILE@d": "a:mma/w2x2/f2x4/k2", "REDUCE@a1": "g8k"}
        assert gnb.knob_distance(a, b) == 2


class TestSpecAndKey:
    def test_spec_is_sorted_and_order_stable(self):
        assert gnb.knob_spec({"B": "2", "A": "1"}) == gnb.knob_spec({"A": "1", "B": "2"}) == "A=1,B=2"

    def test_spec_rejects_grammar_breaking_values(self):
        with pytest.raises(ValueError):
            gnb.knob_spec({"A": "1,2"})
        with pytest.raises(ValueError):
            gnb.knob_spec({"A": "x=y"})

    def test_point_key_stable_and_discriminating(self):
        k = gnb.point_key("RTX 4090", "M2048xN512xK3840_fp16", "A=1")
        assert k == gnb.point_key("RTX 4090", "M2048xN512xK3840_fp16", "A=1")
        assert k != gnb.point_key("RTX 5090", "M2048xN512xK3840_fp16", "A=1")
        assert k != gnb.point_key("RTX 4090", "M2048xN512xK3840_fp16", "A=2")


class TestPickBatch:
    def test_seeded_and_bounded(self):
        remaining = {"g1": [("k1", "s1"), ("k2", "s2")], "g2": [("k3", "s3")]}
        gid_a, pts_a = gnb.pick_batch(random.Random(7), dict(remaining), batch=5)
        gid_b, pts_b = gnb.pick_batch(random.Random(7), dict(remaining), batch=5)
        assert (gid_a, pts_a) == (gid_b, pts_b)  # deterministic under a seed
        assert len(pts_a) <= len(remaining[gid_a])  # batch capped by availability

    def test_group_choice_tracks_remaining_counts(self):
        # 9:1 remaining split — the big group must dominate the draws, so a
        # time-truncated run samples the pool near its true distribution.
        remaining = {"big": [(f"k{i}", f"s{i}") for i in range(90)], "small": [("kx", "sx")] * 10}
        rng = random.Random(0)
        picks = [gnb.pick_batch(rng, remaining, batch=1)[0] for _ in range(300)]
        big_share = picks.count("big") / len(picks)
        assert 0.8 < big_share < 1.0

    def test_empty_groups_are_never_picked(self):
        remaining = {"empty": [], "full": [("k", "s")]}
        assert gnb.pick_batch(random.Random(0), remaining, batch=2)[0] == "full"


@dataclass
class _FakeBase:  # stands in for GoldenConfig (the shared non-shape fields)
    name: str = "g"
    dynamic: bool = False


@dataclass
class _FakeReduce(_FakeBase):
    M: int = 4096
    K: int = 512
    dtype: str = "fp32"


_BASE_FIELDS = frozenset({"name", "dynamic"})


class TestGroupId:
    def test_matmul_keeps_the_legacy_spelling(self):
        @dataclass
        class Matmul(_FakeBase):
            M: int = 2048
            N: int = 512
            K: int = 3840
            dtype: str = "fp16"
            trans_b: bool = True

        gid = gnb._group_id("matmul", Matmul(dynamic=True), _BASE_FIELDS)
        assert gid == "M2048xN512xK3840_fp16_dyn_tb"

    def test_generic_kind_derives_from_own_fields(self):
        assert gnb._group_id("reduce", _FakeReduce(), _BASE_FIELDS) == "reduce_M4096_K512_dtypefp32"
        assert gnb._group_id("reduce", _FakeReduce(dynamic=True), _BASE_FIELDS) == "reduce_M4096_K512_dtypefp32_dyn"

    def test_unknown_new_kind_never_crashes(self):
        # The resilience contract: a golden kind added after this script was written
        # (new discriminator, new shape fields) still gets a stable, distinct id.
        @dataclass
        class Novel(_FakeBase):
            rows: int = 7
            fused: bool = True

        gid = gnb._group_id("novel_kind", Novel(), _BASE_FIELDS)
        assert gid == "novel_kind_rows7_fusedTrue"
        assert gid != gnb._group_id("novel_kind", Novel(rows=8), _BASE_FIELDS)


class TestAssignSlice:
    def test_own_wins_even_when_cross_is_closer(self):
        assert gnb.assign_slice(2, 0, max_dist=2) == "own"

    def test_cross_when_only_cross_is_within_dist(self):
        assert gnb.assign_slice(None, 1, max_dist=2) == "cross"
        assert gnb.assign_slice(5, 2, max_dist=2) == "cross"

    def test_tail_when_no_anchor_is_within_dist(self):
        assert gnb.assign_slice(None, None, max_dist=2) == "tail"
        assert gnb.assign_slice(3, 4, max_dist=2) == "tail"

    def test_boundary_distance_is_inclusive(self):
        assert gnb.assign_slice(2, None, max_dist=2) == "own"
        assert gnb.assign_slice(None, 2, max_dist=2) == "cross"


class TestPickSlice:
    SHARES = {"own": 0.6, "cross": 0.25, "tail": 0.15}

    def test_deterministic_under_a_seed(self):
        remaining = {s: {"g": [("k", "s")]} for s in gnb.SLICES}
        picks_a = [gnb.pick_slice(random.Random(7), self.SHARES, remaining) for _ in range(5)]
        picks_b = [gnb.pick_slice(random.Random(7), self.SHARES, remaining) for _ in range(5)]
        assert picks_a == picks_b

    def test_empty_slice_is_never_picked(self):
        remaining = {"own": {}, "cross": {"g": [("k", "s")]}}
        rng = random.Random(0)
        assert all(gnb.pick_slice(rng, self.SHARES, remaining) == "cross" for _ in range(50))

    def test_draws_track_shares(self):
        remaining = {s: {"g": [("k", "s")]} for s in gnb.SLICES}
        rng = random.Random(0)
        picks = [gnb.pick_slice(rng, self.SHARES, remaining) for _ in range(1000)]
        own_share = picks.count("own") / len(picks)
        assert 0.5 < own_share < 0.7  # tracks the 60% share

    def test_exhausted_slice_share_flows_to_the_rest(self):
        remaining = {"cross": {"g": [("k", "s")]}, "tail": {"g": [("k", "s")]}}
        rng = random.Random(0)
        picks = [gnb.pick_slice(rng, self.SHARES, remaining) for _ in range(1000)]
        cross_share = picks.count("cross") / len(picks)
        assert 0.5 < cross_share < 0.75  # 0.25 / (0.25 + 0.15) = 0.625, own's share redistributed


class TestTailSample:
    def test_cap_respected_and_deterministic(self):
        specs = [f"A={i}" for i in range(50)]
        out = gnb.tail_sample(specs, cap=10)
        assert len(out) == 10
        assert out == gnb.tail_sample(specs, cap=10)

    def test_input_order_independent(self):
        specs = [f"A={i}" for i in range(20)]
        assert gnb.tail_sample(specs, cap=5) == gnb.tail_sample(list(reversed(specs)), cap=5)

    def test_stable_subset_on_fixed_input(self):
        # The hash ranking is a frozen contract: a changed selection would orphan
        # ledger entries mid-coverage. Pin the exact subset for one fixed input.
        specs = [f"A={i}" for i in range(10)]
        assert gnb.tail_sample(specs, cap=3) == ["A=5", "A=9", "A=3"]

    def test_cap_larger_than_pool_keeps_everything(self):
        specs = ["A=1", "A=2"]
        assert sorted(gnb.tail_sample(specs, cap=10)) == specs


class TestNormalizeShares:
    def test_normalizes_to_fractions(self):
        shares = gnb.normalize_shares(60, 25, 15)
        assert shares == {"own": 0.6, "cross": 0.25, "tail": 0.15}
        assert sum(shares.values()) == 1.0

    def test_zero_share_allowed_for_a_slice(self):
        assert gnb.normalize_shares(1, 0, 0) == {"own": 1.0, "cross": 0.0, "tail": 0.0}

    def test_rejects_negative_and_all_zero(self):
        with pytest.raises(ValueError):
            gnb.normalize_shares(-1, 2, 1)
        with pytest.raises(ValueError):
            gnb.normalize_shares(0, 0, 0)


class TestLedger:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "ledger.json"
        ledger = gnb.load_ledger(path)
        gnb.mark(ledger, "p1", "O1", "ok", spec="A=1")
        gnb.save_ledger(path, ledger)
        again = gnb.load_ledger(path)
        assert gnb.opt_state(again, "p1", "O1") == {"status": "ok", "attempts": 1}
        assert again["points"]["p1"]["spec"] == "A=1"

    def test_corrupt_or_missing_file_starts_fresh(self, tmp_path):
        assert gnb.load_ledger(tmp_path / "absent.json")["points"] == {}
        bad = tmp_path / "bad.json"
        bad.write_text("[]")
        assert gnb.load_ledger(bad)["points"] == {}

    def test_terminal_statuses_stick(self):
        ledger = gnb.load_ledger(Path("/nonexistent"))
        assert gnb.needs_run(ledger, "p", "O1", max_attempts=2)  # fresh point
        for status in sorted(gnb.TERMINAL):
            gnb.mark(ledger, f"p_{status}", "O1", status)
            assert not gnb.needs_run(ledger, f"p_{status}", "O1", max_attempts=2)

    def test_non_terminal_retries_up_to_attempt_cap(self):
        ledger = gnb.load_ledger(Path("/nonexistent"))
        gnb.mark(ledger, "p", "O3", "timeout")
        assert gnb.needs_run(ledger, "p", "O3", max_attempts=2)  # one attempt left
        gnb.mark(ledger, "p", "O3", "timeout")
        assert not gnb.needs_run(ledger, "p", "O3", max_attempts=2)

    def test_opts_tracked_independently(self):
        ledger = gnb.load_ledger(Path("/nonexistent"))
        gnb.mark(ledger, "p", "O1", "ok")
        assert not gnb.needs_run(ledger, "p", "O1", max_attempts=2)
        assert gnb.needs_run(ledger, "p", "O3", max_attempts=2)
