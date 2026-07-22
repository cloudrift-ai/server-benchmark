"""scripts/golden_neighbor_bench.py — the pure selection/resume logic (no GPU, no emmy imports).

Covers the component-aware knob distance, the order-stable point spec/key, the
remaining-proportional randomized batch sampling, and the ledger's resume semantics
(terminal statuses stick, non-terminal ones retry up to the attempt cap).
"""

import random
import sys
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
