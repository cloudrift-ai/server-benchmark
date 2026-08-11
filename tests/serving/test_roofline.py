"""Boot roofline audit (serving/roofline.py) — decision logic and advisory-only contract.
Pure CPU: the CUDA-touching measurement helpers are stubbed."""

import logging

from emmy.serving import roofline
from emmy.serving.roofline import audit_boot_programs, flag_ratio

GB = 1e9


class _Prog:
    def __init__(self, const_bytes, input_weight_bytes=0):
        self.program = object()
        self.const_bytes = const_bytes
        self.input_weight_bytes = input_weight_bytes

    @property
    def weight_bytes(self):
        return self.const_bytes + self.input_weight_bytes


def test_flag_ratio_thresholds():
    # 100 MB at 1 TB/s → floor 100 µs. 10x is the boundary: at it, no flag; above it, flag.
    assert flag_ratio(1000.0, 100_000_000, 1000 * GB) is None  # exactly 10x → silent
    floor_us, ratio = flag_ratio(1001.0, 100_000_000, 1000 * GB)
    assert abs(floor_us - 100.0) < 1e-6
    assert ratio > 10.0
    # The incident shape: ~46 MB of weights, ~700 GB/s → floor ~66 µs; measured 10,085 µs → ~153x.
    verdict = flag_ratio(10_085.0, 46_000_000, 700 * GB)
    assert verdict is not None
    assert verdict[1] > 100.0


def test_flag_ratio_skips_tiny_and_degenerate():
    assert flag_ratio(1e6, 1_000, 1000 * GB) is None  # floor below MIN_FLOOR_US → skip
    assert flag_ratio(1e6, 0, 1000 * GB) is None  # no weights at all
    assert flag_ratio(1e6, 1_000_000, 0.0) is None  # broken bandwidth measurement


def test_audit_counts_weight_inputs(monkeypatch, caplog):
    """An expert program holds no constants — its weights arrive as per-launch INPUTS. Counting
    only the constant side would give it a zero floor and audit nothing."""
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    monkeypatch.setattr(roofline, "time_program_us", lambda program: 50_000.0)  # 500x its 100 µs floor
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("moe.expert.one", _Prog(0, input_weight_bytes=100_000_000))])
    assert len(caplog.records) == 1
    assert "moe.expert.one" in caplog.text


def test_audit_warns_on_outlier_and_stays_quiet_on_healthy(monkeypatch, caplog):
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: 1000 * GB)
    times = iter([50_000.0, 150.0])  # slow program then healthy program, both floor 100 µs
    monkeypatch.setattr(roofline, "time_program_us", lambda program: next(times))
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.post.decode.m8", _Prog(100_000_000)), ("L0.pre.decode.m8", _Prog(100_000_000))])
    assert len(caplog.records) == 1
    assert "L0.post.decode.m8" in caplog.text
    assert "emmy tune" in caplog.text


def test_audit_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(roofline, "measure_copy_bw", lambda: (_ for _ in ()).throw(RuntimeError("no gpu")))
    with caplog.at_level(logging.WARNING, logger="emmy.serving.roofline"):
        audit_boot_programs([("L0.pre.decode.m8", _Prog(100_000_000))])
    assert not caplog.records  # swallowed to debug level — a boot warning is never a boot blocker
