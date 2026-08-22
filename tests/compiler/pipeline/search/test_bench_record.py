"""``bench_leaves`` — turning a benched graph into rows the tune DB can compare.

Every row carries two structural facts that arrive by different routes: its POOL KEY, digested from the
pre-descent offer site, and its FEATURES, taken from the kernel that actually ran. Nothing made them agree, and
when they disagree the row accounts for a piece of its pool's work while claiming to be a rival of the whole —
which is how a 5.9 µs ``rms_norm`` kernel came to be scored against the 131 ms fused megakernel it was split out
of.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.pipeline.search.bench_record import FAIL_SENTINEL_US, bench_leaves

# The fused rms_norm->linear megakernel and one kernel of the same op's unfused realization — the pair every
# mixed pool in the RTX 5090 freeze turned out to hold.
WHOLE = {"S_reduce_add": 2.0, "S_ext_free_prod": 69632.0, "S_ext_reduce_prod": 14745600.0, "S_ext_n_reduce_axis": 2.0}
PIECE = {"S_reduce_add": 1.0, "S_ext_free_prod": 30720.0, "S_ext_reduce_prod": 3840.0, "S_ext_n_reduce_axis": 1.0}


@dataclass
class _Launch:
    time_ms: float
    samples: tuple = ()


@dataclass
class _Bench:
    per_launch: list


@dataclass
class _Node:
    op: object
    inputs: tuple = ()


class _Graph:
    """The three things ``bench_leaves`` asks a compiled graph for."""

    def __init__(self, ops: list[CudaOp]) -> None:
        self.nodes = {f"k{i}": _Node(op) for i, op in enumerate(ops)}

    def topological_order(self) -> list[str]:
        return list(self.nodes)

    def producer(self, buf: str):
        return self.nodes[buf]


@dataclass
class _Site:
    """The offer op a kernel lowered from, as ``_offer_site`` reads it: a dialect tag and a knob row.

    A stub rather than a real ``LoopOp`` because a real one validates its body at construction, and a valid
    body would be pages of noise covering nothing this module does — ``bench_leaves`` never looks past these
    two attributes."""

    knobs: dict
    dialect: str = "loop"
    source: object = None


def _kernel(site_knobs: dict, own_knobs: dict) -> CudaOp:
    """A CUDA kernel whose source chain bottoms at a loop-dialect offer site."""
    return CudaOp(knobs=own_knobs, source=_Site(site_knobs))


def test_one_kernel_realizing_its_whole_site_records_a_row():
    op = _kernel({**WHOLE, "S_n_load": 6.0}, {**WHOLE, "TILE": "mma_m16n8k16_f16_f32/f2x2/k4", "WORK": "w2x8"})
    (leaf,) = bench_leaves(_Graph([op]), _Bench([_Launch(0.25, (0.25, 0.25))]))

    assert leaf.value_us == 250.0
    assert leaf.knobs["WORK"] == "w2x8"
    assert leaf.n_samples == 2  # a single-kernel group keeps its own bench stats


def test_a_split_piece_is_not_recorded_against_the_whole_ops_pool(caplog):
    """The defect this check exists for. A structural fork splits the site, and one piece's own source chain
    still bottoms at the UN-split site — so without the check it is filed as a rival of the whole op and its
    latency compared against a config doing thousands of times more work."""
    piece = _kernel({**WHOLE, "S_n_load": 6.0}, {**PIECE, "REDUCE": "coop", "WORK": "t128"})

    with caplog.at_level("WARNING"):
        leaves = bench_leaves(_Graph([piece]), _Bench([_Launch(0.0059)]))

    assert leaves == []
    assert "do not match the offer site" in caplog.text


def test_a_failed_bench_is_screened_the_same_way():
    """A ``bench_fail`` row is a durable negative example, so it is just as wrong to file it under a pool whose
    work it never attempted."""
    good = _kernel({**WHOLE, "S_n_load": 6.0}, {**WHOLE, "WORK": "w2x8"})
    piece = _kernel({**WHOLE, "S_n_load": 6.0}, {**PIECE, "WORK": "t128"})

    assert [leaf.value_us for leaf in bench_leaves(_Graph([good]), None, status="bench_fail")] == [FAIL_SENTINEL_US]
    assert bench_leaves(_Graph([piece]), None, status="bench_fail") == []


def test_kernels_sharing_a_site_are_summed_into_one_row():
    """A fragment kernel's own tiny latency must never become the site's row, so a site's kernels contribute
    ONE leaf valued at their summed launch time. The extent check must not fire here: the group's knob
    identity is its most-tunable op, which is the kernel realizing the site."""
    site = {**WHOLE, "S_n_load": 6.0}
    main = _kernel(site, {**WHOLE, "TILE": "mma_m16n8k16_f16_f32/f2x2/k4", "WORK": "w2x8", "STAGE": "d1/sync"})
    epilogue = _kernel(site, {**WHOLE, "WORK": "w1x1"})
    (leaf,) = bench_leaves(_Graph([main, epilogue]), _Bench([_Launch(0.25), _Launch(0.01)]))

    assert leaf.value_us == 260.0
    assert leaf.knobs["STAGE"] == "d1/sync"  # the most-tunable op is the group's knob identity
    assert leaf.n_samples is None  # cross-kernel samples do not align, so a summed variance would be fiction


def test_two_sites_stay_two_rows():
    """Distinct offer sites never pool, whatever their kernels' latencies."""
    a = _kernel({**WHOLE, "S_n_load": 6.0}, {**WHOLE, "WORK": "w2x8"})
    b = _kernel({**PIECE, "S_n_load": 5.0}, {**PIECE, "WORK": "w1x8"})
    leaves = bench_leaves(_Graph([a, b]), _Bench([_Launch(0.25), _Launch(0.006)]))

    assert len({leaf.op_sig for leaf in leaves}) == 2
    assert sorted(leaf.value_us for leaf in leaves) == [6.0, 250.0]
