"""The schedule pool's addressable-space contract.

Large catalog products are tested by index and by narrow pins. Tests must not flatten a live
schedule space merely to assert its cardinality or ordered digest: doing so turns catalog growth
into test time and recreates the eager traversal the production scheduler deliberately avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.pipeline.passes.lowering.tile import _pool, _schedule
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pool import PoolSample

_CC = (12, 0)

#: The knob pins the enumeration reads off the environment. A host with one set would enumerate a
#: narrowed pool and fail every digest here for a reason that has nothing to do with the traversal.
_PIN_VARS = ("EMMY_KNOBS", "EMMY_PLACE", "EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_RASTER")


def _matmul_graph(m: int, n: int, k: int, dtype: str) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _sdpa_graph(b: int = 1, h: int = 2, s: int = 64, d: int = 32) -> Graph:
    """The fused streaming cell: one primary and one pool over its computed score edge."""
    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (Dim(b), Dim(h), Dim(s), Dim(d)), dtype="f16"), node_id=name)
    g.add_node(SdpaOp(is_causal=False), ["q", "k", "v"], Tensor("o", (Dim(b), Dim(h), Dim(s), Dim(d)), dtype="f16"), node_id="o")
    g.inputs, g.outputs = ["q", "k", "v"], ["o"]
    return g


def _code_graph(code: str) -> Graph:
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415

    return graph_from_code(code)[0]


_NORM = "(lambda t: t*torch.rsqrt((t.float()*t.float()).mean(-1,keepdim=True)+1e-6).to(t.dtype))"

FIXTURES = {
    "scalar_matmul": lambda: _matmul_graph(128, 128, 128, "f32"),
    "warp_matmul": lambda: _matmul_graph(128, 128, 128, "f16"),
    "reduce_matvec": lambda: _code_graph("torch.nn.functional.linear(torch.randn(1, 4096), torch.randn(512, 4096))"),
    "fused_norm_linear": lambda: _code_graph(
        f"torch.nn.functional.linear({_NORM}(torch.randn(128, 256, dtype=torch.float16)), torch.randn(256, 256, dtype=torch.float16))"
    ),
    "flash_pair": _sdpa_graph,
}


def _spaces(graph, monkeypatch) -> list:
    """Every :class:`~._pool.PoolSpace` the graph builds, in enumeration order."""
    seen: list = []
    original = _schedule._space

    def spy(terms):
        seen.append(space := original(terms))
        return space

    monkeypatch.setattr(_schedule, "_space", spy)
    ctx = dc_replace(Context.from_target(_CC), pool_sample=PoolSample(rows=8, seed=0))
    enumerate_graph(graph, ctx)
    return seen


@pytest.fixture
def unpinned(monkeypatch):
    for var in _PIN_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_real_spaces_are_addressable_without_exhaustion(case, unpinned, monkeypatch) -> None:
    spaces = _spaces(FIXTURES[case](), monkeypatch)
    assert spaces
    for space in spaces:
        assert len(space) > 0
        for index in {0, len(space) // 2, len(space) - 1}:
            assert space[index] == space[index]


@pytest.mark.parametrize("case, tile_sites, reduce_sites", (("fused_norm_linear", 1, 2), ("flash_pair", 2, 3)))
def test_computed_fold_sites_remain_addressable(case, tile_sites, reduce_sites, unpinned, monkeypatch) -> None:
    space = _spaces(FIXTURES[case](), monkeypatch)[0]
    assert sum(key == "TILE" or key.startswith("TILE@") for key in space.keys) == tile_sites
    assert sum(key == "REDUCE" or key.startswith("REDUCE@") for key in space.keys) == reduce_sites


def _pin_paired_mma(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    # The score's N tile is the value contraction's streamed K block.
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f1x1")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")


def test_sdpa_fold_tree_offers_a_paired_mma_row(unpinned, monkeypatch) -> None:
    _pin_paired_mma(monkeypatch)
    monkeypatch.setenv("EMMY_REDUCE", "")
    space = _spaces(_sdpa_graph(), monkeypatch)[0]
    row = dict(space[0])
    assert sum(key.startswith("TILE@") and "mma_" in value for key, value in row.items()) == 2


def test_paired_sdpa_honors_a_grid_reduce_partition(unpinned, monkeypatch) -> None:
    _pin_paired_mma(monkeypatch)
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    space = _spaces(_sdpa_graph(), monkeypatch)[0]
    row = dict(space[0])
    assert row["REDUCE"] == row["REDUCE@a3"] == row["REDUCE@pj"] == "g2k"


def test_reduce_space_keeps_combined_atomic_and_deferred_forks(unpinned, monkeypatch) -> None:
    """Inspect the lazy partition index: no full schedule-space traversal is needed."""
    space = _spaces(_matmul_graph(64, 64, 64, "f32"), monkeypatch)[0]
    choices = {value for value, _ in space.partition("REDUCE")}
    assert {"", "g2a", "g2k"} <= choices


# --- the space itself: two traversals of one structure -------------------------------------------


@dataclass(frozen=True)
class _FakeRow:
    """The whole contract :mod:`_pool` has with a row — spelled knobs, the stamps its one open axis
    offers, and the derived width. Nothing else, which is why the space needs no ``_schedule``
    import and cannot acquire one by accident."""

    knobs: dict
    stages: tuple = ()

    @property
    def width(self) -> int:
        return len(self.stages) or 1


def _fake_space() -> _pool.PoolSpace:
    """A deliberately RAGGED space: segments of different sizes, rows of different widths, one row
    with no open axis at all, and two launch orders — so every radix in the arithmetic is exercised
    and none of them can be a coincidence of being 1."""
    rows_a = [
        _FakeRow({"TILE": "t0"}, ({"STAGE": ""}, {"STAGE": "d1/smem"}, {"STAGE": "d2/smem"})),
        _FakeRow({"TILE": "t1"}, ({"STAGE": ""},)),
        _FakeRow({"TILE": "t2"}),
    ]
    rows_b = [_FakeRow({"TILE": "t3"}, ({"STAGE": ""}, {"STAGE": "d1/smem"}))]
    return _pool.PoolSpace.build(
        ("TILE", "STAGE"),
        {"TILE": "", "STAGE": ""},
        [
            _pool.Segment.build(rows_a, {"WORK": ""}, [{"RASTER": ""}, {"RASTER": "gm8"}]),
            _pool.Segment.build(rows_b, {"WORK": "w2x2"}, [{"RASTER": ""}]),
            _pool.Segment.build([], {"WORK": "t128"}, [{"RASTER": ""}]),
        ],
    )


def test_the_two_traversals_are_one_traversal() -> None:
    """``list(space)`` and ``space[i]`` decode the same radices, so they must agree element for
    element — the no-drift guarantee, which is structural rather than a discipline."""
    space = _fake_space()
    assert len(space) == (3 + 1 + 1) * 2 + 2 * 1
    assert len(space) == len(list(space))
    assert list(space) == [space[i] for i in range(len(space))]
    assert space[-1] == space[len(space) - 1]
    with pytest.raises(IndexError):
        space[len(space)]


def test_the_size_is_known_before_any_candidate_exists(monkeypatch) -> None:
    """``len`` is a prefix-sum lookup: it must not build a candidate. That is what lets the row
    budget be checked before 400k dicts exist, and what makes an indexed sample exact."""
    space = _fake_space()

    def explode(*_args):
        raise AssertionError("len() must not spell a candidate")

    monkeypatch.setattr(_pool, "spell", explode)
    assert len(space) == 12
    assert all(len(seg) for seg in space.segments), "an empty segment must never enter the space"
    with pytest.raises(AssertionError):
        space[0]


def test_a_large_structural_partition_does_not_read_candidates() -> None:
    """Fork grouping delegates to the row product even when the product has millions of rows."""

    class Rows:
        closed = True

        def __init__(self, size: int, groups=()):
            self.size = size
            self.groups = groups

        def __len__(self):
            return self.size

        def __getitem__(self, _index):
            raise AssertionError("structural partition read a candidate")

        def partition(self, key):
            assert key == "TILE"
            return self.groups

    mma = Rows(3_000_000)
    scalar = Rows(7_000_000)
    rows = Rows(10_000_000, (("mma", mma), ("", scalar)))
    space = _pool.PoolSpace.build(("TILE",), {"TILE": ""}, [_pool.Segment.build(rows, {"WORK": "w1x1"}, ({},))])

    groups = dict(space.partition("TILE"))
    assert len(groups["mma"]) == 3_000_000
    assert len(groups[""]) == 7_000_000


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_a_real_space_addresses_what_it_iterates(case, unpinned, monkeypatch) -> None:
    """Boundary indices of live spaces are stable without flattening those spaces."""
    spaces = _spaces(FIXTURES[case](), monkeypatch)
    assert spaces
    for space in spaces:
        assert space[0] == next(iter(space))
        assert space[-1] == space[len(space) - 1]
