"""The enumerated schedule pool as a SPACE — its traversal pinned element-wise.

Written against the recursion ``_enumerate`` ran BEFORE the pool became an addressable space, so
the space's own traversal is gated by behaviour that predates it. Two properties, one per test:

- **the traversal itself** — every fixture kind's ordered row sequence is digested here, so a
  rewrite that reorders, drops, adds or re-spells a single row fails. The site keys and the row
  count travel beside the digest, so a failure says WHICH of the three moved before the opaque
  hash does;
- **space-vs-deploy agreement** — the rows the enumeration emits are exactly the leaves
  ``020_schedule``'s fork actually offers a compile, compared as a sorted
  :func:`~emmy.compiler.pipeline.knob.canonical_row_key` multiset. This ties the space to the
  DEPLOY surface rather than to itself: an enumeration that agreed only with its own digest could
  drift away from the fork tree without either test noticing.

The fixtures span every shape the walk dispatches on: the scalar and warp contraction tiers (the
warp one over both raster orders), a pure reduce, a fused norm→linear (a computed operand cone
with its own nested statistic site) and the fused streaming flash cell (one primary, one pool;
the grouped placement inverse remains a separately priced structural sibling).

**If a digest fails after a deliberate catalog or legality change**, re-record it — the digest
pins the traversal, not the catalog. Print the new triple from a scratch run and update
:data:`EXPECTED`; a digest change with no intended catalog change is the regression this exists
to catch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.pipeline.knob import canonical_row_key
from emmy.compiler.pipeline.passes.lowering.tile import _pool, _schedule
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph

_CC = (12, 0)

#: The knob pins the enumeration reads off the environment. A host with one set would enumerate a
#: narrowed pool and fail every digest here for a reason that has nothing to do with the traversal.
_PIN_VARS = ("EMMY_KNOBS", "EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_RASTER")


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

#: ``fixture -> (site keys, row count, ordered-row digest)`` per pool the graph enumerates, in
#: enumeration order. Recorded against the pre-space recursion — see the module docstring.
EXPECTED: dict[str, list[tuple[tuple[str, ...], int, str]]] = {
    "scalar_matmul": [(("TILE", "STAGE", "REDUCE"), 17988, "b30d5d0f9070cd89d130ff8fad132a84")],
    "warp_matmul": [(("TILE", "STAGE", "REDUCE"), 74926, "67b94c73a84918b48aa104e1575f2399")],
    "reduce_matvec": [(("TILE", "STAGE", "REDUCE"), 20, "bc42c1d8f8640471f226c25327e6d792")],
    "fused_norm_linear": [(("TILE", "STAGE@a1", "STAGE", "REDUCE@a1", "REDUCE"), 21495, "3931bd58b58a61f03789de5e68ea4747")],
    # Over-budget paired score + value rows are intentionally absent; other fixture pools do not
    # carry concurrently-live contraction fragments and remain byte-identical.
    "flash_pair": [(("TILE", "STAGE", "REDUCE"), 3457, "b3ecfbf96e98237f7e1fe8ee258d6dec")],
}


def _digest(rows) -> str:
    """The ordered row sequence as one hash — key order inside a row is not part of the identity
    (a row IS a mapping), row order is."""
    h = hashlib.blake2b(digest_size=16)
    for row in rows:
        h.update(repr(sorted((str(k), str(v)) for k, v in row.items())).encode())
        h.update(b"\n")
    return h.hexdigest()


def _enumerated(graph, monkeypatch) -> tuple[list[tuple[tuple[str, ...], list[dict]]], list[dict]]:
    """Every pool ``_enumerate`` builds while ``graph`` resolves, in enumeration order, plus the
    rows the resolved fork tree hands a decider (:func:`enumerate_graph`, the one live-fork
    capture the fit and the record evaluator already share)."""
    pools: list[tuple[tuple[str, ...], list[dict]]] = []
    original = _schedule._enumerate

    def spy(terms, *args, **kwargs):
        rows, keys, total = original(terms, *args, **kwargs)
        pools.append((tuple(keys), [dict(r) for r in rows]))
        return rows, keys, total

    monkeypatch.setattr(_schedule, "_enumerate", spy)
    deploy = enumerate_graph(graph, Context.from_target(_CC)).rows
    return pools, deploy


def _spaces(graph, monkeypatch) -> list:
    """Every :class:`~._pool.PoolSpace` the graph builds, in enumeration order."""
    seen: list = []
    original = _schedule._space

    def spy(terms):
        seen.append(space := original(terms))
        return space

    monkeypatch.setattr(_schedule, "_space", spy)
    enumerate_graph(graph, Context.from_target(_CC))
    return seen


@pytest.fixture
def unpinned(monkeypatch):
    for var in _PIN_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_the_traversal_is_pinned_element_wise(case, unpinned, monkeypatch) -> None:
    pools, _ = _enumerated(FIXTURES[case](), monkeypatch)
    got = [(keys, len(rows), _digest(rows)) for keys, rows in pools]
    want = EXPECTED[case]
    assert [k for k, _, _ in got] == [k for k, _, _ in want], "the fork's site keys moved"
    assert [n for _, n, _ in got] == [n for _, n, _ in want], "the enumeration's row count moved"
    assert got == want, "the enumeration emits different rows, or the same rows in a different order"


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_the_space_agrees_with_the_deploy_surface(case, unpinned, monkeypatch) -> None:
    pools, deploy = _enumerated(FIXTURES[case](), monkeypatch)
    space = [row for _, rows in pools for row in rows]
    assert sorted(map(canonical_row_key, space)) == sorted(map(canonical_row_key, deploy)), (
        "the enumerated space and the fork's leaves are the same set of candidates or the space is a fiction"
    )


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


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_a_real_space_addresses_what_it_iterates(case, unpinned, monkeypatch) -> None:
    """The same equality over the LIVE spaces, whose radices are whatever the catalogs offer."""
    spaces = _spaces(FIXTURES[case](), monkeypatch)
    assert spaces
    for space in spaces:
        assert len(space) == len(list(space))
        assert list(space) == [space[i] for i in range(len(space))]
