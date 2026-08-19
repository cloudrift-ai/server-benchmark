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
with its own nested statistic site) and the streaming flash pair (two primaries, two pools).

**If a digest fails after a deliberate catalog or legality change**, re-record it — the digest
pins the traversal, not the catalog. Print the new triple from a scratch run and update
:data:`EXPECTED`; a digest change with no intended catalog change is the regression this exists
to catch.
"""

from __future__ import annotations

import hashlib

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.pipeline.knob import canonical_row_key
from emmy.compiler.pipeline.passes.lowering.tile import _schedule
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
    """The streaming pair: the hoisted score edge and the derived P@V, two primaries and so two
    pools, the second carrying the chain's own ``REDUCE@<axis>`` statistic site."""
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
    "warp_matmul": [(("TILE", "STAGE", "REDUCE"), 122308, "38eb4fdc94a7be1bb8cc889f3dd01462")],
    "reduce_matvec": [(("TILE", "STAGE", "REDUCE"), 20, "bc42c1d8f8640471f226c25327e6d792")],
    "fused_norm_linear": [(("TILE", "STAGE@a1", "STAGE", "REDUCE@a1", "REDUCE"), 21495, "3931bd58b58a61f03789de5e68ea4747")],
    "flash_pair": [
        (("TILE", "STAGE", "REDUCE"), 23726, "1a95fa4c2e0823132dfd854c3d46ded0"),
        (("TILE", "STAGE", "REDUCE@a2", "REDUCE"), 4061, "f73d264419f9deda8253bdd0a805dd17"),
    ],
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
        rows, keys = original(terms, *args, **kwargs)
        pools.append((tuple(keys), [dict(r) for r in rows]))
        return rows, keys

    monkeypatch.setattr(_schedule, "_enumerate", spy)
    deploy = enumerate_graph(graph, Context.from_target(_CC))
    return pools, deploy


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
