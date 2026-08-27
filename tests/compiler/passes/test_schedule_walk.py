"""The schedule walk's enumeration contracts — what outlived the deleted addressable product.

The walk has no product space and no ``space[i]``: one recursive traversal IS the enumeration, the
prescan is the one place a catalog question is asked, and sampling is a reservoir over the leaf
stream (``search/pool.py``). What survives of the old space's contract is pinned here: one prescan
per term, computed and derived fold sites keyed as schedule sites, the paired mma row on the flash
pair, and the structural split fork standing beside the walk with both finalize arms on offer.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.passes.lowering.tile import _schedule
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pool import PoolSample

_CC = (12, 0)

#: The knob pins the enumeration reads off the environment. A host with one set would enumerate a
#: narrowed pool and fail the offer assertions here for a reason that has nothing to do with the
#: traversal.
_PIN_VARS = ("EMMY_KNOBS", "EMMY_PLACE", "EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_RASTER")


def _matmul_graph(m: int, n: int, k: int, dtype: str) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _sdpa_graph(b: int = 1, h: int = 2, s: int = 64, d: int = 32) -> Graph:
    """The fused streaming cell: one primary and one derived site over its computed score edge."""
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


def _rows(graph) -> list[dict]:
    """The sampled candidate rows of every schedule fork ``graph`` opens.

    The sample makes the walk exhaust its leaf stream once inside ``schedule`` (the reservoir
    retains nothing proportional to the pool), so these tests observe a complete traversal without
    flattening a live space into test memory."""
    ctx = dc_replace(Context.from_target(_CC), pool_sample=PoolSample(rows=8, seed=0))
    return enumerate_graph(graph, ctx).rows


@pytest.fixture
def unpinned(monkeypatch):
    for var in _PIN_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("case", sorted(FIXTURES))
def test_the_prescan_asks_each_catalog_question_once(case, unpinned, monkeypatch) -> None:
    """Every catalog question is asked ONCE per term — the invariant the walk rests on: options are
    a function of the node and the live pins, the prescan fills ``_State.options`` once, and every
    branch expansion only READS the memo. Asking a site what it offers per branch instead
    re-resolves every stage once per member of a list those same catalogs produce — seconds per
    lowered kernel while the fork above reads a single row. The sampled enumeration exhausts every
    leaf, so a reintroduced per-branch re-ask shows up here as a repeated question, not as a slow
    test somebody eventually notices."""
    asked: list[tuple] = []
    original = _schedule._options

    def spy(state, node):
        asked.append((state.tile, node))  # strong refs, so ids below cannot alias freed objects
        return original(state, node)

    monkeypatch.setattr(_schedule, "_options", spy)
    assert _rows(FIXTURES[case]())
    assert asked, "the fixture built no catalog at all"
    keys = [(id(tile), id(node)) for tile, node in asked]
    repeats = len(keys) - len(set(keys))
    assert not repeats, f"_options was asked the same question {repeats} time(s) over ({len(keys)} calls)"


@pytest.mark.parametrize("case, tile_sites, reduce_sites", (("fused_norm_linear", 1, 2), ("flash_pair", 2, 3)))
def test_computed_fold_sites_are_keyed_schedule_sites(case, tile_sites, reduce_sites, unpinned) -> None:
    """A computed cone's fold and a derived site (flash's synthesized PV) are real schedule sites:
    every row spells them, keyed by the tree-path codec, so nothing nested is silently undecided."""
    row = _rows(FIXTURES[case]())[0]
    assert sum(key == "TILE" or key.startswith("TILE@") for key in row) == tile_sites
    assert sum(key == "REDUCE" or key.startswith("REDUCE@") for key in row) == reduce_sites


def test_sdpa_fold_tree_offers_a_paired_mma_row(unpinned, monkeypatch) -> None:
    """The walk reaches a row where BOTH flash contractions ride the tensor core — the score's N
    tile feeding the value contraction's streamed K block through the fragment seam."""
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    # The score's N tile is the value contraction's streamed K block.
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f1x1")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE", "")
    rows = _rows(_sdpa_graph())
    assert any(sum(key.startswith("TILE@") and "mma_" in str(value) for key, value in row.items()) == 2 for row in rows)


def test_the_split_fork_offers_atomic_and_deferred_arms(unpinned) -> None:
    """The old product kept the cross-CTA partitions as combined rows; they are now STRUCTURAL
    siblings of the unsplit tree (``035_split_reduce``), and a fold that admits both finalizes
    still sees the atomic and the deferred arm offered together, beside the unsplit one."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    offered: set[str] = set()

    def decide(fp):
        if any(getattr(option, "structural", False) for option in fp.options):
            offered.update(str(v) for option in fp.options for k, v in option.knobs.items() if family_of(k) == "REDUCE")
        return next(iter_leaves(fp.options))

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target(_CC)).resolve(_matmul_graph(64, 64, 64, "f32"), decide)
    assert {"", "g2a", "g2k"} <= offered


def test_the_twisted_carrier_split_offers_only_the_deferred_arm(unpinned, monkeypatch) -> None:
    """The cross-CTA split composes with the paired-mma flash cell. Asserted at the offer
    function: after a cut fork decides on the same kernel, the engine's structural replay resolves
    a later structural fork inline (the count evidence is per-op, not per-rule — see
    ``_replay_structural_decision``), so the split's siblings never reach a decide on this graph —
    the offer itself is the observable. The atomic
    arm is refused on the carrier's ARITY (``atomicAdd`` folds one additive state; the twisted
    carrier streams three), while the deferred workspace arm slices the multi-component carrier.
    And the pieces re-schedule their paired sites: under the pinned deferred split the partial's
    row still spells BOTH mma contractions."""
    from types import SimpleNamespace

    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._split import split_forks
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    ctx = Context.from_target(_CC)
    captured: list[TileOp] = []

    def keep(fp):
        op = fp.root_op
        if isinstance(op, TileOp) and op.op is not None and not op.place.is_mapped and not captured:
            captured.append(op)
        return next(iter_leaves(fp.options))

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(_sdpa_graph(), keep)
    assert captured, "the flash cell must reach the tile passes as one fused kernel"
    offers = split_forks(None, SimpleNamespace(op=captured[0]))
    assert offers is not None
    spellings = {str(v) for offer in offers for v in offer.knobs.values()}
    assert "g2k" in spellings, "the deferred workspace arm must slice the twisted carrier"
    assert not any(s.startswith("g") and s.endswith("a") for s in spellings), (
        "atomicAdd folds ONE additive state; the three-component twisted carrier has no atomic arm"
    )

    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_TILE@A3", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_TILE@PJ", "mma_m16n8k16_f16_f32/f1x1")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target(_CC)).resolve(
        _sdpa_graph(), lambda fp: next(iter_leaves(fp.options))
    )
    partial = next(n.op for nid, n in out.nodes.items() if nid.endswith("__partial") and isinstance(n.op, TileOp))
    assert partial.place.is_mapped, "the partial piece must decide its own row"
    assert sum(key.startswith("TILE@") and "mma_" in str(value) for key, value in partial.knobs.items()) == 2, (
        "the sliced piece must still spell both paired mma sites"
    )
