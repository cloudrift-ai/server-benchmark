"""The capability-derived cold lead for the structural placement fork.

A computed-operand contraction whose target offers only ``materialized_edges_only`` atoms has no
hardware contraction tier in its fused form; cold, the lead selects the fork option that cuts the
computed edge (the ``PLACE@a`` seam). A target with an inline-capable atom keeps the fused
default, and a dtype with no atom at all keeps the functional fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Assign, Load
from emmy.compiler.ir.tile import Channel, Fold, TileOp
from emmy.compiler.ir.tile.ir import Placement
from emmy.compiler.pipeline.fork import OptionFork
from emmy.compiler.pipeline.passes.lowering.tile._atomize import make_cone
from emmy.compiler.pipeline.search.policy.greedy import _cold_placement_lead


def _placement_fork(dtype: str) -> list:
    cone = make_cone(
        [
            Load(name="g_e", input="g", index=(Var("m"), Var("k"))),
            Assign(name="g_s", op="silu", args=("g_e",)),
        ],
        "k",
    )
    node = Fold.contraction(
        k_axis=Axis("k", 64),
        a=cone,
        channels=(Channel(b=Load(name="w_e", input="w", index=(Var("k"), Var("n"))), acc="acc0"),),
    )
    inputs = {"g": Tensor("g", (64, 64), dtype), "w": Tensor("w", (64, 64), dtype)}
    fused = TileOp(op=node, place=Placement(free=(Axis("m", 64), Axis("n", 64))), inputs=inputs)
    return [
        OptionFork(option=fused, knobs={"PLACE@a": "fuse"}),
        OptionFork(option=Graph(), knobs={"PLACE@a": "cut"}),
    ]


def test_cold_lead_cuts_the_computed_edge_when_atoms_are_materialized_only() -> None:
    leaves = _placement_fork("f16")
    fp = SimpleNamespace(ctx=Context.from_target((7, 0)))
    lead = _cold_placement_lead(fp, leaves)
    assert lead is leaves[1]


def test_cold_lead_keeps_fused_when_an_inline_capable_atom_exists() -> None:
    leaves = _placement_fork("f16")
    fp = SimpleNamespace(ctx=Context.from_target((9, 0)))
    assert _cold_placement_lead(fp, leaves) is None


def test_cold_lead_keeps_the_functional_fallback_without_any_atom() -> None:
    leaves = _placement_fork("f64")
    fp = SimpleNamespace(ctx=Context.from_target((7, 0)))
    assert _cold_placement_lead(fp, leaves) is None
