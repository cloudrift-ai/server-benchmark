"""A chain-form root's direct members bind partitioned: the provider chain (workspace loads, the
rsqrt captures) emits ahead of the strided loop, shared by every lane; the merge broadcasts; the
close is lane-distributed (DeepSeek-V4 post4096's composed-cut pieces)."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, TreeHalve, WarpShuffle
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, StridedLoop, Write
from emmy.compiler.ir.tile import ReducePlan, TileOp
from emmy.compiler.ir.tile.ops import Sched
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

_PLACE = Placement(free=(Axis("m", 4),))


def _with_slice(tile: TileOp, family: str, node, value) -> TileOp:
    """Attach one schedule slice by rebuilding — ops are frozen; a slice map is assembled before
    construction, never written after."""
    schedule = dict(tile.schedule)
    Sched(tile.op, schedule, place=tile.place).put(family, node, value)
    return replace(tile, schedule=schedule)


def _reduce(axis: str, extent: int, acc: str, factor: str, src: str) -> Fold:
    """``acc = Σ_axis src[m, axis] · factor`` — the member fold, its ``factor`` captured from the
    provider chain ahead of it (never defined inside the loop)."""
    body = Body(
        (
            Load(name=f"{src}_e", input=src, index=(Var("m"), Var(axis))),
            Assign(name=f"{src}_scaled", op="multiply", args=(f"{src}_e", factor)),
            Accum(name=acc, value=f"{src}_scaled", op="add", axes=(axis,)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis(axis, extent), body=body, role=AxisRole.PLANAR))
    assert red is not None
    return red


def _chain_tile(plan: ReducePlan, post: tuple = ()) -> TileOp:
    red = _reduce("k", 128, "acc", "scale", "x")
    chain = (Load(name="ws", input="cutbuf", index=(Var("m"),)), Assign(name="scale", op="rsqrt", args=("ws",)))
    results = post[-1].defines() if post else ("acc",)
    root = Fold.projection(body=Body((*chain, red, *post)), results=results)
    assert not root.operands, "the fed member must remain in the body for this shape to be under test"
    return _with_slice(TileOp(op=root, place=_PLACE), "REDUCE", red, plan)


def _flat(stmts) -> list:
    out = []
    for s in stmts:
        out.append(s)
        for b in s.nested():
            out.extend(_flat(b))
    return out


def _names_read(stmts) -> set[str]:
    return {n for s in stmts for n in (*s.deps(), *(v for e in s.exprs() for v in e.free_vars()))}


def test_a_partitioned_chain_member_emits_the_provider_chain_ahead_of_the_strided_loop() -> None:
    bound = factorize(_chain_tile(ReducePlan.of(coop=64)), root=None)
    stmts = list(bound.body)
    loops = [i for i, s in enumerate(stmts) if isinstance(s, StridedLoop)]
    assert len(loops) == 1, "the partitioned member emits one strided fold"
    i_ws = next(i for i, s in enumerate(stmts) if isinstance(s, Load) and s.input == "cutbuf")
    assert i_ws < loops[0], "the provider chain must precede the partitioned loop"
    strided = stmts[loops[0]]
    assert strided.start == Var("k_co") and strided.step.value == 64
    assert any(isinstance(s, (WarpShuffle, TreeHalve, Smem)) for s in _flat(stmts)), "the cross-thread combine must close the fold"
    assert any(a.name == "k_co" and a.extent.as_static() == 64 for a in bound.axes)
    assert bound.block_threads == 64
    guard = next(s for s in stmts if isinstance(s, Cond) and "k_co" in s.deps())
    assert any(isinstance(g, Write) for g in _flat(guard.body)), "the scalar store closes guarded to lane 0"


def test_an_ilp_only_chain_member_shares_the_providers_unrenamed() -> None:
    bound = factorize(_chain_tile(ReducePlan.of(reg=2)), root=None)
    read = _names_read(_flat(bound.body))
    assert "scale" in read and "scale__r1" not in read, "the provider value is one shared read"
    defined = {n for s in _flat(bound.body) for n in s.defines()}
    assert "acc__r1" in defined, "the second ILP chain must exist"
    assert bound.block_threads is None and not any(a.name.endswith("_co") for a in bound.axes)


def test_post_members_close_after_the_merge() -> None:
    post = (Assign(name="out", op="multiply", args=("acc", "scale")),)
    bound = factorize(_chain_tile(ReducePlan.of(coop=64), post=post), root=None)
    flat = _flat(bound.body)
    combines = [i for i, s in enumerate(flat) if isinstance(s, (WarpShuffle, TreeHalve))]
    assert combines, "the cross-thread combine must close the fold"
    i_post = next(i for i, s in enumerate(flat) if isinstance(s, Assign) and "out" in s.defines())
    assert combines[0] < i_post, "the trailing segment reads the merged carrier"


def _two_member_tile(plan_a: ReducePlan, plan_b: ReducePlan) -> TileOp:
    red_a = _reduce("k", 128, "acc", "scale", "x")
    red_b = _reduce("j", 256, "acc2", "mid", "y")
    chain = (Load(name="ws", input="cutbuf", index=(Var("m"),)), Assign(name="scale", op="rsqrt", args=("ws",)))
    mid = Assign(name="mid", op="multiply", args=("acc", "scale"))
    root = Fold.projection(body=Body((*chain, red_a, mid, red_b)), results=("acc2",))
    tile = _with_slice(TileOp(op=root, place=_PLACE), "REDUCE", red_a, plan_a)
    return _with_slice(tile, "REDUCE", red_b, plan_b)


def test_two_partitioned_members_share_one_lane_axis() -> None:
    bound = factorize(_two_member_tile(ReducePlan.of(coop=64), ReducePlan.of(coop=64)), root=None)
    stmts = list(bound.body)
    loops = [i for i, s in enumerate(stmts) if isinstance(s, StridedLoop)]
    assert len(loops) == 2, "each partitioned member emits its own strided fold"
    i_mid = next(i for i, s in enumerate(stmts) if isinstance(s, Assign) and "mid" in s.defines())
    assert loops[0] < i_mid < loops[1], "the between-members segment emits in body order"
    assert [s.start for s in stmts if isinstance(s, StridedLoop)] == [Var("k_co"), Var("k_co")]
    assert [a.name for a in bound.axes if a.name.endswith("_co")] == ["k_co"]


def test_an_ilp_only_member_beside_a_cooperating_one_starts_at_zero() -> None:
    """An ILP-only member folds its whole axis on EVERY lane — no lane start, no cross-thread
    combine of its own — so its result is identical per lane and the segments after it stay
    deterministic. Only the cooperating member contributes the shared lane axis."""
    bound = factorize(_two_member_tile(ReducePlan.of(coop=64), ReducePlan.of(reg=2)), root=None)
    stmts = list(bound.body)
    strided = [s for s in stmts if isinstance(s, StridedLoop)]
    assert [(s.axis.name, s.start, s.step.value) for s in strided] == [("k", Var("k_co"), 64), ("j", Literal(0, "int"), 2)]
    assert [a.name for a in bound.axes if a.name.endswith("_co")] == ["k_co"]
    assert bound.block_threads == 64
    assert "acc2__r1" in {n for s in _flat(bound.body) for n in s.defines()}, "the ILP-only member still replicates"
