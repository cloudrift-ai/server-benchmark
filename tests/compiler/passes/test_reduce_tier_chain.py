"""A root's reduce member binds partitioned: the provider chain it closes over (workspace loads, the
rsqrt captures) emits ahead of the strided loop, shared by every lane; the merge broadcasts; the
close is lane-distributed (DeepSeek-V4 post4096's composed-cut pieces). The provider is the member's
OPERAND — a term closes over the values it reads — so the hoist is ``Fold.lower``'s own."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, TreeHalve, WarpShuffle
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.schedule import Placement, Raster, Reduce, Stage, Tile, Work, derive_inventory
from emmy.compiler.ir.schedule.base import Schedule
from emmy.compiler.ir.schedule.classic import (
    ClassicMaterialization,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
)
from emmy.compiler.ir.stmt import Assign, Cond, Load, Loop, StridedLoop, Write
from emmy.compiler.ir.tile import TileOp, blockify
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize
from tests.compiler.terms import contraction, projection, reduction, slab

_M = Axis("m", 4)
_K, _J = Axis("k", 128), Axis("j", 256)
_PLACE = Placement(free=(_M,))


def _stamped(root: Fold, plans: dict, axes: tuple = (_K,)) -> TileOp:
    """A scheduled ``TileOp`` carrying ``plans`` (member node → :class:`Reduce`) as its accepted
    classic assignment. The kernel ``WORK`` derives from the widest cooperative plan, because the
    assignment's own validation requires the inventory to realize the node choices."""
    tile = TileOp(op=root, place=_PLACE, axes=(_M, *axes))
    tile = replace(tile, blocks=blockify(tile))
    by_site = {tile.node_id(node): plan for node, plan in plans.items()}
    nodes = {
        site: ProjectionSchedule(Tile()) if tile.views[site].axis is None else ReductionSchedule(Tile(), by_site.get(site, Reduce()))
        for site in tile.node_sites
    }
    coop = max((plan.coop for plan in plans.values()), default=1)
    work = derive_inventory((Tile(),), coop=coop) or Work()
    edges = {edge: EdgeSchedule(Stage.direct()) for edge in tile.edge_sites}
    return replace(
        tile,
        schedule=Schedule(KernelSchedule(work, Raster.parse("")), nodes, edges),
        materialization=ClassicMaterialization({}, {}),
    )


def _provider() -> Fold:
    """The provider chain — a workspace row read and its rsqrt — as the cone a member closes over."""
    return projection((), (Load(name="ws", input="cutbuf", index=(Var("m"),)), Assign(name="scale", op="rsqrt", args=("ws",))))


def _reduce(axis: Axis, acc: str, factor: Fold, src: str) -> Fold:
    """``acc = Σ_axis src[m, axis] · factor`` — the member fold, ``factor`` an operand it closes over
    (never defined inside the loop)."""
    scaled = Assign(name=f"{acc}__v", op="multiply", args=(f"{src}_e", factor.exposes[0]))
    return reduction(axis, (slab(f"{src}_e", src, "m", axis.name), factor), (scaled,), (acc,))


def _chain_tile(plan: Reduce, post: tuple = ()) -> TileOp:
    provider = _provider()
    red = _reduce(_K, "acc", provider, "x")
    root = projection((red, provider), post, results=post[-1].defines() if post else ("acc",))
    return _stamped(root, {red: plan})


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
    bound = factorize(_chain_tile(Reduce.of(coop=64)), root=None)
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
    bound = factorize(_chain_tile(Reduce.of(reg=2)), root=None)
    read = _names_read(_flat(bound.body))
    assert "scale" in read and "scale__r1" not in read, "the provider value is one shared read"
    defined = {n for s in _flat(bound.body) for n in s.defines()}
    assert "acc__r1" in defined, "the second ILP chain must exist"
    assert bound.block_threads is None and not any(a.name.endswith("_co") for a in bound.axes)


def test_post_members_close_after_the_merge() -> None:
    post = (Assign(name="out", op="multiply", args=("acc", "scale")),)
    bound = factorize(_chain_tile(Reduce.of(coop=64), post=post), root=None)
    flat = _flat(bound.body)
    combines = [i for i, s in enumerate(flat) if isinstance(s, (WarpShuffle, TreeHalve))]
    assert combines, "the cross-thread combine must close the fold"
    i_post = next(i for i, s in enumerate(flat) if isinstance(s, Assign) and "out" in s.defines())
    assert combines[0] < i_post, "the trailing segment reads the merged carrier"


def test_a_transposed_band_on_a_chain_member_binds_serial() -> None:
    """The ``coop-t`` band's σ-substitution and guarded close assume the fold is the kernel ROOT,
    so the chain arm cannot realize one. It must fall to the degenerate serial arm — realizing it
    as a PLAIN coop band would mint one kernel from two knob spellings."""
    bound = factorize(_chain_tile(Reduce.of(coop=32, coop_transposed=True)), root=None)
    flat = _flat(bound.body)
    assert not any(isinstance(s, StridedLoop) for s in flat), "a transposed band is not offered the chain arm"
    assert any(isinstance(s, Loop) for s in flat), "the member still folds serially per cell"
    assert not any(a.name.endswith("_co") for a in bound.axes)
    assert bound.block_threads is None


def _two_member_root() -> tuple[Fold, Fold, Fold]:
    """Two members: ``red_b`` folds ``y`` scaled by ``mid = acc · scale``, the cone over ``red_a``'s
    state and the shared provider — so the second member closes over the first."""
    provider = _provider()
    red_a = _reduce(_K, "acc", provider, "x")
    mid = projection((red_a, provider), (Assign(name="mid", op="multiply", args=("acc", "scale")),))
    red_b = _reduce(_J, "acc2", mid, "y")
    return red_a, red_b, projection((red_b,), results=("acc2",))


def _two_member_tile(plan_a: Reduce, plan_b: Reduce) -> TileOp:
    red_a, red_b, root = _two_member_root()
    return _stamped(root, {red_a: plan_a, red_b: plan_b}, axes=(_K, _J))


def test_two_partitioned_members_share_one_lane_axis() -> None:
    bound = factorize(_two_member_tile(Reduce.of(coop=64), Reduce.of(coop=64)), root=None)
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
    bound = factorize(_two_member_tile(Reduce.of(coop=64), Reduce.of(reg=2)), root=None)
    stmts = list(bound.body)
    strided = [s for s in stmts if isinstance(s, StridedLoop)]
    assert [(s.axis.name, s.start, s.step.value) for s in strided] == [("k", Var("k_co"), 64), ("j", Literal(0, "int"), 2)]
    assert [a.name for a in bound.axes if a.name.endswith("_co")] == ["k_co"]
    assert bound.block_threads == 64
    assert "acc2__r1" in {n for s in _flat(bound.body) for n in s.defines()}, "the ILP-only member still replicates"


def test_a_contraction_chain_member_binds_through_the_chain_arm() -> None:
    """The reduce binder reads only the member's axis and its :class:`Reduce`, never its algebra, so a
    CONTRACTION member partitions its K exactly like a monoid fold — a contraction is a monoid with
    a ⊗ lift, and the reduce domain hands both the same catalog through one projection."""
    provider = _provider()
    cone = projection(
        (provider,),
        (Load(name="a_e", input="A", index=(Var("m"), Var("k"))), Assign(name="a_scaled", op="multiply", args=("a_e", "scale"))),
    )
    con = contraction(_K, cone, (Load(name="b_e", input="B", index=(Var("k"),)), "acc"))
    bound = factorize(_stamped(projection((con,), results=("acc",)), {con: Reduce.of(coop=64)}), root=None)

    flat = _flat(list(bound.body))
    strided = [s for s in flat if isinstance(s, StridedLoop)]
    assert [(s.axis.name, s.start, s.step.value) for s in strided] == [("k", Var("k_co"), 64)]
    assert [a.name for a in bound.axes if a.name.endswith("_co")] == ["k_co"]
    assert bound.block_threads == 64
    assert any(isinstance(s, (WarpShuffle, TreeHalve, Smem)) for s in flat), "the cross-thread combine must close the fold"


def test_the_walk_offers_and_the_binder_realizes_two_partitioned_members(monkeypatch) -> None:
    """End-to-end through the ACTUAL enumeration, no direct stamping: a chain kernel with two
    reduce members enumerates rows where both members carry the coop band — a row holds ONE worker
    inventory, which is what forces the shared width the binder's coop-agreement assert leans
    on — and the materialized leaf binds through the chain arm with one shared lane axis striding
    both folds."""
    import importlib  # noqa: PLC0415

    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves  # noqa: PLC0415

    classic_forks = importlib.import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule").classic_forks
    for var in ("EMMY_KNOBS", "EMMY_PLACE", "EMMY_TILE", "EMMY_WORK", "EMMY_STAGE", "EMMY_REDUCE", "EMMY_RASTER"):
        monkeypatch.delenv(var, raising=False)
    _, _, root = _two_member_root()
    tile = TileOp(op=root, place=_PLACE, axes=(_M, _K, _J), name="k_two_member_probe", knobs={})
    tile = replace(tile, blocks=blockify(tile))

    leaves = list(iter_leaves(classic_forks(tile, tile.name, {}, Context.from_target((12, 0)))))
    keys = sorted({key for leaf in leaves for key in leaf.knobs if key.split("@", 1)[0] == "REDUCE"})
    assert len(keys) == 2, f"both members must be keyed schedule sites: {keys}"
    both = [leaf for leaf in leaves if all(str(leaf.knobs.get(key, "")) == "coop" for key in keys)]
    assert both, "the enumeration must offer a row partitioning BOTH members"

    (scheduled_tile,) = both[0].expand()
    bound = factorize(scheduled_tile, root=None)
    strided = [s for s in bound.body if isinstance(s, StridedLoop)]
    assert len(strided) == 2, "each partitioned member emits its own strided fold"
    lanes = [a for a in bound.axes if a.name.endswith("_co")]
    assert len(lanes) == 1, "cooperating members share ONE lane axis"
    width = lanes[0].extent.as_static()
    assert [s.step.value for s in strided] == [width, width], "both folds stride by the one inventory's width"
    assert [s.start for s in strided] == [Var(lanes[0].name), Var(lanes[0].name)]
    assert bound.block_threads == width
