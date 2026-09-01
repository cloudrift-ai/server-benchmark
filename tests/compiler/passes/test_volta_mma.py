"""Target selection and source lowering for the Volta m8n8k4 MMA family."""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_value
from emmy.compiler.pipeline.search.space import MAX_FRAGMENT_REGISTERS, warp_tile_moves
from emmy.compiler.target import set_target
from tests.compiler.helpers import pin_classic

VOLTA = "mma_m8n8k4_f16_f32"
AMPERE = "mma_m16n8k16_f16_f32"

# Every instruction a newer target has and sm_70 does not — the one list the source assertions share.
NEWER_INSTRUCTIONS = ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16")


def _graph(*, m: int = 16, n: int = 16, k: int = 4, trans: bool = False) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), dtype=F16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (n, k) if trans else (k, n), dtype=F16), node_id="b")
    graph.add_node(LinearOp() if trans else MatmulOp(), ["a", "b"], Tensor("c", (m, n), dtype=F16), node_id="c")
    graph.inputs, graph.outputs = ["a", "b"], ["c"]
    return graph


def _norm_linear_graph(*, m: int = 16, n: int = 16, k: int = 16) -> Graph:
    """A fused norm→linear: the projection's ``a`` edge is the norm's producer CONE, not a load."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(m), Dim(k)), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("wn", (Dim(k),), dtype=F16), node_id="wn")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(n), Dim(k)), dtype=F16), node_id="w")
    graph.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(m), Dim(k)), dtype=F16), node_id="xn")
    graph.add_node(LinearOp(), ["xn", "w"], Tensor("y", (Dim(m), Dim(n)), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "wn", "w"], ["y"]
    return graph


def _pin(monkeypatch, atom: str, *, tile: str = "f1x1", stage: str = "") -> None:
    pin_classic(monkeypatch, {"TILE": f"{atom}/{tile}"})
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    pin_classic(monkeypatch, {"STAGE": stage})
    pin_classic(monkeypatch, {"REDUCE": ""})


def _source(graph: Graph, ctx: Context) -> tuple[str, dict]:
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    kernels = [node.op for node in lowered.nodes.values() if getattr(node.op, "kernel_source", None)]
    assert len(kernels) == 1
    return kernels[0].kernel_source, kernels[0].knobs


@pytest.mark.parametrize(
    ("cc", "caps"),
    [
        ((7, 0), (True, False, False, False, False, False)),
        ((7, 5), (True, True, False, False, False, False)),
        ((8, 0), (False, True, True, False, True, False)),
        ((8, 9), (False, True, True, False, True, True)),
        ((9, 0), (False, True, True, True, True, True)),
    ],
)
def test_context_instruction_capabilities(cc, caps) -> None:
    ctx = Context(compute_capability=cc)
    assert (
        ctx.has_volta_mma,
        ctx.has_ldmatrix,
        ctx.has_cp_async,
        ctx.has_tma,
        ctx.has_bf16_mma,
        ctx.has_fp8_mma,
    ) == caps


def test_volta_atom_separates_logical_and_instruction_shapes() -> None:
    atom = ATOM_REGISTRY[VOLTA]
    assert atom.shape == (16, 16, 4)
    assert atom.ptx_shape == (8, 8, 4)
    assert tuple(atom.fragment_nregs(role) for role in ("a", "b", "c")) == (2, 2, 8)
    assert atom.fragment_layout == "m8n8k4"
    assert atom.sync_copy_staging and not atom.c_to_a_repack

    # The PTX C-fragment map covers the logical 16x16 output exactly once.
    coords = []
    for lane in range(32):
        comp = (lane & 15) >> 2
        row0 = (comp >> 1) * 8 + (lane >> 4) * 4 + (lane & 1)
        col0 = (comp & 1) * 8 + (lane & 2)
        coords.extend((row0 + (elem & 2), col0 + (elem & 4) + (elem & 1)) for elem in range(8))
    assert len(set(coords)) == 16 * 16
    assert set(coords) == {(row, col) for row in range(16) for col in range(16)}


def test_atom_selection_is_target_specific() -> None:
    assert atoms_for(F16, ctx=Context(compute_capability=(7, 0))) == (VOLTA,)
    assert atoms_for(F16, ctx=Context(compute_capability=(7, 5))) == (VOLTA,)
    assert atoms_for(F16, ctx=Context(compute_capability=(8, 0))) == (AMPERE,)
    assert atoms_for(F16, ctx=Context(compute_capability=(12, 0))) == (AMPERE,)
    assert atoms_for(F16) == (AMPERE,)  # registry inspection retains the established modern view


def test_volta_warp_tiles_respect_accumulator_register_budget() -> None:
    volta = warp_tile_moves((VOLTA,))
    ampere = warp_tile_moves((AMPERE,))
    assert volta and ampere
    assert max(p.reg_m * p.reg_n for p in volta) == 16
    assert max(p.reg_m * p.reg_n for p in ampere) == 32
    assert all(p.reg_m * p.reg_n * p.atom.accumulator_registers_per_lane <= MAX_FRAGMENT_REGISTERS for p in volta + ampere)


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_source_uses_only_the_volta_mma_family(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA)
    src, knobs = _source(_graph(trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == ""
    assert "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32" in src
    assert "unsigned _a0[2]" in src and "unsigned _b0[2]" in src
    assert "float _c0_0[8]" in src
    assert ("emmy_mma884_load_b_gmem_trans(_b0" in src) == trans
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16", ".bf16", ".e4m3", ".e5m2"):
        assert forbidden not in src


def test_sm70_m1_linear_synthesizes_a_masked_mma_row(monkeypatch) -> None:
    """A literal unit output row is still the M side of a plain m1xKxN contraction.

    RESTORED: the decode row is the only shape a V100 deployment runs at serving time, and without
    it the m=1 linear falls to the scalar path with no assertion anywhere noticing.
    """
    _pin(monkeypatch, VOLTA, tile="f1x1", stage="")
    src, knobs = _source(_graph(m=1, n=16, k=16, trans=True), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert knobs["WORK"] == "w1x1"
    assert "emmy_mma_m8n8k4_f16_f32" in src
    assert "_um_b" in src and "< (1)" in src


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_sync_copy_stages_fragments_without_newer_instructions(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA, stage="d1/smem")
    src, knobs = _source(_graph(k=16, trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "STAGE") == "d1/smem"
    # Every staged slab now declares its fill-chunk alignment, so match the declaration from the
    # dtype on — the attribute between ``__shared__`` and the type is not what this test is about.
    assert "__half _a_smem[64]" in src
    assert "__half _b_smem[64]" in src
    assert "emmy_mma884_load_a_smem(_a0, &_a_smem" in src
    b_helper = "emmy_mma884_load_b_smem_trans" if trans else "emmy_mma884_load_b_smem"
    assert f"{b_helper}(_b0, &_b_smem" in src
    assert src.count("__syncthreads();") == 2
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16"):
        assert forbidden not in src


def test_sm70_sync_copy_composes_ring_and_register_pipelines(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f1x1/k2", stage="d2/smem/p2")
    src, knobs = _source(_graph(k=32), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1/k2"
    assert family_value(knobs, "STAGE") == "d2/smem/p2"
    assert "__half _a_smem[256]" in src
    assert "__half _b_smem[256]" in src
    for fragment in ("_a0_s0", "_a0_s1", "_b0_s0", "_b0_s1"):
        assert fragment in src
    assert src.count("emmy_mma_m8n8k4_f16_f32(_c0_0") == 2
    assert "cp.async" not in src and "ldmatrix" not in src


def test_sm70_register_tile_keeps_the_volta_fragment_layout_through_the_reroll(monkeypatch) -> None:
    """A register tile wide enough to ROLL back into a loop still drains and stores as Volta.

    The rolled form rebuilds every stmt through ``Stmt.rewrite``; dropping ``fragment_layout``
    there re-defaulted the drain to the modern ``ldmatrix`` and the store to the m16n8k16 C map —
    an uncompilable, wrong-layout sm_70 kernel that the ``f1x1`` cases above cannot see."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    monkeypatch.setenv("EMMY_LOOPIFY", "2")
    src, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    assert "unsigned _a[2][2]" in src  # the ROLLED fragment family (count > 1)
    assert "emmy_mma884_load_a_smem(_a[" in src
    assert "emmy_mma884_load_b_smem(_b[" in src
    assert "const int _vr = " in src and "const int _vc = " in src  # the m8n8k4 C-fragment store map
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


@pytest.mark.parametrize("stage", ["d1/smem", "d2/smem"])
def test_sm70_computed_a_edge_stages_through_the_smem_compute_fill(monkeypatch, stage) -> None:
    """A COMPUTED ``a`` edge reaches the Volta mma tier: the fill evaluates the norm cone into the
    A slab the Volta shared gather reads, and the materialized B peer rides the BLOCKING vector
    copy — sm_70 has no ``cp.async`` to fly it under the fill."""
    _pin(monkeypatch, VOLTA, tile="f1x1", stage=stage)
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    src, knobs = _source(_norm_linear_graph(), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == stage
    assert "emmy_mma884_load_a_smem(_a0, &_a_smem" in src
    assert "emmy_mma884_load_b_smem_trans(_b0, &_b_smem" in src
    assert "rsqrtf" in src  # the norm cone itself, evaluated into the A slab
    assert "_b_copy0" in src  # the peer's blocking vector load/store
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


def test_modern_computed_a_edge_keeps_the_cp_async_peer_copy(monkeypatch) -> None:
    """The same fused edge on a cp.async target still flies its peer copy asynchronously — the
    blocking copy is the sm_70 fallback, not a new default."""
    _pin(monkeypatch, AMPERE, tile="f1x1", stage="d1/smem")
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    src, _ = _source(_norm_linear_graph(k=32), Context(compute_capability=(8, 0)))
    assert "emmy_cp_async" in src
    assert "ldmatrix" in src and "rsqrtf" in src
    assert "emmy_mma884" not in src


def test_modern_mma_source_does_not_gain_the_volta_prelude(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    src, _ = _source(_graph(k=16), Context(compute_capability=(8, 0)))
    assert "mma.sync.aligned.m16n8k16" in src
    assert "emmy_mma884" not in src and "mma.sync.aligned.m8n8k4" not in src


def test_sm70_modern_atom_pin_restricts_the_schedule_to_empty(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    out = Pipeline.build(TILE_PASSES).run(_graph(k=16), ctx=Context(compute_capability=(7, 0)))
    tile = next(node.op for node in out.nodes.values() if isinstance(node.op, TileOp))

    assert not tile.place.is_mapped and tile.schedule is None


@pytest.mark.parametrize("stage", ["d1/smem-async", "d1/smem-tma"])
def test_sm70_newer_stage_pin_refuses(monkeypatch, stage) -> None:
    _pin(monkeypatch, VOLTA, stage=stage)
    with pytest.raises(ValueError, match="requires sm_"):
        Pipeline.build(TILE_PASSES).run(_graph(), ctx=Context(compute_capability=(7, 0)))


def test_requested_target_reaches_nvcc_arch_and_cubin_key(monkeypatch) -> None:
    monkeypatch.setattr(nvcc, "_toolkit_tag", lambda: "toolkit")
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    try:
        set_target((7, 0))
        arch70 = nvcc.device_arch(False)
        key70 = nvcc._cubin_key("source", "kernel", arch70)
        set_target((8, 0))
        arch80 = nvcc.device_arch(False)
        key80 = nvcc._cubin_key("source", "kernel", arch80)
    finally:
        set_target(None)
    assert (arch70, arch80) == ("sm_70", "sm_80")
    assert key70 != key80
