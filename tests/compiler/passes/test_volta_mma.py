"""Target selection and source lowering for the Volta m8n8k4 MMA family."""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.search.space import MAX_FRAGMENT_REGISTERS, warp_tile_moves
from emmy.compiler.target import set_target

VOLTA = "mma_m8n8k4_f16_f32"
AMPERE = "mma_m16n8k16_f16_f32"


def _graph(*, m: int = 16, n: int = 16, k: int = 4, trans: bool = False) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), dtype=F16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (n, k) if trans else (k, n), dtype=F16), node_id="b")
    graph.add_node(LinearOp() if trans else MatmulOp(), ["a", "b"], Tensor("c", (m, n), dtype=F16), node_id="c")
    graph.inputs, graph.outputs = ["a", "b"], ["c"]
    return graph


def _pin(monkeypatch, atom: str, *, tile: str = "f1x1", stage: str = "") -> None:
    monkeypatch.setenv("EMMY_TILE", f"{atom}/{tile}")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_REDUCE", "")


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
    assert atom.materialized_edges_only and atom.sync_copy_staging and not atom.c_to_a_repack

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
    assert knobs["TILE"] == f"{VOLTA}/f1x1"
    assert knobs["STAGE"] == ""
    assert "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32" in src
    assert "unsigned _a0[2]" in src and "unsigned _b0[2]" in src
    assert "float _c0_0[8]" in src
    assert ("emmy_mma884_load_b_gmem_trans(_b0" in src) == trans
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16", ".bf16", ".e4m3", ".e5m2"):
        assert forbidden not in src


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_sync_copy_stages_fragments_without_newer_instructions(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA, stage="d1/sync")
    src, knobs = _source(_graph(k=16, trans=trans), Context(compute_capability=(7, 0)))
    assert knobs["STAGE"] == "d1/sync"
    assert "__shared__ __half _a_smem[64]" in src
    assert "__shared__ __half _b_smem[64]" in src
    assert "emmy_mma884_load_a_smem(_a0, &_a_smem" in src
    b_helper = "emmy_mma884_load_b_smem_trans" if trans else "emmy_mma884_load_b_smem"
    assert f"{b_helper}(_b0, &_b_smem" in src
    assert src.count("__syncthreads();") == 2
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16"):
        assert forbidden not in src


def test_sm70_sync_copy_composes_ring_and_register_pipelines(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f1x1/k2", stage="d2/sync/p2")
    src, knobs = _source(_graph(k=32), Context(compute_capability=(7, 0)))
    assert knobs["TILE"] == f"{VOLTA}/f1x1/k2"
    assert knobs["STAGE"] == "d2/sync/p2"
    assert "__shared__ __half _a_smem[256]" in src
    assert "__shared__ __half _b_smem[256]" in src
    for fragment in ("_a0_s0", "_a0_s1", "_b0_s0", "_b0_s1"):
        assert fragment in src
    assert src.count("emmy_mma_m8n8k4_f16_f32(_c0_0") == 2
    assert "cp.async" not in src and "ldmatrix" not in src


def test_modern_mma_source_does_not_gain_the_volta_prelude(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    src, _ = _source(_graph(k=16), Context(compute_capability=(8, 0)))
    assert "mma.sync.aligned.m16n8k16" in src
    assert "emmy_mma884" not in src and "mma.sync.aligned.m8n8k4" not in src


def test_sm70_rejects_a_pinned_modern_atom(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    with pytest.raises(ValueError, match="has_mma_m16n8k16.*unavailable on sm_70"):
        Pipeline.build(TILE_PASSES).run(_graph(k=16), ctx=Context(compute_capability=(7, 0)))


@pytest.mark.parametrize(("stage", "message"), [("d1/cp", "cp.async requires sm_80"), ("d1/tma", "TMA requires sm_90")])
def test_sm70_rejects_a_pinned_newer_stage(monkeypatch, stage, message) -> None:
    _pin(monkeypatch, VOLTA, stage=stage)
    with pytest.raises(ValueError, match=message):
        Pipeline.build(TILE_PASSES).run(_graph(), ctx=Context(compute_capability=(7, 0)))


def test_requested_target_reaches_nvcc_arch_and_cache_key(monkeypatch) -> None:
    monkeypatch.setattr(nvcc, "_toolkit_tag", lambda: "toolkit")
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    try:
        set_target((7, 0))
        arch70 = nvcc.device_arch(False)
        key70 = nvcc._cache_key("source", "kernel", arch70)
        set_target((8, 0))
        arch80 = nvcc.device_arch(False)
        key80 = nvcc._cache_key("source", "kernel", arch80)
    finally:
        set_target(None)
    assert (arch70, arch80) == ("sm_70", "sm_80")
    assert key70 != key80
