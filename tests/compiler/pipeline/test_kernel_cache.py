"""The session kernel cache — fetch a finished lowering instead of re-lowering.

What these pin: (a) REPLAY EQUALITY — a twin kernel compiled through a cache hit renders the
byte-identical CUDA source a fresh compile produces (modulo nothing: the io rebind restores the
twin's own buffer names before rendering); (b) the cache is OPT-IN (no ``Context.kernel_cache``,
no behavior change) and greedy-only (the tune search strips it); (c) a multi-kernel origin
poisons its key rather than serving a fragment."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.kernel_cache import POISON, KernelCache
from emmy.compiler.pipeline.search.db import SearchDB
from tests.compiler.helpers import pin_classic


def _matmul(x: str, w: str, o: str) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor(x, (64, 64), "f16"), node_id=x)
    g.add_node(InputOp(), [], Tensor(w, (64, 64), "f16"), node_id=w)
    g.add_node(MatmulOp(), [x, w], Tensor(o, (64, 64), "f16"), node_id=o)
    g.inputs, g.outputs = [x, w], [o]
    return g


def _cuda_ops(graph: Graph) -> list[CudaOp]:
    return [n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp)]


def _compile(graph: Graph, ctx: Context) -> list[CudaOp]:
    return _cuda_ops(Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=SearchDB()))


def _pin_direct_matmul(monkeypatch) -> None:
    """Keep cache tests on one complete schedule; their subject is replay, not enumeration."""
    pin_classic(monkeypatch, {"WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""})


def test_twin_replay_renders_byte_identical_source(monkeypatch) -> None:
    _pin_direct_matmul(monkeypatch)
    cache = KernelCache()
    ctx = replace(Context.from_target((12, 0)), kernel_cache=cache)

    (first,) = _compile(_matmul("x0", "w0", "o0"), ctx)
    assert cache.hits == 0 and len(cache._store) == 1

    (replayed,) = _compile(_matmul("x1", "w1", "o1"), ctx)
    assert cache.hits >= 1, "the twin must fetch the finished lowering"

    (fresh,) = _compile(_matmul("x1", "w1", "o1"), replace(Context.from_target((12, 0)), kernel_cache=None))
    assert replayed.kernel_source == fresh.kernel_source, "replay must render the byte-identical kernel"
    assert replayed.arg_order == fresh.arg_order
    assert (replayed.grid, replayed.block, replayed.smem_bytes) == (fresh.grid, fresh.block, fresh.smem_bytes)
    assert first.kernel_source != fresh.kernel_source, "the premise: the twins spell different buffer names"


def test_without_a_cache_nothing_changes(monkeypatch) -> None:
    _pin_direct_matmul(monkeypatch)
    ctx = Context.from_target((12, 0))
    assert ctx.kernel_cache is None
    (op,) = _compile(_matmul("x0", "w0", "o0"), ctx)
    assert op.kernel_source


def test_tune_search_strips_the_cache() -> None:
    from emmy.compiler.pipeline.search.policy.mcts import TuningSearch

    ctx = replace(Context.from_target((12, 0)), kernel_cache=KernelCache())
    assert TuningSearch.prepare_ctx(TuningSearch.__new__(TuningSearch), ctx).kernel_cache is None


def test_multi_kernel_origin_poisons_the_key() -> None:
    import types

    from emmy.compiler.ir.kernel import KernelOp

    cache = KernelCache()
    a, b = KernelOp(), KernelOp()
    origin = types.SimpleNamespace(inputs={}, outputs={})
    cache.harvest("k", a, origin)
    assert cache._store["k"] is not POISON
    cache.harvest("k", b, origin)
    assert cache._store["k"] is POISON, "a second kernel from one origin means a multi-kernel lowering"
