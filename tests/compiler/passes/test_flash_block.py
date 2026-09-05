"""FlashAttention-2 out of the blocked carrier — the emission the block exists to reach.

Blocking separates a twisted carrier's two monoids so its expectation channel reads as a
contraction. What that has to buy is one kernel per attention: the score contraction into C
fragments, a fragment row reduce for the block pivot, the weight applied to those fragments, and
the ``P·V`` mma against the streamed slab — the score computed ONCE per block and read twice.

The corpus replays pinned schedules as data; the emitted SOURCE is what only Python can ask, so
the mma count, the shared score and the block loop are asserted here.
"""

from __future__ import annotations

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline

_SDPA = """
import torch.nn.functional as F
q = torch.randn(1, 4, 128, 32, dtype=torch.float16)
F.scaled_dot_product_attention(q, q.clone(), q.clone())
"""

#: The row the cold greedy deploys on an RTX 5090 — pinned so the assertions below read one
#: schedule rather than whatever evidence happens to be on the machine. The two tiles inside the
#: block are the only ones that fit it: the channel's K-step consumes the 64-wide block in one
#: trip (``k4`` of a k16 atom) and the score's eight register columns cover it.
_FLASH_ROW = {
    "EMMY_KNOBS": ",".join(
        (
            "TILE@map.1/twist.2/inner=mma_m16n8k16_f16_f32/f2x4/k4",
            "STAGE@map.1/twist.2/inner=d2/smem",
            "TILE@map.1/twist.2/inner.1/map.1/inner=mma_m16n8k16_f16_f32/f2x8/k4",
            "TILE@map.1/twist.1/reduce.1/inner=mma_m16n8k16_f16_f32/f2x8",
            "WORK=w4x1",
        )
    )
}


@pytest.fixture(scope="module")
def flash_source() -> str:
    """The one compile every assertion below reads — the row costs a real nvcc-free lowering, and
    five of them is four too many."""
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    with pinned_knobs(dict(pair.split("=", 1) for pair in _FLASH_ROW["EMMY_KNOBS"].split(","))):
        return _lowered()


def _lowered() -> str:
    graph, _, _ = graph_from_code(_SDPA)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    kernels = [node.op for node in lowered.nodes.values() if getattr(node.op, "kernel_source", None)]
    assert len(kernels) == 1, [node.op for node in lowered.nodes.values()]
    return kernels[0].kernel_source


def test_both_halves_of_attention_reach_the_tensor_cores(flash_source: str) -> None:
    """``Q·K`` and ``P·V`` are both mma. Neither is on main's term: the score is the only
    contraction a twisted carrier holds, and the value channel is a coefficient of its ⊕."""
    assert flash_source.count("mma.sync") >= 2


def test_the_block_is_one_loop_over_the_stream(flash_source: str) -> None:
    """One K loop over the 128-key stream in blocks, not a loop per channel: the outer axis strides
    and the two inner passes are the block's own program."""
    assert flash_source.count("for (int _ks = 0; _ks < 128;") == 1


def test_the_score_is_computed_once_per_block(flash_source: str) -> None:
    """The pivot's pass and the weight's read ONE score. Two ``Q·K`` mma loops in the block would
    be the two-pass form the twisted merge already gives without any of this."""
    assert flash_source.count("_fold_c0_") > 0
    assert flash_source.count("emmy_mma_load_a_gmem(_fold_a0") == 1


def test_the_pivot_is_a_fragment_row_reduce(flash_source: str) -> None:
    """The block's ⊕ over the score fragments is a warp shuffle tree, not a serial loop — which is
    what makes the pivot's pass free once the score is already in registers."""
    assert "__shfl_xor_sync" in flash_source
    assert "fmaxf" in flash_source


def test_the_weight_tile_is_stored_for_the_value_mma(flash_source: str) -> None:
    """``P`` reaches the second mma through the A slab: the channel's A is a COMPUTED cone, so the
    compute fill writes the fragments the drain reads back."""
    assert "_a_smem" in flash_source
