"""Unit tests for the per-kernel cost estimate and its training corpus.

CPU-only throughout: every case either builds a tensor by hand or replays a recorded golden
through the loop passes, and neither needs a card.
"""

from __future__ import annotations

import math

import pytest

from emmy import gpu
from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.dtype import F32, F16x2
from emmy.compiler.pipeline.search import kernel_cost
from emmy.compiler.pipeline.search.data.shape import stamped_flops
from emmy.compiler.tensor import Tensor

_5090 = gpu.by_name("NVIDIA GeForce RTX 5090")


class _Op:
    """The surface :mod:`kernel_cost` reads: stamped knobs plus resolved io."""

    def __init__(self, knobs, inputs, outputs):
        self.knobs, self.inputs, self.outputs = knobs, inputs, outputs


def _matmul_stamps(m: int, n: int, k: int, *, symbolic_m: bool = False) -> dict[str, float]:
    """The histogram ``loop/stamp`` writes for an ``(m,k)@(k,n)`` contraction.

    A symbolic axis is EXCLUDED from the extent products and counted separately — mirroring
    ``ShapeKey.from_matmul``, which documents why: the stamped histogram is the op side's only
    identity and it does not know the hint."""
    return {
        "S_ext_free_prod": float(n if symbolic_m else m * n),
        "S_ext_free_max": float(0 if symbolic_m else max(m, n)),
        "S_ext_reduce_prod": float(k),
        "S_ext_reduce_max": float(k),
        "S_ext_n_free_axis": 1.0 if symbolic_m else 2.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_n_symbolic_axis": 1.0 if symbolic_m else 0.0,
        "S_loop_depth": 3.0,
        "S_dtype_f32": 2.0,
        "S_reduce_add": 1.0,
        "S_pw_multiply": 1.0,
        "S_n_distinct_input": 2.0,
    }


def _t(name: str, *shape, dtype=F32) -> Tensor:
    return Tensor(name, tuple(Dim(d) for d in shape), dtype)


# --- work and traffic -------------------------------------------------------------------------


def test_flops_and_bytes_of_a_known_matmul():
    """A 512-cube fp32 contraction: 2*M*N*K of work, and three square buffers of traffic."""
    op = _Op(_matmul_stamps(512, 512, 512), {"a": _t("a", 512, 512), "b": _t("b", 512, 512)}, {"c": _t("c", 512, 512)})
    assert stamped_flops(op.knobs) == 2 * 512**3
    assert kernel_cost.kernel_bytes(op) == 3 * 512 * 512 * 4


def test_cutting_a_kernel_shows_up_as_the_workspace_it_materializes():
    """The property the fuse-or-cut decision is priced on, and the reason traffic is in the floor
    at all.

    A cut materializes its intermediate as a workspace buffer — ``passes/lowering/tile/_cut.py``
    opens by saying so — which the producing piece writes and the consuming piece reads. The
    fused form keeps that value in registers and never pays for it. So the cut pair must cost
    exactly two crossings of the workspace more than the fused kernel; if it did not, the
    estimate would price both arms identically and have no mechanism for its own decision."""
    a, b, c = _t("a", 256, 256), _t("b", 256, 256), _t("c", 256, 256)
    w = _t("w", 256, 256)  # the intermediate a cut has to materialize
    fused = _Op(_matmul_stamps(256, 256, 256), {"a": a, "b": b}, {"c": c})
    produce = _Op(_matmul_stamps(256, 256, 256), {"a": a, "b": b}, {"w": w})
    consume = _Op(_matmul_stamps(256, 256, 256), {"w": w}, {"c": c})
    cut_total = kernel_cost.kernel_bytes(produce) + kernel_cost.kernel_bytes(consume)
    assert cut_total - kernel_cost.kernel_bytes(fused) == 2 * 256 * 256 * 4


def test_bytes_does_not_double_count_a_packed_pair():
    """``Tensor.shape`` is the STORED shape, so a packed pair's halved last axis already accounts
    for its two logical elements. Multiplying by ``logical_elems`` as well would double it."""
    packed = _t("w", 64, 32, dtype=F16x2)
    assert packed.dtype.logical_elems == 2  # the trap this pins
    op = _Op(_matmul_stamps(64, 64, 64), {"w": packed}, {"o": _t("o", 1)})
    assert kernel_cost.kernel_bytes(op) == 64 * 32 * packed.dtype.nbytes + 1 * 4


def test_a_symbolic_axis_is_sized_at_its_hint_on_both_terms():
    """THE dynamic-golden defect, pinned on both halves of the floor.

    A ``.dynM`` kernel stamps only its static axes, so work read straight off the stamps
    under-counts by the hint — 512-fold — and its traffic would too if the tensor's symbolic dim
    were skipped. Both must be sized at the hint the bench used, or a fifth of the corpus prices
    against a shape that never ran."""
    m = DEFAULT_SEQ_HINT
    static = _Op(_matmul_stamps(m, 64, 128), {"a": _t("a", m, 128), "b": _t("b", 128, 64)}, {"c": _t("c", m, 64)})
    dyn = _Op(
        _matmul_stamps(m, 64, 128, symbolic_m=True),
        {"a": Tensor("a", (Dim("num_tokens"), Dim(128)), F32), "b": _t("b", 128, 64)},
        {"c": Tensor("c", (Dim("num_tokens"), Dim(64)), F32)},
    )
    assert stamped_flops(dyn.knobs) == stamped_flops(static.knobs) == 2 * m * 64 * 128
    assert kernel_cost.kernel_bytes(dyn) == kernel_cost.kernel_bytes(static)
    assert kernel_cost.t_roofline_us(dyn, _5090) == kernel_cost.t_roofline_us(static, _5090)


# --- the floor --------------------------------------------------------------------------------


def test_floor_is_the_largest_of_the_three_terms():
    op = _Op(_matmul_stamps(512, 512, 512), {"a": _t("a", 512, 512), "b": _t("b", 512, 512)}, {"c": _t("c", 512, 512)})
    compute = 2 * 512**3 / (_5090.peak_fp32_tflops * 1e12) * 1e6
    traffic = 3 * 512 * 512 * 4 / (_5090.mem_bw_gbps * 1e9) * 1e6
    assert kernel_cost.t_roofline_us(op, _5090) == pytest.approx(max(compute, traffic, kernel_cost.MIN_SCALE_US))


def test_a_tiny_kernel_falls_back_to_the_minimum_scale():
    """Both physical terms go to nanoseconds on a kernel this small, and dividing a real latency
    by that yields a ratio in the tens of thousands. The floor keeps the scale finite — and sits
    below the fastest latency the corpus records, so it never puts a real row under its own
    denominator."""
    op = _Op(_matmul_stamps(4, 4, 4), {"a": _t("a", 4, 4), "b": _t("b", 4, 4)}, {"c": _t("c", 4, 4)})
    assert kernel_cost.t_roofline_us(op, _5090) == kernel_cost.MIN_SCALE_US


def test_a_term_with_no_recorded_input_drops_out_rather_than_reading_as_zero():
    """A card with no bandwidth contributes no traffic term; a kernel whose stamps do not certify
    a work formula contributes no arithmetic term. Neither may floor the estimate at zero."""
    bare = gpu.GpuSpec(name="Synthetic no-peaks", compute_capability=(9, 0), sm_count=132, smem_per_sm=233472)
    op = _Op(_matmul_stamps(4096, 4096, 4096), {"a": _t("a", 4096, 4096)}, {"c": _t("c", 4096, 4096)})
    assert kernel_cost.t_roofline_us(op, bare) == kernel_cost.MIN_SCALE_US
    norm = _Op({**_matmul_stamps(1024, 1024, 16), "S_loop_depth": 2.0}, {"a": _t("a", 1024, 1024)}, {"o": _t("o", 1024, 1024)})
    assert stamped_flops(norm.knobs) is None  # the certificate fails, so no arithmetic term
    assert kernel_cost.t_roofline_us(norm, _5090) > kernel_cost.MIN_SCALE_US  # traffic still binds


# --- the feature row --------------------------------------------------------------------------


def test_the_row_reads_the_kernel_and_never_the_fork_that_made_it():
    """The invariance the whole design rests on: a kernel minted by a placement cut and the same
    kernel standing alone must price identically, or a model fitted on these features learns
    provenance instead of physics.

    Enforced by construction — only ``S_*`` stamps are read, and those are written at birth in
    recognition, before the first schedule fork is offered. So schedule knobs and a ``PLACE``
    decision are both invisible here, which is what this asserts."""
    io = ({"a": _t("a", 256, 256), "b": _t("b", 256, 256)}, {"c": _t("c", 256, 256)})
    plain = _Op(_matmul_stamps(256, 256, 256), *io)
    from_a_cut = _Op(
        {**_matmul_stamps(256, 256, 256), "PLACE@a": "cut", "TILE": "f4x8", "WORK": "t32x8", "STAGE": "d3/smem-tma"},
        *io,
    )
    assert kernel_cost.kernel_row(plain, _5090, precision_trading=False) == kernel_cost.kernel_row(
        from_a_cut, _5090, precision_trading=False
    )


def test_the_row_carries_no_knob_features_and_no_raw_peaks():
    """Knob columns exist to tell candidates apart INSIDE a pool; this estimate never looks inside
    one. The raw peaks are excluded for a different reason: ``t_roofline`` already contains them,
    so a column would hand a tree a route to re-derive the per-card constant and undo the
    normalization the floor exists to provide."""
    op = _Op(_matmul_stamps(256, 256, 256), {"a": _t("a", 256, 256)}, {"c": _t("c", 256, 256)})
    row = kernel_cost.kernel_row(op, _5090, precision_trading=False)
    assert not [k for k in row if k.split("@")[0] in ("TILE", "WORK", "STAGE", "REDUCE", "RASTER", "PLACE")]
    assert not [k for k in row if k.startswith(("D_", "MMA_"))]
    assert not [k for k in row if "peak" in k or "tflops" in k]
    assert "S_n_mma" not in row  # structurally zero on every stamped row
    assert "H_opt" not in row  # the corpus is one compile regime throughout


def test_precision_trading_is_the_only_thing_separating_a_fast_math_twin():
    """A fast-math kernel and its standard sibling carry identical stamps and share a card, so
    without this column they would be feature-identical rows with different labels."""
    op = _Op(_matmul_stamps(256, 256, 256), {"a": _t("a", 256, 256)}, {"c": _t("c", 256, 256)})
    std = kernel_cost.kernel_row(op, _5090, precision_trading=False)
    fm = kernel_cost.kernel_row(op, _5090, precision_trading=True)
    assert std != fm
    assert {k for k in std if std[k] != fm[k]} == {"R_precision_trading"}


def test_unknowable_values_are_nan_not_zero():
    """``nan`` means "not knowable here", which a tree splits on separately from a real zero —
    the convention the online prior's featurizer states."""
    norm = _Op({**_matmul_stamps(1024, 1024, 16), "S_loop_depth": 2.0}, {"a": _t("a", 1024, 1024)}, {"o": _t("o", 1024, 1024)})
    row = kernel_cost.kernel_row(norm, _5090, precision_trading=False)
    assert math.isnan(row["R_log_flops"]) and math.isnan(row["R_intensity"])
    assert not math.isnan(row["R_log_bytes"])  # traffic is always knowable


# --- the corpus -------------------------------------------------------------------------------
#
# Driven by passing records to ``build_rows`` rather than by swapping the module global: the
# parameter is the seam, so these read real recorded kernels without hiding the corpus from
# anything else running concurrently.


def _corpus():
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS

    return GOLDEN_RECORDS


def _same_pool_candidates():
    """Records that plausibly share a pool, found WITHOUT asking for one.

    ``GoldenRecord.pool_group`` lowers the record to compose its kernel identity, so asking it of
    all 1285 costs ~8 s. Name and card are free, and records sharing a pool necessarily share
    both — so narrowing on those first turns a corpus-wide lowering into a handful."""
    import collections

    by = collections.defaultdict(list)
    for r in _corpus():
        by[(r.gpu_name, r.name)].append(r)
    return max(by.values(), key=len)


def test_several_goldens_over_one_pool_become_one_row_at_the_best_label():
    """A pool is the unit, not a golden. Competing recordings of one kernel are competing
    measurements of one best, so the label is their minimum — which is also why a duplicate
    recording cannot drag a row: "best anyone achieved" only ever improves."""
    from emmy.compiler.pipeline.search.data.cost_dataset import _records_by_pool, build_rows

    members = max(_records_by_pool(_same_pool_candidates()).values(), key=len)
    assert len(members) > 1, "corpus no longer has a multi-record pool to exercise"
    rows, _ = build_rows(members)
    assert len(rows) == 1
    assert rows[0].best_us == min(m.emmy_us for m in members)
    assert rows[0].members == len(members)


def test_a_golden_recording_no_schedule_is_dropped_and_reported():
    """The fabricated-label class: with no schedule family recorded, such a golden silently
    matches option 0 of whatever pool it opens, so a fit would train on that as the verified
    optimum. Dropped — but named in the skip list, never lost quietly."""
    from emmy.compiler.pipeline.search.data.cost_dataset import _records_no_schedule, build_rows

    # Per record, not per pool: the predicate reads knobs only, so this needs no lowering.
    bad = [r for r in _corpus() if _records_no_schedule([r])]
    assert bad, "corpus no longer has a schedule-less golden to exercise"
    rows, skipped = build_rows(bad)
    assert rows == []
    assert {name for _, name, _ in skipped} == {m.name for m in bad}
    assert all(reason == "no schedule family recorded" for _, _, reason in skipped)


def test_a_fast_math_twin_stays_a_separate_row():
    """The pin is part of the pool key, so a fast-math kernel and its standard sibling are two
    rows — as they must be, since they are separately measured and reach different latencies.
    ``R_precision_trading`` is what keeps them from being feature-identical rows with conflicting
    labels."""
    from emmy.compiler.pipeline.search.data.cost_dataset import build_rows

    pairs = {r.name: r for r in _corpus()}
    fm_name = next(n for n in pairs if n.endswith(".fm") and n[:-3] in pairs)
    rows, _ = build_rows([pairs[fm_name], pairs[fm_name[:-3]]])
    assert len(rows) == 2
    assert {row.features["R_precision_trading"] for row in rows} == {0.0, 1.0}


def test_every_row_carries_a_fold_key_that_ignores_the_card():
    """Holding out by anything card-specific lets a model see a kernel on one card and be scored
    on it on another — and 86% of this corpus sits on two cards."""
    from emmy.compiler.pipeline.search.data.cost_dataset import build_rows

    by_name = {r.name: r for r in _corpus()}
    same_kernel = [r for r in _corpus() if r.name == "matmul.square.512"]
    assert len({r.gpu_name for r in same_kernel}) > 1, "corpus no longer records this kernel on two cards"
    rows, _ = build_rows(same_kernel)
    assert len({row.gpu for row in rows}) > 1  # distinct cards ...
    assert len({row.fold for row in rows}) == 1  # ... one fold
    assert by_name  # keeps the fixture honest if the corpus is re-recorded


def test_lowering_a_kernel_keeps_the_stamps_the_row_was_built_from():
    """The corpus reads a kernel after the loop passes; an ordinary compile prices it after
    ``lowering/tile``. Those must be the same stamps, or a model fitted here would be queried on
    a different reading of the same kernel.

    They are, and by construction: lifting a ``LoopOp`` to a ``TileOp`` carries its knobs across,
    and ``IdentityStrategy._stamp`` skips an op that already has stamps. The one addition is
    ``S_warp_eligible``, which ``lowering/tile/040_schedule`` contributes because only the
    scheduler knows whether a warp tile is on offer.

    (A cut fork's pieces do differ — but they are new kernels with their own bodies and their own
    shapes, which is what a cut IS, not the same kernel read two ways.)"""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.fork import iter_leaves
    from emmy.compiler.pipeline.knob import STRUCT_PREFIX
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, _target_kernel_nodes

    record = next(r for r in GOLDEN_RECORDS if r.name.startswith("layer0.i006.k_linear_mean_reduce"))
    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    lowered, nodes = _target_kernel_nodes(record)
    before = kernel_cost.kernel_stamps(nodes[0].op)

    out = Run(Pipeline.build(TILE_PASSES), ctx).resolve(record.target_program.copy(), lambda fp: next(iter_leaves(fp.options)))
    terminal = out[0] if isinstance(out, tuple) else out
    stamped = [
        kernel_cost.kernel_stamps(n.op) for n in terminal.nodes.values() if any(k.startswith(STRUCT_PREFIX) for k in (n.op.knobs or {}))
    ]

    assert len(stamped) == 1, "the keep-fused arm is one kernel"
    after = stamped[0]
    assert set(after) - set(before) == {"S_warp_eligible"}
    assert {k: v for k, v in after.items() if k != "S_warp_eligible"} == before


def test_the_invariance_holds_on_a_real_reconstructed_kernel():
    """The same property as the hand-built case, on a kernel the compiler actually produced.

    A real golden lowers to an op carrying only its ``S_*`` stamps — written in recognition,
    before any schedule fork — so this adds the marks a cut-minted kernel would carry on top and
    asserts the row does not move. The hand-built case pins the mechanism; this one pins that a
    real op's knob dict holds nothing else the row would pick up."""
    from dataclasses import replace

    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, _target_kernel_nodes

    record = next(r for r in GOLDEN_RECORDS if r.name == "matmul.square.512")
    lowered, nodes = _target_kernel_nodes(record)
    node = nodes[0]
    standalone = node.op.with_io(lowered, node)
    assert not [k for k in standalone.knobs if not k.startswith("S_")], "a freshly lowered op should carry stamps only"

    spec = gpu.by_name(record.gpu_name)
    as_cut_piece = replace(standalone, knobs={**standalone.knobs, "PLACE@a": "cut", "TILE": "f4x8", "WORK": "t16x8"})
    assert kernel_cost.kernel_row(as_cut_piece, spec, precision_trading=False) == kernel_cost.kernel_row(
        standalone, spec, precision_trading=False
    )


def test_every_symbolic_golden_is_traced_at_the_default_hint():
    """The assumption that keeps the floor's two halves commensurable.

    A symbolic axis is excluded from the ``S_*`` extent products — deliberately, so a kernel's
    identity does not depend on the size it was tuned at — and re-enters as a hint factor. But the
    work term reads the stamps, which carry no dim, so it applies ``DEFAULT_SEQ_HINT`` as a
    constant; the traffic term reads the real dims. Those give the same answer only while every
    symbolic axis carries the default.

    That holds for all 212 dynamic goldens today. If it stops holding — a corpus traced with a
    different ``--seq-len`` — the two halves would size the same kernel differently and this fails,
    which is the point: the fix would be to pass the hint into the work term, not to discover the
    disagreement from a skewed fit."""
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, _target_kernel_nodes

    checked = 0
    for record in GOLDEN_RECORDS:
        if not record.structural_features.get("S_ext_n_symbolic_axis"):
            continue
        lowered, nodes = _target_kernel_nodes(record)
        if len(nodes) != 1:
            continue
        op = nodes[0].op.with_io(lowered, nodes[0])
        for tensor in (*op.inputs.values(), *op.outputs.values()):
            for dim in tensor.shape:
                if not dim.is_static:
                    assert dim.hint == DEFAULT_SEQ_HINT, f"{record.name}: hint {dim.hint}"
                    checked += 1
        if checked > 40:  # a sample: the whole corpus lowers in ~10s, this is a unit test
            break
    assert checked, "corpus no longer has a symbolic-axis golden to exercise"
