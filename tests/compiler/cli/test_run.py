"""CLI tests for ``emmy run`` — accuracy check + ``--bench`` table.

Accuracy failures (``max_diff >= 1.0``) make ``emmy run`` exit
non-zero, so ``rc == 0`` is the accuracy assertion.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
import torch  # used by test_bind_inputs_preserves_int_dtype

from ..conftest import requires_cuda


def _randn(shape: str, dtype, scale: float | None = None) -> str:
    """Build a ``torch.randn(...)`` snippet for the given dtype.

    ``shape`` is a comma-joined dim list as it would appear inside the
    parens. fp16 inputs are scaled down (default 0.1) so reductions
    stay in fp16's representable range; fp32 inputs use the raw value.
    """
    if dtype.name == "f16":
        s = 0.1 if scale is None else scale
        return f"(torch.randn({shape}, dtype=torch.float16) * {s})"
    return f"torch.randn({shape})"


def test_run_no_code_errors(run_cli):
    rc, stdout, stderr = run_cli("run")
    assert rc != 0
    # argparse complains about the missing required ``--code`` flag.
    assert "code" in (stdout + stderr).lower() or "ir" in (stdout + stderr).lower()


def test_run_code_and_ir_mutually_exclusive(run_cli, tmp_path):
    fake_ir = tmp_path / "fake.json"
    fake_ir.write_text("{}")
    rc, stdout, stderr = run_cli("run", "--code", "torch.zeros(4)", "--ir", str(fake_ir))
    assert rc != 0
    assert "mutually exclusive" in (stdout + stderr).lower()


def test_run_input_and_code_mutually_exclusive(run_cli):
    rc, stdout, stderr = run_cli("run", "some/model", "--code", "torch.zeros(4)")
    assert rc != 0
    assert "mutually exclusive" in (stdout + stderr).lower()


def test_pinned_knobs_sets_and_restores_env(monkeypatch):
    """``_pinned_knobs`` pins ``EMMY_<KNOB>`` for the block, then restores the
    prior environment — removing keys that were unset, restoring preexisting ones
    (the golden-bench A/B relies on this to compile a pinned variant cleanly)."""
    import os

    from emmy.commands.run import _pinned_knobs

    monkeypatch.delenv("EMMY_TILE", raising=False)
    monkeypatch.setenv("EMMY_STAGE", "preexisting")
    with _pinned_knobs({"TILE": "n32x8/f2x4", "STAGE": "d2/cp", "WARP_SPECIALIZE": False}):
        assert os.environ["EMMY_TILE"] == "n32x8/f2x4"
        assert os.environ["EMMY_STAGE"] == "d2/cp"
        assert os.environ["EMMY_WARP_SPECIALIZE"] == "False"
    assert "EMMY_TILE" not in os.environ  # was unset → removed
    assert os.environ["EMMY_STAGE"] == "preexisting"  # restored
    assert "EMMY_WARP_SPECIALIZE" not in os.environ


def _symbolic_input_graph():
    """Frontend-ish graph with one symbolic input (``x``, seq axis 1) and one
    static input (``w``) — enough for ``_hint_sized_inputs``' input pairing."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, Dim("seq_len"), 8)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (4, 8)), node_id="w")
    g.inputs = ["x", "w"]
    g.outputs = ["x"]
    return g


def test_hint_sized_inputs_tiles_symbolic_axes():
    """A symbolic input axis grows to its Dim hint (DEFAULT_SEQ_HINT) by tiling
    the trace-time values; static inputs pass through untouched (the full-model table
    must bench torch at the same hint shape emmy benches at)."""
    from emmy.commands.run import _hint_sized_inputs
    from emmy.compiler.dim import DEFAULT_SEQ_HINT

    x = torch.randn(1, 32, 8)
    w = torch.randn(4, 8)
    args, kwargs, sym_env = _hint_sized_inputs(_symbolic_input_graph(), (x, w), {})
    assert sym_env == {"seq_len": DEFAULT_SEQ_HINT}
    assert kwargs == {}
    rx, rw = args
    assert rx.shape == (1, DEFAULT_SEQ_HINT, 8)
    assert rw is w  # static input untouched
    # Values are tiled repeats of the trace input, not fresh randoms.
    assert torch.equal(rx[:, :32], x)
    assert torch.equal(rx[:, 32:64], x)


def test_hint_sized_inputs_static_graph_is_noop():
    from emmy.commands.run import _hint_sized_inputs

    from ..conftest import matmul_graph

    a, b = torch.randn(4, 8), torch.randn(8, 4)
    args, kwargs, sym_env = _hint_sized_inputs(matmul_graph(4, 8, 4), (a, b), {})
    assert sym_env == {}
    assert args[0] is a and args[1] is b


def test_hint_sized_inputs_resizes_kwargs_in_nested_structure():
    """Tensors inside kwargs containers (HF's ``position_embeddings`` tuple
    style) are paired in ``_flatten_tensors`` order and rebuilt in place."""
    from emmy.commands.run import _hint_sized_inputs
    from emmy.compiler.dim import Dim
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, Dim("seq_len"), 8)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("cos", (Dim("seq_len"), 4)), node_id="cos")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sin", (Dim("seq_len"), 4)), node_id="sin")
    g.inputs = ["x", "cos", "sin"]
    g.outputs = ["x"]

    x, cos, sin = torch.randn(1, 32, 8), torch.randn(32, 4), torch.randn(32, 4)
    args, kwargs, sym_env = _hint_sized_inputs(g, (x,), {"position_embeddings": (cos, sin)})
    assert args[0].shape == (1, 512, 8)
    rcos, rsin = kwargs["position_embeddings"]
    assert isinstance(kwargs["position_embeddings"], tuple)
    assert rcos.shape == (512, 4) and rsin.shape == (512, 4)
    assert torch.equal(rsin[:32], sin)


def test_tile_to_non_divisible_and_dtype():
    from emmy.commands.run import _tile_to

    t = torch.tensor([[1, 2]], dtype=torch.long)
    out = _tile_to(t, 1, 5)
    assert out.dtype == torch.long
    assert out.tolist() == [[1, 2, 1, 2, 1]]
    assert _tile_to(t, 1, 2) is t  # already sized → identity


def test_symbolic_bench_note():
    from emmy.commands.run import _symbolic_bench_note

    assert _symbolic_bench_note({}) is None
    note = _symbolic_bench_note({"seq_len": 512})
    assert "seq_len=512" in note and "symbolic" in note


def test_build_torch_fns_resets_dynamo_before_compile(monkeypatch):
    """Persistent bench processes compile a fresh closure of the SAME torch_ref
    ``fn`` code object per row; dynamo's per-code-object recompile limit then
    silently degrades torch.compile to eager after ~8 rows, corrupting the
    tcompile column. ``_build_torch_fns`` must reset dynamo before each compile."""
    import torch._dynamo

    from emmy.commands.run import _build_torch_fns

    calls: list[bool] = []
    real_reset = torch._dynamo.reset
    monkeypatch.setattr(torch._dynamo, "reset", lambda: (calls.append(True), real_reset())[1])

    module = torch.nn.Linear(4, 4)
    fns = _build_torch_fns(module, (torch.randn(2, 4),), {}, warmup=0, backends={"tcompile"})
    assert calls, "dynamo must be reset before the bench compile"
    assert "torch.compile" in fns

    calls.clear()
    fns = _build_torch_fns(module, (torch.randn(2, 4),), {}, warmup=0, backends={"eager"})
    assert not calls, "eager-only benches must not touch dynamo"
    assert "Eager PyTorch" in fns


@requires_cuda
def test_run_golden_bench_shows_benched_golden_row(run_cli):
    """``run --golden NAME --bench`` compiles + benches the recorded golden (knobs
    pinned) and prints it as a ``golden NAME``-labeled row in the kernel table, plus the
    ``greedy (isolated)`` twin — the greedy graph re-benched through the pinned-row path,
    the baseline the golden rows compare against."""
    rc, stdout, stderr = run_cli("run", "--golden", "matmul.square.512", "--bench")
    assert rc == 0, f"stderr: {stderr}"
    assert "golden matmul.square.512" in stdout, stdout
    assert "greedy (isolated)" in stdout, stdout


def test_run_ab_requires_bench(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.zeros(4)", "--ab", "BM=8")
    assert rc == 2
    assert "--ab requires --bench" in (stdout + stderr)


def test_run_ab_requires_relowerable_input(run_cli):
    """``--ab`` on a model-ID positional has no code / IR to re-lower per config."""
    rc, stdout, stderr = run_cli("run", "some/model", "--ab", "BM=8", "--bench")
    assert rc == 2
    assert "re-lowerable" in (stdout + stderr)


def test_run_ab_rejects_malformed_spec(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.zeros(4)", "--bench", "--ab", "BM8")
    assert rc == 2
    assert "missing '='" in (stdout + stderr)


def test_ab_samples_parse_label_and_shape():
    """``_ab_samples`` parses each spec with the ``EMMY_KNOBS`` grammar, labels
    the row with the raw spec, and marks it shapeless (the ``_print_kernel_stats``
    cue to nest by kernel ``S_*`` signature instead of a golden matmul shape)."""
    from emmy.commands.run import _ab_samples

    (s,) = _ab_samples(["tile=n16x8, STAGE=d2/cp"])
    assert s.knobs == {"TILE": "n16x8", "STAGE": "d2/cp"}  # names uppercased, whitespace tolerated
    assert s.name == "ab tile=n16x8, STAGE=d2/cp"
    assert s.shape is None
    assert s.dynamic is None
    # A --dynamic run stamps its specs on the pseudo-sample so the A/B re-trace
    # builds the same symbolic graph as the greedy run.
    (d,) = _ab_samples(["TILE=n16x8"], dynamic=["seq_len@x0:0"])
    assert d.dynamic == ("seq_len@x0:0",)


def test_bench_golden_variants_retraces_with_dynamic_spec(monkeypatch):
    """A pinned sample carrying ``dynamic`` specs (a dynamic golden) re-traces its
    snippet with the matching ``torch.export`` dynamic_shapes, so the pinned kernel
    is the same masked-tile artifact as the greedy run; a static sample re-traces
    with ``dynamic_shapes=None``."""
    from types import SimpleNamespace

    from emmy.commands import trace as tmod
    from emmy.commands.run import _bench_golden_variants
    from emmy.compiler.graph import Graph

    seen = []

    def fake_graph_from_code(code, dynamic_shapes=None):
        seen.append(dynamic_shapes)
        return Graph(), "slug", (None, (), {})

    monkeypatch.setattr(tmod, "graph_from_code", fake_graph_from_code)

    async def fake_bench_pinned_async(g, *, run_inputs=None, run_inputs_key=None, warmup, num_iters):
        return SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]), None

    backend = SimpleNamespace(compile=lambda g: g, bench_pinned_async=fake_bench_pinned_async)
    dyn = SimpleNamespace(name="g.dynM", knobs={"TILE": "n16x8/f2x2"}, shape=None, dynamic=("seq_len@x0:0",))
    static = SimpleNamespace(name="g", knobs={"TILE": "n16x8/f2x2"}, shape=None, dynamic=None)
    benches = asyncio.run(
        _bench_golden_variants(backend, "torch.matmul(torch.randn(8,8), torch.randn(8,8))", [dyn, static], warmup=1, iters=1)
    )
    assert len(benches) == 2
    assert seen[0] is not None and 0 in seen[0]["x0"]  # {x0: {0: Dim(seq_len)}}
    assert seen[1] is None


def test_intensity_floor_flags_impossible_row(monkeypatch):
    """The finding-4 gate: a benched row whose CONFIG-implied FLOP/s exceeds the device's
    recorded peak is flagged (the sixth sweep's 8.2 µs "2 PFLOP/s" 2048³ golden row); a
    plausible latency passes, and an unregistered device degrades to no gate. The gate reads
    ``Sample.flops`` (``GoldenConfig.flops()`` — never overestimated, hint-free) rather than
    reconstructing from the ShapeKey: the join key excludes symbolic axes on the matmul side
    but includes them on the reduce-tier side, so the old hint-multiplier reconstruction
    flagged every reduce-tier ``.dynM`` replay 512× over (the golden-audit false positive)."""
    from types import SimpleNamespace

    from emmy import gpu
    from emmy.commands.run import _intensity_floor_flag

    monkeypatch.setattr(gpu, "live_name", lambda: "NVIDIA GeForce RTX 5090")
    s = SimpleNamespace(flops=2.0 * 2048**3, dtype="fp32")
    assert "impossible" in _intensity_floor_flag(s, 8.2)  # 2 PFLOP/s on a 104.8 TFLOP/s card
    assert _intensity_floor_flag(s, 300.0) is None  # ~57 TFLOP/s — plausible
    # A dynamic golden's flops() already sizes the symbolic axis at its benched hint.
    d = SimpleNamespace(flops=2.0 * 512**3, dtype="fp32")
    assert "impossible" in _intensity_floor_flag(d, 0.5)
    assert _intensity_floor_flag(d, 9.0) is None
    # Flopless (--ab) rows and unknown devices are ungateable.
    assert _intensity_floor_flag(SimpleNamespace(flops=None, dtype="fp32"), 1.0) is None
    monkeypatch.setattr(gpu, "live_name", lambda: "Some Unknown GPU")
    assert _intensity_floor_flag(s, 8.2) is None


def test_wrong_answer_flag_catches_bad_pinned_output():
    """The pinned-row output gate: identical outputs pass, reduction-reorder noise passes
    the loose 5% tolerance, a silently-wrong kernel (the g2a skipped-finalize class) flags,
    and a missing / mis-shaped output flags rather than crashing."""
    import numpy as np

    from emmy.commands.run import _wrong_answer_flag

    ref = {"o": np.full((4, 4), 100.0)}
    assert _wrong_answer_flag({"o": ref["o"].copy()}, ref) is None
    assert _wrong_answer_flag({"o": ref["o"] + 1.0}, ref) is None  # 1% off — reorder noise
    assert "wrong-answer" in _wrong_answer_flag({"o": ref["o"] * 0.5}, ref)
    assert "missing" in _wrong_answer_flag({}, ref)
    assert "shape" in _wrong_answer_flag({"o": np.zeros((2, 2))}, ref)


def test_unreproducible_pin_flag(monkeypatch):
    """The realized-vs-pinned gate: a pin the compile silently dropped (the fallback
    substituted the planner's own pick — the retired ``w2x1`` hd128 flash form) flags
    with the pinned and realized values; a pin realized on ANY kernel of a multi-kernel
    lowering passes; a bare pin matches its axis-stamped ``@``-keyed realizations
    (``TILE@d``, multi-axis ``TILE@dd``/``TILE@pj``, the never-collapsed ``PLACE@fold``);
    value compare is registry-canonical (``knob.values_equal``); a registered family
    with no stamp anywhere is ungateable (a reloaded ``--ir`` graph drops serialized
    knobs, so absence ≠ dropped), while an unregistered family flags as a pin typo.
    A synthetic registry (mirroring space.py's TILE/STAGE/PLACE/FAST_EXP declarations)
    keeps the test independent of module-load order."""
    from emmy.commands.run import _unreproducible_pin_flag
    from emmy.compiler.pipeline import knob as knob_mod
    from emmy.compiler.pipeline.knob import Knob, KnobType

    monkeypatch.setattr(
        knob_mod,
        "_REGISTRY",
        {
            "TILE": Knob("TILE", KnobType.STR, off=""),
            "STAGE": Knob("STAGE", KnobType.STR, off=""),
            "PLACE": Knob("PLACE", KnobType.STR),
            "FAST_EXP": Knob("FAST_EXP", KnobType.BOOL, off=False),
        },
    )

    # Honored pin — realized exactly.
    assert _unreproducible_pin_flag({"TILE": "w2x1/f1x8"}, [{"TILE": "w2x1/f1x8"}]) is None
    # Silently swapped pin — the greedy-vs-greedy case the gate exists for.
    flag = _unreproducible_pin_flag({"TILE": "w2x1/f1x8"}, [{"TILE": "w4x2/f2x4"}])
    assert "unreproducible pin" in flag and "TILE=w2x1/f1x8" in flag and "w4x2/f2x4" in flag
    # A REGISTERED family with no stamp on any kernel is ungateable, not a miss: on a
    # partially re-lowered --ir reload the stamp may have been serialized away (a full
    # compile OFF-fills declared knobs, so a dropped pin still shows as (off)/conflict).
    assert _unreproducible_pin_flag({"STAGE": "k8"}, [{"TILE": "w2x1"}]) is None
    # An UNREGISTERED family with no stamp is a typo in the pin — flagged.
    assert "(unset)" in _unreproducible_pin_flag({"TIEL": "w2x1"}, [{"TILE": "w2x1"}])
    # Multi-kernel lowering (split main + finalize): honored on the second kernel.
    assert _unreproducible_pin_flag({"STAGE": "k8"}, [{"TILE": "w2x1"}, {"STAGE": "k8"}]) is None
    # Bare pin vs single-axis @-keyed realization.
    assert _unreproducible_pin_flag({"TILE": "n16x8/f2x2"}, [{"TILE@d": "n16x8/f2x2"}]) is None
    # Bare pin vs a MULTI-axis realization (flash stamps two TILE@ keys — no collapse).
    assert _unreproducible_pin_flag({"TILE": "w4x1/f1x16"}, [{"TILE@dd": "w4x1/f1x16", "TILE@pj": "w4x1/f1x16"}]) is None
    # Bare PLACE (the golden YAML spelling) vs the fold/cone-stamped realization.
    assert _unreproducible_pin_flag({"PLACE": "fuse"}, [{"PLACE@fold": "fuse", "PLACE@cone": "fuse"}]) is None
    # An @-keyed pin whose axis the re-lowering renamed: a genuine miss, but the
    # diagnostic names the family's realized value instead of (unset).
    flag = _unreproducible_pin_flag({"TILE@dd": "w4x1/f1x16"}, [{"TILE@d2": "w2x1/f1x8"}])
    assert "TILE@d2=w2x1/f1x8" in flag and "(unset)" not in flag
    # Registry-canonical value compare (bool knob pinned via the string grammar).
    assert _unreproducible_pin_flag({"FAST_EXP": "true"}, [{"FAST_EXP": True}]) is None
    # OFF values are "declined", not conflicts: the honored axis wins, the off-stamped
    # sibling never pollutes the diagnostic...
    assert _unreproducible_pin_flag({"TILE": "w2x1"}, [{"TILE": ""}, {"TILE@d": "w2x1"}]) is None
    # ...and a family realized ONLY as off reports (off), not the empty string.
    assert "realized (off)" in _unreproducible_pin_flag({"STAGE": "d2/tma/ring"}, [{"STAGE": ""}])
    # No kernel knobs → ungateable, not a flag — [] and all-empty dicts alike.
    assert _unreproducible_pin_flag({"TILE": "w2x1"}, []) is None
    assert _unreproducible_pin_flag({"TILE": "w2x1"}, [{}]) is None
    assert _unreproducible_pin_flag({"TILE": "w2x1"}, [{}, {}]) is None


def test_bench_golden_variants_unmatched_pin_fails_row_without_benching(monkeypatch):
    """End-to-end through ``_bench_golden_variants``: a pinned config whose compiled
    kernels realized different knobs FAILS its row loudly before any bench — status
    ``pin_unmatched``, no bench (benching the fallback realization would measure the
    planner's own pick under the pin's name, the sweep-misleading silent-degrade class) —
    while a config whose pin was honored benches clean, so one bad pin never blocks the
    remaining rows."""
    from types import SimpleNamespace

    from emmy.commands import trace as tmod
    from emmy.commands.run import _bench_golden_variants
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    monkeypatch.setattr(tmod, "graph_from_code", lambda code, dynamic_shapes=None: (object(), "slug", (None, (), {})))

    def graph_with(knobs):
        g = Graph()
        g.add_node(op=CudaOp(kernel_name="k", knobs=knobs), inputs=[], output=Tensor("o", (4,)), node_id="n0")
        return g

    compiled = iter([graph_with({"TILE": "w4x2/f2x4"}), graph_with({"TILE": "w2x1/f1x8"})])
    benched: list = []

    async def fake_bench_pinned_async(g, *, run_inputs=None, run_inputs_key=None, warmup, num_iters):
        benched.append(g)
        return SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]), None

    backend = SimpleNamespace(compile=lambda g: next(compiled), bench_pinned_async=fake_bench_pinned_async)
    dropped = SimpleNamespace(name="g.dropped", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None)
    honored = SimpleNamespace(name="g.honored", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None)
    benches = asyncio.run(_bench_golden_variants(backend, "torch.matmul(a, b)", [dropped, honored], warmup=1, iters=1))
    assert len(benches) == 2
    assert benches[0].status == "pin_unmatched" and benches[0].bench is None
    assert any("unreproducible pin" in f and "NOT benched" in f for f in benches[0].flags)
    assert len(benched) == 1  # only the honored row spent GPU time
    assert benches[1].status == "ok" and benches[1].flags == [] and benches[1].bench is not None


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("benchmark run stage exceeded 10.0s"),
        RuntimeError("bench worker exceeded 100.0s wall budget — SIGKILL'd, stream cleaned"),
    ],
    ids=["budget", "hung-worker-sigkill"],
)
def test_bench_golden_variants_survives_row_bench_fail(monkeypatch, failure):
    """A pinned config whose worker job fails is kept as a ``bench_fail`` row (never
    dropped — the table / ``--json`` must show why) and the remaining rows still bench.
    A HANG is not a special case anymore: it dies with the SIGKILL'd worker child and
    surfaces as the same RuntimeError — the parent's CUDA context is never touched, so
    there is no escalation path and no ``os._exit``."""
    from types import SimpleNamespace

    from emmy.commands import trace as tmod
    from emmy.commands.run import _bench_golden_variants
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    monkeypatch.setattr(tmod, "graph_from_code", lambda code, dynamic_shapes=None: (object(), "slug", (None, (), {})))

    def graph_with(knobs):
        g = Graph()
        g.add_node(op=CudaOp(kernel_name="k", knobs=knobs), inputs=[], output=Tensor("o", (4,)), node_id="n0")
        return g

    compiled = iter([graph_with({"TILE": "w2x1/f1x8"}), graph_with({"TILE": "w2x1/f1x8"})])
    calls = iter([failure, None])

    async def fake_bench_pinned_async(g, *, run_inputs=None, run_inputs_key=None, warmup, num_iters):
        exc = next(calls)
        if exc is not None:
            raise exc
        return SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]), None

    backend = SimpleNamespace(compile=lambda g: next(compiled), bench_pinned_async=fake_bench_pinned_async)
    rows = [
        SimpleNamespace(name="g.slow", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None),
        SimpleNamespace(name="g.fine", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None),
    ]
    benches = asyncio.run(_bench_golden_variants(backend, "torch.matmul(a, b)", rows, warmup=1, iters=1))
    assert len(benches) == 2
    assert benches[0].status == "bench_fail" and benches[0].bench is None
    assert any("bench_fail" in f for f in benches[0].flags)
    assert benches[1].status == "ok" and benches[1].bench is not None


def test_bench_golden_variants_wrong_answer_gate_uses_worker_outputs(monkeypatch):
    """With a greedy reference supplied, each row's worker job re-executes the pinned
    config on the greedy run's inputs and the RETURNED outputs feed the wrong-answer
    gate: matching outputs stay clean, a silently-wrong kernel flags. Without a
    reference no execution is requested (``run_inputs is None``)."""
    from types import SimpleNamespace

    import numpy as np

    from emmy.commands import trace as tmod
    from emmy.commands.run import _bench_golden_variants
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    monkeypatch.setattr(tmod, "graph_from_code", lambda code, dynamic_shapes=None: (object(), "slug", (None, (), {})))

    def graph_with(knobs):
        g = Graph()
        g.add_node(op=CudaOp(kernel_name="k", knobs=knobs), inputs=[], output=Tensor("o", (4,)), node_id="n0")
        return g

    compiled = iter([graph_with({"TILE": "w2x1/f1x8"}), graph_with({"TILE": "w2x1/f1x8"})])
    ref_outputs = {"n0": np.full((4,), 100.0)}
    row_outputs = iter([{"n0": ref_outputs["n0"].copy()}, {"n0": ref_outputs["n0"] * 0.5}])
    seen_inputs: list = []

    async def fake_bench_pinned_async(g, *, run_inputs=None, run_inputs_key=None, warmup, num_iters):
        seen_inputs.append(run_inputs)
        outs = next(row_outputs) if run_inputs is not None else None
        return SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]), outs

    backend = SimpleNamespace(compile=lambda g: next(compiled), bench_pinned_async=fake_bench_pinned_async)
    rows = [
        SimpleNamespace(name="g.good", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None),
        SimpleNamespace(name="g.bad", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None),
    ]
    ref = ({"x": np.zeros((4,))}, ref_outputs)
    benches = asyncio.run(_bench_golden_variants(backend, "torch.matmul(a, b)", rows, warmup=1, iters=1, ref=ref))
    assert seen_inputs == [ref[0], ref[0]]  # the greedy inputs rode each row's job
    assert benches[0].flags == []
    assert any("wrong-answer" in f for f in benches[1].flags)


def test_bench_greedy_isolated_ok_and_bench_fail():
    """The isolated greedy re-bench ships the ALREADY-compiled greedy graph through
    ``bench_pinned_async`` — pinned-row timing semantics, no re-trace / recompile — so
    pinned speedups get a comparable baseline (the interleaved greedy number is ~7% off
    pinned-row semantics). A worker failure is kept as a ``bench_fail`` row and must not
    raise (the pinned rows still bench after it)."""
    from types import SimpleNamespace

    from emmy.commands.run import _bench_greedy_isolated

    compiled = object()
    benched: list = []

    async def ok_bench(g, *, warmup, num_iters):
        benched.append(g)
        return SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]), None

    gb = asyncio.run(_bench_greedy_isolated(SimpleNamespace(bench_pinned_async=ok_bench), compiled, warmup=1, iters=1))
    assert benched == [compiled]  # the deployed graph itself rode the worker job
    assert gb.status == "ok" and gb.bench is not None and gb.flags == []
    assert gb.sample.name == "greedy (isolated)" and gb.sample.shape is None

    async def hung_bench(g, *, warmup, num_iters):
        raise RuntimeError("bench worker exceeded 100.0s wall budget — SIGKILL'd, stream cleaned")

    gb = asyncio.run(_bench_greedy_isolated(SimpleNamespace(bench_pinned_async=hung_bench), compiled, warmup=1, iters=1))
    assert gb.status == "bench_fail" and gb.bench is None
    assert any("bench_fail" in f for f in gb.flags)


@requires_cuda
def test_run_golden_bench_json_record(run_cli, tmp_path):
    """``run --bench --golden NAME --json PATH`` writes the machine-readable A/B record:
    backends, the greedy kernel rows, and one pinned entry per recorded golden config with
    its integrity ``flags`` field and recorded reference latencies."""
    import json

    out = tmp_path / "ab.json"
    rc, stdout, stderr = run_cli("run", "--golden", "matmul.square.512", "--bench", "--warmup", "2", "--iters", "5", "--json", str(out))
    assert rc == 0, f"stderr: {stderr}"
    rec = json.loads(out.read_text())
    assert rec["golden"] == "matmul.square.512"
    assert rec["greedy"]["kernels"] and rec["greedy"]["total_us"] > 0
    # The pinned-comparable greedy baseline: the same graph re-benched emmy-only.
    assert rec["greedy"]["isolated"]["status"] == "ok" and rec["greedy"]["isolated"]["total_us"] > 0
    assert rec["pinned"] and all(p["kind"] == "golden" for p in rec["pinned"])
    for p in rec["pinned"]:
        assert "flags" in p and p["total_us"] > 0 and p["pinned_knobs"]
        assert p["recorded_emmy_us"] > 0 and p["recorded_ref_us"] > 0
    # Eager + emmy backend rows made it into the record.
    assert any("Eager" in k for k in rec["backends"])


_NCU_CSV = """"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","k_matmul_abc123","gpu__time_duration.sum","nsecond","10,000"
"0","k_matmul_abc123","sm__warps_active.avg.pct_of_peak_sustained_active","%","41.5"
"0","k_matmul_abc123","smsp__inst_executed_pipe_lsu.sum","inst","100"
"1","k_matmul_abc123","gpu__time_duration.sum","nsecond","12,000"
"1","k_matmul_abc123","sm__warps_active.avg.pct_of_peak_sustained_active","%","42.5"
"1","k_matmul_abc123","smsp__inst_executed_pipe_lsu.sum","inst","100"
"2","ampere_sgemm_128x64_nn","gpu__time_duration.sum","nsecond","8,000"
"2","ampere_sgemm_128x64_nn","sm__warps_active.avg.pct_of_peak_sustained_active","%","80.0"
"2","ampere_sgemm_128x64_nn","launch__registers_per_thread","register/thread","90"
"""


def test_parse_ncu_csv_keeps_both_sides_and_aggregates():
    """The parser keeps the reference (cuBLAS / aten) rows beside the ``k_*`` rows
    (the comparison table needs both) and aggregates multi-launch rows: ``.sum`` /
    ``smsp__*`` counters add up, percentages average."""
    from emmy.commands.run import _ncu_units, _parse_ncu_csv

    parsed = _parse_ncu_csv(_NCU_CSV)
    assert set(parsed) == {"k_matmul_abc123", "ampere_sgemm_128x64_nn"}
    dep = parsed["k_matmul_abc123"]
    assert dep["gpu__time_duration.sum"] == 22000.0  # summed over the two launches
    assert dep["sm__warps_active.avg.pct_of_peak_sustained_active"] == 42.0  # averaged
    assert dep["smsp__inst_executed_pipe_lsu.sum"] == 200.0
    assert _ncu_units(_NCU_CSV)["gpu__time_duration.sum"] == "nsecond"


def test_print_ncu_compare_renders_dep_then_ref(capsys):
    """The compare table puts emmy rows above the reference rows, carries the
    duration unit in the header, and dashes out metrics a side didn't report."""
    from emmy.commands.run import _ncu_units, _parse_ncu_csv, _print_ncu_compare

    _print_ncu_compare(_parse_ncu_csv(_NCU_CSV), _ncu_units(_NCU_CSV))
    out = capsys.readouterr().out
    assert "ncu compare" in out
    assert out.index("k_matmul_abc123") < out.index("ampere_sgemm_128x64_nn")  # dep side first
    assert "dur (nsecond)" in out
    lines = [ln for ln in out.splitlines() if "ampere_sgemm" in ln]
    assert lines and lines[0].split()[-1] == "90"  # regs lands in the last column
    assert "-" in lines[0]  # unreported metrics dash out


@requires_cuda
def test_run_ab_bench_shows_pinned_row(run_cli):
    """``run --code ... --bench --ab KNOBS`` benches the pinned config and prints it
    as an ``ab KNOBS``-labeled row in the kernel table. The pin must REALIZE (a valid
    scalar-tile spelling for this shape) — an unmatched pin now fails its row loudly
    and exits non-zero instead of benching the planner's pick under the pin's name."""
    rc, stdout, stderr = run_cli(
        "run", "--code", "torch.matmul(torch.randn(64, 64), torch.randn(64, 64))", "--bench", "--ab", "TILE=n16x16/f2x2"
    )
    assert rc == 0, f"stderr: {stderr}"
    assert "ab TILE=n16x16/f2x2" in stdout, stdout


@requires_cuda
def test_run_ab_bench_unmatched_pin_fails_loudly(run_cli):
    """An ``--ab`` pin that matches no offered row (a knob family that does not exist)
    is NOT benched — the row fails loudly (``unreproducible pin`` on the row, no timing)
    and the run exits non-zero, instead of silently benching the planner's own pick
    under the pin's name (the pin-spelling trap the golden sweeps kept hitting)."""
    rc, stdout, _stderr = run_cli(
        "run", "--code", "torch.matmul(torch.randn(64, 64), torch.randn(64, 64))", "--bench", "--ab", "BM=8,BN=16"
    )
    assert rc != 0, "an unmatched --ab pin must fail the run"
    assert "unreproducible pin" in stdout, stdout
    assert "! ab BM=8,BN=16" in stdout, stdout


@requires_cuda
def test_run_code_rmsnorm_accuracy(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.RMSNorm(64)({_randn('1,8,64', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_rmsnorm_via_pow_neg_half(run_cli):
    """Gemma-style RMSNorm normalization uses ``torch.pow(ms, -0.5)`` (not
    ``rsqrt``); the exponent arrives as a broadcast constant. Guards the
    ``030_pow`` regression where every ``pow`` was squared — here that would
    compute ``x * (mean+eps)²`` and fail the eager comparison."""
    code = "x = torch.randn(2,64,256); torch.mul(x, torch.pow(torch.mean(torch.pow(x,2),-1,keepdim=True)+1e-6, -0.5))"
    rc, _, stderr = run_cli("run", "--code", code)
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_accuracy(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.matmul({_randn('16,32', dtype)}, {_randn('32,16', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_target_override(run_cli):
    """``--target sm_80`` gates lowering to the cp.async path (no TMA); the kernel still runs
    on the live device and must match eager, so ``rc == 0`` is the accuracy assertion."""
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(256, 256), torch.randn(256, 256))", "--target", "sm_80")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_rmsnorm_blockify(run_cli, dtype):
    """Wide hidden + ≥16 rows triggers blockify on the row axis. Regression
    test: cooperative load step must match the actual thread count, not
    BLOCK_SIZE=256, or staged-weight indices get skipped."""
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.RMSNorm(2048)({_randn('1,32,2048', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_softmax_blockify(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.functional.softmax({_randn('32,2048', dtype)}, dim=-1)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_blockify(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.matmul({_randn('64,128', dtype)}, {_randn('128,64', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
@pytest.mark.parametrize("fk", [2, 4, 8])
@pytest.mark.parametrize("br", [None, 1])
def test_run_code_rmsnorm_fk_accuracy(run_cli, monkeypatch, fk, br):
    """FK register-tiles the reduce axis into ``fk`` independent accumulators +
    a cross-accumulator fold. Pin FK
    (and optionally BR=1 for the pure-serial scope) and confirm the folded
    reduction still matches eager — ``rc == 0`` is the accuracy assertion."""
    monkeypatch.setenv("EMMY_FK", str(fk))
    if br is not None:
        monkeypatch.setenv("EMMY_BR", str(br))
    rc, _, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(2048)(torch.randn(4,32,2048))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
@pytest.mark.parametrize("fk", [2, 4, 8])
def test_run_code_fp16_matmul_window_accuracy(run_cli, monkeypatch, fk):
    """fp16 scalar matmul half2 accumulation window:
    pin MMA off + an even FK window and confirm the windowed half2 accumulate +
    fp32 flush matches eager within fp16 tolerance — ``rc == 0`` asserts it."""
    monkeypatch.setenv("EMMY_MMA", "0")
    monkeypatch.setenv("EMMY_FK", str(fk))
    rc, _, stderr = run_cli("run", "--code", "torch.randn(256,256,dtype=torch.float16) @ torch.randn(256,256,dtype=torch.float16)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
@pytest.mark.parametrize("br", [None, 1])
def test_run_code_softmax_fk_accuracy(run_cli, monkeypatch, br):
    """Softmax carries both a ``max`` and a ``sum`` reduce, so FK exercises the
    ``fmaxf`` and ``+`` cross-accumulator folds together. ``rc == 0`` asserts
    the pinned-FK kernel matches eager."""
    monkeypatch.setenv("EMMY_FK", "4")
    if br is not None:
        monkeypatch.setenv("EMMY_BR", str(br))
    rc, _, stderr = run_cli("run", "--code", "torch.softmax(torch.randn(4,32,2048), dim=-1)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_linear_blockify(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.nn.Linear(2048, 2048, bias=False)(torch.randn(1, 32, 2048))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_k_chunked(run_cli):
    """Matmul with K large enough to exercise the K-chunked SGEMM path
    (BK=64). Regression: the K_o outer loop is syntactically a free
    Loop (no immediate Accum) but the Init for the running accumulator
    must still land at the surrounding Tile body so it persists across
    K_o iterations."""
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(128, 2048), torch.randn(2048, 128))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_k_chunked(run_cli):
    """SDPA: the per-output free loop (head_dim) wraps a reduce loop +
    a Write — its body has a Write so it is *not* a reduce-passthrough
    and the per-output accumulator must reset per iteration. Pairs
    with the matmul case above to cover both branches of the recursive
    reduce-crossing rule."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,2,32,64), torch.randn(1,2,32,64), torch.randn(1,2,32,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_tinyllama_per_head(run_cli):
    """Per-head SDPA at TinyLlama-block-seq=512 dimensions, mirroring the
    ``k_scaled_dot_product_attention_reduce_reduce.json`` kernel in
    ``experiments/kernel_dataset/tinyllama_block_seq512`` (M=512, K=512,
    N=64). The K=512 reduction does not fit a full smem slab once
    register-tile + double-buffer apply, so this exercises the chunked
    blockify + staging path on the per-head shape."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,1,512,64), torch.randn(1,1,512,64), torch.randn(1,1,512,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_seq1024_dynamic_smem(run_cli):
    """SDPA at seq_len=1024, 32 heads: the Q·Kᵀ kernel needs ~50 KB of
    smem after register-tile + double-buffer + bank-pad — well past the
    48 KB static cap. Pins the dynamic-smem pool path: kernel must
    declare ``extern __shared__ ... _smem_pool[]``, the launch must pass
    ``shared_mem=smem_bytes``, and ``cudaFuncSetAttribute(MaxDynamicShared
    MemorySize)`` must opt this kernel into the device's larger dynamic
    allowance."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,32,1024,64), torch.randn(1,32,1024,64), torch.randn(1,32,1024,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_tinyllama_full(run_cli):
    """Full multi-head TinyLlama-block-seq=512 SDPA (1 batch × 32 heads ×
    512 × 64). Regression: the blockify + staging interaction
    over-allocated per-block smem (PTXAS rejected the kernel with
    ``uses too much shared data``, 0xc600 > 0xc000 = 49152 cap)."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,32,512,64), torch.randn(1,32,512,64), torch.randn(1,32,512,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_bench_prints_table(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "--bench", "--warmup", "2", "--iters", "5")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Eager PyTorch" in log
    assert "Emmy" in log
    assert "vs Eager" in log


# --- --ir mode -------------------------------------------------------------


def _dump_ir(project_root: Path, code: str, stage: str, out_dir: Path) -> Path:
    """Run ``emmy compile --code <code> --dump-dir`` and return the path
    to the JSON dump for the requested stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [sys.executable, "-m", "emmy.emmy", "compile", "--code", code, "--dump-dir", str(out_dir), "--ir", stage],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert rc.returncode == 0, rc.stderr
    # Stage dumps are named ``NN_<stage>.json`` — pick the matching one.
    candidates = sorted(out_dir.glob(f"*_{stage.replace('/', '_')}*.json"))
    candidates = [p for p in candidates if not p.name.endswith(".rules.json") and not p.name.endswith(".kernels.json")]
    assert candidates, f"no dump matched stage={stage} in {out_dir}: {list(out_dir.iterdir())}"
    return candidates[-1]


@requires_cuda
def test_run_ir_loop_stage(run_cli, project_root, tmp_path):
    """``emmy run --ir <loop.json>`` loads loop IR and runs the
    remaining tile / kernel / cuda passes, executes with random inputs."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "loop", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded loop IR" in log
    assert "lowering/tile" in log


@requires_cuda
def test_run_positional_json_like_ir(run_cli, project_root, tmp_path):
    """A ``.json`` passed as the positional input takes the same IR path as ``--ir``."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "loop", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    assert "Loaded loop IR" in (stdout + stderr)


@requires_cuda
def test_run_ir_tile_stage(run_cli, project_root, tmp_path):
    """Tile-IR JSON loads and runs only the kernel + cuda tail."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded tile IR" in log
    assert "lowering/kernel" in log
    # tile-stage already ran lowering/tile, so it should NOT be in the tail list.
    assert "running tail passes: ['lowering/kernel'" in log


@requires_cuda
def test_run_ir_kernel_stage(run_cli, project_root, tmp_path):
    """Kernel-IR JSON loads and runs only the cuda tail."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "kernel", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded kernel IR" in log
    assert "running tail passes: ['lowering/cuda']" in log


@requires_cuda
def test_run_ir_cuda_stage_no_tail(run_cli, project_root, tmp_path):
    """Already-lowered cuda IR has no remaining passes."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "cuda", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded cuda IR" in log
    assert "running tail passes: (none)" in log


@requires_cuda
def test_run_ir_bench(run_cli, project_root, tmp_path):
    """``--bench`` with ``--ir`` prints just the emmy latency row
    (no eager reference is available for partial-IR mode)."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path), "--bench", "--warmup", "2", "--iters", "5")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Emmy" in log
    assert "Latency (us)" in log


@requires_cuda
def test_run_ir_seed_reproducible(run_cli, project_root, tmp_path):
    """Two runs with the same seed produce the same output mean."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    runs = []
    for _ in range(2):
        rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path), "--seed", "42")
        assert rc == 0
        # Output line: "Output rms_norm: shape=... finite=True mean=<value>"
        for line in (stdout + stderr).splitlines():
            if "mean=" in line:
                runs.append(line.split("mean=")[-1].strip())
                break
    assert len(runs) == 2 and runs[0] == runs[1], runs


def test_run_ir_invalid_json(run_cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc, _, stderr = run_cli("run", "--ir", str(bad))
    assert rc != 0


def test_run_ir_missing_file(run_cli, tmp_path):
    rc, _, stderr = run_cli("run", "--ir", str(tmp_path / "does_not_exist.json"))
    assert rc != 0


@requires_cuda
def test_run_code_dynamic_seq_len(run_cli):
    """``run --code --dynamic seq_len@x:1`` traces with torch.export's
    dynamic_shapes, compiles to a single ``int seq_len``-arg kernel,
    runs it at the canonical shape, and checks accuracy against eager."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.RMSNorm(64)(torch.randn(1,8,64))",
        "--dynamic",
        "seq_len@x:1",
    )
    assert rc == 0, f"stderr: {stderr}"


# ---------------------------------------------------------------------------
# _bind_inputs: integer-dtype preservation
# ---------------------------------------------------------------------------


def test_bind_inputs_preserves_int_dtype():
    """``_bind_inputs`` must cast each torch input to the numpy dtype
    that matches the graph's declared ``Tensor.dtype`` — not blanket-
    cast to float32 as it did before integer placeholders (``input_ids``,
    ``position_ids``) became part of whole-model traces. A float32 cast
    of int64 indices would silently corrupt the embedding-lookup path."""
    import numpy as np

    from emmy.commands.run import _bind_inputs
    from emmy.compiler import dtype as dt
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("input_ids", (1, 8), dt.I64), node_id="input_ids")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("position_ids", (1, 8), dt.I32), node_id="position_ids")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("activations", (1, 8, 16), dt.F32), node_id="activations")
    g.inputs = ["input_ids", "position_ids", "activations"]

    class _EmptyModule:
        def named_parameters(self, remove_duplicate=True):
            return iter(())

        def named_buffers(self, remove_duplicate=True):
            return iter(())

    input_ids = torch.zeros((1, 8), dtype=torch.long)
    position_ids = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    activations = torch.randn(1, 8, 16)

    bound = _bind_inputs(g, _EmptyModule(), (input_ids, position_ids, activations), {})

    assert bound["input_ids"].dtype == np.int64
    assert bound["position_ids"].dtype == np.int32
    assert bound["activations"].dtype == np.float32
    # Values must round-trip without precision loss.
    np.testing.assert_array_equal(bound["position_ids"], np.arange(8, dtype=np.int32)[None, :])


def test_bind_inputs_arity_mismatch_raises():
    """Binding failures RAISE (with the real cause) instead of ``sys.exit`` — the function
    also runs inside the bench worker child, where an exit(1) reached the parent as an
    opaque ``SystemExit(1)`` with the cause stranded in the child's log stream."""
    from emmy.commands.run import _bind_inputs
    from emmy.compiler import dtype as dt
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,), dt.F32), node_id="x")
    g.inputs = ["x"]

    class _EmptyModule:
        def named_parameters(self, remove_duplicate=True):
            return iter(())

        def named_buffers(self, remove_duplicate=True):
            return iter(())

    with pytest.raises(RuntimeError, match="arity mismatch"):
        _bind_inputs(g, _EmptyModule(), (), {})


def test_compare_wall_s_scales_with_kernel_count():
    """The comparison job's SIGKILL wall cap derives from the workload — a fixed cap
    false-fails big-model runs whose child legitimately pays a first nvcc compile per
    kernel (bounded by the in-child compile budget) before ever benching."""
    from types import SimpleNamespace

    from emmy.commands.run import _compare_wall_s
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.cuda.ir import CudaOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    for i in range(3):
        g.add_node(op=CudaOp(kernel_name=f"k{i}"), inputs=[], output=Tensor(f"o{i}", (4,)), node_id=f"n{i}")
    backend = SimpleNamespace(bench_compile_timeout_s=30.0, bench_run_timeout_s=10.0)
    assert _compare_wall_s(g, backend, base_s=240.0) == 240.0 + 3 * 30.0 + 10.0
    # A kernel-less graph still gets one compile budget of headroom.
    assert _compare_wall_s(Graph(), backend, base_s=60.0) == 60.0 + 30.0 + 10.0


def test_launch_order_cuda_nodes_pairs_by_topo_not_dict_order():
    """The per-kernel table pairs ``bench.per_launch`` times to kernels by launch index,
    and the backend launches in ``graph.topological_order()`` — NOT graph dict order.
    Node-splitting passes insert a producer (the split ``__partial``) after its consumer
    in dict order, which used to cross-label the rows: the 4 µs norm kernel printed the
    1573 µs matmul-partial's time and vice versa (nsys-verified on the gemma-4-12B
    QK-norm reproducer). ``_launch_order_cuda_nodes`` must return topo order regardless
    of insertion order."""
    from emmy.commands.run import _launch_order_cuda_nodes
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.cuda import CudaOp

    def _cuda_op(name, args):
        return CudaOp(kernel_source="", kernel_name=name, arg_order=args, grid=(1, 1, 1), block=(1, 1, 1))

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (8,)), node_id="A")
    g.add_node(op=_cuda_op("k_producer", ("A", "T")), inputs=["A"], output=Tensor("T", (8,)), node_id="T")
    g.add_node(op=_cuda_op("k_consumer", ("T", "C")), inputs=["T"], output=Tensor("C", (8,)), node_id="C")
    g.inputs = ["A"]
    g.outputs = ["C"]
    # Simulate the split pass's dict layout: the consumer precedes its producer in insertion order.
    g.nodes = {nid: g.nodes[nid] for nid in ("A", "C", "T")}

    names = [n.op.kernel_name for n in _launch_order_cuda_nodes(g)]
    assert names == ["k_producer", "k_consumer"]


def test_accuracy_check_fails_scrambled_output_passes_outliers():
    """The accuracy verdict's three clauses each do their one job (the PR #354 sync-fill
    incident: a swizzle fill/drain mismatch SCRAMBLED the A slab and the old fp16 form —
    ``max ≤ peak OR mean ≤ peak`` — passed the permuted output at mean_diff 15% of peak):

    - a PERMUTED fp16 output must FAIL (systematic corruption — the tight mean gate), even
      though its max error can sit near the loose fp16 outlier ceiling;
    - a correct fp16 output with small noise plus a few atomic-reorder-style OUTLIERS must
      PASS (the near-zero mean vouches for them — the split-K escape hatch);
    - a correct low-noise output must PASS the plain path.

    The verdict is a pure return value (no print / exit — it runs in the bench worker
    child, whose stdout is the pickle protocol channel)."""
    import numpy as np
    import torch

    from emmy.commands.run import _check_accuracy

    rng = np.random.default_rng(0)
    eager = torch.from_numpy((rng.standard_normal(1 << 16) * 20).astype(np.float32))

    def verdicts(arr):
        return _check_accuracy({"o": arr}, eager) is not None

    base = eager.numpy()
    # Scrambled (permutation of the correct values): must FAIL.
    assert verdicts(rng.permutation(base).astype(np.float16)), "a permuted output must fail the mean gate"
    # Correct + tiny noise + a few big outliers (the split-K atomic-reorder class): must PASS.
    noisy = base + rng.standard_normal(base.shape) * 0.01
    noisy[:4] += np.abs(base).max() * 0.5  # outliers past the fp32 max ceiling, near the fp16 one
    assert not verdicts(noisy.astype(np.float16)), "outliers with a near-zero mean must pass (escape hatch)"
    # Correct low-noise: must PASS.
    assert not verdicts((base + rng.standard_normal(base.shape) * 0.001).astype(np.float16))


def test_write_ab_json_greedy_bench_fail_and_record_knobs(tmp_path):
    """The ``--json`` record survives a failed greedy row: the greedy block carries
    ``status: bench_fail`` + ``error`` with null timings, pinned rows carry their own
    ``status``, and every kernel row exposes ``record_knobs`` — the realized tuning knobs
    with EVERY schedule family explicitly stamped (OFF included), the map a golden
    recording copies verbatim so no family is left to the planner's replay-time fill."""
    import json
    from types import SimpleNamespace

    import emmy.compiler.pipeline.knob as knob_mod
    import emmy.compiler.pipeline.search.space  # noqa: F401 — the schedule codec declarations
    from emmy.commands.run import _GoldenBench, _write_ab_json
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    knob_mod.reset_registry()  # the lazy registry may predate the space import above

    def graph_with(knobs):
        g = Graph()
        g.add_node(op=CudaOp(kernel_name="k", knobs=knobs), inputs=[], output=Tensor("o", (4,)), node_id="n0")
        return g

    greedy_graph = graph_with({"TILE": "a:mma_m16n8k16_f16_f32/w1x1/f1x1", "REDUCE": "g2k"})
    unmatched = _GoldenBench(
        SimpleNamespace(name="g.unmatched", knobs={"TILE": "w2x1/f1x8"}, shape=None, dynamic=None, latency_us=None, ref_us=None),
        graph_with({"TILE": "w4x2/f2x4"}),
        None,
        ["unreproducible pin: TILE=w2x1/f1x8 realized w4x2/f2x4 — row NOT benched"],
        "pin_unmatched",
    )
    args = SimpleNamespace(
        json=str(tmp_path / "ab.json"), code="torch.matmul(a, b)", input=None, ir=None, golden=None, dynamic=None, warmup=1, iters=1
    )
    _write_ab_json(args, {}, greedy_graph, None, [unmatched], greedy_fail="greedy bench failed: hung")
    rec = json.loads((tmp_path / "ab.json").read_text())
    assert rec["greedy"]["status"] == "bench_fail" and "hung" in rec["greedy"]["error"]
    assert rec["greedy"]["total_us"] is None
    krow = rec["greedy"]["kernels"][0]
    assert krow["us"] is None
    # record_knobs: realized values + explicit OFF for families the compile never stamped.
    assert krow["record_knobs"]["REDUCE"] == "g2k"
    for fam in ("STAGE", "WSPEC", "RASTER"):
        assert krow["record_knobs"][fam] == ""
    row = rec["pinned"][0]
    assert row["status"] == "pin_unmatched" and row["total_us"] is None
    assert any("NOT benched" in f for f in row["flags"])


def test_print_kernel_stats_greedy_bench_fail_row(capsys):
    """Degraded kernel table: ``bench=None`` + ``greedy_fail`` prints the greedy kernels
    with ``--`` timings, a ``bench_fail`` TOTAL, the failure reason as a ``!`` line, and the
    pinned rows still render — the exact view the golden workflow needs when the greedy pick
    hangs (previously the whole run aborted before any pinned row printed)."""
    from types import SimpleNamespace

    from emmy.commands.run import _GoldenBench, _print_kernel_stats
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    def graph_with(knobs):
        g = Graph()
        g.add_node(op=CudaOp(kernel_name="k_pinned", knobs=knobs), inputs=[], output=Tensor("o", (4,)), node_id="n0")
        return g

    greedy = Graph()
    greedy.add_node(op=CudaOp(kernel_name="k_greedy", knobs={"TILE": "w2x1/f1x8"}), inputs=[], output=Tensor("o", (4,)), node_id="n0")
    pinned_bench = SimpleNamespace(min_ms=0.5, time_ms=0.5, per_launch=[], e2e_min_ms=None)
    ok_row = _GoldenBench(
        SimpleNamespace(name="ab TILE=w4x2/f2x4", knobs={"TILE": "w4x2/f2x4"}, shape=None, dynamic=None),
        graph_with({"TILE": "w4x2/f2x4"}),
        pinned_bench,
        [],
    )
    _print_kernel_stats(greedy, None, golden_benches=[ok_row], greedy_fail="greedy bench failed: hung")
    outp = capsys.readouterr().out
    assert "bench_fail" in outp and "greedy bench failed: hung" in outp
    assert "k_greedy" in outp and "ab TILE=w4x2/f2x4" in outp


def _iso_bench_fixtures():
    """One-kernel greedy graph + interleaved bench (500 µs) + isolated re-bench (400 µs)
    ``_GoldenBench`` — the ``greedy_iso`` display/json inputs."""
    from types import SimpleNamespace

    from emmy.commands.run import _GoldenBench
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.cuda.ir import CudaOp

    greedy = Graph()
    greedy.add_node(op=CudaOp(kernel_name="k_greedy", knobs={"TILE": "w2x1/f1x8"}), inputs=[], output=Tensor("o", (4,)), node_id="n0")
    bench = SimpleNamespace(min_ms=0.5, time_ms=0.5, per_launch=[SimpleNamespace(idx=0, samples=[0.5], time_ms=0.5)], e2e_min_ms=None)
    iso_bench = SimpleNamespace(min_ms=0.4, time_ms=0.4, per_launch=[SimpleNamespace(idx=0, samples=[0.4], time_ms=0.4)], e2e_min_ms=None)
    iso_sample = SimpleNamespace(name="greedy (isolated)", knobs={}, shape=None, dynamic=None)
    return greedy, bench, _GoldenBench(iso_sample, greedy, iso_bench, [])


def test_print_kernel_stats_greedy_isolated_rows(capsys):
    """With ``greedy_iso`` present, each greedy kernel row gets a ``greedy (isolated)``
    twin with the emmy-only re-bench's per-launch timing (same graph — indexes align 1:1),
    the pinned-comparable baseline. A failed iso re-bench prints as a ``!`` note only —
    its rows would just duplicate the greedy geometry with ``--`` timings."""
    from emmy.commands.run import _GoldenBench, _print_kernel_stats

    greedy, bench, iso = _iso_bench_fixtures()
    _print_kernel_stats(greedy, bench, golden_benches=[], greedy_iso=iso)
    outp = capsys.readouterr().out
    assert "greedy (isolated)" in outp and "400.0" in outp and "500.0" in outp

    failed = _GoldenBench(iso.sample, greedy, None, ["bench_fail: hung"], "bench_fail")
    _print_kernel_stats(greedy, bench, golden_benches=[], greedy_iso=failed)
    outp = capsys.readouterr().out
    assert "! greedy (isolated): bench_fail: hung" in outp
    assert outp.count("greedy (isolated)") == 1  # the note, no `--` twin rows


def test_write_ab_json_greedy_isolated_block(tmp_path):
    """``greedy.isolated`` rides the ``--json`` record when pinned rows benched: the
    emmy-only re-bench of the greedy graph, shaped like a pinned row (``status`` /
    ``total_us`` / ``kernels`` / ``flags``) — the block pinned ``total_us`` compares
    against, where the greedy block's own number is interleaved with torch (~7% apart)."""
    import json
    from types import SimpleNamespace

    from emmy.commands.run import _write_ab_json

    greedy, bench, iso = _iso_bench_fixtures()
    args = SimpleNamespace(
        json=str(tmp_path / "ab.json"), code="torch.matmul(a, b)", input=None, ir=None, golden=None, dynamic=None, warmup=1, iters=1
    )
    _write_ab_json(args, {}, greedy, bench, [], greedy_iso=iso)
    rec = json.loads((tmp_path / "ab.json").read_text())
    assert rec["greedy"]["total_us"] == 500.0
    iso_rec = rec["greedy"]["isolated"]
    assert iso_rec["status"] == "ok" and iso_rec["total_us"] == 400.0
    assert iso_rec["kernels"][0]["us"] == 400.0 and iso_rec["flags"] == []
