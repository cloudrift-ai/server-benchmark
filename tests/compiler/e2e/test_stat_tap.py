"""The row-statistic TAP (``PLACE@stat`` — loop-level tap fusion, the inverted resting state).

Covers the contracts the feature rests on:

- the ``RowAccum`` stmt survives the generic σ-rewrite with EVERY field intact (the
  ``RegStore.atomic`` silent-degrade bug class — a field dropped in the registered ``_rewrite``
  turns the fold numerically wrong with no loud failure);
- ``010_tap_row_stat`` fissions the assembled norm at loop level (the producer gains the
  ``__sq`` aux output + the atomic tap; the sweep drops to a ``T__sq`` reader), including the
  per-head MIXED-RADIX form (a 2-D aux buffer, no flat-address algebra), and the stamped
  structural identity stays TAP-BLIND;
- the ``fuse`` cut-out ROUND-TRIPS: the default compile of a tapped pair produces kernels
  byte-identical to a compile with the fission rule disabled (the never-fissioned pipeline) —
  the bit-parity contract that keeps every golden MATCH across the inversion;
- the pinned sink realizes (aux slot-1 output, atomic row fold, wide sweep) and stays
  numerically correct, on the pointwise host and across a split-K relocation (the tap rides the
  FINALIZE, never a partial);
- an input-norm (no in-graph producer) never taps and compiles the ordinary fused form.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context  # noqa: F401 — parity with sibling e2e modules

from ..conftest import requires_cuda

# A gelu producer: expensive enough that fusion's reduce-heavy multi-load guard keeps it a
# SEPARATE pointwise kernel, so the norm assembles standalone and the tap rule fissions it.
_SNIPPET = (
    "nw = torch.randn(1024,dtype=torch.float16)\n"
    "x = torch.randn(32,1024,dtype=torch.float16)\n"
    "F.rms_norm(F.gelu(x, approximate='tanh'), (1024,), nw)"
)
# The per-head (mixed-radix) form: the norm reads the producer's flat (32, 4096) output through
# a (32, 16, 256) view, so the tap's row coordinate is ``index // 256`` and the aux is 2-D.
_PERHEAD = (
    "nw = torch.randn(256,dtype=torch.float16)\n"
    "x = torch.randn(32,4096,dtype=torch.float16)\n"
    "F.rms_norm(F.gelu(x, approximate='tanh').view(32,16,256), (256,), nw)"
)


def _feed(graph, seed: int = 3, scale: float = 0.5) -> dict:
    rng = np.random.default_rng(seed)
    feed = {}
    for name in graph.inputs:
        shape = tuple(d.as_static() for d in graph.nodes[name].output.shape)
        feed[name] = (rng.standard_normal(shape) * scale).astype(np.float16)
    return feed


def test_rowaccum_rewrite_preserves_fields():
    """The registered ``rewrite`` handler must reconstruct EVERY field — a dropped ``dst`` or
    ``index`` entry would silently mis-bucket the row stat; the σ must reach the index exprs."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.ir.stmt import RowAccum

    s = RowAccum(dst="t__sq", index=(Var("a0"), BinaryExpr("/", Var("a1"), Literal(64, "int"))), value="v1")
    rename = {"v1": "v1__r"}
    out = s.rewrite(lambda n: rename.get(n, n), Sigma({"a0": BinaryExpr("+", Var("a0"), Literal(1, "int"))}))
    assert isinstance(out, RowAccum)
    assert out.dst == "t__sq" and out.value == "v1__r" and len(out.index) == 2
    # σ applied: the a0 coord is now (a0 + 1) — the expr must differ from the input's.
    assert out.index[0] != s.index[0] and "a0" in set(out.index[0].free_vars())


def _loop_stage(code: str):
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import LOOP_PASSES
    from emmy.compiler.pipeline.pipeline import Pipeline

    graph = graph_from_code(code)[0]
    out = Pipeline.build(LOOP_PASSES).run(graph)
    return out[0] if isinstance(out, tuple) else out


def test_tap_fission_at_loop_level():
    """The fission fires at merge-fixpoint: the producer gains the ``__sq`` aux output (slot 1)
    with the atomic tap in its body; the norm drops to a sweep reading it — and the stamped
    structural identity is TAP-BLIND (featurizing the tapped body equals the stripped body's)."""
    from emmy.compiler.ir.loop.tap import has_taps, strip_taps
    from emmy.compiler.ir.stmt import Body
    from emmy.compiler.pipeline.passes.loop.stamp._stamp import structure_features

    g = _loop_stage(_SNIPPET)
    gelu = g.nodes["gelu"]
    assert [t.name for t in gelu.outputs] == ["gelu", "gelu__sq"]
    assert has_taps(gelu.op.body)
    sweep = g.nodes["rms_norm"]
    assert "gelu__sq" in sweep.inputs
    assert not any(s.is_reduce for s in sweep.op.body.loops), "the sweep must carry no statistic reduce"
    assert structure_features(gelu.op.body) == structure_features(Body(strip_taps(gelu.op.body)))


def test_tap_fission_perhead_mixed_radix():
    """The per-head norm over a flattened axis taps into a 2-D aux buffer — the row coordinate
    derives positionally (``index // W``), no flat-address bijection proof."""
    g = _loop_stage(_PERHEAD)
    gelu = g.nodes["gelu"]
    assert [t.name for t in gelu.outputs] == ["gelu", "gelu__sq"]
    assert tuple(d.as_static() for d in gelu.outputs[1].shape) == (32, 16)


def test_input_norm_never_taps():
    """An input-norm has no in-graph producer — the fission never matches and the graph carries
    no ``__sq`` buffer."""
    code = "nw = torch.randn(1024,dtype=torch.float16)\nx = torch.randn(32,1024,dtype=torch.float16)\nF.rms_norm(x, (1024,), nw)"
    g = _loop_stage(code)
    assert not [t.name for n in g.nodes.values() for t in n.outputs if t.name.endswith("__sq")]


def _cuda_sources(code: str, *, drop_rule: str | None = None) -> dict[str, str]:
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import CUDA_PASSES
    from emmy.compiler.pipeline.pipeline import _PASSES_DIR, Pipeline, _strip_rule_prefix

    graph = graph_from_code(code)[0]
    select = None
    if drop_rule is not None:
        select = set()
        for name in CUDA_PASSES:
            for f in (_PASSES_DIR / name).glob("*.py"):
                if f.name != "__init__.py" and not f.name.startswith("_"):
                    select.add(_strip_rule_prefix(f.stem))
        select -= {drop_rule}
    g = Pipeline.build(CUDA_PASSES, select=select).run(graph)
    if isinstance(g, tuple):
        g = g[0]
    return {nid: n.op.kernel_source for nid, n in g.nodes.items() if getattr(n.op, "kernel_source", None)}


@requires_cuda
@pytest.mark.parametrize("code", [_SNIPPET, _PERHEAD], ids=["plain", "perhead"])
def test_fuse_cutout_round_trip_bit_parity(code):
    """THE round-trip contract: the default (option-0 ``fuse``) compile of a tapped pair — tap
    fused at loop level, peeled at recognition, cut back out at tile lowering — produces kernels
    BYTE-IDENTICAL to the never-fissioned pipeline (the fission rule filtered out). This is what
    keeps every golden MATCH and every kernel name stable across the resting-state inversion."""
    baseline = _cuda_sources(code, drop_rule="tap_row_stat")
    fissioned = _cuda_sources(code)
    assert set(baseline) == set(fissioned), f"kernel sets differ: {set(baseline) ^ set(fissioned)}"
    for nid in baseline:
        assert baseline[nid] == fissioned[nid], f"kernel {nid!r} diverged from the never-fissioned compile"


@requires_cuda
def test_stat_tap_pinned_sink_e2e(monkeypatch):
    """Pinned sink on the gelu→rms_norm pair: the producer keeps the ``__sq`` aux output (slot 1)
    with the atomic row fold attached, the norm deploys as a wide pointwise sweep, and the output
    matches the fp32 reference."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("EMMY_PLACE@STAT", "sink")
    graph = graph_from_code(_SNIPPET)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    bufs = [buf for node in compiled.nodes.values() for buf in node.buffer_names()]
    sq = [b for b in bufs if b.endswith("__sq")]
    assert sq, f"no __sq aux buffer — the sink did not realize (buffers: {bufs})"
    producer = compiled.producer(sq[0])
    assert len(producer.outputs) == 2 and producer.buffer_names()[1] == sq[0], "the row stat must be output slot 1 of its producer"
    assert "atomicAdd" in producer.op.kernel_source, "the producer fold must accumulate the row stat atomically"
    feed = _feed(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]], dtype=np.float32)
    x, nw = feed["x"].astype(np.float32), feed["nw"].astype(np.float32)
    y = 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    ref = y / np.sqrt((y * y).mean(axis=-1, keepdims=True) + 1e-6) * nw
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=5e-2, atol=5e-2)


@requires_cuda
def test_stat_tap_sink_relocates_across_split(monkeypatch):
    """Sink + a deferred split-K pin on a linear→rms_norm(+residual) pair: ``030_split_reduce``
    relocates the tap onto the FINALIZE (the partial carries none), the aux buffer stays output
    slot 1 of the finalize, and numerics hold — the split-K sites the old realizer refused."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("EMMY_PLACE@STAT", "sink")
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    code = (
        "nw = torch.randn(1024,dtype=torch.float16)\n"
        "w = torch.randn(1024,1024,dtype=torch.float16)\n"
        "r = torch.randn(32,1024,dtype=torch.float16)\n"
        "x = torch.randn(32,1024,dtype=torch.float16)\n"
        "F.rms_norm(F.linear(x, w), (1024,), nw) + r"
    )
    graph = graph_from_code(code)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    sq = [b for n in compiled.nodes.values() for b in n.buffer_names() if b.endswith("__sq")]
    assert sq, "the relocated tap must survive the split"
    fin = compiled.producer(sq[0])
    assert not fin.id.endswith("__partial"), "the tap must ride the finalize, never the partial"
    assert len(fin.outputs) == 2
    feed = _feed(graph, seed=5, scale=0.1)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]], dtype=np.float32)
    named = {n: feed[n].astype(np.float32) for n in graph.inputs}
    lin = named["x"] @ named["w"].T
    ref = lin / np.sqrt((lin * lin).mean(axis=-1, keepdims=True) + 1e-6) * named["nw"] + named["r"]
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=8e-2, atol=8e-2)


@requires_cuda
def test_stat_tap_refuses_input_norm(monkeypatch):
    """An input-norm has no in-graph producer — under the global sink pin nothing taps (no
    ``__sq`` node) and the kernel compiles/runs in its ordinary fused form."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("EMMY_PLACE@STAT", "sink")
    code = "nw = torch.randn(1024,dtype=torch.float16)\nx = torch.randn(32,1024,dtype=torch.float16)\nF.rms_norm(x, (1024,), nw)"
    graph = graph_from_code(code)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    all_bufs = [buf for node in compiled.nodes.values() for buf in node.buffer_names()]
    assert not [b for b in all_bufs if b.endswith("__sq")], "input-norm must not tap"
    feed = _feed(graph, seed=4)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]], dtype=np.float32)
    x, nw = feed["x"].astype(np.float32), feed["nw"].astype(np.float32)
    ref = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6) * nw
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=5e-2, atol=5e-2)
