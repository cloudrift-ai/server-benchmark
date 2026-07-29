"""The row-statistic sink (``PLACE@stat=sink`` — ``025_sink_row_reduce``).

Covers the three contracts the feature rests on:

- the ``RowAccum`` stmt survives the generic σ-rewrite with EVERY field intact (the
  ``RegStore.atomic`` silent-degrade bug class — a field dropped in the registered ``_rewrite``
  turns the epilogue numerically wrong with no loud failure);
- the pinned e2e realizes the sink (producer gains the ``__sq`` aux output + the atomic
  row-fold epilogue; the norm re-emits as a wide sweep) and stays numerically correct;
- an input-norm (no in-graph producer) under the same pin refuses the sink and compiles the
  ordinary fused form.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.context import Context  # noqa: F401 — parity with sibling e2e modules

from ..conftest import requires_cuda

# A gelu producer: expensive enough that fusion's reduce-heavy multi-load guard keeps it a
# SEPARATE pointwise kernel (a cold matmul producer may deploy unsplit — no plain store to sink
# into — so a matmul pair would need environment-dependent split pins).
_SNIPPET = (
    "nw = torch.randn(1024,dtype=torch.float16)\n"
    "x = torch.randn(32,1024,dtype=torch.float16)\n"
    "F.rms_norm(F.gelu(x, approximate='tanh'), (1024,), nw)"
)


def test_rowaccum_rewrite_preserves_fields():
    """The registered ``rewrite`` handler must reconstruct EVERY field — ``n`` or ``dst`` dropped
    would silently mis-bucket the row stat; the σ must reach the ``flat`` expr."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.ir.stmt import RowAccum

    s = RowAccum(dst="t__sq", flat=BinaryExpr("+", Var("a1"), BinaryExpr("*", Var("a0"), Literal(64, "int"))), n=64, value="v1")
    rename = {"v1": "v1__r"}
    out = s.rewrite(lambda n: rename.get(n, n), Sigma({"a0": BinaryExpr("+", Var("a0"), Literal(1, "int"))}))
    assert isinstance(out, RowAccum)
    assert out.dst == "t__sq" and out.n == 64 and out.value == "v1__r"
    # σ applied: the a0 term is now (a0 + 1) * 64 — the expr must differ from the input's.
    assert out.flat != s.flat and "a0" in set(out.flat.free_vars())


def test_sinkable_stat_binding_gates():
    """The structural probe: a well-formed norm form binds; breaking any gate (non-additive fold,
    a second trailing loop, a symbolic reduce) declines."""
    from emmy.compiler.pipeline.passes.lowering.tile._sink import bind_sinkable_stat

    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
    from emmy.compiler.ir.tile import Map, Reduction
    from emmy.compiler.tensor import Tensor

    m, n = Axis(name="a0", extent=Dim(32)), Axis(name="a1", extent=Dim(64))
    load = Load(name="x0", input="t", index=(Var("a0"), Var("a1")))
    sq = Assign(name="v0", op=ElementwiseImpl("multiply"), args=("x0", "x0"))
    acc = Accum(name="acc0", value="v0", op=ElementwiseImpl("add"), axes=("a1",))
    red = Reduction(carrier=acc.as_carrier(), axis=n, partial=Body((load, sq, acc)), role=AxisRole.PLANAR)
    sweep = Loop(
        axis=n,
        body=Body((Load(name="x1", input="t", index=(Var("a0"), Var("a1"))), Write(output="o", index=(Var("a0"), Var("a1")), value="x1"))),
    )
    node = Map(body=Body((sweep,)), sources=(red,))
    inputs = {"t": Tensor("t", (Dim(32), Dim(64)), F16)}
    b = bind_sinkable_stat(node, (m,), inputs)
    assert b is not None and b.src == "t" and b.n == 64 and b.rows == 32 and b.acc == "acc0"
    # A max fold is not additive — decline.
    acc_max = Accum(name="acc0", value="v0", op=ElementwiseImpl("maximum"), axes=("a1",))
    red_max = Reduction(carrier=acc_max.as_carrier(), axis=n, partial=Body((load, sq, acc_max)), role=AxisRole.PLANAR)
    assert bind_sinkable_stat(Map(body=Body((sweep,)), sources=(red_max,)), (m,), inputs) is None
    # A second trailing loop strands the re-emitted sweep on the row-only grid — decline.
    assert bind_sinkable_stat(Map(body=Body((sweep, sweep)), sources=(red,)), (m,), inputs) is None


@requires_cuda
def test_stat_sink_pinned_e2e(monkeypatch):
    """Pinned sink on a linear→rms_norm pair: the producer gains the ``__sq`` aux write (atomic
    row fold, zero-init'd per launch), the norm re-emits as a wide pointwise sweep, and the
    output matches the fp32 reference."""
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
    sources = {nid: node.op.kernel_source for nid, node in compiled.nodes.items() if getattr(node.op, "kernel_source", None)}
    producer_src = next((s for s in sources.values() if sq[0] in s), "")
    assert "atomicAdd" in producer_src, "the producer epilogue must accumulate the row stat atomically"
    rng = np.random.default_rng(3)
    feed = {}
    for name in graph.inputs:
        shape = tuple(d.as_static() for d in graph.nodes[name].output.shape)
        feed[name] = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    out_name = graph.outputs[0]
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[out_name], dtype=np.float32)
    by_shape = {feed[n].shape: feed[n].astype(np.float32) for n in graph.inputs}
    x, nw = by_shape[(32, 1024)], by_shape[(1024,)]
    y = 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    ref = y / np.sqrt((y * y).mean(axis=-1, keepdims=True) + 1e-6) * nw
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=5e-2, atol=5e-2)


@requires_cuda
def test_stat_sink_refuses_input_norm(monkeypatch):
    """An input-norm has no in-graph producer — under the global sink pin the realizer refuses
    (no ``__sq`` node) and the kernel compiles/runs in its ordinary fused form."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("EMMY_PLACE@STAT", "sink")
    code = "nw = torch.randn(1024,dtype=torch.float16)\nx = torch.randn(32,1024,dtype=torch.float16)\nF.rms_norm(x, (1024,), nw)"
    graph = graph_from_code(code)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    all_bufs = [buf for node in compiled.nodes.values() for buf in node.buffer_names()]
    assert not [b for b in all_bufs if b.endswith("__sq")], "input-norm must not sink"
    rng = np.random.default_rng(4)
    feed = {}
    for name in graph.inputs:
        shape = tuple(d.as_static() for d in graph.nodes[name].output.shape)
        feed[name] = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    out_name = graph.outputs[0]
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[out_name], dtype=np.float32)
    by_shape = {feed[n].shape: feed[n].astype(np.float32) for n in graph.inputs}
    x, nw = by_shape[(32, 1024)], by_shape[(1024,)]
    ref = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6) * nw
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=5e-2, atol=5e-2)
