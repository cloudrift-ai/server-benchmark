"""Pack save/load on a real GPU (``backend/pack.py``): a compiled program stored as a
binary-keyed plan must load without its sources and produce identical outputs; any
disqualifier (key mismatch, evicted cubin) must return ``None`` — the recompile fallback."""

import json

import numpy as np
import pytest

from emmy.compiler.backend.cuda.program import CompiledProgram
from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.pack import load_pack, pack_path, save_pack
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

from ..conftest import requires_cuda

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]

_KEY = {"kind": "test", "model": "test/pack-model", "max_seq_len": 32}


def _plan():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (32, 64)), node_id="x")
    g.add_node(op=ElementwiseOp(op="exp"), inputs=["x"], output=Tensor("y", (32, 64)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return plan_from_graph(Pipeline.build(CUDA_PASSES).run(g))


def _run(plan, x):
    with gpu_lock():
        prog = CompiledProgram.build_from_plan(plan, {"x": x})
        prog.run_once()
        return prog.outputs()["y"]


def test_pack_round_trip_runs_identically(tmp_path):
    plan = _plan()
    x = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    ref = _run(plan, x)

    pdir = pack_path(tmp_path, _KEY)
    save_pack(pdir, {"trunk": plan}, key=_KEY)
    loaded = load_pack(pdir, key=_KEY)
    assert loaded is not None
    stored = loaded["trunk"]
    # The stored plan is binary-keyed with no sources — the boot that never runs codegen.
    for spec in stored.kernels.values():
        assert spec.source is None and spec.binary_key
    np.testing.assert_allclose(_run(stored, x), ref)
    np.testing.assert_allclose(ref, np.exp(x), rtol=1e-3, atol=1e-4)


def test_pack_key_mismatch_falls_back(tmp_path):
    pdir = pack_path(tmp_path, _KEY)
    save_pack(pdir, {"trunk": _plan()}, key=_KEY)
    assert load_pack(pdir, key={**_KEY, "max_seq_len": 64}) is None


def test_pack_path_separates_environments(tmp_path, monkeypatch):
    """Two lanes differing only in ENVIRONMENT must not share a directory. The precision gate
    is the live case: ``FAST_MATH`` changes which kernel forks the compile enumerates but not
    the serving-shape key, so a shared path let the second warm silently overwrite the first
    (found baking a multi-shape image, 2026-08-02: 5 directories for 8 shapes, and the pinned
    boot then mismatched its own pack)."""
    from emmy.compiler.pipeline.search.space import FAST_MATH

    std = pack_path(tmp_path, _KEY)
    monkeypatch.setenv(f"EMMY_{FAST_MATH.name}", "1")
    fm = pack_path(tmp_path, _KEY)
    assert std != fm, "the precision gate must reach the pack directory, not only the manifest"
    # ...and each still round-trips under its own lane.
    save_pack(fm, {"trunk": _plan()}, key=_KEY)
    assert load_pack(fm, key=_KEY) is not None


def test_pack_missing_cubin_falls_back(tmp_path, monkeypatch):
    # A private cubin cache so evicting the pack's cubins can't race parallel test workers.
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path / "cubin"))
    pdir = pack_path(tmp_path, _KEY)
    save_pack(pdir, {"trunk": _plan()}, key=_KEY)
    manifest = json.loads((pdir / "manifest.json").read_text())
    assert load_pack(pdir, key=_KEY) is not None
    for f in (tmp_path / "cubin").glob("*.cubin"):
        f.unlink()
    assert manifest["programs"], "sanity: pack stored at least one program"
    assert load_pack(pdir, key=_KEY) is None
