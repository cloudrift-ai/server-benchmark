"""Cross-process source determinism — the contract the content-addressed cubin cache
(``nvcc._cache_key``: sha1 over source among others) stands on.

Rendered kernel source must be byte-identical across interpreter launches: an
address- or seed-derived token in the text re-keys the kernel on every boot, which
silently defeats the on-disk cubin cache (each server start recompiles) and makes a
prebuilt-cache image impossible. Found on a real RTX 5090 release run: ~270 of ~580
serving kernels re-keyed per boot — the vectorized-store temp name embedded
``id(self)``. The compile here runs in two SUBPROCESSES (fresh address space and hash
seed — an in-process double compile cannot see this class of bug), off-GPU (CUDA
hidden; sources render without a device), and the traced pointwise reshape is pinned
to the production ``f2`` register-strip schedule so it must emit a vectorized store.
The explicit pin keeps coverage independent of deploy-policy and tune-database picks.
"""

import subprocess
import sys

_SNIPPET = """
import hashlib
import torch
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.trace.torch import trace_module

class M(torch.nn.Module):
    def forward(self, x):
        return (x + 1).reshape(8, 4, 16)

g = trace_module(M().eval(), (torch.zeros(8, 64),))
with pinned_knobs({"TILE": "f2"}):
    c = CudaBackend().compile(g)
for _nid, node in sorted(c.nodes.items()):
    src = getattr(node.op, "kernel_source", None)
    if src:
        marker = "VS" if "_vs_" in src else "--"
        print(node.op.kernel_name, marker, hashlib.sha1(src.encode()).hexdigest())
"""


def _render_once(tmp_path, tag):
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONHASHSEED", None)  # each subprocess gets its own hash seed — the point
    out = subprocess.run([sys.executable, "-c", _SNIPPET], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, f"render {tag} failed: {out.stderr[-800:]}"
    return out.stdout


def test_kernel_source_identical_across_processes(tmp_path):
    a = _render_once(tmp_path, "a")
    b = _render_once(tmp_path, "b")
    assert "VS" in a, "the traced module no longer emits a vectorized store — the covered path is gone, repick the module"
    assert a == b, f"kernel sources differ across processes (cache re-keys every boot):\n--- a\n{a}\n--- b\n{b}"
