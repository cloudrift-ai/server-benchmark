from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler import dtype
from emmy.compiler.backend.cuda.program import CompiledProgram
from emmy.compiler.backend.plan import BufferSpec
from emmy.compiler.dim import Dim


def _program(*, tma=False):
    bufs = [
        BufferSpec("x", (Dim(2), Dim(3)), dtype.F32, "input"),
        BufferSpec("tmp", (Dim(2), Dim(3)), dtype.F32, "scratch"),
        BufferSpec("y", (Dim(2), Dim(3)), dtype.F32, "output"),
    ]
    compiled = SimpleNamespace(
        bufs=bufs,
        buf_by_name={buf.name: buf for buf in bufs},
        launches=[SimpleNamespace(tma_descriptors=(object(),) if tma else ())],
    )
    arrays = {buf.name: np.zeros((2, 3), dtype=np.float32) for buf in bufs}
    return CompiledProgram(compiled=compiled, arrays=arrays, descs={})


def test_run_once_external_binds_exact_device_arrays_only_for_the_launch(monkeypatch):
    program = _program()
    original = dict(program.arrays)
    external_x = np.ones((2, 3), dtype=np.float32)
    external_y = np.empty((2, 3), dtype=np.float32)
    seen = []

    def launch():
        seen.append((program.arrays["x"], program.arrays["y"]))

    monkeypatch.setattr(program, "run_once", launch)
    program.run_once_external({"x": external_x, "y": external_y})

    assert seen == [(external_x, external_y)]
    assert program.arrays == original


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("missing", np.empty((2, 3), dtype=np.float32), "unknown buffer"),
        ("tmp", np.empty((2, 3), dtype=np.float32), "expected input/output"),
        ("x", np.empty((3, 2), dtype=np.float32), r"expected \(2, 3\)"),
        ("x", np.empty((2, 3), dtype=np.float16), "expected float32"),
        ("x", np.empty((2, 6), dtype=np.float32)[:, ::2], "C-contiguous"),
    ],
)
def test_run_once_external_rejects_incompatible_bindings(name, value, error):
    with pytest.raises((KeyError, ValueError), match=error):
        _program().run_once_external({name: value})


def test_run_once_external_rejects_tma_pointer_rebinding():
    with pytest.raises(ValueError, match="TMA descriptors"):
        _program(tma=True).run_once_external({"x": np.empty((2, 3), dtype=np.float32)})
