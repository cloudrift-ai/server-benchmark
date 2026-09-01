from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from tests.compiler.realization import helpers


def test_built_loads_the_lowered_program_through_nvcc(monkeypatch) -> None:
    source = SimpleNamespace(copy=lambda: "source-copy")
    case = SimpleNamespace(pinned={}, record=SimpleNamespace(target_program=source))
    lowered = object()
    seen = []

    monkeypatch.setattr("emmy.compiler.backend.cuda.backend.CudaBackend.compile", lambda self, graph, *, ctx: lowered)
    monkeypatch.setattr("emmy.compiler.backend.cuda.program.CompiledProgram.build", lambda graph, feed: seen.append((graph, feed)))
    monkeypatch.setattr("emmy.compiler.backend.gpu_lock.gpu_lock", nullcontext)
    monkeypatch.setattr(helpers, "seeded_inputs", lambda program: {"x": program})

    assert helpers.built(case) is lowered
    assert seen == [(lowered, {"x": source})]


def test_bench_command_keeps_structural_input_pins_in_the_ab_row() -> None:
    case = SimpleNamespace(
        pinned={"FAST_MATH": False, "PLACE@a7": "cut", "WORK": "w1x1"},
        record=SimpleNamespace(name="k_example"),
        path=Path("case.yaml"),
    )

    command = helpers.bench_command(case, Path("result.json"))

    assert command[command.index("--ab") + 1] == "FAST_MATH=False,PLACE@a7=cut,WORK=w1x1"
