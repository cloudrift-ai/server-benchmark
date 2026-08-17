"""Tests for incremental Loop IR body construction."""

from emmy.compiler.ir.loop.builder import LoopBuilder
from emmy.compiler.ir.loop.ir import Assign, Axis, Loop, Scope


class _CountingSet(set[str]):
    """Set that exposes how many candidate names ``fresh`` probes."""

    def __init__(self, values=()) -> None:
        super().__init__(values)
        self.probes = 0

    def __contains__(self, value) -> bool:
        self.probes += 1
        return super().__contains__(value)


def test_fresh_advances_suffix_cursor_without_changing_spelling() -> None:
    builder = LoopBuilder({"value"})
    used = _CountingSet(builder._used)
    builder._used = used

    names = [builder.fresh("value") for _ in range(1_000)]

    assert names[:3] == ["value_s1", "value_s2", "value_s3"]
    assert names[-1] == "value_s1000"
    assert used.probes == 2_000


def test_fresh_cursor_skips_names_reserved_by_another_hint() -> None:
    builder = LoopBuilder({"value"})

    assert builder.fresh("value") == "value_s1"
    assert builder.fresh("value_s2") == "value_s2"
    assert builder.fresh("value") == "value_s3"


def test_finish_retains_prepend_order_across_scopes() -> None:
    a0 = Axis("a0", 4)
    a1 = Axis("a1", 8)
    builder = LoopBuilder(set())

    builder.insert(Assign("root_0", "copy", ("x",)), Scope())
    builder.insert(Assign("inner_0", "copy", ("x",)), Scope((a0, a1)))
    builder.insert(Assign("root_1", "copy", ("x",)), Scope())
    builder.insert(Assign("inner_1", "copy", ("x",)), Scope((a0, a1)))

    assert builder.finish() == (
        Assign("root_1", "copy", ("x",)),
        Loop(
            a0,
            (
                Loop(
                    a1,
                    (
                        Assign("inner_1", "copy", ("x",)),
                        Assign("inner_0", "copy", ("x",)),
                    ),
                ),
            ),
        ),
        Assign("root_0", "copy", ("x",)),
    )
