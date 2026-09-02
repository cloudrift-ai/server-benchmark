"""Loop body construction and fresh-name allocation."""

from emmy.compiler.ir.loop.builder import LoopBuilder


class _LookupBudget(set):
    """Set whose deterministic probe budget catches suffix rescans without wall-clock assertions."""

    def __init__(self, budget: int):
        super().__init__()
        self.budget = budget
        self.lookups = 0

    def __contains__(self, value):
        self.lookups += 1
        if self.lookups > self.budget:
            raise AssertionError(f"fresh-name allocation exceeded {self.budget} membership probes")
        return super().__contains__(value)


def test_many_same_hint_allocations_stay_within_a_linear_probe_budget():
    count = 50_000
    builder = LoopBuilder(set())
    used = _LookupBudget(3 * count)
    builder._used = used

    last = None
    for _ in range(count):
        last = builder.fresh("v")

    assert last == f"v_s{count - 1}"
    assert used.lookups < 2 * count


def test_fresh_names_keep_lowest_available_suffix_across_hint_collisions():
    builder = LoopBuilder({"v", "v_s1", "v_s3"})

    assert builder.fresh("v") == "v_s2"
    assert builder.fresh("v_s4") == "v_s4"
    assert builder.fresh("v") == "v_s5"


def test_insert_descends_by_axis_and_prepends_at_the_leaf():
    """The builder's tree contract: descent creates missing Loops, a repeated scope reuses its
    Loop, and each level keeps prepend order — newest first, a fresh nested Loop ahead of the
    stmts inserted before it."""
    from emmy.compiler.ir.loop.ir import Axis, Loop, Scope
    from emmy.compiler.ir.stmt.leaves import Assign

    a, b = Axis("a", 4), Axis("b", 8)
    builder = LoopBuilder(set())
    builder.insert(Assign(name="v0", op="copy", args=("x",)), Scope(enclosing=(a,)))
    builder.insert(Assign(name="v1", op="copy", args=("x",)), Scope(enclosing=(a, b)))
    builder.insert(Assign(name="v2", op="copy", args=("x",)), Scope(enclosing=(a,)))
    builder.insert(Assign(name="v3", op="copy", args=("x",)), Scope(enclosing=()))

    body = builder.finish()
    assert [type(s).__name__ for s in body] == ["Assign", "Loop"]
    assert body[0].name == "v3"
    outer = body[1]
    assert isinstance(outer, Loop) and outer.axis == a
    assert [getattr(s, "name", type(s).__name__) for s in outer.body] == ["v2", "Loop", "v0"]
    inner = outer.body[1]
    assert inner.axis == b and [s.name for s in inner.body] == ["v1"]
