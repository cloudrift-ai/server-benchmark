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
