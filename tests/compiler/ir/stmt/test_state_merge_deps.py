"""``StateMerge.deps()`` completeness — the read-set contract the chain fold / read counters rely on.

``deps`` used to return only ``state_b``, hiding any other outer name the merge program reads
(``merge`` is a ``tuple[Stmt, ...]``, not a nested ``Body``, so ``nested()`` never surfaces it —
``deps`` is the ONLY channel these reads reach consumers through). The fixed contract: ``state_b``
first (the old prefix, order preserved), then every remaining external read in first-use order;
carried state names and program-internal temps stay excluded (the ``Accum`` convention).
"""

from __future__ import annotations

from emmy.compiler.ir.stmt.algebra import StateMerge
from emmy.compiler.ir.stmt.leaves import Accum, Assign


def _merge_with_outer_read() -> StateMerge:
    # A two-component (m, l) combine whose rescale reads an OUTER name ``scale`` — the shape of
    # the under-report: ``scale`` is neither carried state, nor state_b, nor an internal temp.
    return StateMerge(
        state=("m", "l"),
        merge=(
            Assign(name="t", op="subtract", args=("m", "m__o")),
            Assign(name="t2", op="multiply", args=("t", "scale")),
            Accum(name="l", value="l__o"),
            Assign(name="m", op="maximum", args=("m", "t2")),
        ),
        state_b=("m__o", "l__o"),
    )


def test_deps_reports_outer_reads_beyond_state_b() -> None:
    deps = _merge_with_outer_read().deps()
    assert "scale" in deps, f"outer read must be reported: {deps}"


def test_deps_keeps_state_b_prefix_and_excludes_internals() -> None:
    sm = _merge_with_outer_read()
    deps = sm.deps()
    assert deps[: len(sm.state_b)] == sm.state_b, f"state_b must stay the (ordered) prefix: {deps}"
    assert "t" not in deps and "t2" not in deps, f"program-internal temps are not deps: {deps}"
    assert "m" not in deps and "l" not in deps, f"carried state lives in defines(), not deps: {deps}"
    assert set(deps) == {"m__o", "l__o", "scale"}
