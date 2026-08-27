"""``BinaryExpr.simplify`` over a pair-packed weight address: separating the row axis from ``k``.

An NVFP4 weight constant stores two 4-bit codes per byte, so the loader spells its address as the
FLAT element offset divided by 2 and wrapped back into the packed row: ``((n·K + k) / 2) % (K/2)``.
Read literally that expression carries ``n`` inside a division, which makes ``n`` unrecoverable to
every consumer that asks "does this index still mention the row axis outside a div/mod" — the
``loop/canonicalize`` axis re-fusion asks exactly that, and its refusal leaves a packed matmul
without a contraction to bind.

The value does separate: ``n·K`` is a multiple of the divisor, so the quotient is ``n·(K/2) + k/2``
and the modulo keeps only ``k/2``. What the fold needs from the loop extents is NON-NEGATIVITY, not a
tight bound — a negative row index would make the floor division round the wrong way. These tests pin
the separation, its value-equivalence at ranges wider than the declared one, and the refusal when no
range proves the sign.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expr import BinaryExpr, Interval, Literal, SimplifyCtx, Var

K, N, BLOCK = 4096, 12288, 16  # reduce extent, row extent, NVFP4 scale block


def _lit(v: int) -> Literal:
    return Literal(v, "int")


def _ctx(k_hi: int = K - 1) -> SimplifyCtx:
    return SimplifyCtx({"n": Interval(0, N - 1), "k": Interval(0, k_hi)}, {})


def _flat() -> BinaryExpr:
    """``n·K + k`` — the unpacked element offset the packed address is built from."""
    return BinaryExpr("+", BinaryExpr("*", Var("n"), _lit(K)), Var("k"))


def _same_value(a, b, ctx_desc: str) -> None:
    for nv in (0, 1, 7, 1023, N - 1):
        for kv in (0, 1, 2, 15, 16, 4095):
            env = {"n": nv, "k": kv}
            assert a.eval(env) == b.eval(env), f"{ctx_desc}: mismatch at n={nv} k={kv}"


def test_packed_modulo_drops_the_row_axis():
    """``((n·K + k) / 2) % (K/2) → k/2``: the row axis leaves the expression entirely."""
    e = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    s = e.simplify(_ctx())
    assert "n" not in s.free_vars(), f"row axis survived: {s.pretty()}"
    _same_value(e, s, "packed modulo")


def test_packed_quotient_is_the_row_axis():
    """The companion division ``((n·K + k) / 2) / (K/2) → n`` — the same decomposition read
    from its quotient side."""
    e = BinaryExpr("/", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    s = e.simplify(_ctx())
    assert s == Var("n"), f"expected the bare row axis, got {s.pretty()}"


def test_scale_block_index_drops_the_row_axis():
    """The e4m3 block-scale companion ``((n·K + k) / BLOCK) % (K/BLOCK)`` separates the same way —
    one packed weight feeds both addresses, so both must fold or neither helps."""
    e = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(BLOCK)), _lit(K // BLOCK))
    s = e.simplify(_ctx())
    assert "n" not in s.free_vars(), f"row axis survived: {s.pretty()}"
    _same_value(e, s, "block-scale index")


def test_row_major_flatten_of_the_packed_load_loses_its_residue():
    """The whole question `_access_ok` asks: flatten ``w[n, packed]`` over shape ``(N, K/2)`` and
    check that no division holds the row axis. ``n·(K/2) + ((n·K + k)/2) % (K/2)``."""
    packed = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    flat = BinaryExpr("+", BinaryExpr("*", Var("n"), _lit(K // 2)), packed)
    s = flat.simplify(_ctx())
    assert "%" not in s.pretty(), f"residue survived: {s.pretty()}"
    _same_value(flat, s, "row-major flatten")


def test_no_range_does_not_fold():
    """With no range at all the operands are not provably non-negative, so the fold must decline
    rather than commit to a rounding direction it cannot justify."""
    e = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    assert e.simplify(SimplifyCtx.empty()) == e


def test_signed_row_axis_does_not_fold():
    """A row index that may go negative rounds the floor division the other way; the fold declines
    on the sign alone, which is the only thing it asks the ranges for."""
    e = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    signed = SimplifyCtx({"n": Interval(-1, N - 1), "k": Interval(0, K - 1)}, {})
    assert e.simplify(signed) == e


@pytest.mark.parametrize("k_hi", [K - 1, K, K + 7, 3 * K])
def test_wider_reduce_range_stays_value_exact(k_hi: int):
    """A ``k`` range wider than the row stride does not have to defeat the fold — the row term is a
    multiple of the divisor either way — but whatever comes back must agree with the original over
    the range it was folded under."""
    e = BinaryExpr("%", BinaryExpr("/", _flat(), _lit(2)), _lit(K // 2))
    s = e.simplify(_ctx(k_hi=k_hi))
    for nv in (0, 1, 2, 37):
        for kv in (0, 1, min(K - 1, k_hi), k_hi):
            env = {"n": nv, "k": kv}
            assert e.eval(env) == s.eval(env), f"k_hi={k_hi}: mismatch at n={nv} k={kv}: {s.pretty()}"
