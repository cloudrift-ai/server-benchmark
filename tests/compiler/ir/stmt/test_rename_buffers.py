"""``rename_buffers`` — the setter counterpart of ``external_reads`` / ``external_writes``.

What these pin: every buffer-bearing leaf renames its field (and only it), the body-level
rename reaches leaves through wrappers, and ``canonicalize_buffer_names`` (behind
``Body.structural_key``) covers EVERY external-buffer field — a kernel-stage body differing
only in a staged source buffer's name must key identically."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write, ZeroPrologue
from emmy.compiler.ir.stmt.normalize import canonicalize_buffer_names


def _body(x: str = "x", out: str = "o", zero: str = "acc_buf") -> Body:
    inner = Body(
        (
            Load(name="v", input=x, index=(Var("k"),)),
            Assign(name="w", op=ElementwiseImpl("relu"), args=("v",)),
            Accum(name="acc", value="w", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    return Body(
        (
            ZeroPrologue(dst=zero, words=16),
            Loop(axis=Axis("k", Dim(8)), body=inner),
            Write(output=out, index=(), value="acc"),
        )
    )


def test_body_rename_reaches_nested_leaves_and_every_buffer_field() -> None:
    renamed = _body().rename_buffers({"x": "x9", "o": "o9", "acc_buf": "z9"})
    reads = {n for s in renamed.iter() for n in s.external_reads()}
    writes = {n for s in renamed.iter() for n in s.external_writes()}
    assert reads == {"x9"} and writes == {"o9", "z9"}


def test_rename_is_identity_off_the_mapping() -> None:
    body = _body()
    assert body.rename_buffers({"unrelated": "name"}) == body


def test_canonicalization_covers_non_load_write_buffer_fields() -> None:
    a, b = _body(zero="acc_buf"), _body(zero="differently_named")
    assert canonicalize_buffer_names(a) == canonicalize_buffer_names(b)
    assert a.structural_key() == b.structural_key()
