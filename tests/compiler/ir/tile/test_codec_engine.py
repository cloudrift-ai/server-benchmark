"""The generic codec engine — desugar, decode/encode round-trips, order-free binding.

These exercise :mod:`emmy.compiler.ir.schedule` directly on synthetic schemas (the real
``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC`` schemas are covered through their IR codec classes).
The contract: every malformed input raises ``ValueError`` (never another type), the wire format is
byte-identical across ``decode``→``encode``, and adding a param is append-compatible
(``w2x2`` ≡ ``w:2x2``, ``w2x2:q8`` ≡ ``w:2x2,q8``).
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.schedule import (
    Emit,
    Field,
    FieldKind,
    Schema,
    decode,
    desugar,
    encode,
)

# A synthetic schema spanning every kind: a TUPLE arity-1 with a suffix enum (g), a TUPLE arity-2
# with trailing suppression (n) and without (w), a CHOICE (transport), a FLAG (ring), a GROUP (p
# with a named int param q). Mirrors the real codecs' shapes without importing them.
_G = Field("g", FieldKind.TUPLE, suffix=(("a", "atomic"), ("k", "kernel")), suffix_default="kernel")
_N = Field("n", FieldKind.TUPLE, arity=2)
_W = Field("w", FieldKind.TUPLE, arity=2, suppress_trailing=False)
_T = Field("t", FieldKind.CHOICE, choices=(("sync", "sync"), ("cp", "cp.async")), default="sync")
_R = Field("ring", FieldKind.FLAG, emit=Emit.TRUE)
_Q = Field("q", FieldKind.TUPLE)
_P = Field("p", FieldKind.GROUP, params=(_Q,))

SCH = Schema("TEST", (_G, _N, _W, _T, _R, _P))

# A second schema for the ALWAYS-emit fields (mirrors STAGE's d<depth>/transport, always spelled).
_D = Field("d", FieldKind.TUPLE, emit=Emit.ALWAYS)
_TA = Field("t", FieldKind.CHOICE, choices=(("sync", "sync"), ("cp", "cp.async")), default="sync", emit=Emit.ALWAYS)
ALWAYS_SCH = Schema("ALW", (_D, _TA))


@pytest.mark.parametrize(
    ("glued", "canonical"),
    [
        ("w2x2", "w:2x2"),
        ("g2a", "g:2a"),
        ("d3", "d:3"),
        ("p2:q8", "p:2,q8"),
        ("cp", "cp"),  # bare CHOICE token — no glued value, unchanged
        ("ring", "ring"),  # bare FLAG token — unchanged
        ("a:mma_m16n8k16_f16_f32", "a:mma_m16n8k16_f16_f32"),  # already colon-form
    ],
)
def test_desugar(glued: str, canonical: str) -> None:
    assert desugar(glued) == canonical


@pytest.mark.parametrize(
    "spec",
    ["", "g2a", "g2k", "n4x2", "w2x2", "cp", "ring", "g4a/w2x2/cp/ring", "p2", "p2:q8"],
)
def test_round_trip(spec: str) -> None:
    assert encode(SCH, decode(SCH, spec)) == spec


@pytest.mark.parametrize("spec", ["d1/sync", "d3/cp"])
def test_always_emit_round_trip(spec: str) -> None:
    # ALWAYS fields (STAGE-style d/transport) render even at their defaults — canonical is "d1/sync".
    assert encode(ALWAYS_SCH, decode(ALWAYS_SCH, spec)) == spec
    assert encode(ALWAYS_SCH, decode(ALWAYS_SCH, "")) == "d1/sync"


def test_order_free_binding() -> None:
    # Fields decode by token regardless of input order; encode re-emits in schema order.
    assert encode(SCH, decode(SCH, "cp/w2x2/g4a")) == "g4a/w2x2/cp"


def test_glued_equals_colon_form() -> None:
    # w2x2 ≡ w:2x2, and appending a param is the extension path.
    assert decode(SCH, "w2x2") == decode(SCH, "w:2x2")
    assert decode(SCH, "p2:q8") == decode(SCH, "p2:q8")


def test_suffix_enum_decodes_and_defaults() -> None:
    assert decode(SCH, "g2a")["g"] == (2, "atomic")
    assert decode(SCH, "g2")["g"] == (2, "kernel")  # default suffix when the letter is omitted


def test_choice_and_flag() -> None:
    assert decode(SCH, "cp")["t"] == "cp.async"
    assert decode(SCH, "ring")["ring"] is True
    assert decode(SCH, "")["ring"] is False
    assert decode(SCH, "")["t"] == "sync"  # CHOICE default


def test_group_positional_and_named() -> None:
    g = decode(SCH, "p2:q8")["p"]
    assert g[""] == 2 and g["q"] == 8


@pytest.mark.parametrize("spec", ["g0", "n0x4", "w2x2x2", "zzz", "p2:bad9"])
def test_malformed_raises_value_error(spec: str) -> None:
    with pytest.raises(ValueError, match="TEST"):
        decode(SCH, spec)


@pytest.mark.parametrize("spec", ["g2/g4", "sync/cp", "w2x2/cp/w4x4", "ring/ring"])
def test_repeated_field_raises(spec: str) -> None:
    """Binding is order-free, so a repeated token has no last-one-wins reading a caller could
    have meant: ``sync/cp`` would silently deploy the cp.async pipeline a ``sync`` pin named."""
    with pytest.raises(ValueError, match="spelled twice"):
        decode(SCH, spec)


def test_absent_fields_take_their_declared_defaults() -> None:
    """No field is mandatory: an absent node decodes to the kind's default, which is what lets a
    codec add a field without invalidating every stored value spelled before it."""
    v = decode(SCH, "")
    assert v == {"g": (1, "kernel"), "n": (1, 1), "w": (1, 1), "t": "sync", "ring": False, "p": None}


def test_expect_hint_is_derived_from_the_fields() -> None:
    """The parse-error hint is DERIVED — a hand-written copy falsifies itself the first time a
    field moves. Every field contributes exactly one fragment, in schema order."""
    assert SCH.expect == "expect g<n>[a|k] / n<n>[x<m>] / w<n>x<m> / sync|cp / ring / p<n>[:<param>,...]"
    assert ALWAYS_SCH.expect == "expect d<n> / sync|cp"
