"""The generated exp-family algebra (``carrier.exp_merge`` / ``exp_combine_states``) must
reproduce the original hand-authored flash / online-softmax combine
programs — the frozen golden below — so the hand-authored bodies could be deleted. Compared by
value-numbering (inline temps to per-output expression trees), since the generator chooses its own
temp names and statement order.
"""

from __future__ import annotations

import pytest

from emmy.compiler.dtype import F32
from emmy.compiler.ir.pure import carrier as _carrier
from emmy.compiler.ir.pure.carrier import UnstableCarrierError, exp_combine_states, exp_merge, exp_rescale
from emmy.compiler.ir.stmt import Accum, Assign


def _exp_family(score, extra, state):
    """The (merge, combine_states, state_b) surface of one exp-family spec — what the retired
    ``Algebra.exp_family`` bundled."""
    other = tuple(f"{n}__o" for n in state)
    return exp_merge(state, (score, *extra), key=state[0]), exp_combine_states(state, other), other


_COMMUTATIVE = {"add", "multiply", "maximum"}

# FROZEN GOLDEN — the original hand-authored flash / online-softmax combine programs (the
# log-sum-exp recurrence). The spec-driven generator must reproduce these; this is the safety
# net that guards the generator after the hand-written builders were deleted. Do NOT regenerate
# these from the builders (that would make the check circular) — they are a literal snapshot.


def _t(suf: str) -> str:
    return f"m_i__{suf}"


_HAND_FLASH_MERGE = (
    Assign(_t("mx"), "maximum", ("m_i", "s_e")),
    Assign(_t("dm"), "subtract", ("m_i", _t("mx"))),
    Assign(_t("al"), "exp", (_t("dm"),)),
    Assign(_t("ds"), "subtract", ("s_e", _t("mx"))),
    Assign(_t("p"), "exp", (_t("ds"),)),
    Assign(_t("lm"), "multiply", ("l_i", _t("al"))),
    Accum(name="l_i", value=_t("p"), op="add", base=_t("lm"), dtype=F32),
    Assign(_t("om"), "multiply", ("O_i", _t("al"))),
    Assign(_t("pv"), "multiply", (_t("p"), "v_e")),
    Accum(name="O_i", value=_t("pv"), op="add", base=_t("om"), dtype=F32),
    Accum(name="m_i", value="s_e", op="maximum", dtype=F32),
)
_HAND_FLASH_COMBINE_STATES = (
    Assign(_t("cmx"), "maximum", ("m_i", "m_i__o")),
    Assign(_t("cda"), "subtract", ("m_i", _t("cmx"))),
    Assign(_t("ca"), "exp", (_t("cda"),)),
    Assign(_t("cdb"), "subtract", ("m_i__o", _t("cmx"))),
    Assign(_t("cb"), "exp", (_t("cdb"),)),
    Assign(_t("cla"), "multiply", ("l_i", _t("ca"))),
    Assign(_t("clb"), "multiply", ("l_i__o", _t("cb"))),
    Assign("l_i", "add", (_t("cla"), _t("clb"))),
    Assign(_t("coa"), "multiply", ("O_i", _t("ca"))),
    Assign(_t("cob"), "multiply", ("O_i__o", _t("cb"))),
    Assign("O_i", "add", (_t("coa"), _t("cob"))),
    Assign("m_i", "copy", (_t("cmx"),)),
)
# online softmax = flash minus the O / value-expectation channel, state (m, d), temps keyed on m.
_HAND_SOFTMAX_MERGE = (
    Assign("m__mx", "maximum", ("m", "s")),
    Assign("m__dm", "subtract", ("m", "m__mx")),
    Assign("m__al", "exp", ("m__dm",)),
    Assign("m__ds", "subtract", ("s", "m__mx")),
    Assign("m__p", "exp", ("m__ds",)),
    Assign("m__dl", "multiply", ("d", "m__al")),
    Accum(name="d", value="m__p", op="add", base="m__dl", dtype=F32),
    Accum(name="m", value="s", op="maximum", dtype=F32),
)
_HAND_SOFTMAX_COMBINE_STATES = (
    Assign("m__cmx", "maximum", ("m", "m__o")),
    Assign("m__cda", "subtract", ("m", "m__cmx")),
    Assign("m__ca", "exp", ("m__cda",)),
    Assign("m__cdb", "subtract", ("m__o", "m__cmx")),
    Assign("m__cb", "exp", ("m__cdb",)),
    Assign("m__cda2", "multiply", ("d", "m__ca")),
    Assign("m__cdb2", "multiply", ("d__o", "m__cb")),
    Assign("d", "add", ("m__cda2", "m__cdb2")),
    Assign("m", "copy", ("m__cmx",)),
)


def _canon(prog: tuple, state: tuple[str, ...]) -> dict:
    """Value-numbering equivalence: inline every temp to express each STATE-OUTPUT write as a
    canonical expression tree over the fixed input names (state = old carried value, plus
    state_b/score/value), so two programs that compute the same values compare equal regardless
    of temp names or statement order. ``copy`` is transparent; an ``Accum`` is ``op(base|name,
    value)`` with its commutative args sorted."""
    fixed = set(state)
    temp: dict[str, object] = {}  # temp name -> its Assign
    for s in prog:
        if isinstance(s, Assign) and s.name not in fixed:
            temp[s.name] = s

    def resolve(name: str):
        s = temp.get(name)
        if s is None:  # a fixed input leaf (state old-value / state_b / score / value)
            return name
        if s.op.name == "copy":
            return resolve(s.args[0])
        args = [resolve(a) for a in s.args]
        if s.op.name in _COMMUTATIVE:
            args = sorted(args, key=repr)
        return (s.op.name, tuple(args))

    out: dict[str, object] = {}
    for s in prog:
        if s.name not in fixed:
            continue
        if isinstance(s, Accum):
            base = s.base if s.base is not None else s.name
            args = sorted([resolve(base), resolve(s.value)], key=repr)
            out[s.name] = (s.op.name, tuple(args))
        elif s.op.name == "copy":
            out[s.name] = resolve(s.args[0])
        else:
            args = [resolve(a) for a in s.args]
            if s.op.name in _COMMUTATIVE:
                args = sorted(args, key=repr)
            out[s.name] = (s.op.name, tuple(args))
    return out


def _merge_is_seedable(prog: tuple, state: tuple[str, ...]) -> bool:
    """Each accumulator channel is written by a ``base``-``Accum`` (seed rides ``op.identity``)
    and the pivot by a ``maximum`` ``Accum`` — the streaming-fold shape the loop seeds."""
    writes = {s.name: s for s in prog if s.name in set(state)}
    pivot = writes[state[0]]
    if not (isinstance(pivot, Accum) and pivot.op.name == "maximum"):
        return False
    return all(
        isinstance(writes[n], Accum) and writes[n].op.name == "add" and writes[n].base is not None and writes[n].dtype is not None
        for n in state[1:]
    )


def test_generated_flash_matches_handwritten():
    state = ("m_i", "l_i", "O_i")
    merge, combine_states, state_b = _exp_family("s_e", (1.0, "v_e"), state)
    assert _canon(merge, state) == _canon(_HAND_FLASH_MERGE, state)
    assert _canon(combine_states, state) == _canon(_HAND_FLASH_COMBINE_STATES, state)
    assert state_b == ("m_i__o", "l_i__o", "O_i__o")
    assert _merge_is_seedable(merge, state)


def test_generated_online_softmax_matches_handwritten():
    state = ("m", "d")
    merge, combine_states, state_b = _exp_family("s", (1.0,), state)
    assert _canon(merge, state) == _canon(_HAND_SOFTMAX_MERGE, state)
    assert _canon(combine_states, state) == _canon(_HAND_SOFTMAX_COMBINE_STATES, state)
    assert state_b == ("m__o", "d__o")
    assert _merge_is_seedable(merge, state)


def test_certificate_rejects_unstable_combine():
    # A mis-specced carrier whose rescale subtracts the WRONG max (a bare maximum that does not
    # include the numerator) must be rejected, not silently emitted.
    bad = (
        Assign("mx", "maximum", ("a", "b")),
        Assign("dm", "subtract", ("z", "mx")),  # z ∉ {a, b} → exp(z − max(a,b)) not provably ≤ 0
        Assign("e", "exp", ("dm",)),
    )
    with pytest.raises(UnstableCarrierError):
        _carrier._certify(bad)


def test_every_exp_arg_is_nonpositive_by_construction():
    # The certificate runs inside the builders; if it passed, every exp is x − max(…, x, …).
    state = ("m_i", "l_i", "O_i")
    merge, combine_states, _ = _exp_family("s_e", (1.0, "v_e"), state)
    for prog in (merge, combine_states):
        defs = {s.name: s for s in prog if isinstance(s, Assign)}
        exps = [s for s in prog if isinstance(s, Assign) and s.op.name == "exp"]
        assert exps  # the LSE carrier has rescales
        for e in exps:
            sub = defs[e.args[0]]
            assert sub.op.name == "subtract"


def test_softmax_is_flash_minus_the_expectation_channel():
    # Structural: the (m, d) softmax combine_states is the (m, l) projection of flash's — same
    # generator, one fewer accumulator channel.
    _, flash_cs, _ = _exp_family("s", (1.0, "v"), ("m", "l", "O"))
    _, soft_cs, _ = _exp_family("s", (1.0,), ("m", "d"))
    # both have exactly one exp-rescale pair (a, b) in combine_states
    n_exp = lambda p: sum(isinstance(s, Assign) and s.op.name == "exp" for s in p)  # noqa: E731
    assert n_exp(soft_cs) == n_exp(flash_cs) == 2


def test_rescale_is_the_factor_the_merge_puts_on_the_carried_state():
    # The ψ-rescale named on its own must BE the factor ``exp_merge`` applies to every carried
    # channel — otherwise an accumulator outside the fold's state (attention's output tile) would
    # advance by a different law than the state beside it. Structural: the merge's ``Accum`` base
    # for each channel is ``<channel> · exp(pivot − max(pivot, arrival))``, and the rescale is
    # exactly that ``exp``.
    prog, advanced = exp_rescale("psi", "m", "s", key="k")
    defs = {s.name: s for s in prog}
    assert defs[defs["psi"].args[0]].op.name == "subtract"
    assert defs[defs["psi"].args[0]].args == ("m", advanced)
    assert defs[advanced].op.name == "maximum" and defs[advanced].args == ("m", "s")

    merge = exp_merge(("m", "d"), ("s", 1.0), key="m")
    mdefs = {s.name: s for s in merge if isinstance(s, Assign)}
    base = next(s.base for s in merge if isinstance(s, Accum) and s.name == "d")
    factor = mdefs[base].args[1] if mdefs[base].args[0] == "d" else mdefs[base].args[0]
    assert mdefs[base].op.name == "multiply"
    assert mdefs[factor].op.name == "exp"
    arg = mdefs[mdefs[factor].args[0]]
    assert arg.op.name == "subtract" and arg.args[0] == "m"
    assert mdefs[arg.args[1]].op.name == "maximum" and set(mdefs[arg.args[1]].args) == {"m", "s"}


def test_rescale_is_certified_like_every_generated_combine():
    # The rescale's ``exp`` argument is ``pivot − max(pivot, arrival)`` — provably ≤ 0 by the same
    # rule; a hand-built one that subtracts a foreign max is rejected.
    exp_rescale("psi", "m", "s", key="k")  # certifies inside the builder
    with pytest.raises(UnstableCarrierError):
        _carrier._certify((Assign("mx", "maximum", ("a", "b")), Assign("d", "subtract", ("m", "mx")), Assign("psi", "exp", ("d",))))
