"""Scoped tuning-knob pins and realized-pin validation."""

from __future__ import annotations

import contextlib
import os

from emmy import config
from emmy.compiler.ir.classic_schedule import CLASSIC_FAMILIES
from emmy.compiler.ir.schedule import Level, Reduce, Work
from emmy.compiler.pipeline.knob import axis_of, family_of, get, is_off_value, pin_key_matches, values_equal

#: A synthetic thread inventory so ``Reduce.parse`` accepts a ``coop`` token here. The width
#: value never matters — ``spell`` is site-local and drops it — but a count > 1 is load-bearing:
#: at ``units=(1, 1)`` the parsed ``coop`` collapses to 1 and ``spell`` drops the token entirely,
#: silently reading ``g2k/coop`` as ``""``.
_ANY_THREAD_WORK = Work(kind="thread", units=(1, 32))


def _stampable_reduce(want: str) -> str | None:
    """The part of a ``REDUCE`` pin a kernel can still stamp, or ``None`` if it carries no
    cross-CTA stage. Read through :class:`Reduce` — the same typed reading the schedule
    walk's pin path consumes the ``g`` half with — so the rule has one statement; a value the
    codec does not parse answers ``None`` and is probed as-is.

    A cross-CTA split is realized by REPLACING the kernel: the structural ``035_split_reduce``
    fork mints brand-new
    pieces and ``knob.consume_kernel_row`` strips their schedule row, so no piece may carry the
    ``g<n>`` it came from (``test_split_fresh_kernels`` asserts that outright). The receipt is
    structural — the piece's reduce axis is a slice of the parent — and knob stamps cannot show
    it.

    Only the cross-CTA stage is invisible. The rest of the value (``coop`` / ``r<n>``) is decided
    by the piece on its own body and stamped there, so it stays gateable.
    """
    try:
        plan = Reduce.parse(str(want), _ANY_THREAD_WORK)
    except ValueError:
        return None
    if not plan.needs_split:
        return None
    return Reduce(tuple(st for st in plan.stages if st.level is not Level.GRID)).spell()


def unreproducible_pin_flag(pinned: dict, kernel_knobs: list[dict], *, reject_conflicts: bool = False) -> str | None:
    """Describe pins not realized by any compiled CUDA kernel, or return ``None``.

    A registered family with no realized key is ungateable because serialized IR
    can omit knob stamps. Declared OFF values mean not-applicable rather than a
    conflicting realization; an unknown absent family remains a likely typo.
    ``reject_conflicts`` additionally rejects any matching child scope that decided
    a different non-OFF value, even when another child realized the requested pin.
    """
    if not any(kernel_knobs):
        return None
    misses: list[str] = []
    for name, want in pinned.items():
        fam = family_of(name)
        if fam == "PLACE":
            continue  # graph placement is consumed by a splice, not stamped on either resulting kernel
        probe = want
        if fam == "REDUCE":
            # Likewise a realized cross-CTA split — but only its ``g<n>`` stage is structural,
            # so gate whatever the pieces still stamp.
            rest = _stampable_reduce(want)
            if rest is not None:
                if not rest:
                    continue
                probe = rest
        others: list[str] = []
        conflicts: list[str] = []
        saw_off = False
        hit = False
        for raw in kernel_knobs:
            for key, got in raw.items():
                if family_of(key) != fam:
                    continue
                same_scope = pin_key_matches(name, key)
                if same_scope and values_equal(name, probe, got):
                    hit = True
                elif is_off_value(fam, got):
                    saw_off = True
                else:
                    spell = f"{key}={got}" if key != name else str(got)
                    if spell not in others:
                        others.append(spell)
                    if same_scope and spell not in conflicts:
                        conflicts.append(spell)
            if hit and not reject_conflicts:
                break
        if hit and (not reject_conflicts or not conflicts):
            continue
        if not others and not saw_off and get(fam) is not None and fam not in CLASSIC_FAMILIES:
            continue
        ran_values = conflicts if reject_conflicts and conflicts else others
        ran = "/".join(ran_values) if ran_values else ("(off)" if saw_off else "(unset)")
        misses.append(f"{name}={want} realized {ran}")
    return f"unreproducible pin: {'; '.join(misses)}" if misses else None


@contextlib.contextmanager
def pinned_knobs(knobs: dict):
    """Temporarily publish ``knobs`` as authoritative environment pins.

    Axis-scoped keys ride both their programmatic ``EMMY_<KNOB@site>`` splat and the raw
    ``EMMY_KNOBS`` aggregate because ``@`` is not a portable shell-variable name.
    """
    saved: dict[str, str | None] = {}
    try:
        scoped = []
        for name, value in knobs.items():
            key = config.knob_var(name)
            saved[key] = os.environ.get(key)
            os.environ[key] = str(value)
            if axis_of(name) is not None:
                scoped.append(f"{name}={value}")
        if scoped:
            saved[config.KNOBS] = os.environ.get(config.KNOBS)
            aggregate = config.knobs_aggregate()
            os.environ[config.KNOBS] = ",".join(part for part in (aggregate, *scoped) if part)
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
