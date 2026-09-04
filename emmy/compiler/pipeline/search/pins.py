"""Scoped tuning-knob pins and realized-pin validation."""

from __future__ import annotations

import contextlib
import os

from emmy import config
from emmy.compiler.ir.schedule import Level, Reduce, Work
from emmy.compiler.ir.schedule.classic import CLASSIC_FAMILIES
from emmy.compiler.pipeline.knob import axis_of, family_of, get, is_off_value, pin_key_matches, values_equal

#: A synthetic thread inventory so ``Reduce.parse`` accepts a ``coop`` token here. The width
#: value never matters — ``spell`` is site-local and drops it — but a count > 1 is load-bearing:
#: at ``units=(1, 1)`` the parsed ``coop`` collapses to 1 and ``spell`` drops the token entirely,
#: silently reading ``g2k/coop`` as ``""``.
_ANY_THREAD_WORK = Work(kind="thread", units=(1, 32))


def parse_reduce(spec: str) -> Reduce | None:
    """A ``REDUCE`` spelling read through :class:`Reduce` with the synthetic thread inventory, so a
    ``coop`` token parses; ``None`` when the codec does not parse it."""
    try:
        return Reduce.parse(str(spec), _ANY_THREAD_WORK)
    except ValueError:
        return None


def spelled_arm(options, row) -> tuple[object, dict[str, str]] | None:
    """The kernel-set arm a knob row spells among a cut-pass fork's ``options``, as ``(option, its
    knobs)`` — or ``None`` when the row decides nothing at this fork.

    At a twist fork the row spells the arm carrying its ``TWIST`` value, the carrier when it
    carries none. At a placement fork the row spells the first offered seam it marks ``cut`` (a bare
    ``PLACE=cut`` takes the root-most offered seam), the fuse arm when it marks no seam ``cut`` —
    a schedule row with no ``PLACE`` key says the kernel it decorates ran fused — and nothing when
    the seams it marks are not on this kernel's ballot. At a split fork it spells the offered plan
    whose cross-CTA half equals its ``REDUCE`` value's, and the unsplit arm when that value carries
    no such half or the row carries no ``REDUCE`` at all — a schedule row measured the kernel
    whole. One reading for both consumers:
    the deploy's evidence pick reads every measured row of the kernel through it, and the golden
    replay follows a record's route and knobs through it. Nothing is installed on the kernel — the
    arm is the pass's own offer, and the pieces it mints are brand-new kernels whose own forks
    consult their own rows."""
    from emmy.compiler.pipeline.fork import leaf_knobs  # noqa: PLC0415

    arms = [(option, {str(key): str(value) for key, value in leaf_knobs(option).items()}) for option in options]
    keys = {key for _, knobs in arms for key in knobs}
    if "TWIST" in keys:
        # At a twist fork the row spells the arm carrying its ``TWIST`` value; a row with none
        # measured the fused carrier, the arm offered first.
        want = row.get("TWIST")
        return next(((option, knobs) for option, knobs in arms if want is None or knobs.get("TWIST") == str(want)), None)
    if any(family_of(key) == "PLACE" for key in keys):
        route = {str(key): str(value) for key, value in row.items() if family_of(str(key)) == "PLACE"}
        cuts = {key for key, value in route.items() if value == "cut"}
        for option, knobs in arms:
            if any(value == "cut" and (key in cuts or "PLACE" in cuts) for key, value in knobs.items()):
                return option, knobs
        if cuts:
            return None  # a cut this kernel does not offer: a stale spelling, or another kernel's seam
        return next(((option, knobs) for option, knobs in arms if knobs.get("PLACE") == "fuse"), None)
    key = next((key for key in sorted(keys) if family_of(key) == "REDUCE"), None)
    if key is None:
        return None
    value = row.get(key, row.get("REDUCE"))
    want = parse_reduce(value) if value is not None else None
    if value is not None and want is None:
        return None
    for option, knobs in arms:
        got = parse_reduce(knobs[key]) if knobs.get(key) else None
        if got is None:
            if want is None or not want.needs_split:
                return option, knobs  # the row measured the kernel whole: no cross-CTA split
        elif want is not None and want.needs_split and got.cta == want.cta and got.finalize == want.finalize:
            return option, knobs
    return None


def stampable_reduce(want: str) -> str | None:
    """The part of a ``REDUCE`` pin a kernel can still stamp, or ``None`` if it carries no
    cross-CTA stage. Read through :class:`Reduce` — the same typed reading the schedule
    walk's pin path consumes the ``g`` half with — so the rule has one statement; a value the
    codec does not parse answers ``None`` and is probed as-is.

    A cross-CTA split is realized by REPLACING the kernel: the structural ``030_cut``
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
            rest = stampable_reduce(want)
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
