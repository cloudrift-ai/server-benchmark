"""Scoped tuning-knob pins and realized-pin validation."""

from __future__ import annotations

import contextlib
import os

from emmy import config
from emmy.compiler.pipeline.knob import family_of, get, is_off_value, pin_key_matches, values_equal


def unreproducible_pin_flag(pinned: dict, kernel_knobs: list[dict]) -> str | None:
    """Describe pins not realized by any compiled CUDA kernel, or return ``None``.

    A registered family with no realized key is ungateable because serialized IR
    can omit knob stamps. Declared OFF values mean not-applicable rather than a
    conflicting realization; an unknown absent family remains a likely typo.
    """
    if not any(kernel_knobs):
        return None
    misses: list[str] = []
    for name, want in pinned.items():
        fam = family_of(name)
        if fam == "PLACE":
            continue  # a realized cut is visible structurally, not as a knob stamp
        others: list[str] = []
        saw_off = False
        hit = False
        for raw in kernel_knobs:
            for key, got in raw.items():
                if family_of(key) != fam:
                    continue
                if pin_key_matches(name, key) and values_equal(name, want, got):
                    hit = True
                elif is_off_value(fam, got):
                    saw_off = True
                else:
                    spell = f"{key}={got}" if key != name else str(got)
                    if spell not in others:
                        others.append(spell)
            if hit:
                break
        if hit:
            continue
        if not others and not saw_off and get(fam) is not None:
            continue
        ran = "/".join(others) if others else ("(off)" if saw_off else "(unset)")
        misses.append(f"{name}={want} realized {ran}")
    return f"unreproducible pin: {'; '.join(misses)}" if misses else None


@contextlib.contextmanager
def pinned_knobs(knobs: dict):
    """Temporarily publish ``knobs`` as authoritative ``EMMY_<KNOB>`` pins."""
    saved: dict[str, str | None] = {}
    try:
        for name, value in knobs.items():
            key = config.knob_var(name)
            saved[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
