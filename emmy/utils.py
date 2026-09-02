"""Small, dependency-free helpers shared across the package."""

from __future__ import annotations

import functools


class cached_method:  # noqa: N801 — a decorator, spelled like ``cached_property``
    """A method memoized on its instance, per argument tuple — :func:`functools.cached_property`'s
    storage (the instance ``__dict__``, which a frozen dataclass allows) under a private slot,
    while the attribute stays a call. A derived read of an immutable term is computed once per
    (hashable) argument tuple and dies with the instance — unlike a ``functools.cache`` keyed on
    ``self``, which hashes the whole term per call and holds every term ever built for the life of
    the process. The owner's ``__getstate__`` may strip the slot (``Lambda`` does), like any memo
    here."""

    def __init__(self, fn):
        self.fn = fn
        self.slot = f"_{fn.__name__}__memo"
        functools.update_wrapper(self, fn)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        def bound(*args):
            memo = instance.__dict__.setdefault(self.slot, {})
            if args not in memo:
                memo[args] = self.fn(instance, *args)
            return memo[args]

        return bound


__all__ = ["cached_method"]
