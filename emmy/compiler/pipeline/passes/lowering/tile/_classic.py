"""Clean-slate boundary for classic schedule reconstruction.

The rebuilt scheduler has one semantic candidate-space contract: kernel, node, and edge domains
are projected independently from static offers, and enumeration is exactly the compatible subset
of their Cartesian product. Traversal order may change evaluation cost, never membership. Until
that implementation exists, every entry point fails through :class:`ClassicScheduleUnavailable`.
"""

from __future__ import annotations


class ClassicScheduleUnavailable(RuntimeError):
    """Classic scheduling has not yet been reconstructed for this term."""


class PinRefused(ValueError):
    """A classic schedule pin may be realizable only after a structural rewrite."""


def schedule(tile, name: str, knobs: dict, ctx):
    """Refuse classic scheduling until the independent-domain implementation is rebuilt."""
    del tile, name, knobs, ctx
    raise ClassicScheduleUnavailable("classic scheduling is unavailable during clean-slate reconstruction")


def _removed(*args, **kwargs):
    """Fail a directly exercised former scheduler seam through the reconstruction boundary."""
    del args, kwargs
    raise ClassicScheduleUnavailable("this classic scheduler seam has not been reconstructed")


_atom_families = _removed
_fold_states = _removed
_fragment_epilogue_ok = _removed
_kstep_refusal = _removed
_node_refusal = _removed
_options = _removed
_reduce_moves = _removed
_split_store_refusal = _removed
cone_seam = _removed


__all__ = ["ClassicScheduleUnavailable", "PinRefused", "schedule"]
