"""The identities a measurement is stored and looked up under — one module, two value objects.

Every key the measurement store uses is one of these, and every field of one is a column of the
table it keys. Adding an element to an identity is adding a field here: the column follows and the
digest is recomputed from the columns beside it, so there is no second place where a key is
assembled by hand and no way for a writer and a reader to spell the same regime differently.

- :class:`Regime` — WHERE a measurement holds: the card, its capability, the nvcc flags split at
  the opt level, and the input pins the compile ran under. µs is per card, so the card is part of
  the identity: without it a measurement taken on one sm_89 card answers a deploy on another.
- :class:`OpIdentity` — WHAT a kernel is, knob-free: the canonical Loop-IR body it computes and
  the io it is bound to. Deliberately **not** the rendered CUDA source, and its digest is
  deliberately stage-free: a golden's identity is minted by lifting a recorded target to Loop IR
  while the live compile mints it at a Tile fork, and the deploy join only exists because both
  spell the same Loop-IR content. ``dialect`` rides along as a descriptive column.

Each carries its ``digest`` as a computed property over its own columns — the store's primary key
for it, never a stored field and never passed in: a key a caller could supply is a key that can
disagree with the row it sits on, and the point of unpacking an identity into columns is that the
key is a function of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

from emmy.compiler import structural
from emmy.compiler.structural import Structural


@dataclass(frozen=True)
class Regime(Structural):
    """The measurement regime — the ``context`` table's columns.

    ``gpu`` is :meth:`Context.hardware_id` (the PCIe product name, else a digest of the
    device-physical ``H_*`` regime). ``opt_level`` is the split nvcc cicc opt level, never the raw
    flag string, so ``""`` and ``"-Xcicc -O3"`` are the one regime they physically are;
    ``nvcc_flags`` keeps every OTHER flag verbatim, because a flag like ``--use_fast_math``
    genuinely changes codegen and must stay its own partition. ``pins`` is the input-pin regime
    (:func:`live_pin_regime`) — the marker BOOLs a compile ran under, spelled ``NAME=0/1`` in
    name order; an unpinned marker is absent, which reads as its default.
    """

    gpu: str
    sm_major: int
    sm_minor: int
    opt_level: int
    nvcc_flags: str
    pins: str

    @cached_property
    def digest(self) -> str:
        """The ``context`` row's primary key — the inherited :meth:`Structural.structural_key`
        walk over every field above, cached (sound outright: frozen, and the fields are the whole
        of it). The class deliberately does NOT override ``structural_key``: :func:`structural.form`
        delegates to any type that overrides it, so an override asking for the field walk would
        call itself."""
        return self.structural_key()

    @classmethod
    def of(cls, ctx) -> Regime:
        """The regime ``ctx`` compiles in — the ONE derivation, shared by the key and the row."""
        from emmy.compiler.context import split_opt_level  # noqa: PLC0415

        major, minor = ctx.compute_capability
        opt, residual = split_opt_level(ctx.compile_flags)
        return cls(
            gpu=ctx.hardware_id(),
            sm_major=major,
            sm_minor=minor,
            opt_level=opt,
            nvcc_flags=residual,
            pins=live_pin_regime(),
        )


def live_pin_regime() -> str:
    """The ambient input pins as ``Regime.pins`` — exactly the complement of the decision.

    A marker BOOL is dropped from a measurement's knob row (``knob.tuning_knob_items``: the
    realized fork is identified by what it enables, not by the gate that allowed it), so it has to
    land here — otherwise a kernel measured under ``EMMY_VECTORIZE_LOADS=0`` shares a key with the
    same kernel measured under the default and keep-best silently picks between two different
    pieces of code. The precision gates resolve
    through their own :func:`precision_pin` precedence (own pin > the ``FAST_MATH`` umbrella >
    unset), so the regime is spelled by the rule that decided what the compile could enumerate.
    """
    from emmy.compiler.pipeline.knob import KnobType, registry  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, FAST_EXP, FAST_MATH, FP8_MMA, precision_pin  # noqa: PLC0415

    gates = {knob.name: precision_pin(knob) for knob in (FAST_EXP, F16_MMA_F32_ACC, FP8_MMA)}
    pins = {name: int(value) for name, value in gates.items() if value is not None}
    # The umbrella itself never enters: it is fully resolved into the three gates above, and
    # carrying it too would make ``FAST_MATH=1`` and the three gates pinned by hand two regimes.
    resolved = {*gates, FAST_MATH.name}
    for name, knob in registry().items():
        if name in resolved or knob.type is not KnobType.BOOL or (raw := knob.raw()) is None:
            continue
        pins[name] = int(knob.parse(raw))
    return ",".join(f"{name}={pins[name]}" for name in sorted(pins))


@dataclass(frozen=True)
class OpIdentity(Structural):
    """What a kernel IS, knob-free — the ``op`` table's columns.

    ``body`` is the canonical Loop-IR body digest (``Body.structural_key``: SSA / axis / buffer
    names and commutative-arg order normalized away); ``io`` is the operand dtype / shape
    fingerprint, held as the fingerprint itself and rendered to its column by :attr:`io_json`.
    ``dialect`` is the stage the identity was taken at — a descriptive column, and the one field
    :attr:`digest` leaves out: every stage of one rewrite chain keys off the same Loop-IR content,
    which is what lets a golden minted by lifting a recorded target to Loop IR join a live Tile
    fork.
    """

    dialect: str
    body: str
    io: tuple

    @cached_property
    def digest(self) -> str:
        """The ``op`` row's primary key, and the deploy join key (``identity_key(with_io=True)``).

        Spelled out rather than taking :class:`Regime`'s generic field walk, for one reason worn
        two ways: ``dialect`` is descriptive and must stay out, and this fold is DURABLE — stamped
        identities are checked into the corpus, so restructuring around it must not move it."""
        return structural.digest(self.body, self.io)

    def structural_key(self) -> str:
        """Implements :class:`~emmy.compiler.structural.Structural` — :attr:`digest`. Overriding is
        the point: a nested ``OpIdentity`` renders by its own key, so ``dialect`` cannot reach an
        enclosing digest either."""
        return self.digest

    @cached_property
    def io_json(self) -> str:
        """The ``op.io`` column — the fingerprint as canonical JSON, so the row means something on
        its own. A rendering of the field, never its identity: the digest folds the fingerprint
        itself, so how the column spells it is free to change."""
        return json.dumps(self.io)
