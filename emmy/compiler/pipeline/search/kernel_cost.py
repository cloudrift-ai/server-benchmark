r"""What one kernel costs at best, and the features that predict it — work, traffic, the physical
floor under them, and the feature row built from all three.

This answers a question neither prior asks. A :class:`~.prior.base.Prior` ranks the candidates
*within* one kernel's pool and its score is ordinal — only the order means anything. A structural
fork compares kernel **sets** (keep these operations fused in one kernel, or cut them into
several), and summing ordinal scores across a set is meaningless, which
``policy/greedy._resolved_price`` documents as a known defect. The fix is a per-kernel estimate in
absolute microseconds, so a sum composes with the measured microseconds the evidence tiers supply.

**Fork provenance does not leak in.** Only ``S_*`` stamps are read, and those are written at
birth in recognition, before the first schedule fork is offered — so a kernel minted by a
placement cut and the same kernel standing alone produce identical rows, which is what stops a
model fitted on these features learning provenance instead of physics.

**Why the roofline, and why nominal.** Measured latency across the golden corpus spans nearly
seven orders of magnitude, so a model regressing it directly spends its capacity on between-shape
variance the shape features already explain. Dividing by a physical floor collapses that to the
achieved fraction of the hardware, which is bounded and comparable across kernels and cards.
``serving/roofline.py`` audits whole serving programs against the same two-term floor and records
the residual it sees in practice: healthy tuned programs land ~1-3x their floor. That module
self-calibrates on the live card because it runs at boot with one in hand; this one reads nominal
per-card figures from :mod:`emmy.gpu` because it runs offline over cards nobody has, and a floor
that mixed measured and nominal sources across cards would make a cross-card comparison fit which
cards happened to be measured.

**It is a SCALE, not a strict bound, and the difference is measured.** The intent is the
roofline's — no kernel beats its own arithmetic or its own data movement — but two effects let
real kernels come in under it, and both are systematic rather than noise:

- **L2 residency — the systematic one.** The traffic term divides by DRAM bandwidth, and a
  kernel whose working set fits L2 is not limited by DRAM at all. 71% of the golden corpus fits,
  so this is the common case rather than the exception. Measured directly: of the rows whose
  working set fits, 32% come in under the scale, against 9% of those that spill — the same
  effect read the other way, and why the sub-1.0 ratios concentrate in the small kinds
  (median ratio: rms_norm 0.67, softmax 0.67, against contraction 1.51).
- **Ops that do not touch all of an input — a small tail.** Compulsory traffic assumes each
  input is read once in full, which a gather does not do: an embedding lookup reads a handful of
  rows of a large table, and the corpus's worst case is one, measured at 4.3 us against a
  1181 us scale. Detectable in principle (an integer-dtype operand is the index), but only 10 of
  982 rows are gather-shaped and they scatter in both directions — median ratio 24.7 — so
  abstaining on them would be machinery for one percent of the corpus with no consistent error
  to correct. Left alone deliberately.

Over the 982 golden rows the ratio runs p10 0.41, median 1.86, p90 6.16. State the gain as the
spread it removes, not as a range: dividing halves the log standard deviation, **2.73 to 1.44**,
and 1.44 is what a model has left to fit. The ratio's full range is no narrower than the raw
label's — the tails are the two effects above, not the bulk.

So it is a normalizer, and a caller must not read a sub-1.0 ratio as a broken measurement. ``log``
of this quantity is a FEATURE as well as the divisor, precisely so a fit can correct a systematic
error in it. Note what that does and does not buy: removing a per-kind median recovers only 1.44
to 1.30, because the distortion is mostly WITHIN a kind rather than a per-kind offset — ``fused``
alone carries sd(log) 2.43.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from emmy.compiler.dim import DEFAULT_SEQ_HINT
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.search.data.shape import stamped_flops, stamped_peak_dtype

if TYPE_CHECKING:
    from emmy.compiler.tensor import Tensor
    from emmy.gpu import GpuSpec

#: Floor under :func:`t_roofline_us`, microseconds — there to keep the scale finite, not to model
#: any particular cost. Both physical terms go to nanoseconds on a small enough kernel, and
#: dividing by that produces a ratio in the tens of thousands from a shape nobody cares about.
#:
#: Chosen BELOW the fastest latency the golden corpus records (0.67 us), deliberately: a floor
#: above some real measurement would put those rows under their own denominator for a cost they
#: never paid. Measured over the 982 golden rows, raising it from 0 to 0.5 pulls p90 from 39.4 to
#: 6.2 while leaving the sub-1.0 population exactly where it was — that population comes from the
#: physical terms (see the module docstring), which is the evidence this value is doing the one
#: job it claims and not papering over the other.
#:
#: Host launch overhead is NOT this and is not modelled here: a bench times the kernel through
#: CUDA events around it, so the measured latency never contained it. It becomes real where launch
#: COUNTS differ — one for a fused kernel, two for a cut — which is a property of a fork
#: comparison, and belongs with that comparison when one exists.
MIN_SCALE_US = 0.5


def _tensor_bytes(t: Tensor) -> float:
    """One buffer's storage footprint.

    A symbolic axis sizes at its hint, so a ``.dynM`` kernel prices against the shape its bench
    actually ran. Spelled inline rather than through ``Axis.hint_extent``: that reading has three
    competing homes across in-flight branches (``ir/axis.py`` here, ``ir/tile/identity.py`` on the
    fusion work, a private copy in ``_schedule.py`` on others), and consolidating it is that
    work's call, not this change's. Note also that the two halves of the floor reach the hint
    DIFFERENTLY:
    :func:`~..data.shape.stamped_flops` works from the stamps, which carry no dim, so it applies
    the default hint as a constant. They agree only while every symbolic axis is traced at the
    default — true of all 212 dynamic goldens today, and asserted by a test so that a corpus
    traced at another ``--seq-len`` fails loudly rather than sizing its work and its traffic
    differently.

    ``shape`` is the STORED shape, so this does not consult ``DataType.logical_elems``: a packed
    pair already carries a halved last-axis extent, and multiplying by the pair count as well
    would double-count it. Same convention as the constant-folding pass's ``_shape_nbytes``, which
    is the other place in the tree that turns a shape into a byte count."""
    return float(math.prod(d.as_static() if d.is_static else (d.hint or DEFAULT_SEQ_HINT) for d in t.shape) * t.dtype.nbytes)


def kernel_bytes(op) -> float:
    """The kernel's compulsory memory traffic in bytes: every distinct input read once, every
    output written once.

    Compulsory, so schedule-independent — tiling changes how often a value is *re*-read, never
    whether it must cross the memory bus at least once. That is what makes this a property of the
    kernel rather than of any candidate in its pool, and it is what lets the fuse/cut comparison
    see the difference it exists to price: a cut materializes its intermediate as a workspace
    buffer that one piece writes and the next reads, while the fused kernel keeps it in registers
    and never pays for it here.

    **Takes an op whose io is populated.** A wire-decoded graph seeds placeholder tensors
    (``Tensor(name, (), F32)``), so callers reconstructing a kernel from a golden must pass
    ``node.op.with_io(graph, node)`` — the one spelling of "refresh io from the surrounding
    graph" — rather than an op straight off the decoded graph."""
    return sum(_tensor_bytes(t) for t in op.inputs.values()) + sum(_tensor_bytes(t) for t in op.outputs.values())


def kernel_stamps(op) -> dict[str, float]:
    """The ``S_*`` structural histogram the identity strategy stamped on this op.

    Stamped at birth, in recognition, before the first schedule fork is offered — which is what
    makes every number derived from it fork-independent."""
    return {k: float(v) for k, v in (getattr(op, "knobs", None) or {}).items() if k.startswith(STRUCT_PREFIX)}


def t_roofline_us(op, spec: GpuSpec, stamps: dict | None = None) -> float:
    """The physical time-scale of this kernel on this card, microseconds — the larger of the
    arithmetic time and the data-movement time. See the module docstring for why real kernels can
    come in under it.

    Terms whose inputs are unrecorded drop out rather than reading as zero: a card with no
    bandwidth figure contributes no traffic term, and a kernel whose stamps do not certify a work
    formula (a norm, a softmax, a fused attention — see :func:`~..data.shape.stamped_flops`)
    contributes no arithmetic term. :data:`MIN_SCALE_US` is always present, so a kernel with
    neither still gets a finite scale rather than a division by zero."""
    stamps = kernel_stamps(op) if stamps is None else stamps
    terms = [MIN_SCALE_US]
    flops = stamped_flops(stamps)
    peak = spec.peak_tflops(stamped_peak_dtype(stamps)) if spec else None
    if flops is not None and peak:
        terms.append(flops / (peak * 1e12) * 1e6)  # FLOP / (FLOP/s) -> s -> us
    if spec and spec.mem_bw_gbps:
        terms.append(kernel_bytes(op) / (spec.mem_bw_gbps * 1e9) * 1e6)
    return max(terms)


def kernel_row(op, spec: GpuSpec) -> dict[str, float]:
    """The feature row for one kernel on one card — what a best-latency estimate is regressed on.

    Deliberately absent:

    - **Knob features.** No ``TILE`` / ``WORK`` / ``STAGE``, no ``D_*`` geometry block. That whole
      vocabulary exists to tell candidates apart *within* a pool, and this estimate never looks
      inside one; it predicts what the pool's best achieves.
    - **The raw peaks.** ``t_roofline_us`` already contains them, so a column would be the same
      fact twice. Note this is NOT a claim that the row denies a tree any route to a per-card
      constant: ``H_sm_count`` alone is unique across all five corpus cards (170 / 128 / 80 / 188
      / 76), so one is available whatever the peaks do. Moving predictions between cards is a real
      job and the card columns exist to do it; whether a fit memorizes instead is a question for
      leave-one-card-out, not for the column list.
    - **The kernel KIND.** ``ShapeKey.kind`` is derived from these very stamps, so as a column it
      is the same fact twice. Measured: adding it as a native categorical moves held-out error
      from 0.328 to 0.330 and the fused bias from +0.499 to +0.505 — nothing. It stays the right
      way to GROUP a report; it is not information a model lacks.
    - **``S_n_mma``.** Structurally zero on every stamped row — the stamp pass runs before the
      tile tier emits ``Mma`` statements (``data/shape.py`` documents the trap at length).
    - **``H_opt``.** The corpus this is fitted on is one compile regime throughout.

    - **The precision regime.** Tried and removed. A fast-math kernel and its standard sibling are
      separate rows with separate labels but identical stamps, so 292 of 638 feature-identical
      ``(card, S_*)`` cells hold both — which sounds like a column that must exist. Measured, it
      buys nothing for the decision: both arms of a real fork come from one program under one
      precision setting, so the column is CONSTANT across every comparison the model is asked to
      make and cannot change one (concordance on close pairs 0.726 with it, 0.726 without), and it
      ranked 34th of 42 by importance at 0.3%. Its one real effect was a ~3% bias correction on
      fast-math rows. Against that it was the only feature with no clean compile-time source — in
      training it reads a golden's recorded pins, and at a fork there is no golden — so it risked
      reading 0 at deploy on rows that trained as 1. The 292 cells now carry identical features
      with differing labels, which the fit averages; that cost is accepted deliberately.

    A missing value is ``nan``, never ``0.0``, matching the convention the online prior's
    featurizer states: ``nan`` means "not knowable here", which a tree splits on separately from a
    real zero."""
    stamps = kernel_stamps(op)
    nan = float("nan")
    flops = stamped_flops(stamps)
    traffic = kernel_bytes(op)
    floor = t_roofline_us(op, spec, stamps)

    row: dict[str, float] = dict(stamps)
    row.pop("S_n_mma", None)
    row["R_log_flops"] = math.log(flops) if flops else nan
    row["R_log_bytes"] = math.log(traffic) if traffic > 0 else nan
    row["R_intensity"] = flops / traffic if (flops and traffic > 0) else nan
    row["R_log_roofline_us"] = math.log(floor)
    # Card facts, plus three ratios against them. A tree can read the operands separately, so what
    # these add is nothing WITHIN one card and a comparable quantity ACROSS cards, which is the
    # job the card columns exist for. Not an occupancy model: a real one needs a CTA count, and
    # that is a property of a schedule this row deliberately has no opinion about.
    sm = float(spec.sm_count) if spec and spec.sm_count else nan
    row["H_sm_count"] = sm
    row["H_smem_optin"] = float(spec.smem_optin) if spec else nan
    row["H_cc"] = float(spec.compute_capability[0] * 10 + spec.compute_capability[1]) if spec and spec.compute_capability else nan
    row["H_tc_gen"] = float(spec.tensor_core_gen) if spec and spec.tensor_core_gen else nan
    known_sm = not math.isnan(sm)
    row["R_work_per_sm"] = flops / sm if (flops and known_sm) else nan
    row["R_out_per_sm"] = stamps.get("S_ext_free_prod", nan) / sm if known_sm else nan
    row["R_bytes_over_l2"] = traffic / spec.l2_bytes if (spec and spec.l2_bytes) else nan
    return row
