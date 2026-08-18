"""Stamp a kernel MINTED in the tile dialect with its structural features — the twin of
``loop/stamp/020_stamp_structural_features``, and deliberately a duplicate of it.

The loop-dialect pass stamps every kernel the fusion end produces. It cannot stamp one minted
later: the pass cursor only ever restarts a scan WITHIN the current pass (``Cursor.advance``) and
never returns to an earlier one, so a fragment this pass's rules splice is behind it forever. The
rule has to live in the pass where the minting happens, which is why the work is registered twice
rather than moved — moving it would leave the OUTER search's terminals (finalized ``LoopOp``\\ s,
handed over before tile lowering runs) with no identity at all, and ``two_level``'s ``op_sig`` would
digest an empty set for every kernel in every model.

Idempotent, so the first sweep changes nothing: every op arriving from the loop dialect already
carries ``S_*`` and skips. What it catches is the pass-scan restart after a splice — a placement
cut's fragments, which are freshly-built ``LoopOp``\\ s with no knobs at all. They then reach
``010_recognize`` and ``020_schedule`` carrying their own identity, like any kernel, and no rule
had to be told they were fragments.

**Why ``LoopOp`` and not every kernel-bearing op.** Matching ``TileOp`` too would also catch the
fragments ``010_recognize`` splices as a ``Graph`` rather than rebinding — which have never carried
a structural identity at all, reaching codegen with one ``S_*`` knob, invisible to the prior, the
evidence store and golden matching. Giving them one is right and is owed. But it makes structural
pricing reachable for them for the first time, and one of the placement cuts it then selects is
numerically wrong: a whole-model accuracy test goes to ``max_diff`` 0.27 against a 0.005 gate, on a
legal seam distinct from the one a bare ``PLACE=cut`` pin already raises on. That is a real defect
and its own change. Widen this pattern once it is fixed, and ``030_split_reduce`` can drop its
hand-stamp with it.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.base import Op
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.passes.loop.stamp._stamp import kernel_body, structure_features

PATTERN = [Pattern("root", Op)]


def rewrite(match: Match, root: Node) -> Op | None:
    if any(k.startswith(STRUCT_PREFIX) for k in root.op.knobs):
        raise RuleSkipped("already carries structural features")
    body = kernel_body(root.op)
    if body is None:
        raise RuleSkipped("no kernel body to featurize")
    return replace(root.op, knobs={**root.op.knobs, **structure_features(body, match.graph)})
