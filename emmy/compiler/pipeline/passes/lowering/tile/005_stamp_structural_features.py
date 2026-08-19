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

**Reaching them is a guarantee, not an ordering accident.** The restart only comes when the scan
wraps past the LAST rule, and a fork's option is applied by the CALLER (``Candidate.apply``), which
advances the rule cursor only if the applied match closed its batch. So a cut taken at an UNPINNED
placement fork with another kernel still pending leaves the cursor on ``010_recognize``, which
re-matches the pieces on the next step — ahead of this rule. ``010`` therefore DEFERS an unstamped
``LoopOp`` (``RuleSkipped``): the batch ends, the scan wraps, and the piece is stamped here before
anything recognizes it.

**``LoopOp`` covers every kernel a rule mints**, because every rule mints in the loop dialect: a
placement cut's fragments and a cross-CTA split's pieces are both plain ``LoopOp`` nodes, built that
way precisely so they re-enter the pipeline where a freshly fused kernel does. Nothing needs a
tile-dialect arm, and there is no kernel-minting path left that skips this rule.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.passes.loop.stamp._stamp import structure_features

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if any(k.startswith(STRUCT_PREFIX) for k in root.op.knobs):
        raise RuleSkipped("LoopOp already carries structural features")
    feats = structure_features(root.op.body, match.graph)
    return replace(root.op, knobs={**root.op.knobs, **feats})
