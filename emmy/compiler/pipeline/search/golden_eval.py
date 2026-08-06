"""Golden-config evaluation harness — reconstructs a matmul shape's enumeration
and ranks it with a ``Prior``.

Ranking itself now lives in :mod:`emmy.compiler.pipeline.search.prior`: the
hand-coded :class:`OfflinePrior` (the cold-start linear model over
``features.knob_features``) and the ``OnlinePrior`` are the ONE ranking
path. This module is just the offline *evaluation* glue — given a recorded golden
it enumerates the shape's candidate rows and reports the golden's rank under a
scorer (the ``OfflinePrior`` by default; ``eval online`` passes the online one).
Used by ``emmy eval offline`` / ``eval online`` and the prior diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.features import tile_signature
from emmy.compiler.pipeline.search.prior.fit.rank import dual_rank

# Golden dtype spelling → the graph Tensor dtype name.
_DTYPES = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def _matmul_graph(M: int, N: int, K: int, dtype: str):
    """A bare ``(M, K) @ (K, N)`` matmul frontend graph — the shape a matmul golden records."""
    from emmy.compiler import dtype as _dt  # noqa: PLC0415
    from emmy.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415

    dt = _dt.get(_DTYPES.get(dtype, dtype))
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dt), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dt), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dt), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def enumerate_graph(graph, ctx: Context, *, family: str = "") -> list[dict]:
    """The planner's candidate enumeration for any ``graph`` — the SAME rows the scheduler's fork
    tree offers a live compile, captured by resolving the graph through ``TILE_PASSES`` with a
    decide that flattens each fork's leaves (each leaf's knob row keyed by the CANONICAL codec
    spelling — bare on a single-primary tree, ``TILE@dd``-style on flash — exactly
    what ``tile_signature`` joins a golden against). ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). The one live-fork capture the matmul
    :func:`_enumerate` and the offline fitter (``emmy fit``'s case builder) share."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import Fork, flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    rows: list[dict] = []
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        for leaf in leaves:
            row = dict(getattr(leaf, "knobs", None) or {})
            if any(family_of(k) in wanted for k in row):
                rows.append(row)
        option = fp.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
    return rows


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> list[dict]:
    """Reconstruct the planner's matmul enumeration for a shape (:func:`enumerate_graph` over the
    bare matmul graph, TILE-family rows)."""
    return enumerate_graph(_matmul_graph(M, N, K, dtype), ctx, family="TILE")


def _offline_scorer(M: int, N: int, K: int, ctx: Context, *, dynamic: bool = False) -> Callable[[dict], float]:
    """Default ranker for :func:`evaluate_golden` — the :class:`OfflinePrior`
    folded into the higher-is-better convention ``evaluate_golden`` ranks on
    (``-latency``). Merges the shape / regime features the prior featurizes
    (``S_ext_*`` from ``M``/``N``/``K`` + ``H_*`` from the context) into each row,
    mirroring what the planner stamps in the live pipeline. ``dynamic`` mirrors
    the 992 stamp for a symbolic-M shape: M drops out of the free-dim product and
    ``S_ext_n_symbolic_axis`` is set — the flag the prior selects its masked-tier
    weight set on."""
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415

    ap = OfflinePrior()
    free = float(N) if dynamic else float(M * N)
    base = {**ctx.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        base["S_ext_n_symbolic_axis"] = 1.0
    return lambda r: -ap.score({**base, **r})


def evaluate_golden(
    M: int,
    N: int,
    K: int,
    dtype: str,
    golden_knobs: dict,
    ctx: Context,
    scorer: Callable[[dict], float] | None = None,
    dynamic: bool = False,
) -> tuple[dict, int | None, int, int | None]:
    """Score a matmul shape's full enumeration and return ``(pick, rank, pool,
    rank_optimistic)``: the argmax pick (higher score = better), the recorded golden's
    0-based rank in the scored order (``None`` if the golden isn't in the enumeration —
    pin / dtype mismatch), the enumeration size, and the tie-OPTIMISTIC rank (strictly
    greater rows only; ``None`` with ``rank``). The rank — not whether the #1 pick equals
    the golden — is the metric that matters: it's where the tuner's patience budget
    has to reach. ``rank`` is tie-pessimistic: a row tied with the golden but earlier
    in emission order counts against it, because that is exactly the row a greedy
    argmin deploys instead — rank 0 therefore means "greedy finds the golden", not
    "nothing scores strictly better". The pessimistic-optimistic gap is the tie-plateau
    width at the golden's score — the score-saturation canary (see
    :func:`prior.fit.rank.dual_rank`, the one rank computation both flavors come
    from). ``scorer`` (``row → float``, higher better) defaults to the
    :class:`OfflinePrior` (negated latency); the online-prior diagnostics pass
    ``-prior.mean_score`` instead. Returns ``({}, None, 0, None)`` if nothing
    enumerates.

    A **fast-math** golden (its knobs carry a precision-trading realization —
    ``golden.fast_math_knobs``) self-gates: its row only exists in the enumeration when the
    ``F16_MMA_F32_ACC`` gate is on, so the rank is computed with the gate pinned for the scope
    of this call — otherwise every fast-math entry would false-skip as "recorded knobs not in
    the enumeration". Standard goldens rank against the default (gate-off) enumeration."""
    from contextlib import nullcontext  # noqa: PLC0415

    from emmy.compiler.pipeline.search.golden import fast_math_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC  # noqa: PLC0415

    gate = F16_MMA_F32_ACC.pinned("1") if golden_knobs and fast_math_knobs(golden_knobs) else nullcontext()
    with gate:
        rows = _enumerate(M, N, K, dtype, ctx)
    if not rows:
        return {}, None, 0, None
    if scorer is None:
        scorer = _offline_scorer(M, N, K, ctx, dynamic=dynamic)
    # Match the recorded golden against the native candidate rows by schema-agnostic
    # structural signature (free slots + reduce decomp + atom + stage) — robust to bare vs
    # axis-named key spelling.
    want = tile_signature(golden_knobs) if golden_knobs else None
    gidx = next((i for i, r in enumerate(rows) if tile_signature(r) == want), None) if want else None
    scores = [scorer(r) for r in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    # Both tie semantics from the one shared computation: pessimistic (greedy's argmin
    # breaks score ties by emission order, so a golden tied with an earlier-emitted row
    # loses the deploy — the #364 rule; the old strictly-greater count reported rank 0
    # for EVERY row inside a tie plateau, which is how a saturated prior scored "27/28
    # top-1" while real cold deploys shipped emission-order picks 12-29x off the
    # golden) and optimistic (strictly-greater only), whose gap is the plateau width.
    rank, rank_opt = dual_rank(scores, gidx) if gidx is not None else (None, None)
    return rows[best], rank, len(rows), rank_opt
