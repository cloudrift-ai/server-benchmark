#!/usr/bin/env python
"""Offline fit of the :class:`AnalyticPrior` weights artifact over ``knob_features``.

Motivation
----------
The two-level autotuner explores the post-fusion kernel's knob space with an inner
MCTS that stops on **patience** (N consecutive measured terminals with no new
best). Cold (no learned data) that search is ranked by the :class:`AnalyticPrior` —
a fixed linear model over the engineered ``D_*`` geometry / occupancy features
:func:`features.knob_features` produces. If the golden config sits at rank 800 of 2400,
patience never reaches it. This script fits the linear weights so the golden lands
near the top.

It treats the problem as offline learning-to-rank, over the SINGLE featurization
(``features.knob_features`` — no parallel feature set), covering **every kernel
regime**: fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise.

  1. For every golden (matmul thread / warp / dyn, reduce, pointwise), reconstruct
     the planner's exact candidate enumeration for that mode and locate the golden
     row. Matmul goldens reuse ``analytic._enumerate`` — the SAME gate-narrowed pool
     ``eval analytic`` and the greedy deploy rank over (fp32 → thread tier, fp16/bf16
     → the warp tier alone; the block-DAG rework moved the scalar↔warp choice to a
     structural fork, so a warp golden ranks within the warp tier, not against
     scalar rows). Reduce / pointwise trace the snippet and capture the restored
     schedule fork's rows through the same live-fork capture
     (``analytic.enumerate_graph``).
  2. Featurize every candidate via ``features.knob_features`` (the ``D_*`` engineered
     features plus ``MMA_tier``, the warp/scalar tier discriminator — the ``S_*`` /
     ``H_*`` shape/regime features are constant within a shape, so they drop out of
     a within-shape ranking).
  3. Score candidates with a parameterized linear model and measure the golden's
     rank (top-k coverage, median rank).
  4. Random-search + coordinate-descent the weights to minimize mean ``log2(rank+1)``
     across all goldens (both tiers jointly — the ``D_*`` features are tier-aware,
     so one weight vector serves both).
  5. Write the winning weights (both sets + the carried-over scoring params +
     provenance) to the ``AnalyticPrior`` weights artifact (``--out``, default the
     repo-checked ``search/prior/analytic_weights.json``), and print them.

Run:  ./venv/bin/python scripts/golden_knob_heuristics.py
      ./venv/bin/python scripts/golden_knob_heuristics.py --samples 40000 --out /tmp/candidate.json
"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import numpy as np

from emmy import config, storage
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.analytic import _enumerate, enumerate_graph
from emmy.compiler.pipeline.search.golden import (
    GOLDEN_CONFIGS,
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
)


def _base(ctx: Context, M: int, N: int, K: int, *, dynamic: bool = False) -> dict:
    """The shape / regime features the planner stamps and the prior featurizes —
    merged into each candidate row before ``knob_features`` so the occupancy terms
    (``#CTAs``, waves) and tier-aware BK targets fire. A dynamic (symbolic-M)
    golden mirrors the 992 stamp: the symbolic axis is EXCLUDED from the free-dim
    product and counted in ``S_ext_n_symbolic_axis`` — the same features the
    deployed featurization computes, and the flag the ``AnalyticPrior`` selects
    its dynamic weight set on."""
    free = float(N) if dynamic else float(M * N)
    base = {**ctx.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        base["S_ext_n_symbolic_axis"] = 1.0
    return base


def _snippet_rows(snippet: str, ctx: Context) -> list[dict]:
    """A non-matmul golden's candidate rows — trace the golden's torch snippet
    (``torch.sum`` / ``torch.relu`` have no dedicated frontend op, so the graph comes from the real
    trace) and capture the RESTORED schedule fork's leaf rows via the same live-fork capture the
    matmul path uses (``analytic.enumerate_graph``). A reduce offers its ``REDUCE@<axis>``
    partitions; a pointwise kernel forks nothing today, so its pool is empty and the golden is
    reported un-enumerable (an honest read of the live space, not a reconstruction of the
    demolished one)."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415

    graph = graph_from_code(snippet)[0]
    return enumerate_graph(graph, ctx)


def build_cases() -> list[tuple[str, str, int, list[dict[str, float]]]]:
    """Reconstruct each golden's candidate enumeration, pin the golden's index, and
    featurize every row. Returns ``(name, tier, golden_idx, feats)`` where ``tier``
    is ``"thread"`` / ``"warp"`` / ``"dyn"`` / ``"reduce"`` / ``"pointwise"`` and
    ``feats`` is the per-row ``D_*`` (+ ``MMA_tier``) feature dict.

    Matmul goldens enumerate via ``analytic._enumerate`` — the SAME gate-narrowed
    pool ``eval analytic`` and the greedy deploy rank over (fp32 → thread tier,
    fp16/bf16 → warp tier; the block-DAG rework moved the scalar↔warp choice to a
    structural fork, so a real fp16 matmul ranks within the warp tier alone, no
    scalar rows in the pool). A dynamic (``.dynM``) golden enumerates its hint-sized
    static twin's pool and featurizes with the symbolic-axis stamp (its own weight
    set). Reduce / pointwise goldens trace their snippet and capture the restored
    schedule fork's rows (``_snippet_rows``); a regime the live tree doesn't fork
    (pointwise) reports un-enumerable rather than reconstructing the demolished space.

    Each golden is reconstructed under its OWN card's context
    (``Context.from_target(cap, gpu_name=…)``, mirroring ``Sample.from_golden``):
    the multi-GPU golden set spans cards that differ in compute capability AND in
    SM count at the same cap (RTX 5090 = 170 vs RTX PRO 6000 = 188 vs RTX 4090 =
    128 SMs, the latter at sm_89), so both the candidate enumeration (cp.async /
    TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the
    recording card's regime — not one global cap — for the rank objective to match
    the deployed per-card featurization."""
    cases = []
    for g in GOLDEN_CONFIGS:
        ctx = Context.from_target(tuple(g.compute_cap), gpu_name=g.gpu_name)
        if isinstance(g, ReduceGoldenConfig):
            # The reduce's free axis (the ``M`` rows) maps to the planner's N axis
            # (the tuned ``FN`` register tile sweeps it): trace E_M=1, E_N=M, E_K=K.
            base = _base(ctx, 1, g.M, g.K)
            rows, tier = _snippet_rows(g.snippet(), ctx), "reduce"
        elif isinstance(g, PointwiseGoldenConfig):
            base = _base(ctx, g.M, g.N, 1)
            rows, tier = _snippet_rows(g.snippet(), ctx), "pointwise"
        elif isinstance(g, MatmulGoldenConfig):
            dyn = bool(g.dynamic)
            base = _base(ctx, g.M, g.N, g.K, dynamic=dyn)
            # A fast-math golden's row only exists in the gate-on enumeration (the same
            # self-gating seam as ``analytic.evaluate_golden``) — pin the gate for the
            # reconstruction so the f16-accumulate rows are present and the fit anchors
            # their ``MMA_acc_bits`` discriminator.
            if g.fast_math:
                from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC  # noqa: PLC0415

                with F16_MMA_F32_ACC.pinned("1"):
                    rows, _ = _enumerate(g.M, g.N, g.K, g.dtype, ctx)
            else:
                rows, _ = _enumerate(g.M, g.N, g.K, g.dtype, ctx)
            tier = "dyn" if dyn else ("thread" if g.dtype == "fp32" else "warp")
        else:
            continue
        if not rows:
            print(f"  !! {g.name}: nothing enumerated — skipping")
            continue
        # Match the legacy-recorded golden against the native candidate rows by
        # schema-agnostic structural signature (free-axis slots + reduce decomp +
        # atom) — the candidates speak native ``MOVE@element``, the golden YAML legacy
        # GEMM-letters, so a per-key tuple compare never lines up.
        want = features.tile_signature(g.knobs)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
        if gidx is None:
            print(f"  !! {g.name}: golden not in {len(rows)} candidates — skipping")
            continue
        # Keep ``D_*`` geometry/occupancy plus ``MMA_tier`` (the warp/scalar tier
        # discriminator, where the featurization still emits it) — the ``S_*`` /
        # ``H_*`` shape/regime features are constant within a shape, so they drop out
        # of a within-shape ranking.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if k.startswith("D_") or k == "MMA_tier"} for r in rows]
        cases.append((g.name, tier, gidx, feats))
    return cases


def _matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


def rank_of_golden(scores: np.ndarray, gidx: int) -> int:
    """0-based rank of the golden by descending score. Ties count AGAINST the golden
    (``>=``): greedy deploy breaks score ties by enumeration order, and option-0 is the
    per-cell / gmem-direct row — a tie IS a miss at deploy time. (The old ``>`` tie-optimism
    let a fit with zero ``D_stage_*`` weights report top-1 golden ranks while the deploy pick
    landed on the per-cell row — the 2026-07-02 sweep's 5-15x regressions.)"""
    return int((scores >= scores[gidx]).sum()) - 1


def topk_table(ranks: list[int], ks=(1, 5, 10, 25, 50, 100)) -> str:
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"


def eval_weights(mats: list[np.ndarray], gidx: list[int], w: np.ndarray) -> list[int]:
    return [rank_of_golden(m @ w, gi) for m, gi in zip(mats, gidx, strict=True)]


def objective(ranks: list[int], weights: list[float]) -> float:
    # Weighted mean log2(rank+1): rewards pushing every golden up, dominated by the
    # worst offenders. Per-case ``weights`` balance the tiers (fp16 warp is only
    # ~7/32 cases, so it'd be drowned out unweighted). Lower is better.
    vals = [w * math.log2(r + 1) for r, w in zip(ranks, weights, strict=True)]
    return float(sum(vals) / sum(weights))


def _fit(cases, names, sd_ref, *, seed_w, rng, samples):
    """Random-search + coordinate-descent one weight vector over ``cases``.
    Each fit z-scores over its own candidate pool; ``seed_w`` arrives scaled by
    ``sd_ref`` (``ones`` for a raw-weight seed, the previous fit's ``sd`` to
    chain fits) and is re-scaled into this pool's z-space. Returns
    ``(best_w, best_ranks, mu, sd)`` in this pool's z-space."""
    mats = [_matrix(feats, names) for _, _, _, feats in cases]
    gidx = [gi for _, _, gi, _ in cases]
    tier_n = {t: sum(1 for _, ct, _, _ in cases if ct == t) for _, t, _, _ in cases}
    cw = [1.0 / tier_n[t] for _, t, _, _ in cases]

    # Z-score over this fit's candidate pool so weights are comparable across features.
    allf = np.concatenate(mats, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    sd[sd == 0] = 1.0
    matsz = [(m - mu) / sd for m in mats]

    best_w = seed_w * sd / sd_ref  # re-scale the seed into this pool's z-space
    best_ranks = eval_weights(matsz, gidx, best_w)
    best_obj = objective(best_ranks, cw)
    print("  seed: " + topk_table(best_ranks))

    for _ in range(samples):
        w = rng.standard_normal(len(names))
        ranks = eval_weights(matsz, gidx, w)
        ob = objective(ranks, cw)
        if ob < best_obj:
            best_obj, best_w, best_ranks = ob, w, ranks

    # Coordinate-descent refine around the best.
    step = 1.0
    for _ in range(8):
        improved = False
        for j in range(len(names)):
            for delta in (step, -step):
                w = best_w.copy()
                w[j] += delta
                ranks = eval_weights(matsz, gidx, w)
                ob = objective(ranks, cw)
                if ob < best_obj:
                    best_obj, best_w, best_ranks, improved = ob, w, ranks, True
        if not improved:
            step /= 2

    print("  best: " + topk_table(best_ranks))
    for (name, tier, _, _), r in sorted(zip(cases, best_ranks, strict=True), key=lambda t: -t[1]):
        print(f"    {name:32s} [{tier:6s}] rank={r:5d}")
    return best_w, best_ranks, mu, sd


def _raw_weights(names, best_w, sd) -> dict[str, float]:
    # Fold the z-score into the weights so they apply to RAW features directly:
    # score = ((raw-mu)/sd)·w = raw·(w/sd) - const; the const drops out of ranking.
    return {name: float(best_w[i] / sd[i]) for i, name in enumerate(names) if abs(best_w[i] / sd[i]) > 1e-4}


def _print_weights(var: str, raw_w: dict[str, float]) -> None:
    print(f"\n== {var} ==")
    for name, wv in sorted(raw_w.items(), key=lambda t: -abs(t[1])):
        print(f"    {name!r}: {wv},")


def main() -> None:
    from emmy.compiler.pipeline.search.prior.analytic import _DEFAULT_FILE  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="random weight vectors to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(_DEFAULT_FILE),
        help="Weights artifact to write (default: the repo-checked analytic_weights.json).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Building golden dataset (each golden under its own card's context) ...")
    cases = build_cases()
    names = sorted({k for _, _, _, feats in cases for f in feats for k in f})
    static_cases = [c for c in cases if c[1] != "dyn"]
    dyn_cases = [c for c in cases if c[1] == "dyn"]
    print(f"  {len(static_cases)} static + {len(dyn_cases)} dynamic golden cases, {len(names)} D_* features")

    # Seed: the incumbent artifact's static weights so each search starts from
    # today's ranking and can only improve on it (the dyn fit seeds from the STATIC
    # result — the masked tier shares most of the geometry priors and diverges where
    # the boundary guard / occupancy differences demand). Lenient read, no feat_ver
    # gate: a refit after a featurizer change is exactly when versions mismatch, and
    # a stale key simply seeds 0.0. The scoring params carry through unchanged —
    # this script fits only the linear weights.
    incumbent = storage.read_json(config.analytic_path() or _DEFAULT_FILE)
    if not isinstance(incumbent, dict) or "params" not in incumbent:
        raise SystemExit(f"no incumbent weights artifact to seed from at {config.analytic_path() or _DEFAULT_FILE}")
    seed_weights = incumbent.get("weights", {})
    seed_raw = np.array([seed_weights.get(n, 0.0) for n in names])

    print("\n== static fit (thread / warp / reduce / pointwise tiers) ==")
    static_w, static_ranks, _, static_sd = _fit(static_cases, names, np.ones(len(names)), seed_w=seed_raw, rng=rng, samples=args.samples)
    static_raw = _raw_weights(names, static_w, static_sd)
    _print_weights("weights (static)", static_raw)

    dyn_raw, dyn_note = incumbent.get("weights_dynamic", static_raw), "carried from incumbent (no dynamic cases)"
    if dyn_cases:
        print("\n== dynamic fit (symbolic-axis masked-tile goldens) ==")
        dyn_w, dyn_ranks, _, dyn_sd = _fit(dyn_cases, names, static_sd, seed_w=static_w, rng=rng, samples=args.samples)
        dyn_raw = _raw_weights(names, dyn_w, dyn_sd)
        dyn_note = f"dynamic {topk_table(dyn_ranks)}"
        _print_weights("weights_dynamic", dyn_raw)

    artifact = {
        "feat_ver": features.FEATURIZER_VERSION,
        "kind": "linear",
        "weights": static_raw,
        "weights_dynamic": dyn_raw,
        "params": incumbent["params"],
        "provenance": {
            "fitted": datetime.date.today().isoformat(),
            "script": "scripts/golden_knob_heuristics.py",
            "args": {"samples": args.samples, "seed": args.seed},
            "cases": {"static": len(static_cases), "dynamic": len(dyn_cases)},
            "notes": f"static {topk_table(static_ranks)}; {dyn_note}",
        },
    }
    storage.write_json(Path(args.out), artifact, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
