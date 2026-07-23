#!/usr/bin/env python
"""Budgeted node-data sweep over every golden kind: paired -O1 / -O3 pinned benches, resumably.

The ONE collection flow for the cross-hardware node store (it replaced the ε-greedy
``emmy tune --dataset golden`` phase — search-driven collection over-samples the branches
the incumbent prior already likes and its wall time grows with the golden set; a budgeted
sample of the enumeration gives the future offline prior clean leaf rows at a fixed cost).
Every shape with a recorded golden — matmul, reduce, rms_norm, attention, the fused kinds —
enumerates its candidate pool on the LIVE card, and each selected row is benched twice —
once at ``-Xcicc -O1`` and once at ``-Xcicc -O3`` — via ``emmy run --bench --ab``, whose
default bench-to-node recording lands every clean row in the node store under its regime's
``context_key`` with ``H_opt`` stamped. The -O1/-O3 twins of a point share the knob set,
so they join later on ``op_sig`` + tunables (the -O1→-O3 offset dataset).

The pool is split into three SLICES, sampled by configurable budget shares (60/25/15):

- ``own`` — rows within ``--max-dist`` of the live card's OWN golden anchors: dense label
  support where this card's deploy decisions actually land.
- ``cross`` — rows near OTHER cards' golden anchors (that realize here) and not already
  ``own``: every point is verified-excellent somewhere, so it is either transfer signal or
  the arch-disagreement rows the prior's ``H_* × knob`` interactions train on.
- ``tail`` — a capped, deterministic (hash-ordered, seed-independent) subsample of the rest
  of the enumeration: landscape support so the prior also sees the bad tail.

Within a slice, each batch picks a shape with probability proportional to its REMAINING
point count, then samples uniformly inside it — a time-truncated run yields an
approximately uniform sample of each slice's remaining pool, and repeated runs converge on
full coverage. Progress persists in a JSON ledger (``--ledger``) keyed by (gpu, shape, knob
signature): a rerun — same box or a freshly rented one of the same card — skips terminal
points and continues. ``--budget-s`` bounds a run's wall time (it stops starting new work
once elapsed; the in-flight invocation finishes).

Per batch the driver runs (one invocation per opt level, only the rows still missing it):

    emmy run --code <shape snippet> --bench --bench-backends emmy --warmup 5 --iters 30 \
        --nvcc-flags "-Xcicc -O{1,3}" --ab <row1> --ab <row2> ... --json <receipt>

The run's integrity gates protect the dataset: a pinned row that doesn't realize
(``pin_unmatched``), fails, or trips a flag (wrong answer, intensity floor) is marked
terminal in the ledger and never recorded as a clean measurement. Fast-math shapes pin
``EMMY_FAST_MATH=1`` (the umbrella gate) on their batches so the precision-trading rows
exist in the subprocess's enumeration, mirroring how the offline fit reconstructs those
pools.

Run it on the GPU box, from the repo root (``scripts/remote_node_collect.py`` drives it
remotely and merges the node rows home):

    ./venv/bin/python scripts/golden_neighbor_bench.py --dry-run          # pool stats only
    ./venv/bin/python scripts/golden_neighbor_bench.py --budget-s 14400
    ./venv/bin/python scripts/golden_neighbor_bench.py --filter k_proj --max-dist 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path

O1_FLAGS = "-Xcicc -O1"
O3_FLAGS = "-Xcicc -O3"
OPT_FLAGS = {"O1": O1_FLAGS, "O3": O3_FLAGS}

# Ledger statuses that end a (point, opt) — anything else is retried up to --max-attempts.
TERMINAL = {"ok", "bench_fail", "pin_unmatched", "flagged"}


# --- pure helpers (unit-tested via tests/test_golden_neighbor_bench.py) -----


def knob_distance(a: dict, b: dict) -> int:
    """Component-aware distance between two knob rows: per differing key, a slash codec
    (``TILE=a:.../w2x2/f4x4/k2``) counts its differing ``/`` segments, any other value
    counts 1 — so one TILE pick that changes only the register tile is distance 1, not
    "TILE changed". Missing keys compare as empty (each present segment counts)."""
    d = 0
    for k in set(a) | set(b):
        va, vb = str(a.get(k, "")), str(b.get(k, ""))
        if va == vb:
            continue
        if "/" in va or "/" in vb:
            sa, sb = va.split("/"), vb.split("/")
            sa += [""] * (len(sb) - len(sa))
            sb += [""] * (len(sa) - len(sb))
            d += sum(1 for x, y in zip(sa, sb, strict=True) if x != y)
        else:
            d += 1
    return d


def knob_spec(row: dict) -> str:
    """Render a knob row as the ``K1=V1,K2=V2`` spec ``--ab`` / ``EMMY_KNOBS`` parse.
    Keys are sorted so the spec (and the point key derived from it) is order-stable."""
    items = sorted((k, str(v)) for k, v in row.items())
    for k, v in items:
        if "," in k or "," in v or "=" in v:
            raise ValueError(f"knob {k}={v!r} cannot ride the comma/equals spec grammar")
    return ",".join(f"{k}={v}" for k, v in items)


def point_key(gpu: str, group_id: str, spec: str) -> str:
    """Stable ledger key for one (card, shape, knob row) point."""
    sig = hashlib.sha1(spec.encode()).hexdigest()[:16]
    return f"{gpu}|{group_id}|{sig}"


def pick_batch(rng: random.Random, remaining: dict[str, list], batch: int) -> tuple[str, list]:
    """One batch: a group drawn with probability proportional to its remaining point
    count (so truncated runs sample the pool near-uniformly), then a uniform sample of
    up to ``batch`` of its remaining points. ``remaining`` maps group id → point list."""
    groups = [g for g, pts in remaining.items() if pts]
    weights = [len(remaining[g]) for g in groups]
    gid = rng.choices(groups, weights=weights, k=1)[0]
    pts = remaining[gid]
    return gid, rng.sample(pts, k=min(batch, len(pts)))


SLICES = ("own", "cross", "tail")


def assign_slice(dist_own: int | None, dist_cross: int | None, max_dist: int) -> str:
    """Slice a pool row by its distance to the nearest own-card / cross-card golden
    anchor (``None`` = no anchor of that class realized). ``own`` wins when both are
    within ``max_dist`` — a row near this card's own golden is own-neighborhood data
    even if another card's golden also sits nearby."""
    if dist_own is not None and dist_own <= max_dist:
        return "own"
    if dist_cross is not None and dist_cross <= max_dist:
        return "cross"
    return "tail"


def pick_slice(rng: random.Random, shares: dict[str, float], remaining: dict[str, dict[str, list]]) -> str:
    """Draw a slice by its budget share, renormalized over the slices that still have
    points — an exhausted slice's share flows to the others instead of stalling."""
    slices = [s for s in SLICES if remaining.get(s)]
    weights = [shares[s] for s in slices]
    return rng.choices(slices, weights=weights, k=1)[0]


def tail_sample(specs: list[str], cap: int) -> list[str]:
    """The tail slice's capped subsample: specs ranked by their own sha1 — deterministic
    AND seed/enumeration-order independent, so the selected tail set is stable across
    sessions and the ledger never chases a moving subsample."""
    ranked = sorted(specs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return ranked[:cap]


def normalize_shares(own: float, cross: float, tail: float) -> dict[str, float]:
    """Validate and normalize the slice budget shares to fractions summing to 1."""
    vals = {"own": own, "cross": cross, "tail": tail}
    if any(v < 0 for v in vals.values()) or not any(vals.values()):
        raise ValueError(f"slice shares must be non-negative and not all zero: {vals}")
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def load_ledger(path: Path) -> dict:
    if path.exists():
        obj = json.loads(path.read_text())
        if isinstance(obj, dict) and isinstance(obj.get("points"), dict):
            return obj
    return {"version": 1, "points": {}}


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True))
    tmp.replace(path)


def opt_state(ledger: dict, key: str, opt: str) -> dict:
    return ledger["points"].get(key, {}).get(opt, {})


def needs_run(ledger: dict, key: str, opt: str, max_attempts: int) -> bool:
    st = opt_state(ledger, key, opt)
    if st.get("status") in TERMINAL:
        return False
    return int(st.get("attempts", 0)) < max_attempts


def mark(ledger: dict, key: str, opt: str, status: str, *, spec: str | None = None) -> None:
    pt = ledger["points"].setdefault(key, {})
    if spec is not None:
        pt.setdefault("spec", spec)
    st = pt.setdefault(opt, {})
    st["status"] = status
    st["attempts"] = int(st.get("attempts", 0)) + 1


# --- pool building (imports emmy; needs the repo root on sys.path) ----------


@dataclass
class ShapeGroup:
    """One benchable shape: the golden entries recorded for it (across cards), its live
    enumeration pool, and the points selected from that pool, each tagged with its slice."""

    group_id: str  # e.g. "M2048xN512xK3840_fp16" (+ _dyn / _tb markers) / "reduce_M4096_K4096_dtypefp32"
    snippet: str
    dynamic_specs: list[str]
    fast_math: bool  # any anchor entry is fast-math → pin the fm enumeration gate
    names: list[str] = field(default_factory=list)  # golden entry names, for --filter / logs
    points: list[tuple[str, str, str]] = field(default_factory=list)  # (point_key, spec, slice)


def _group_id(kind: str, g, base_fields: frozenset[str]) -> str:
    """Ledger-stable shape id. Matmul keeps the exact legacy spelling (existing ledgers
    resume on it); every other kind derives generically from its OWN dataclass fields —
    everything beyond the shared ``GoldenConfig`` base (``base_fields``) — so a newly
    added golden kind, or a new shape field on an existing kind, joins the pool without
    touching this script. ``_dyn`` suffixes the symbolic-axis twin, like matmul."""
    if kind == "matmul":
        dyn = "_dyn" if g.dynamic else ""
        tb = "_tb" if g.trans_b else ""
        return f"M{g.M}xN{g.N}xK{g.K}_{g.dtype}{dyn}{tb}"
    parts = "_".join(f"{f.name}{getattr(g, f.name)}" for f in fields(type(g)) if f.name not in base_fields)
    return f"{kind}_{parts}" + ("_dyn" if g.dynamic else "")


def build_groups(max_dist: int, name_filter: str | None, tail_cap: int) -> tuple[list[ShapeGroup], str]:
    """Enumerate every golden shape's candidate pool on the LIVE card and slice its rows:
    ``own`` / ``cross`` by distance to this card's vs other cards' golden anchors
    (anchors included, distance 0; an anchor that doesn't realize here is skipped — that
    skip is the cross slice's realizability filter), plus a ``tail_cap``-bounded
    hash-ordered subsample of the rest. Returns the groups plus the live card identity
    the ledger keys on."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.knob import tuning_knob_items
    from emmy.compiler.pipeline.search.features import tile_signature
    from emmy.compiler.pipeline.search.golden import _KERNEL_CLASSES, GOLDEN_CONFIGS, GoldenConfig, _live_gpu_key
    from emmy.compiler.pipeline.search.golden_eval import _enumerate, enumerate_graph
    from emmy.compiler.pipeline.search.space import FAST_MATH

    ctx = Context.probe()
    gpu = ctx.hardware_id()
    own_key = _live_gpu_key()
    if own_key is None:
        print("[pool] no live CUDA device — every anchor labels cross, pools are not meaningful", flush=True)
    kind_of = {cls: kind for kind, cls in _KERNEL_CLASSES.items()}
    base_fields = frozenset(f.name for f in fields(GoldenConfig))

    by_shape: dict[str, list] = {}
    for g in GOLDEN_CONFIGS:
        by_shape.setdefault(_group_id(kind_of[type(g)], g, base_fields), []).append(g)

    groups: list[ShapeGroup] = []
    for gid, entries in sorted(by_shape.items()):
        names = sorted({e.name for e in entries})
        if name_filter and not any(name_filter in n for n in names):
            continue
        rep = entries[0]
        kind = kind_of[type(rep)]
        if kind == "attention" and rep.dynamic:
            # Masked-flash pins never resolve axis-keyed TILEs (AttentionGoldenConfig.__post_init__) —
            # every --ab here would burn a flash compile into a guaranteed pin_unmatched.
            print(f"[pool] {gid}: dynamic attention — --ab pins don't resolve on the masked flash, group skipped", flush=True)
            continue
        fast_math = any(e.fast_math for e in entries)
        # Mirror the offline fit's reconstruction: fm anchors only exist gate-on.
        gate = FAST_MATH.pinned("1") if fast_math else nullcontext()
        with gate:
            if kind == "matmul":
                rows, _ = _enumerate(rep.M, rep.N, rep.K, rep.dtype, ctx)
            else:
                # A dynamic golden enumerates its hint-sized static twin's pool (the snippet
                # is concrete-sized) — the accepted approximation the offline fit also uses.
                rows = enumerate_graph(graph_from_code(rep.snippet())[0], ctx)
        canon = [{k: str(v) for k, v in tuning_knob_items(r)} for r in rows]
        own_anchors: list[dict] = []
        cross_anchors: list[dict] = []
        for e in entries:
            want = tile_signature(e.knobs)
            idx = next((i for i, r in enumerate(rows) if tile_signature(r) == want), None)
            if idx is None:
                print(f"[pool] {gid}: golden {e.name} ({e.gpu_name}) not in the live enumeration — anchor skipped", flush=True)
                continue
            is_own = own_key is not None and e.gpu_name == own_key[0] and tuple(e.compute_cap) == own_key[1]
            (own_anchors if is_own else cross_anchors).append(canon[idx])
        seen: set[str] = set()
        points: list[tuple[str, str, str]] = []
        tail_specs: list[str] = []
        for row in canon:
            spec = knob_spec(row)
            if spec in seen:
                continue
            seen.add(spec)
            d_own = min((knob_distance(row, a) for a in own_anchors), default=None)
            d_cross = min((knob_distance(row, a) for a in cross_anchors), default=None)
            sl = assign_slice(d_own, d_cross, max_dist)
            if sl == "tail":
                tail_specs.append(spec)
            else:
                points.append((point_key(gpu, gid, spec), spec, sl))
        points += [(point_key(gpu, gid, spec), spec, "tail") for spec in tail_sample(tail_specs, tail_cap)]
        if not points:
            print(f"[pool] {gid}: no candidate rows on this card (forks nothing?) — group dropped", flush=True)
            continue
        groups.append(
            ShapeGroup(
                group_id=gid,
                snippet=rep.snippet(),
                dynamic_specs=rep.dynamic_specs(),
                fast_math=fast_math,
                names=names,
                points=points,
            )
        )
        n = {s: sum(1 for _, _, sl in points if sl == s) for s in SLICES}
        print(
            f"[pool] {gid}: {len(rows)} enumerated, {len(own_anchors)} own / {len(cross_anchors)} cross anchor(s), "
            f"own {n['own']} / cross {n['cross']} / tail {n['tail']} points (dist {max_dist}, tail cap {tail_cap})",
            flush=True,
        )
    return groups, gpu


# --- bench execution --------------------------------------------------------


def run_batch(group: ShapeGroup, specs: list[str], opt: str, args, receipt: Path) -> dict[str, str]:
    """One ``emmy run`` invocation benching ``specs`` pinned at ``opt``'s nvcc flags.
    Returns spec → status (``ok`` / ``bench_fail`` / ``pin_unmatched`` / ``flagged`` /
    ``timeout`` / ``error``), parsed from the ``--json`` receipt (rows come back in
    ``--ab`` order; a non-zero exit with a parseable receipt just means some row
    failed — each row still carries its own verdict)."""
    cmd = [args.emmy, "run", "--code", group.snippet, "--bench", "--bench-backends", "emmy"]
    cmd += ["--warmup", str(args.warmup), "--iters", str(args.iters)]
    cmd += ["--nvcc-flags", OPT_FLAGS[opt], "--json", str(receipt)]
    for d in group.dynamic_specs:
        cmd += ["--dynamic", d]
    for s in specs:
        cmd += ["--ab", s]
    env = dict(os.environ)
    if group.fast_math:
        env["EMMY_FAST_MATH"] = "1"
    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.run_timeout)
    except subprocess.TimeoutExpired:
        return dict.fromkeys(specs, "timeout")
    try:
        payload = json.loads(receipt.read_text())
        ab_rows = [r for r in payload.get("pinned", []) if r.get("kind") == "ab"]
    except (OSError, ValueError):
        return dict.fromkeys(specs, "error")
    out: dict[str, str] = {}
    for spec, row in zip(specs, ab_rows, strict=False):
        status = row.get("status", "error")
        if status == "ok" and row.get("flags"):
            status = "flagged"
        out[spec] = status
    for spec in specs:
        out.setdefault(spec, "error")  # fewer rows than specs — the run died mid-way
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--budget-s", type=float, default=0.0, help="Stop starting new work after this many seconds (0 = run to pool exhaustion)"
    )
    ap.add_argument("--max-dist", type=int, default=2, help="Max knob-component distance from a golden anchor (default 2)")
    ap.add_argument("--batch", type=int, default=6, help="Pinned rows per emmy run invocation (default 6)")
    ap.add_argument("--share-own", type=float, default=60.0, help="Budget share for the own-golden-neighborhood slice (default 60)")
    ap.add_argument("--share-cross", type=float, default=25.0, help="Budget share for the cross-card golden-exchange slice (default 25)")
    ap.add_argument("--share-tail", type=float, default=15.0, help="Budget share for the uniform-tail slice (default 15)")
    ap.add_argument("--tail-cap", type=int, default=200, help="Max tail points per shape, hash-ordered so the set is stable (default 200)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the sampling order (default 0)")
    ap.add_argument("--warmup", type=int, default=5, help="Bench warmup iters — keep >= 5, the node-record quality bar (default 5)")
    ap.add_argument("--iters", type=int, default=30, help="Bench measure iters — keep >= 20, the node-record quality bar (default 30)")
    ap.add_argument("--run-timeout", type=int, default=1800, help="SIGKILL wall cap per emmy run invocation, seconds (default 1800)")
    ap.add_argument("--max-attempts", type=int, default=2, help="Retries for a non-terminal (timeout/error) point per opt (default 2)")
    ap.add_argument(
        "--ledger", default=str(Path.home() / ".cache/emmy/neighbor_bench/ledger.json"), help="Progress ledger JSON (resume state)"
    )
    ap.add_argument(
        "--receipts", default=str(Path.home() / ".cache/emmy/neighbor_bench/receipts"), help="Directory for per-batch --json receipts"
    )
    ap.add_argument("--filter", help="Only shapes with a golden name containing this substring")
    ap.add_argument("--emmy", default="./venv/bin/emmy", help="emmy executable (default ./venv/bin/emmy)")
    ap.add_argument("--dry-run", action="store_true", help="Build the pool, print per-shape and remaining stats, exit")
    args = ap.parse_args()

    start = time.monotonic()
    shares = normalize_shares(args.share_own, args.share_cross, args.share_tail)
    groups, gpu = build_groups(args.max_dist, args.filter, args.tail_cap)
    ledger_path, receipts_dir = Path(args.ledger), Path(args.receipts)
    ledger = load_ledger(ledger_path)

    by_id = {g.group_id: g for g in groups}
    total_pairs = sum(len(g.points) for g in groups)
    totals = {s: sum(1 for g in groups for _, _, sl in g.points if sl == s) for s in SLICES}

    def remaining_map() -> dict[str, dict[str, list]]:
        out: dict[str, dict[str, list]] = {s: {} for s in SLICES}
        for g in groups:
            for key, spec, sl in g.points:
                if any(needs_run(ledger, key, opt, args.max_attempts) for opt in OPT_FLAGS):
                    out[sl].setdefault(g.group_id, []).append((key, spec))
        return {s: m for s, m in out.items() if m}

    def done_pairs() -> int:
        return sum(
            1 for g in groups for key, _, _ in g.points if all(opt_state(ledger, key, opt).get("status") in TERMINAL for opt in OPT_FLAGS)
        )

    remaining = remaining_map()
    left = {s: sum(len(v) for v in remaining.get(s, {}).values()) for s in SLICES}
    print(
        f"[neighbor-bench] card {gpu}: {len(groups)} shapes, {total_pairs} points "
        f"(own {totals['own']} / cross {totals['cross']} / tail {totals['tail']}), "
        f"{sum(left.values())} with work left (own {left['own']} / cross {left['cross']} / tail {left['tail']})",
        flush=True,
    )
    if args.dry_run or not remaining:
        print(f"neighbor-bench done: {done_pairs()}/{total_pairs} points (0 new this run)", flush=True)
        return 0

    receipts_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    new_done = 0
    batch_i = 0
    while remaining:
        if args.budget_s and time.monotonic() - start > args.budget_s:
            print(f"[neighbor-bench] budget of {args.budget_s:.0f}s exhausted — stopping", flush=True)
            break
        sl = pick_slice(rng, shares, remaining)
        gid, pts = pick_batch(rng, remaining[sl], args.batch)
        group = by_id[gid]
        batch_i += 1
        for opt in OPT_FLAGS:
            todo = [(key, spec) for key, spec in pts if needs_run(ledger, key, opt, args.max_attempts)]
            if not todo:
                continue
            receipt = receipts_dir / f"{batch_i:05d}_{gid}_{opt}.json"
            t0 = time.monotonic()
            statuses = run_batch(group, [spec for _, spec in todo], opt, args, receipt)
            for key, spec in todo:
                mark(ledger, key, opt, statuses[spec], spec=spec)
            save_ledger(ledger_path, ledger)
            ok = sum(1 for s in statuses.values() if s == "ok")
            print(
                f"[batch {batch_i}] {sl} {gid} {opt}: {ok}/{len(todo)} ok ({time.monotonic() - t0:.0f}s) "
                f"statuses={sorted(statuses.values())}",
                flush=True,
            )
        new_done += sum(1 for key, _ in pts if all(opt_state(ledger, key, opt).get("status") in TERMINAL for opt in OPT_FLAGS))
        remaining = remaining_map()

    print(f"neighbor-bench done: {done_pairs()}/{total_pairs} points ({new_done} new this run)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
