#!/usr/bin/env python
"""Bench the knob neighborhood of every recorded golden, paired -O1 / -O3, resumably.

The offline prior trains on golden (-O3) records and the tune sweeps measure mostly -O1
rows with a thin -O3 re-bench band — so the dataset holds few points measured at BOTH
opt levels. This driver grows that paired slice: around every recorded golden config it
enumerates the close-by candidate rows (the live card's own enumeration, filtered to a
knob-component distance from the golden's knobs), then benches each selected row twice —
once at ``-Xcicc -O1`` and once at ``-Xcicc -O3`` — via ``emmy run --bench --ab``, whose
default bench-to-node recording lands every clean row in the node store under its regime's
``context_key`` with ``H_opt`` stamped. The -O1/-O3 twins of a point share the knob set,
so they join later on ``op_sig`` + tunables.

Point selection is randomized but distribution-preserving: each batch picks a shape with
probability proportional to its REMAINING point count, then samples uniformly inside it —
so any time-truncated run yields an approximately uniform sample of the whole remaining
pool, and repeated runs converge on full coverage. Progress persists in a JSON ledger
(``--ledger``) keyed by (gpu, shape, knob signature): a rerun — same box or a freshly
rented one of the same card — skips terminal points and continues, so the pool is
eventually exhausted across sessions. ``--budget-s`` bounds a run's wall time (it stops
starting new work once elapsed; the in-flight invocation finishes).

Per batch the driver runs (one invocation per opt level, only the rows still missing it):

    emmy run --code <shape snippet> --bench --bench-backends emmy --warmup 5 --iters 30 \
        --nvcc-flags "-Xcicc -O{1,3}" --ab <row1> --ab <row2> ... --json <receipt>

The run's integrity gates protect the dataset: a pinned row that doesn't realize
(``pin_unmatched``), fails, or trips a flag (wrong answer, intensity floor) is marked
terminal in the ledger and never recorded as a clean measurement. Fast-math golden
anchors pin ``EMMY_F16_MMA_F32_ACC=1`` on their batches so the f16-accumulate rows exist
in the subprocess's enumeration, mirroring how the offline fit reconstructs those pools.

Run it on the GPU box, from the repo root (``scripts/remote_node_tune.py --mode neighbors``
drives it remotely and merges the node rows home):

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
from dataclasses import dataclass, field
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
    enumeration pool, and the neighbor points selected from that pool."""

    group_id: str  # e.g. "M2048xN512xK3840_fp16" (+ _dyn / _tb markers)
    snippet: str
    dynamic_specs: list[str]
    fast_math: bool  # any anchor entry is fast-math → pin the fm enumeration gate
    names: list[str] = field(default_factory=list)  # golden entry names, for --filter / logs
    points: list[tuple[str, str]] = field(default_factory=list)  # (point_key, spec)


def _group_id(g) -> str:
    return f"M{g.M}xN{g.N}xK{g.K}_{g.dtype}" + ("_dyn" if g.dynamic else "") + ("_tb" if g.trans_b else "")


def build_groups(max_dist: int, name_filter: str | None) -> tuple[list[ShapeGroup], str]:
    """Enumerate every matmul golden shape's pool on the LIVE card and keep the rows
    within ``max_dist`` of any recorded golden's knobs (anchors included, distance 0).
    Returns the groups plus the live card identity the ledger keys on. Non-matmul
    goldens are out of scope — the vicinity moveset here is the matmul schedule
    families, mirroring the manual golden-seeding runs."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.knob import tuning_knob_items
    from emmy.compiler.pipeline.search.features import tile_signature
    from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig
    from emmy.compiler.pipeline.search.golden_eval import _enumerate
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC

    ctx = Context.probe()
    gpu = ctx.hardware_id()

    by_shape: dict[str, list[MatmulGoldenConfig]] = {}
    for g in GOLDEN_CONFIGS:
        if isinstance(g, MatmulGoldenConfig):
            by_shape.setdefault(_group_id(g), []).append(g)

    groups: list[ShapeGroup] = []
    for gid, entries in sorted(by_shape.items()):
        names = sorted({e.name for e in entries})
        if name_filter and not any(name_filter in n for n in names):
            continue
        rep = entries[0]
        fast_math = any(e.fast_math for e in entries)
        # Mirror the offline fit's reconstruction: fm anchors only exist gate-on.
        gate = F16_MMA_F32_ACC.pinned("1") if fast_math else nullcontext()
        with gate:
            rows, _ = _enumerate(rep.M, rep.N, rep.K, rep.dtype, ctx)
        canon = [{k: str(v) for k, v in tuning_knob_items(r)} for r in rows]
        anchors = []
        for e in entries:
            want = tile_signature(e.knobs)
            idx = next((i for i, r in enumerate(rows) if tile_signature(r) == want), None)
            if idx is None:
                print(f"[pool] {gid}: golden {e.name} ({e.gpu_name}) not in the live enumeration — anchor skipped", flush=True)
            else:
                anchors.append(canon[idx])
        if not anchors:
            print(f"[pool] {gid}: no anchors on this card — group skipped", flush=True)
            continue
        seen: set[str] = set()
        points: list[tuple[str, str]] = []
        for row in canon:
            if min(knob_distance(row, a) for a in anchors) > max_dist:
                continue
            spec = knob_spec(row)
            if spec in seen:
                continue
            seen.add(spec)
            points.append((point_key(gpu, gid, spec), spec))
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
        print(f"[pool] {gid}: {len(rows)} enumerated, {len(anchors)} anchor(s), {len(points)} points within dist {max_dist}", flush=True)
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
        env["EMMY_F16_MMA_F32_ACC"] = "1"
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
    groups, gpu = build_groups(args.max_dist, args.filter)
    ledger_path, receipts_dir = Path(args.ledger), Path(args.receipts)
    ledger = load_ledger(ledger_path)

    by_id = {g.group_id: g for g in groups}
    total_pairs = sum(len(g.points) for g in groups)

    def remaining_map() -> dict[str, list]:
        out: dict[str, list] = {}
        for g in groups:
            pts = [(key, spec) for key, spec in g.points if any(needs_run(ledger, key, opt, args.max_attempts) for opt in OPT_FLAGS)]
            if pts:
                out[g.group_id] = pts
        return out

    def done_pairs() -> int:
        return sum(
            1 for g in groups for key, _ in g.points if all(opt_state(ledger, key, opt).get("status") in TERMINAL for opt in OPT_FLAGS)
        )

    remaining = remaining_map()
    n_left = sum(len(v) for v in remaining.values())
    print(f"[neighbor-bench] card {gpu}: {len(groups)} shapes, {total_pairs} points, {n_left} with work left", flush=True)
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
        gid, pts = pick_batch(rng, remaining, args.batch)
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
                f"[batch {batch_i}] {gid} {opt}: {ok}/{len(todo)} ok ({time.monotonic() - t0:.0f}s) statuses={sorted(statuses.values())}",
                flush=True,
            )
        new_done += sum(1 for key, _ in pts if all(opt_state(ledger, key, opt).get("status") in TERMINAL for opt in OPT_FLAGS))
        remaining = remaining_map()

    print(f"neighbor-bench done: {done_pairs()}/{total_pairs} points ({new_done} new this run)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
