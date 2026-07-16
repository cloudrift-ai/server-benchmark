"""Measurement freeze — a digest-pinned, leaf-only snapshot of the tune DB's ``node`` table.

The node DB is a live store (tunes and merges write into it), so a model fit directly from
it is not reproducible: two runs of the same fitter can see different data. A *freeze* is
one local JSONL file extracted from the DB — line 1 a provenance header, then one JSON
object per kept leaf row — whose sha256 digest pins exactly which measurements a fit saw.
The fit becomes a pure function of (repo, freeze digest).

What freezes (see :func:`freeze_reason`): every **leaf** in the current featurizer
vocabulary that passes the physical-plausibility predicates, ``bench_fail`` leaves
included (a "doesn't build/launch here" is a durable negative example). Branch rows never
freeze — a branch's value-of-position bound is a run-artifact of one search's coverage
over the *historical* fork-tree topology; leaves are complete points in knob space, valid
under any tree organization, and prefix rows are re-synthesized at fit time under the
current fork structure. Accordingly a freeze stores **no tree schema**: no ``parent_key``,
no ``depth``, no ``visits``.

Determinism contract: freezing the same DB twice yields the same digest. Rows are sorted
by ``(gpu, op_sig, node_key)`` (total — ``node_key`` is the table's unique key) and
serialized with sorted keys and fixed separators; the digest covers exactly the raw row
bytes after the header line, so the header's ``created_at`` never perturbs it and the
loader catches any corruption by re-hashing the bytes it read. The header reserves both
featurizer version axes (``knob_ver`` / ``encoding_ver``, equal today) so a future
knob-spelling vs feature-encoding version split needs no format migration.

:func:`load_freeze` hard-errors on a foreign file, a ``feat_ver`` mismatch, or a digest
mismatch — never a silent fallback (the :mod:`~emmy.compiler.pipeline.search.prior.offline`
artifact-loading semantics). :func:`load_node_rows` is the interchange seam: it sniffs a
path and yields ``NodeRow``s from either a live sqlite DB or a freeze, so the nodes-dataset
consumers (``eval online --dataset nodes``, ``Dataset.from_node_rows`` /
``fold_node_rows``) accept both. Loaded freeze rows carry ``parent_key=None`` /
``depth=0`` / ``visits=0``: the fork-regret diagnostics skip parentless rows and the
golden-anchored descent treats ``depth=0`` rows as tree-less (its absence rule renders
"no fork-tree data"), so a freeze degrades to the leaf-level metrics instead of
inventing fork groups.

Produced by ``scripts/freeze_node_store.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.db import NodeRow

logger = logging.getLogger(__name__)

FREEZE_KIND = "emmy-node-freeze"
FREEZE_VER = 1

_SQLITE_MAGIC = b"SQLite format 3\x00"


def freeze_reason(row: NodeRow) -> str | None:
    """Why ``row`` is excluded from a measurement freeze, or ``None`` to keep it.

    THE freeze sanity filter, and nothing else — keep every leaf spelled in the current
    featurizer vocabulary that passes the shared plausibility predicates. ``bench_fail``
    leaves reach the keep path by construction: both predicates return ``None`` for
    non-``ok`` rows, so failures are kept as negative examples without a special case."""
    from emmy.compiler.pipeline.search.db import implausible_value_reason, impossible_kernel_reason  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    if row.is_leaf is not True:
        return "non-leaf (branch or pre-enrichment row)"
    if row.feat_ver != FEATURIZER_VERSION:
        return f"stale feat_ver {row.feat_ver} != current {FEATURIZER_VERSION}"
    reason = implausible_value_reason(row)
    if reason is not None:
        return f"implausible value: {reason}"
    reason = impossible_kernel_reason(row)
    if reason is not None:
        return f"impossible kernel: {reason}"
    return None


def _row_line(row: NodeRow) -> bytes:
    """One freeze row as its canonical JSONL line (bytes) — sorted keys, fixed
    separators, no NaN tokens (``allow_nan=False`` hard-errors instead of emitting
    nonstandard JSON): this exact spelling is what makes freeze-twice determinism
    hold. Bytes end-to-end so the digested bytes ARE the written bytes — no platform
    newline translation or locale encoding can slip between them."""
    d = {
        "node_key": row.node_key,
        "context_key": row.context_key,
        "op_sig": row.op_sig,
        "gpu": row.gpu,
        "features": row.features,
        "value_us": row.value_us,
        "variance": row.variance,
        "n_samples": row.n_samples,
        "status": row.status,
        "run_id": row.run_id,
        "measured_at": row.measured_at,
    }
    return (json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _repo_commit() -> str:
    """``git rev-parse HEAD`` of the checkout this module runs from, ``-dirty``-suffixed
    when the tree has uncommitted changes; ``"unknown"`` outside a repo / without git."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10)
        if dirty.returncode != 0:
            return "unknown"  # dirtiness undeterminable — never stamp a clean-tree claim we can't back
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_freeze(db_path: Path | str, out_path: Path | str, *, note: str = "") -> dict:
    """Read the node DB at ``db_path`` read-only, filter through :func:`freeze_reason`,
    and atomically write the freeze to ``out_path``. Returns the header dict (so a caller
    reports counts + digest without re-reading the file). Hard-errors when nothing
    survives the filter — a zero-row freeze means the wrong DB, not an empty dataset."""
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    rows = load_node_rows(db_path)  # the one node-reading seam — also lets an existing freeze re-freeze
    kept = []
    dropped: Counter[str] = Counter()
    for row in rows:
        reason = freeze_reason(row)
        if reason is None:
            kept.append(row)
        else:
            dropped[reason.split(":")[0]] += 1
    for reason, n in dropped.most_common():
        logger.info("[freeze] dropped %d row(s): %s", n, reason)
    if not kept:
        raise RuntimeError(
            f"no freezable leaf rows in {db_path} — wrong DB, or its rows predate the current featurizer vocabulary "
            f"(re-collect with the collect-node-data flow)"
        )
    kept.sort(key=lambda r: (r.gpu, r.op_sig, r.node_key))
    lines = [_row_line(r) for r in kept]
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line)
    per_gpu = Counter(r.gpu for r in kept)
    header = {
        "kind": FREEZE_KIND,
        "freeze_ver": FREEZE_VER,
        "feat_ver": FEATURIZER_VERSION,
        "knob_ver": FEATURIZER_VERSION,
        "encoding_ver": FEATURIZER_VERSION,
        "repo_commit": _repo_commit(),
        "source_db": str(Path(db_path).resolve()),
        "policy_note": note,
        "run_ids": sorted({r.run_id for r in kept if r.run_id}),
        "counts": {
            "rows": len(kept),
            "ok": sum(1 for r in kept if r.status == "ok"),
            "bench_fail": sum(1 for r in kept if r.status == "bench_fail"),
            "per_gpu": {g: per_gpu[g] for g in sorted(per_gpu)},
        },
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": digest.hexdigest(),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    # Bytes end-to-end: the digested bytes are exactly the written bytes — no platform
    # newline translation or locale encoding between the hash and the file.
    with tmp.open("wb") as fh:
        fh.write((json.dumps(header, sort_keys=True) + "\n").encode())
        fh.writelines(lines)
    tmp.replace(out)
    return header


def load_freeze(path: Path | str) -> tuple[dict, list[NodeRow]]:
    """Parse + verify the freeze at ``path`` → ``(header, leaf-only NodeRows)``. Hard
    ``RuntimeError`` — never a silent fallback — on a foreign/corrupt header, a
    ``freeze_ver`` or ``feat_ver`` mismatch, or a row-payload digest mismatch. Rows come
    back with ``parent_key=None`` / ``depth=0`` / ``visits=0`` (a freeze stores no tree
    schema) and ``feat_ver`` stamped from the header."""
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    p = Path(path)
    head, _sep, tail = p.read_bytes().partition(b"\n")
    try:
        header = json.loads(head)
    except (json.JSONDecodeError, UnicodeDecodeError):
        header = None
    regen = "re-freeze with scripts/freeze_node_store.py"
    if not isinstance(header, dict) or header.get("kind") != FREEZE_KIND:
        raise RuntimeError(f"{p} is not a measurement freeze (no {FREEZE_KIND!r} header line) — {regen}")
    if header.get("freeze_ver") != FREEZE_VER:
        raise RuntimeError(f"measurement freeze {p} has freeze_ver={header.get('freeze_ver')!r}, this code reads {FREEZE_VER} — {regen}")
    found = header.get("feat_ver")
    if found != FEATURIZER_VERSION:
        raise RuntimeError(
            f"measurement freeze {p} has feat_ver={found!r}, expected {FEATURIZER_VERSION} — its rows are spelled in a "
            f"different featurizer vocabulary; {regen}"
        )
    digest = hashlib.sha256(tail).hexdigest()
    if digest != header.get("sha256"):
        raise RuntimeError(f"measurement freeze {p} is corrupt: row payload sha256 {digest} != header {header.get('sha256')!r} — {regen}")
    rows = []
    for line in tail.splitlines():
        if not line:
            continue
        d = json.loads(line)
        rows.append(
            NodeRow(
                node_key=d["node_key"],
                parent_key=None,
                context_key=d["context_key"],
                op_sig=d["op_sig"],
                features=d["features"],
                value_us=d["value_us"],
                depth=0,
                gpu=d["gpu"],
                visits=0,
                is_leaf=True,
                variance=d["variance"],
                n_samples=d["n_samples"],
                status=d["status"],
                run_id=d["run_id"],
                measured_at=d["measured_at"],
                feat_ver=found,
            )
        )
    return header, rows


def load_node_rows(path: Path | str) -> list[NodeRow]:
    """Node rows from ``path``, whatever it is — a live sqlite tune DB (sniffed by the
    sqlite magic bytes) or a measurement freeze. The one seam that makes the two
    interchangeable for every ``Iterable[NodeRow]`` consumer; anything else fails loudly
    through :func:`load_freeze`'s header check."""
    p = Path(path)
    with p.open("rb") as fh:
        magic = fh.read(len(_SQLITE_MAGIC))
    # An empty file IS a valid (empty) sqlite DB — sqlite creates the file empty before
    # the first write (an aborted tune) — so it takes the DB branch and yields no rows.
    if magic == _SQLITE_MAGIC or not magic:
        from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

        db = SearchDB.open_readonly(p)
        try:
            return list(db.iter_nodes())
        finally:
            db.close()
    return load_freeze(p)[1]
