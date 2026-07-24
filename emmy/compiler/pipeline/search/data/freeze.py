"""Measurement freeze — a digest-pinned, golden-spelled snapshot of the tune DB's ``node`` table.

The node DB is a live store (tunes and merges write into it), so a model fit directly from
it is not reproducible: two runs of the same fitter can see different data. A *freeze* is
a local snapshot extracted from the DB whose digest pins exactly which measurements a fit
saw — the fit becomes a pure function of (repo, freeze digest).

Freeze v2 is a DIRECTORY of per-GPU YAML files mirroring the ``goldens/`` layout — a
``gpu_name`` / ``compute_cap`` header plus a ``configs`` list — beside a ``manifest.json``
carrying the provenance header and the content digests. Each row is spelled like a golden
entry: the declarative ``shape_spec`` captured at collection time (``kernel`` + shape
fields + ``dynamic`` — ``run --bench --record-shape``, driven by the golden-neighborhood
sweep), the verbatim TUNABLE knobs, and the measurement extension block (``value_us``,
``opt`` — the nvcc cicc lane, ``status``, ``variance`` / ``n_samples``, ``run_id`` /
``measured_at``). Nothing featurized is persisted: the loader re-derives ``H_*`` from the
gpu registry card-faithfully (``Context.from_target``, ``H_opt`` overridden from ``opt``)
and the full ``S_*`` histogram by re-tracing the kind's snippet
(:func:`~..data.sample.traced_s_features`; arithmetic fallback with a warning), so an
encoding-only featurizer change never quarantines a freeze — re-loading IS the refit's
re-featurization.

What freezes (see :func:`freeze_reason`): every **leaf** in the current featurizer
vocabulary that passes the physical-plausibility predicates AND carries a declarative
``shape_spec`` — identity-less legacy rows (tune-written, pre-identity benches) stay in
the DB but never freeze. ``bench_fail`` leaves are kept as durable negatives. Branch rows
never freeze and no tree schema is stored (no ``parent_key`` / ``depth`` / ``visits``):
prefix rows are re-synthesized at fit time under the current fork structure.

Determinism contract: freezing the same DB twice yields the same digests. Every row
serializes to one canonical JSON line (sorted keys, fixed separators, ``allow_nan=False``);
rows within a file sort by that line (total, content-derived order); the per-file
``sha256`` covers exactly those lines — NOT the YAML bytes, so integrity is content-level
and immune to YAML style — and the manifest's top-level ``sha256`` folds the sorted
per-file digests. ``created_at`` never enters any digest.

:func:`load_freeze` hard-errors — never a silent fallback — on a missing/foreign/corrupt
manifest, a ``freeze_ver`` mismatch, a manifest-listed file missing, a per-file digest
mismatch, or a row that fails to instantiate its golden kind class. There is deliberately
NO load-time ``feat_ver`` gate (the v1 rule): features are re-derived by live code, so the
stored ``feat_ver`` / ``knob_ver`` / ``encoding_ver`` are provenance, not a contract.
:func:`load_node_rows` is the interchange seam: it sniffs a path and yields ``NodeRow``s
from either a live sqlite DB (file) or a freeze (directory), so the nodes-dataset
consumers (``eval online --dataset nodes``, ``Dataset.from_node_rows`` /
``fold_node_rows``) accept both. Loaded rows carry ``parent_key=None`` / ``depth=0`` /
``visits=0``: the fork diagnostics skip parentless rows and the golden-anchored descent
treats them as tree-less, so a freeze degrades to the leaf-level metrics. A v1 JSONL
freeze is refused with a re-freeze pointer — freezes are regenerable from the DB.

Produced by ``scripts/freeze_node_store.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.db import NodeRow

logger = logging.getLogger(__name__)

FREEZE_KIND = "emmy-node-freeze"
FREEZE_VER = 2
MANIFEST_NAME = "manifest.json"

_SQLITE_MAGIC = b"SQLite format 3\x00"

# The measurement extension block of a freeze row — every payload key that is NOT part of
# the declarative shape_spec (the loader splits a row back into the two halves by this set).
_EXTENSION_FIELDS = frozenset({"knobs", "value_us", "opt", "status", "variance", "n_samples", "run_id", "measured_at"})


def freeze_reason(row: NodeRow) -> str | None:
    """Why ``row`` is excluded from a measurement freeze, or ``None`` to keep it.

    THE freeze sanity filter, and nothing else — keep every identity-carrying leaf
    spelled in the current featurizer vocabulary that passes the shared plausibility
    predicates. ``bench_fail`` leaves reach the keep path by construction: both
    predicates return ``None`` for non-``ok`` rows, so failures are kept as negative
    examples without a special case. The ``shape_spec`` requirement is the whole
    legacy-exclusion mechanism: only rows recorded with a declarative identity
    (``run --bench --record-shape``) can spell a golden-format freeze row."""
    from emmy.compiler.pipeline.search.db import implausible_value_reason, impossible_kernel_reason  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    if row.is_leaf is not True:
        return "non-leaf (branch or pre-enrichment row)"
    if row.feat_ver != FEATURIZER_VERSION:
        return f"stale feat_ver {row.feat_ver} != current {FEATURIZER_VERSION}"
    if row.shape_spec is None:
        return "no declarative identity (legacy tune row)"
    if "H_cc" not in row.features or "H_opt" not in row.features:
        return "missing H_* regime stamps"
    reason = implausible_value_reason(row)
    if reason is not None:
        return f"implausible value: {reason}"
    reason = impossible_kernel_reason(row)
    if reason is not None:
        return f"impossible kernel: {reason}"
    return None


def _row_payload(row: NodeRow) -> dict:
    """One freeze row as its YAML entry dict — the golden-entry spelling: the declarative
    ``shape_spec`` flattened in, the verbatim tunable knobs, and the measurement
    extension block. Nothing featurized: ``H_*`` / ``S_*`` are dropped here and
    re-derived at load."""
    from emmy.compiler.pipeline.search.data.sample import _split_by_prefix  # noqa: PLC0415

    tunable, _ctx, _s = _split_by_prefix(row.features)
    return {
        **row.shape_spec,
        "knobs": tunable,
        "value_us": row.value_us,
        "opt": int(row.features["H_opt"]),
        "status": row.status,
        "variance": row.variance,
        "n_samples": row.n_samples,
        "run_id": row.run_id,
        "measured_at": row.measured_at,
    }


def _row_line(payload: dict) -> bytes:
    """A payload's canonical JSON line (bytes) — sorted keys, fixed separators, no NaN
    tokens (``allow_nan=False`` hard-errors instead of emitting nonstandard JSON). This
    exact spelling is the row sort order AND the digested content, so determinism and
    integrity are content-level, immune to YAML style."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _gpu_filename(gpu_name: str, cap: tuple[int, int]) -> str:
    """The per-GPU YAML file name, mirroring the ``goldens/`` convention —
    e.g. ``nvidia_geforce_rtx_4090_sm89.yaml``."""
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "unknown_gpu"
    return f"{slug}_sm{cap[0]}{cap[1]}.yaml"


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


def write_freeze(db_path: Path | str, out_dir: Path | str, *, note: str = "") -> dict:
    """Read the node store at ``db_path`` (a live DB or an existing freeze — the
    :func:`load_node_rows` seam) read-only, filter through :func:`freeze_reason`, and
    atomically write the freeze DIRECTORY at ``out_dir``: one per-``(gpu, compute_cap)``
    YAML file plus ``manifest.json``. Returns the manifest dict (so a caller reports
    counts + digest without re-reading). Hard-errors when nothing survives the filter —
    a zero-row freeze means the wrong DB (or an identity-less legacy store), not an
    empty dataset. An existing ``out_dir`` is replaced only when it is itself a freeze
    (has a manifest) — anything else is refused rather than deleted."""
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    rows = load_node_rows(db_path)
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
            f"no freezable leaf rows in {db_path} — wrong DB, or its rows carry no declarative identity / predate the "
            f"current featurizer vocabulary (re-collect with the collect-node-data flow)"
        )

    # Group per (gpu, compute_cap) — cap comes off the row's stamped H_cc (major*10+minor),
    # the same encoding Context.features writes.
    by_card: dict[tuple[str, tuple[int, int]], list] = {}
    for row in kept:
        cap = divmod(int(row.features["H_cc"]), 10)
        by_card.setdefault((row.gpu, cap), []).append(row)

    out = Path(out_dir)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    files: dict[str, dict] = {}
    for (gpu, cap), card_rows in sorted(by_card.items(), key=lambda kv: kv[0]):
        payloads = sorted((_row_payload(r) for r in card_rows), key=_row_line)
        digest = hashlib.sha256()
        for p in payloads:
            digest.update(_row_line(p))
        name = _gpu_filename(gpu, cap)
        if name in files:
            raise RuntimeError(f"freeze file name collision: {name} (cards {files[name]['gpu_name']!r} and {gpu!r})")
        doc = {"gpu_name": gpu, "compute_cap": list(cap), "configs": payloads}
        (tmp / name).write_text(yaml.safe_dump(doc, sort_keys=True, width=120))
        files[name] = {
            "gpu_name": gpu,
            "compute_cap": list(cap),
            "rows": len(payloads),
            "sha256": digest.hexdigest(),
        }

    top = hashlib.sha256()
    for name in sorted(files):
        top.update(files[name]["sha256"].encode())
    per_gpu = Counter(r.gpu for r in kept)
    manifest = {
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
        "files": files,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": top.hexdigest(),
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if out.exists():
        if not (out.is_dir() and (out / MANIFEST_NAME).exists()):
            shutil.rmtree(tmp)
            raise RuntimeError(f"{out} exists and is not a measurement freeze — refusing to replace it")
        shutil.rmtree(out)
    tmp.replace(out)
    return manifest


def load_freeze(path: Path | str) -> tuple[dict, list[NodeRow]]:
    """Parse + verify the freeze directory at ``path`` → ``(manifest, leaf-only
    NodeRows)`` with features RE-DERIVED by live code (see the module docstring). Hard
    ``RuntimeError`` — never a silent fallback — on any integrity failure. Identity keys
    of a loaded row: ``op_sig`` is the canonical JSON of its ``shape_spec`` (a stable
    declarative op id — the ``fold_node_rows(by="op")`` key, shared by a shape's
    -O1/-O3 twins), ``context_key`` spells ``cap<maj>.<min>-O<opt>``."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data.sample import traced_s_features  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import _KERNEL_CLASSES  # noqa: PLC0415
    from emmy.compiler.structural import digest as key_digest  # noqa: PLC0415

    p = Path(path)
    regen = "re-freeze with scripts/freeze_node_store.py"
    mpath = p / MANIFEST_NAME
    if not mpath.exists():
        raise RuntimeError(f"{p} is not a measurement freeze (no {MANIFEST_NAME}) — {regen}")
    try:
        manifest = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"measurement freeze {p} has a corrupt manifest: {e} — {regen}") from e
    if not isinstance(manifest, dict) or manifest.get("kind") != FREEZE_KIND:
        raise RuntimeError(f"{p} is not a measurement freeze (manifest kind != {FREEZE_KIND!r}) — {regen}")
    if manifest.get("freeze_ver") != FREEZE_VER:
        raise RuntimeError(f"measurement freeze {p} has freeze_ver={manifest.get('freeze_ver')!r}, this code reads {FREEZE_VER} — {regen}")

    rows: list[NodeRow] = []
    h_cache: dict[tuple[str, tuple[int, int]], dict] = {}
    s_cache: dict[str, dict] = {}
    for name in sorted(manifest.get("files", {})):
        info = manifest["files"][name]
        fpath = p / name
        if not fpath.exists():
            raise RuntimeError(f"measurement freeze {p} is missing {name} (listed in the manifest) — {regen}")
        doc = yaml.safe_load(fpath.read_text())
        gpu_name, cap = doc["gpu_name"], tuple(doc["compute_cap"])
        payloads = doc.get("configs") or []
        file_digest = hashlib.sha256()
        for payload in payloads:
            file_digest.update(_row_line(payload))
        if file_digest.hexdigest() != info.get("sha256"):
            raise RuntimeError(f"measurement freeze {p} is corrupt: {name} row digest != manifest — {regen}")

        if (gpu_name, cap) not in h_cache:
            # Card-faithful H_* off the gpu registry (the Sample.from_golden recipe);
            # H_opt is overridden per row — from_target reads the ENV nvcc flags.
            h_cache[(gpu_name, cap)] = Context.from_target(cap, gpu_name=gpu_name).features()
        h_base = h_cache[(gpu_name, cap)]

        for payload in payloads:
            spec = {k: v for k, v in payload.items() if k not in _EXTENSION_FIELDS}
            spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
            if spec_json not in s_cache:
                cls = _KERNEL_CLASSES.get(spec.get("kernel"))
                if cls is None:
                    raise RuntimeError(f"measurement freeze {p}: {name} row has unknown kernel kind {spec.get('kernel')!r} — {regen}")
                try:
                    cfg = cls(
                        name="freeze", gpu_name=gpu_name, compute_cap=cap, knobs={}, **{k: v for k, v in spec.items() if k != "kernel"}
                    )
                except (TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"measurement freeze {p}: {name} row does not instantiate {spec.get('kernel')!r}: {e} — {regen}"
                    ) from e
                traced = traced_s_features(cfg.snippet(), tuple(cfg.dynamic_specs()), cfg.shape_key())
                if traced is None:
                    logger.warning(
                        "[freeze] %s: no traced op keys to the %s shape — falling back to arithmetic S_* extents", name, spec.get("kernel")
                    )
                s_cache[spec_json] = dict(traced) if traced is not None else cfg.shape_key().s_features_arith()
            knobs = payload["knobs"] or {}
            opt = int(payload["opt"])
            features = {**h_base, "H_opt": float(opt), **s_cache[spec_json], **knobs}
            rows.append(
                NodeRow(
                    node_key=key_digest("freeze2", gpu_name, spec_json, opt, tuple(sorted((k, str(v)) for k, v in knobs.items()))),
                    parent_key=None,
                    context_key=f"cap{cap[0]}.{cap[1]}-O{opt}",
                    op_sig=spec_json,
                    features=features,
                    value_us=payload["value_us"],
                    depth=0,
                    gpu=gpu_name,
                    visits=0,
                    is_leaf=True,
                    variance=payload["variance"],
                    n_samples=payload["n_samples"],
                    status=payload["status"],
                    run_id=payload["run_id"],
                    measured_at=payload["measured_at"],
                    feat_ver=FEATURIZER_VERSION,  # features were just derived by live code
                    shape_spec=spec,
                )
            )
    return manifest, rows


def load_node_rows(path: Path | str) -> list[NodeRow]:
    """Node rows from ``path``, whatever it is — a live sqlite tune DB (a file, sniffed
    by the sqlite magic bytes) or a measurement freeze (a directory). The one seam that
    makes the two interchangeable for every ``Iterable[NodeRow]`` consumer; anything
    else fails loudly. A v1 JSONL freeze is recognized and refused with a re-freeze
    pointer — no v1 reader survives (freezes are regenerable from the DB)."""
    p = Path(path)
    if p.is_dir():
        return load_freeze(p)[1]
    with p.open("rb") as fh:
        head = fh.read(max(len(_SQLITE_MAGIC), 4096))
    # An empty file IS a valid (empty) sqlite DB — sqlite creates the file empty before
    # the first write (an aborted tune) — so it takes the DB branch and yields no rows.
    if head[: len(_SQLITE_MAGIC)] == _SQLITE_MAGIC or not head:
        from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

        db = SearchDB.open_readonly(p)
        try:
            return list(db.iter_nodes())
        finally:
            db.close()
    first_line = head.partition(b"\n")[0]
    try:
        old = json.loads(first_line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        old = None
    if isinstance(old, dict) and old.get("kind") == FREEZE_KIND:
        raise RuntimeError(
            f"{p} is a v1 JSONL measurement freeze — superseded by the per-GPU YAML freeze directory (freeze_ver "
            f"{FREEZE_VER}); re-freeze with scripts/freeze_node_store.py"
        )
    raise RuntimeError(f"{p} is neither a sqlite node DB nor a measurement freeze directory")
