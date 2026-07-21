"""Pack — an on-disk bundle of execution plans for one model × GPU × serving shape.

Layout::

    pack/
      manifest.json        # validity key + environment tags + provenance + program index
      plan/<program>.json  # one serialized ExecutionPlan per compiled program (plan.py)

Kernel binaries are NOT stored here: each plan references its cubins by content-addressed
cache key into the shared ``EMMY_CUBIN_CACHE``, so multiple packs (and repeated layers within
one pack) dedupe to the same cubin files, and the docker bake ships pack + cubin cache + model
snapshot together. Weights are external too (the HF checkpoint), rebound by
``WeightSpec.source_path`` at load.

Validity: a pack loads only when its manifest matches the caller's ``key`` (model identity +
serving shape — composed by the runner) AND the current environment (backend, device arch,
nvcc toolkit tag + flags — the same tags the cubin cache keys on) AND every referenced cubin
still exists. **Any mismatch or error returns ``None`` and the caller falls back to the full
compile path** — a stale or damaged pack costs a recompile, never a wrong result. Compiler
version is deliberately NOT part of validity (a pack keeps serving its frozen snapshot);
``PLAN_FORMAT_VERSION`` gates the runtime contract instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from pathlib import Path

from emmy.compiler.backend.plan import (
    PLAN_FORMAT_VERSION,
    ExecutionPlan,
    KernelSpec,
    plan_from_dict,
    plan_to_dict,
)

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


def _environment() -> dict:
    """The environment half of the validity key — the same tags the cubin cache keys on
    (arch / toolkit / flags), so a pack never references cubins the current toolchain
    wouldn't have produced. Probes the live GPU."""
    from emmy.compiler.backend.cuda import nvcc  # noqa: PLC0415

    return {
        "backend": "cuda",
        "arch": nvcc.device_arch(False),
        "toolkit": nvcc._toolkit_tag(),
        "nvcc_flags": nvcc.effective_flags(),
    }


def _safe_name(program: str) -> str:
    return re.sub(r"[^-\w.]", "_", program)


def pack_path(root: Path | str, key: dict) -> Path:
    """The directory for one pack under the ``EMMY_PACK_DIR`` root: a human-readable model
    label plus a digest of the full validity key, so packs for different models / serving
    shapes coexist under one root."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    label = _safe_name(str(key.get("model", "model")).split("/")[-1])
    return Path(root) / f"{label}-{digest}"


def save_pack(pack_dir: Path | str, plans: dict[str, ExecutionPlan], *, key: dict, provenance: dict | None = None) -> Path:
    """Write a pack: resolve every kernel to its content-addressed cubin key (compiling into
    the cache on a miss — normally a hit, the caller just built these programs) and store the
    plans binary-keyed with no sources, plus the manifest. Requires the live GPU (arch probe).

    ``key`` is the caller-composed validity dict (model id / config hash / serving shape);
    ``provenance`` is informational only (compiler rev, tune state) and never checked."""
    from emmy.compiler.backend.cuda import nvcc  # noqa: PLC0415

    root = Path(pack_dir)
    (root / "plan").mkdir(parents=True, exist_ok=True)
    names = {p: _safe_name(p) for p in plans}
    if len(set(names.values())) != len(names):
        raise ValueError(f"pack: program names collide after sanitization: {sorted(plans)}")

    index: dict[str, str] = {}
    for program, plan in plans.items():
        kernels: dict[str, KernelSpec] = {}
        for kname, spec in plan.kernels.items():
            if spec.source is None:
                raise ValueError(f"pack: kernel {kname!r} of program {program!r} carries no source — cannot resolve a cubin key")
            cubin = nvcc.compile_to_cubin(spec.source, kname, arch=nvcc.device_arch(spec.uses_tma))
            kernels[kname] = KernelSpec(source=None, binary_key=cubin.stem, uses_tma=spec.uses_tma)
        stored = dataclasses.replace(plan, kernels=kernels)
        rel = f"plan/{names[program]}.json"
        (root / rel).write_text(json.dumps(plan_to_dict(stored)))
        index[program] = rel

    manifest = {
        "format": PLAN_FORMAT_VERSION,
        "environment": _environment(),
        "key": json.loads(json.dumps(key)),
        "programs": index,
        "provenance": provenance or {},
    }
    (root / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    logger.info("[pack] saved %d program plan(s) to %s", len(index), root)
    return root


def load_pack(pack_dir: Path | str, *, key: dict) -> dict[str, ExecutionPlan] | None:
    """Load a pack's plans, or ``None`` when anything disqualifies it — no manifest, format /
    environment / key mismatch, an unparsable plan, or a referenced cubin gone from the cache.
    ``None`` means "boot the full compile path"; the reason is logged."""
    from emmy import config  # noqa: PLC0415

    root = Path(pack_dir)
    try:
        manifest = json.loads((root / _MANIFEST).read_text())
    except (OSError, ValueError):
        logger.info("[pack] no readable manifest at %s — full compile", root)
        return None
    try:
        if manifest.get("format") != PLAN_FORMAT_VERSION:
            logger.info("[pack] format %r != runtime %r — full compile", manifest.get("format"), PLAN_FORMAT_VERSION)
            return None
        env = _environment()
        if manifest.get("environment") != env:
            logger.info("[pack] environment mismatch (pack %r vs current %r) — full compile", manifest.get("environment"), env)
            return None
        want = json.loads(json.dumps(key))
        if manifest.get("key") != want:
            logger.info("[pack] key mismatch (pack %r vs wanted %r) — full compile", manifest.get("key"), want)
            return None
        plans: dict[str, ExecutionPlan] = {}
        cache = config.cubin_cache_dir()
        for program, rel in manifest.get("programs", {}).items():
            plan = plan_from_dict(json.loads((root / rel).read_text()))
            for kname, spec in plan.kernels.items():
                if spec.binary_key is None or not (cache / f"{spec.binary_key}.cubin").exists():
                    logger.info("[pack] cubin for kernel %r of %r missing from cache — full compile", kname, program)
                    return None
            plans[program] = plan
    except Exception:  # noqa: BLE001 — a damaged pack must cost a recompile, never a crash
        logger.warning("[pack] failed to load %s — full compile", root, exc_info=True)
        return None
    logger.info("[pack] loaded %d program plan(s) from %s", len(plans), root)
    return plans
