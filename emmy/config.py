"""Single source of truth for ``EMMY_*`` environment-variable handling.

Every read or write of a ``EMMY_*`` config var goes through this module.
It is intentionally stdlib-only (no ``emmy`` imports) so that ``knob.py``
— imported transitively by every pipeline pass — can depend on it without a
cycle.

Design contract:

- ``os.environ`` stays the backing store. Bench-worker subprocesses
  (``backend/cuda/program.py``) and the ncu child (``commands/run.py``) spawn
  with ``env=dict(os.environ)``, so anything written here propagates to them;
  tests monkeypatch ``os.environ`` directly, so getters must read it **live**.
- Getters read ``os.environ`` on every call (never cache at import).
- The one setter, :func:`set_nvcc_flags`, centralizes the ``--flag > env >
  default`` override that used to live in the CLI layer, so every callsite
  (CLI, programmatic, tests) shares it.

Out of scope: provider/secret vars (``HF_TOKEN``, ``CLOUDRIFT_*``, ``GCP_*``,
``NO_COLOR``) — those stay at their use sites, and ``emmy/redact.py`` owns
secret redaction. The dynamic ``EMMY_<KNOB>`` namespace is owned by
``compiler/pipeline/knob.py``; it borrows :data:`PREFIX` / :func:`knob_var` and
the parse primitives here but keeps its own descriptor logic.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path

# --- Var-name constants (the single source of truth for spellings) ---------

PREFIX = "EMMY_"
TUNE_DB = "EMMY_TUNE_DB"
ONLINE_FILE = "EMMY_ONLINE_FILE"
OFFLINE_FILE = "EMMY_OFFLINE_FILE"
NVCC_FLAGS = "EMMY_NVCC_FLAGS"
DEBUG = "EMMY_DEBUG"
DUMP_DIR = "EMMY_DUMP_DIR"
KNOBS = "EMMY_KNOBS"
TUNE_PATIENCE = "EMMY_TUNE_PATIENCE"
TUNE_EPS = "EMMY_TUNE_EPS"
O3_TOL = "EMMY_O3_TOL"
OFFLINE_TILT = "EMMY_OFFLINE_TILT"
BENCH_BACKENDS = "EMMY_BENCH_BACKENDS"
CUBIN_CACHE = "EMMY_CUBIN_CACHE"
PACK_DIR = "EMMY_PACK_DIR"
NO_NVCC = "EMMY_NO_NVCC"
GPU_LOCK = "EMMY_GPU_LOCK"
NCU_CHILD = "EMMY_NCU_CHILD"
SERVING_STATIC = "EMMY_SERVING_STATIC"
SERVING_BATCHED = "EMMY_SERVING_BATCHED"
GEN_DECODE_BUCKET = "EMMY_GEN_DECODE_BUCKET"
GEN_M1_TIER = "EMMY_GEN_M1_TIER"
GEN_ALIAS_ATTN = "EMMY_GEN_ALIAS_ATTN"
GEN_PREFILL_BUCKET = "EMMY_GEN_PREFILL_BUCKET"
READABLE = "EMMY_READABLE"

_CACHE_ROOT = Path.home() / ".cache" / "emmy"

# The 2026-07 prior rename (analytic → offline, learned → online) respelled three
# env vars. The old spellings keep working with a one-time DeprecationWarning —
# they live in shell profiles and remote-run scripts, unlike the Python names.
_LEGACY_VARS = {
    ONLINE_FILE: "EMMY_PRIOR_FILE",
    OFFLINE_FILE: "EMMY_ANALYTIC_FILE",
    OFFLINE_TILT: "EMMY_ANALYTIC_TILT",
}


def _env_aliased(name: str) -> str | None:
    """Live ``os.environ`` read that also honors the var's pre-rename legacy
    spelling (the new name wins when both are set; a legacy hit warns)."""
    raw = os.environ.get(name)
    if raw is not None:
        return raw
    legacy = _LEGACY_VARS.get(name)
    raw = os.environ.get(legacy) if legacy else None
    if raw is not None:
        warnings.warn(f"{legacy} is deprecated — use {name}", DeprecationWarning, stacklevel=3)
    return raw


def knob_var(name: str) -> str:
    """The ``EMMY_<NAME>`` env-var key for a knob named ``name``.

    Sole place the knob-name → env-var join lives. Used by
    :class:`~emmy.compiler.pipeline.knob.Knob` (via ``Knob.env``) and the
    ``EMMY_KNOBS`` splat."""
    return f"{PREFIX}{name.upper()}"


def knob_raw(name: str) -> str | None:
    """Raw string value of the knob env var ``EMMY_<NAME>``, or ``None`` if
    unset. The per-type decode (INT / BOOL / BINMASK) stays in the ``Knob``
    descriptor (``compiler/pipeline/knob.py``); this is just the env read."""
    return os.environ.get(knob_var(name))


def knobs_aggregate() -> str:
    """Raw ``EMMY_KNOBS`` aggregate string (``""`` if unset)."""
    return _str(KNOBS)


def set_knob(name: str, value: str, *, overwrite: bool = True) -> bool:
    """Write ``EMMY_<NAME>=value`` into ``os.environ`` (so pipeline passes
    and bench subprocesses see it). With ``overwrite=False`` only writes when the
    key is absent. Returns ``True`` iff a write happened."""
    key = knob_var(name)
    if not overwrite and key in os.environ:
        return False
    os.environ[key] = value
    return True


def unset_knob(name: str) -> None:
    """Remove the ``EMMY_<NAME>`` env var (no-op when absent) — the restore half of a
    scoped knob pin (``Knob.pinned``)."""
    os.environ.pop(knob_var(name), None)


# --- Shared parse primitives -----------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def _bool(name: str, default: bool = False) -> bool:
    """Truthy env read. ``{"1","true","yes","on"}`` (case-insensitive) → True;
    unset → ``default``; anything else → False."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def int_env(name: str, default: int) -> int:
    """Int env read. Empty / unset / unparseable → ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    """Float env read. Empty / unset / unparseable → ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --- Typed getters (read os.environ live) ----------------------------------


def tune_db_path() -> Path:
    """Autotune SQLite cache path: ``EMMY_TUNE_DB`` → ``~/.cache/emmy/autotune.db``.

    The shared resolution for ``compile`` / ``run`` / ``tune`` / ``knobs`` and
    for :class:`CudaBackend` constructed with ``tune_db="auto"``. The path is
    advisory — the engine only opens it when the file exists."""
    override = os.environ.get(TUNE_DB)
    return Path(override) if override else _CACHE_ROOT / "autotune.db"


def online_path() -> Path:
    """Online-prior checkpoint file: ``EMMY_ONLINE_FILE`` (legacy
    ``EMMY_PRIOR_FILE``) → ``~/.cache/emmy/online.json``. A single JSON file (not
    the tune DB) holding the one global prior; ``tune`` writes it, ``compile`` /
    ``run`` read it. A pre-rename ``prior.json`` already in the cache keeps being
    used (and written) so existing checkpoints survive the rename."""
    override = _env_aliased(ONLINE_FILE)
    if override:
        return Path(override)
    path = _CACHE_ROOT / "online.json"
    legacy = _CACHE_ROOT / "prior.json"
    return legacy if legacy.exists() and not path.exists() else path


@contextmanager
def online_file_override(path: str | Path | None):
    """Temporarily point ``EMMY_ONLINE_FILE`` at ``path`` (``None`` is a no-op).

    The golden drift audit uses this with a nonexistent path so a compile's
    evidence hierarchy sees NO machine-local online prior / reservoir — the
    golden tier plus the repo-shipped offline prior are the only inputs, making
    MATCH/DRIFT verdicts machine-independent."""
    if path is None:
        yield
        return
    prev = os.environ.get(ONLINE_FILE)
    os.environ[ONLINE_FILE] = str(path)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(ONLINE_FILE, None)
        else:
            os.environ[ONLINE_FILE] = prev


def offline_path() -> Path | None:
    """Offline-prior weights artifact override: ``EMMY_OFFLINE_FILE`` (legacy
    ``EMMY_ANALYTIC_FILE``) → ``None``.

    ``None`` means the repo-checked default (``offline_weights.json`` next to
    ``search/prior/offline.py`` — package-relative, so it resolves there, not
    here). Swap in a candidate fit for an A/B by pointing this at another
    artifact; a version-mismatched or missing file is a hard error, never a
    silent fallback."""
    override = _env_aliased(OFFLINE_FILE)
    return Path(override) if override else None


def nvcc_flags() -> str:
    """Extra nvcc flags for this compile (``EMMY_NVCC_FLAGS``, ``""`` if unset).

    Read fresh each call so a per-invocation override (set via
    :func:`set_nvcc_flags`) and the bench-worker subprocess (which inherits the
    env) see the same value, and so the flags fold into cache keys."""
    return _str(NVCC_FLAGS)


@contextmanager
def nvcc_flags_override(flags: str | None):
    """Temporarily swap ``EMMY_NVCC_FLAGS`` for one compile (e.g. re-benching
    a tune winner at ``-Xcicc -O3``). ``None`` is a no-op. Since ``nvcc_flags`` is
    read fresh and folds into the cubin cache key, this transparently selects the
    right (and separately-cached) cubin. Applied in whichever process compiles —
    the bench worker reads it off the request and wraps its own compile."""
    if flags is None:
        yield
        return
    prev = os.environ.get(NVCC_FLAGS)
    os.environ[NVCC_FLAGS] = flags
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(NVCC_FLAGS, None)
        else:
            os.environ[NVCC_FLAGS] = prev


def debug_enabled() -> bool:
    """``EMMY_DEBUG`` — per-launch debug dump path in the CUDA backend."""
    return _bool(DEBUG)


def readable() -> bool:
    """``EMMY_READABLE`` — batch-enable the readability-only codegen policies (currently ``LOOPIFY``),
    which re-spell the emitted CUDA for legible ``--ir`` listings but are SASS-identical. Default off;
    the ``compile`` CLI turns it on (so inspection output is readable) while ``tune`` / ``run`` /
    ``bench`` leave it off to keep production codegen — and compiler work — minimal."""
    return _bool(READABLE)


def set_readable(on: bool, *, overwrite: bool = False) -> None:
    """Set ``EMMY_READABLE`` in ``os.environ`` so the pipeline (and bench subprocesses) see it. With
    ``overwrite=False`` (the default), an explicit user setting wins — the ``compile`` CLI calls this
    to default readability ON without clobbering ``EMMY_READABLE=0``."""
    if overwrite or READABLE not in os.environ:
        os.environ[READABLE] = "1" if on else "0"


def dump_dir() -> Path | None:
    """``EMMY_DUMP_DIR`` as an expanded ``Path``, or ``None`` when unset."""
    raw = os.environ.get(DUMP_DIR)
    return Path(raw).expanduser() if raw else None


def tune_patience(default: int = 50) -> int:
    """``EMMY_TUNE_PATIENCE`` — inner-MCTS patience fallback for ``tune``."""
    return int_env(TUNE_PATIENCE, default)


def tune_eps(default: float = 0.0) -> float:
    """``EMMY_TUNE_EPS`` — inner-MCTS ε-greedy exploration fraction: the
    probability a selection step descends a uniformly random child instead of the
    PUCT argmax. Opt-in (default ``0`` = deterministic PUCT): on the fp16 sweep it
    didn't recover the lost configs (the gap is a tune-path eligibility issue, not
    selection) and pure randomness regresses
    tuning, so it's a knob for shapes where the heuristic order is known-bad, not a
    default."""
    return float_env(TUNE_EPS, default)


def o3_tol(default: float = 0.15) -> float:
    """``EMMY_O3_TOL`` — tolerance band (fraction of the best -O1 latency)
    within which a tuned config is also re-benched at -O3 for a deployable prior
    sample. ``0.15`` = re-bench everything within 15% of the best -O1."""
    return float_env(O3_TOL, default)


def offline_tilt(default: float = 0.3) -> float:
    """``EMMY_OFFLINE_TILT`` (legacy ``EMMY_ANALYTIC_TILT``) — exponent ``W`` of
    the cold ``OfflinePrior`` multiplier in :meth:`FallbackPrior.score` (selection
    only): the online µs are tilted by ``offline**W`` so the heuristic's ranking
    nudges PUCT exploration toward configs it favors without overriding the online
    scale (``W=0`` = pure online, large ``W`` = offline dominates). See the method
    docstring."""
    raw = _env_aliased(OFFLINE_TILT)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def serving_static(default: bool = False) -> bool:
    """``EMMY_SERVING_STATIC`` — opt into the serving plugin's fully-static
    program: **static extents for both batch and seq_len**. Off (default) keeps the
    symbolic-seq path; on builds ONE static ``(max_num_seqs, max_seq_len)``
    program (batch from vLLM's ``--max-num-seqs``, seq from ``--max-model-len``) and
    runs each scheduler step as a padded batched forward. Only efficient for
    fixed-length workloads — it pads every sequence to ``max_seq_len``; prefer
    ``EMMY_SERVING_BATCHED`` (pads only to the step's longest sequence). A deliberate
    opt-in, not a default. See `serving/ARCHITECTURE.md`."""
    return _bool(SERVING_STATIC, default)


def serving_batched(default: bool = False) -> bool:
    """``EMMY_SERVING_BATCHED`` — opt into the serving plugin's **batched
    symbolic-seq** program: batch extent static at vLLM's ``--max-num-seqs``, seq_len
    symbolic. Each scheduler step runs as ONE batched forward padded to the step's
    longest sequence (not to ``max_seq_len`` — the static mode's waste). Off (default)
    keeps the one-sequence-per-forward path. An opt-in because the shared buffer set is
    allocated at ``(max_num_seqs, max_seq_len)`` capacity — pair it with a sane
    ``--max-num-seqs`` (vLLM's default is 256) and workload-sized ``--max-model-len``.
    ``EMMY_SERVING_STATIC`` takes precedence when both are set.
    See `serving/ARCHITECTURE.md`."""
    return _bool(SERVING_BATCHED, default)


def gen_prefill_bucket(default: int = -1) -> int:
    """``EMMY_GEN_PREFILL_BUCKET`` — the generative runner's static prefill-chunk M.
    The default ``-1`` means "vLLM's ``max_num_batched_tokens``" (chunked prefill fills
    steps to it whenever the queue is deep, so the static twin runs exact grids on the
    hot chunk width); **0 disables** the twin (prefill stays on the symbolic masked-tile
    programs). See `serving/gen_runner.py`."""
    return int_env(GEN_PREFILL_BUCKET, default)


def gen_m1_tier(default: int = 1) -> int:
    """``EMMY_GEN_M1_TIER`` — build and route the static M=1 (gemv-class) decode twins for
    T=1 steps (default 1 = ON since 2026-07-24). The tier's matvecs run the transposed
    ``g<w>k/b<n>t`` partition at the k-major serving layout's gemv floor (o_proj 8.1 µs,
    qkv 14, down 76, gate_up 142 on the 5090), and the flip criterion held e2e: c=1 TPOT
    17.92/18.98 beats the bucket-32 path's 18.0/19.1 with c=8/c=64 unchanged (m1 routes
    only T==1). Set 0 to fall back to bucket-padded decode at T=1. The remaining ~1.6 ms
    to stock is the split-chain / kernel-count / seam work of
    plans/decode-parity-closers.md. See `serving/gen_runner.py`."""
    return int_env(GEN_M1_TIER, default)


def gen_alias_attn(default: int = 1) -> int:
    """``EMMY_GEN_ALIAS_ATTN`` — write vLLM's paged-attention output DIRECTLY into the M=1 post
    twin's ``attn_out`` input backing (default 1 = ON since 2026-07-24). Kills the per-layer
    protective D2D upload copy at T=1 decode: ``upload_prefix_device`` self-copy-skips on
    pointer equality. The alias holds across steps by construction — the m1 program's input
    arrays are allocated once, and the outer whole-step decode capture fixes the launch order
    (attention writes, the post program reads, before the next replay overwrites). Flip
    verdict: c=1 TPOT 17.89→17.83 (256) / 18.92→18.88 (4K) with c=64 / 4K c=8 unchanged and
    greedy completions token-identical to the copy path. Set 0 to restore the copy path.
    See ``serving/vllm_model_gen.py``."""
    return int_env(GEN_ALIAS_ATTN, default)


def gen_decode_bucket(default: int = 16) -> int:
    """``EMMY_GEN_DECODE_BUCKET`` — the generative runner's static decode-bucket M
    (default 16). Set to **0 to disable** the per-layer decode-bucket programs, which
    roughly halves the runner's on-GPU weight footprint (no static `pre`/`post` twin) at
    the cost of slower decode — needed to fit a large model (e.g. gemma-4-12B) on a
    smaller card. See `serving/gen_runner.py`."""
    return int_env(GEN_DECODE_BUCKET, default)


def bench_backends_raw(cli_value: str | None) -> str:
    """Raw comma-separated bench-backend selection. Precedence: ``cli_value`` >
    ``EMMY_BENCH_BACKENDS`` > ``"eager,emmy"``. Backend-key
    normalization stays at the call site (``run.py:_resolve_backends``)."""
    return cli_value or os.environ.get(BENCH_BACKENDS) or "eager,emmy"


def cubin_cache_dir() -> Path:
    """Content-addressed cubin cache dir: ``EMMY_CUBIN_CACHE`` → ``~/.cache/emmy/cubin``."""
    override = os.environ.get(CUBIN_CACHE)
    return Path(override) if override else _CACHE_ROOT / "cubin"


def pack_dir() -> Path | None:
    """``EMMY_PACK_DIR`` — root directory for execution-plan packs (``backend/pack.py``).
    When set, the serving runner loads a matching pack (skipping trace / pipeline / fork
    resolution / codegen) and writes one after a full compile. ``None`` (unset) disables."""
    override = os.environ.get(PACK_DIR)
    return Path(override) if override else None


def nvcc_disabled() -> bool:
    """``EMMY_NO_NVCC`` — force the cupy/NVRTC path instead of offline nvcc."""
    return _bool(NO_NVCC)


def gpu_lock_path() -> str | None:
    """``EMMY_GPU_LOCK`` path, or ``None`` for the no-op (unset) case."""
    return os.environ.get(GPU_LOCK)


def ncu_child() -> bool:
    """``EMMY_NCU_CHILD`` — set in the ncu-profiled child to prevent
    recursive re-spawning of ncu."""
    return _bool(NCU_CHILD)


# Note: ``EMMY_GROUP_M`` (CTA-swizzle row-group size) used to live here as
# a bespoke getter. It is now a real ``Knob`` descriptor in its owning rule
# (``025_swizzle_blocks.py``) so it shows up in ``emmy knobs`` and reads
# through the descriptor's env path. Env access still routes through this
# module's ``knob_raw`` / ``int_env`` primitives.


# --- Setters (write os.environ so subprocesses inherit) --------------------


def set_nvcc_flags(cli_value: str | None, default: str) -> str:
    """Resolve and publish the effective extra nvcc flags via
    ``EMMY_NVCC_FLAGS`` (the carrier the cubin compiler, the bench-worker
    subprocess, and ``Context.structural_key`` all read).

    Precedence: ``cli_value`` (a ``--nvcc-flags`` override, when not ``None``) >
    a pre-set env var > ``default`` (per-command policy: ``""`` for compile/run,
    ``"-Xcicc -O1"`` for tune). Must run before any compile/bench. Returns the
    effective string."""
    if cli_value is not None:
        os.environ[NVCC_FLAGS] = cli_value
    elif NVCC_FLAGS not in os.environ:
        os.environ[NVCC_FLAGS] = default
    return os.environ.get(NVCC_FLAGS, "")
