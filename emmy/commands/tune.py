"""Autotune CudaOps produced by the lowering pipeline and cache results."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from emmy import config
from emmy.commands.compile import (
    add_diagnostics_args,
    add_input_args,
    add_nvcc_args,
    apply_nvcc_flags,
    format_stage,
    load_or_trace,
    resolve_tune_db,
    setup_pipeline_runtime,
    validate_trace_adapter_args,
)
from emmy.compiler.pipeline import TuningSearch
from emmy.compiler.pipeline.search.working_golden import (
    WorkingGoldenTarget,
    load_working_targets,
    measure_proposals,
    persist_proposal_rankings,
    persist_tune_winner,
    validate_working_gpu,
)

logger = logging.getLogger(__name__)


def register_tune_command(subparsers):
    parser = subparsers.add_parser(
        "tune",
        allow_abbrev=False,
        help=(
            "Bench every CudaOp produced by the lowering pipeline, attribute per-kernel "
            "latency to every ancestor along Op.source, and write the rows to the tuning cache."
        ),
    )
    add_input_args(parser)
    parser.add_argument(
        "--kernel",
        help="With --golden-file, tune only entries whose target name contains this substring.",
    )
    parser.add_argument(
        "--golden-file",
        metavar="PATH",
        help=(
            "Tune every target in a working golden YAML file. Each target is reconstructed from embedded stable "
            "Torch IR plus provenance, or from its Loop IR fallback. Entries with a knobs mapping are measured before MCTS and ranking "
            "results are written back to the working file."
        ),
    )
    parser.add_argument("--output", "-o", help="Output path for the tuned CUDA IR")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Stop after this many consecutive measured variants haven't beaten the current best latency. "
            "Falls back to ``EMMY_TUNE_PATIENCE`` env var, then to 50."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Hard candidate budget per tuned kernel. Every working-golden proposal reserves one slot before MCTS "
            "(including a cached replay); MCTS DB cache hits do not spend its remaining live-measurement slots. "
            "By default patience alone stops the search."
        ),
    )
    parser.add_argument(
        "--ucb-c",
        type=float,
        default=TuningSearch.DEFAULT_UCB_C,
        help=(
            "UCB1 exploration constant. The canonical value is sqrt(2) ≈ 1.414; larger values "
            f"shift the walk toward exploration. Default: {TuningSearch.DEFAULT_UCB_C:.4f}."
        ),
    )
    parser.add_argument(
        "--explore-eps",
        type=float,
        default=None,
        help=(
            "ε-greedy exploration: probability a selection step descends a uniformly random child "
            "instead of the PUCT argmax, perturbing (not replacing) the heuristic order for shapes "
            "where it's known-bad. Falls back to ``EMMY_TUNE_EPS`` env var, then to 0.0 "
            "(deterministic PUCT) — opt-in."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Before tuning, delete the tuning DB and the cubin/kernel caches for a fresh sweep.",
    )
    parser.add_argument(
        "--bench",
        "-b",
        action="store_true",
        help=(
            "After tuning, re-bench the winner at -O3 (deployable numbers, NOT the -O1 ranking pass): the full "
            "compiled model and each individual kernel (via its in-memory frontend provenance slice) vs eager "
            "PyTorch / torch.compile / Emmy, then print a comparison table. Writes an HTML per-kernel chart "
            "to <dump-dir>/kernels.html when a dump dir is set. Can take minutes on a large model."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --bench random inputs (default: 0).")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help=(
            "Tune the post-fusion kernels concurrently across this many GPUs (devices 0..N-1): one in-flight "
            "bench per GPU on a single event loop. Default: single-GPU (serial, behaviorally identical). "
            "Bounded by the number of unique kernels; devices must be homogeneous. ``--devices`` overrides."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated GPU ids to tune across (e.g. ``0,1,3``), the explicit form of ``--gpus``. "
            "Devices must be homogeneous (one perf key per tune). Default: single-GPU."
        ),
    )
    parser.add_argument(
        "--bench-backends",
        default="eager,tcompile,emmy",
        help=(
            "Comma-separated subset of backends to time under --bench: any of ``eager``, ``tcompile`` "
            "(torch.compile), ``emmy``. Default: all three — tune --bench is the deployable comparison, "
            "so torch.compile's ~0.8s JIT is worth paying. ``emmy`` is always included."
        ),
    )
    add_diagnostics_args(parser)
    add_nvcc_args(parser)
    parser.set_defaults(func=handle_tune)


def _tune_offline(args):
    """``emmy tune`` with no op: refit the global online prior on its
    persisted reservoir dataset and print offline diagnostics — no GPU, no
    benching. Answers "can the prior reach the best configs?" over everything
    tuned so far."""
    from emmy import config
    from emmy.compiler.pipeline.search.prior import OnlinePrior, diagnostics

    prior = OnlinePrior.load(seed=args.seed)
    if not prior._dataset:
        logger.error("no prior dataset at %s — run `emmy tune <model>` first", config.online_path())
        sys.exit(1)
    sys.stderr.write(f"[tune] offline refit on {len(prior._dataset)} rows from {config.online_path()}\n")
    prior.fit()  # unconditional re-fit on the whole accumulated dataset
    prior.calibration = prior._reservoir_calibration()  # refresh the trustworthy gate's input alongside the fit
    if prior.calibration is not None:
        verdict = "owns deploys" if prior.trustworthy else "QUARANTINED — deploys stay offline"
        sys.stderr.write(f"[tune] reservoir calibration {prior.calibration:+.2f} → online model {verdict}\n")
    prior.checkpoint()
    sys.stderr.write(diagnostics.report(prior) + "\n")
    sys.stderr.write(diagnostics.golden_prior_eval(prior) + "\n")


def _tune_backend(device_id: int | None = None):
    """The autotune-sweep ``CudaBackend``: benches each variant in a SIGKILL-able
    ``_bench_worker`` **subprocess** (``bench_wall_timeout_s`` set → the isolated
    path in ``benchmark_async``), so a wedged kernel dies
    with the worker and the **parent** CUDA stream stays clean. Tight per-variant
    budgets: tune benches isolated single kernels at -Xcicc -O1 (fast), but the
    big-N warp-MMA variants (fp16 N=28672, ``matmul.mlp_gate_up.h4096``) need ~5 s
    of cicc even then — a 4 s compile budget would record that whole family as
    bench failures and lock it out of the sweep, hence 12 s; 2 s run is ample and the wall SIGKILLs any runaway
    (keeping a ~2 s margin over compile+run). ``device_id`` pins the async bench
    worker to a physical GPU (multi-GPU tune)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    return CudaBackend(bench_compile_timeout_s=12.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=16.0, device_id=device_id)


async def _warm_tune_backends(backends) -> None:
    """Ramp multi-GPU workers without charging startup to a candidate timeout."""
    if len(backends) < 2:
        return
    for start in range(0, len(backends), 2):
        await asyncio.gather(*(backend.warm_async_worker() for backend in backends[start : start + 2]))


def _resolve_devices(args) -> list[int | None]:
    """Resolve ``--gpus`` / ``--devices`` into a device-id list (``--devices`` wins).
    Default ``[None]`` → a single unpinned slot = today's serial behavior. Two or
    more devices must be homogeneous — the tune keys every perf row on one probed
    ``ctx``, so mixed compute capabilities would corrupt the per-op cache."""
    if args.devices:
        try:
            devices: list[int | None] = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        except ValueError:
            logger.error("--devices must be comma-separated GPU ids, e.g. 0,1,3")
            sys.exit(2)
    elif args.gpus is not None:
        if args.gpus < 1:
            logger.error("--gpus must be >= 1")
            sys.exit(2)
        devices = list(range(args.gpus))
    else:
        return [None]
    if len(devices) <= 1:
        return devices or [None]
    _require_homogeneous_devices(devices)
    return devices


def _require_homogeneous_devices(devices: list[int | None]) -> None:
    try:
        import cupy  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 — the live tune path will report the missing runtime
        return
    identities = {}
    for d in devices:
        try:
            props = _device_properties(d)
        except Exception as exc:  # noqa: BLE001
            logger.error("--devices: GPU %s not available (%s)", d, exc)
            sys.exit(2)
        name = props.get("name")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        identities[d] = (props["major"], props["minor"], name)
    if len(set(identities.values())) > 1:
        logger.error("--devices must be homogeneous (one perf key per tune); got GPU identities %s", identities)
        sys.exit(2)


def _device_properties(device_id: int | None) -> dict:
    """CUDA properties for one explicit ordinal (or the active ordinal)."""
    import cupy as cp

    ordinal = cp.cuda.Device().id if device_id is None else device_id
    return cp.cuda.runtime.getDeviceProperties(ordinal)


def _context_for_device(device_id: int | None, *, target: str | None = None):
    """Build the tune context from the physical GPU that runs its benches.

    ``Context.probe`` follows the process-current device and normally describes
    ordinal 0 even when ``--devices 3`` selected another card. Explicit ordinals
    are probed directly, including the canonical SKU identity and physical
    feature vector used by prior and node-store keys.
    """
    from emmy import gpu
    from emmy.compiler.context import Context

    if device_id is None:
        return Context.probe()
    try:
        props = _device_properties(device_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("--devices: GPU %s not available (%s)", device_id, exc)
        sys.exit(2)
    cap = (int(props["major"]), int(props["minor"]))
    if target is not None:
        from emmy.compiler.target import parse_sm

        requested = parse_sm(target)
        if requested != cap:
            logger.error("--target %s cannot be benchmarked on selected GPU %s (sm_%d%d)", target, device_id, cap[0], cap[1])
            sys.exit(2)
    raw_name = props.get("name")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode(errors="replace")
    raw_name = str(raw_name) if raw_name is not None else None
    spec = gpu.by_name(raw_name) if raw_name else None
    gpu_name = spec.name if spec else raw_name
    ctx = Context.from_target(cap, gpu_name=gpu_name)
    fallback = ctx.device_props
    feature_keys = {
        "sm_count": "multiProcessorCount",
        "smem_per_sm": "sharedMemPerMultiprocessor",
        "smem_per_block": "sharedMemPerBlock",
        "regs_per_block": "regsPerBlock",
        "warp_size": "warpSize",
        "total_mem": "totalGlobalMem",
    }
    live_features = {
        key: float(props[prop_key]) if prop_key in props else float(fallback.get(key, 0.0)) for key, prop_key in feature_keys.items()
    }
    return replace(
        ctx,
        sm_count=int(live_features["sm_count"] or ctx.sm_count),
        device_props=live_features,
        gpu_name=gpu_name,
        max_threads_per_cta=int(props.get("maxThreadsPerBlock", ctx.max_threads_per_cta)),
        warp_size=int(live_features["warp_size"] or ctx.warp_size),
    )


def _tune_one(
    args,
    *,
    backends,
    db,
    ctx,
    dump,
    run_id=None,
    proposals=(),
    proposal_ranking_callback=None,
):
    """Trace ``args.code`` / ``args.input`` and run the two-level tune on that one
    graph; return ``(result, bench_bundle)``. Manages the live progress bar (closed
    in ``finally``) and prints the per-op ``done`` summary. Lets ``KeyboardInterrupt``
    and the saturated-queue ``RuntimeError`` (dirty parent stream) **propagate** so
    the caller decides how to exit — called once per target by ``handle_tune``'s
    loop (one shape or the whole golden set). Benching itself is subprocess-isolated
    (see ``_tune_backend``), so the parent process is safe to reuse shape-to-shape.
    ``backends`` is the device-pinned pool (one per GPU; single-element by default)
    fanning the inner per-kernel search across GPUs."""
    import time

    from emmy.commands.tune_progress import TuneProgress
    from emmy.compiler.pipeline.search.two_level import run_two_level_tune

    graph, _, bench_bundle = load_or_trace(args)
    if dump:
        dump.dump_input_graph(graph)
    # Live progress bar — default verbosity on a tty only. Disabled under -v (the
    # [tune] INFO lines show progress instead), -q (errors only), and when stderr is
    # redirected (no \r smearing in piped logs).
    progress = TuneProgress(
        enabled=getattr(args, "verbose", 0) == 0 and not getattr(args, "quiet", False) and sys.stderr.isatty(),
    )
    patience = args.patience if args.patience is not None else config.tune_patience(50)
    explore_eps = args.explore_eps if args.explore_eps is not None else config.tune_eps(0.0)
    t0 = time.monotonic()

    async def run():
        from emmy.compiler.pipeline.search.prior import load_prior

        await _warm_tune_backends(backends)
        prior = load_prior(seed=args.seed)
        rankings = await measure_proposals(
            graph,
            proposals,
            backend=backends[0],
            db=db,
            ctx=ctx,
            max_candidates=getattr(args, "max_candidates", None),
            prior=prior,
        )
        # Working-file feedback is durable as soon as its measurements finish;
        # an interrupted/failed MCTS must not discard already-paid proposal data.
        if proposal_ranking_callback is not None:
            proposal_ranking_callback(rankings)
        budget = getattr(args, "max_candidates", None)
        remaining = None if budget is None else max(0, budget - min(len(proposals), budget))
        result = await run_two_level_tune(
            graph,
            ctx=ctx,
            db=db,
            backends=backends,
            patience=patience,
            ucb_c=args.ucb_c,
            explore_eps=explore_eps,
            dump=dump,
            progress=progress,
            prior_seed=args.seed,
            run_id=run_id,
            max_candidates=remaining,
            prior=prior,
        )
        return result, rankings

    try:
        # Proposal measurements and MCTS share one event loop and backend worker,
        # so an exact seed cannot race a search compile through process-global pins.
        result, rankings = asyncio.run(run())
    finally:
        progress.close()
    sys.stderr.write(f"\n[tune] done: {result.n_terminals} fused terminal(s) in {time.monotonic() - t0:.1f}s\n")
    for block in result.prior_summaries:  # online-prior pick-quality sanity stats
        sys.stderr.write(block + "\n")
    return result, bench_bundle


def _exit_flushed(code: int) -> None:
    """Flush stdio and ``os._exit`` — the tune teardown skips Python finalization
    because a bench-timeout can leave a daemon NVRTC worker thread holding the CUDA
    context, which deadlocks cupy's atexit pool teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _tune_targets(args) -> list[tuple[str, str | None, str | None, list[str] | None]]:
    """The one direct ``(label, code, input, dynamic)`` tune target.

    Multi-target inventory and candidate seeding belong exclusively to a mutable
    ``--golden-file``; the old dataset/canonical-golden target shims are gone.
    """
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    return [(args.code or args.input, args.code, args.input, getattr(args, "dynamic", None))]


def _target_artifact_name(index: int, label: str) -> str:
    """Stable, filesystem-safe directory name for one multi-target artifact set."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label).strip("._") or "target"
    return f"{index:03d}_{safe}"


def _bench_dump(args, *, target_dir: str | None = None):
    """Per-target artifact collector. ``--bench`` reads frontend provenance
    slices from memory; route other dump artifacts through a temp dir when no ``--dump-dir`` was
    given (HTML is only written for a real ``--dump-dir`` / ``EMMY_DUMP_DIR``).
    Returns ``(dump, tmp_dir_or_None)``."""
    from emmy.compiler.pipeline.dump import CompilerDump

    configured = Path(args.dump_dir) if args.dump_dir else config.dump_dir()
    if configured is not None and target_dir is not None:
        dump = CompilerDump(configured / target_dir)
    else:
        dump = CompilerDump.resolve(args.dump_dir)
    if args.bench and dump is None:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="emmy-tune-bench-"))
        try:
            return CompilerDump(dir=tmp), tmp
        except BaseException:
            _cleanup_temp_dump(tmp)
            raise
    return dump, None


def _cleanup_temp_dump(path: Path | None) -> None:
    """Remove only a command-created bench temp directory (never a user dump)."""
    if path is None:
        return
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _select_tune_target(args, target: WorkingGoldenTarget) -> None:
    """Install either a normal trace source or an embedded working-golden program."""
    args.code, args.input, args.dynamic = target.code, target.input, target.dynamic
    if target.program is not None:
        args._golden_graph = target.program
    elif hasattr(args, "_golden_graph"):
        del args._golden_graph


def _tune_working_multi(args, targets, document, *, backends, db, ctx, run_id) -> int:
    """Tune independent working-golden targets concurrently across GPU slots.

    Tracing and proposal pinning remain ordered in the parent process. Once seeds
    are measured, all targets share one event loop, backend-slot queue, DB, and
    prior instance, so one-kernel trace entries occupy different GPUs without
    racing checkpoints or process-global pins.
    """
    from emmy.commands.tune_progress import TuneProgress
    from emmy.compiler.pipeline.search.prior import load_prior
    from emmy.compiler.pipeline.search.two_level import run_two_level_tune

    prepared = []
    temp_dumps: list[Path] = []
    try:
        for index, target in enumerate(targets):
            _select_tune_target(args, target)
            dump, tmp_dump = _bench_dump(args, target_dir=_target_artifact_name(index, target.label))
            if tmp_dump is not None:
                temp_dumps.append(tmp_dump)
            graph, _, bench_bundle = load_or_trace(args)
            if dump:
                dump.dump_input_graph(graph)
            prepared.append((target, graph, bench_bundle, dump, tmp_dump))
    except BaseException:
        for tmp_dump in temp_dumps:
            _cleanup_temp_dump(tmp_dump)
        raise

    patience = args.patience if args.patience is not None else config.tune_patience(50)
    explore_eps = args.explore_eps if args.explore_eps is not None else config.tune_eps(0.0)

    async def run_all():
        await _warm_tune_backends(backends)
        prior = load_prior(seed=args.seed)
        # Pins are process-global. Measure proposals before starting concurrent
        # MCTS tasks, rotating their isolated jobs across the available GPUs.
        for index, (target, graph, _bundle, _dump, _tmp) in enumerate(prepared):
            rankings = await measure_proposals(
                graph,
                target.proposals,
                backend=backends[index % len(backends)],
                db=db,
                ctx=ctx,
                max_candidates=getattr(args, "max_candidates", None),
                prior=prior,
            )
            if target.proposals:
                persist_proposal_rankings(args.golden_file, document, target, rankings)

        slots: asyncio.Queue = asyncio.Queue()
        for backend in backends:
            slots.put_nowait(backend)

        async def tune_target(index, target, graph, dump):
            budget = getattr(args, "max_candidates", None)
            remaining = None if budget is None else max(0, budget - min(len(target.proposals), budget))
            return await run_two_level_tune(
                graph,
                ctx=ctx,
                db=db,
                backends=backends,
                patience=patience,
                ucb_c=args.ucb_c,
                explore_eps=explore_eps,
                dump=dump,
                progress=TuneProgress(enabled=False),
                prior_seed=args.seed + index,
                run_id=run_id,
                max_candidates=remaining,
                prior=prior,
                manage_prior=False,
                backend_slots=slots,
                close_backends=False,
            )

        try:
            results = await asyncio.gather(
                *[tune_target(index, target, graph, dump) for index, (target, graph, _bundle, dump, _tmp) in enumerate(prepared)]
            )
            summaries = [prior.summary("global")] if prior.fitted or prior.trajectory else []
            prior.maybe_refit(force=True)
            prior.checkpoint()
            if results:
                results[0].prior_summaries.extend(summaries)
            return results
        finally:
            for backend in backends:
                aclose = getattr(backend, "aclose_async_worker", None)
                if aclose is not None:
                    await aclose()

    try:
        results = asyncio.run(run_all())
        for (target, _graph, bench_bundle, dump, tmp_dump), result in zip(prepared, results, strict=True):
            _select_tune_target(args, target)
            if result.best_reward is not None:
                if args.output and result.assembled is not None:
                    Path(args.output).write_text(format_stage(result.assembled, "cuda"))
                if args.bench and result.assembled is not None:
                    _run_bench(
                        args,
                        bench_bundle,
                        result.assembled,
                        dump,
                        html_dir=(dump.dir if dump and tmp_dump is None else None),
                        device_id=backends[0].device_id,
                    )
            winner = result.best_reward.searched_winner() if result.best_reward is not None else None
            persist_tune_winner(
                args.golden_file,
                document,
                target,
                winner,
                compile_flags=config.nvcc_flags(),
                replay_plan=getattr(result, "replay_plan", None),
            )
        for block in results[0].prior_summaries if results else []:
            sys.stderr.write(block + "\n")
        sys.stderr.write(f"\n[tune] done: {len(results)}/{len(targets)} working-golden target(s)\n")
        return len(results)
    finally:
        for tmp_dump in temp_dumps:
            _cleanup_temp_dump(tmp_dump)


def handle_tune(args):
    if getattr(args, "max_candidates", None) is not None and args.max_candidates < 1:
        logger.error("--max-candidates must be >= 1")
        sys.exit(2)
    if getattr(args, "kernel", None) and not getattr(args, "golden_file", None):
        logger.error("--kernel requires --golden-file")
        sys.exit(2)
    if not args.code and not args.input and not getattr(args, "golden_file", None):
        # No op to tune → offline mode: refit the online prior on its persisted
        # dataset and print diagnostics (reachability, calibration, golden coverage).
        _tune_offline(args)
        return

    working_document = None
    if getattr(args, "golden_file", None):
        conflicts = []
        if args.code or args.input:
            conflicts.append("--code / positional input")
        if getattr(args, "dynamic", None):
            conflicts.append("--dynamic")
        if conflicts:
            logger.error("--golden-file is mutually exclusive with %s", " / ".join(conflicts))
            sys.exit(2)
        try:
            working_document, targets = load_working_targets(args.golden_file, kernel=args.kernel)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
    else:
        targets = [WorkingGoldenTarget(label, code, inp, dyn) for label, code, inp, dyn in _tune_targets(args)]
    if len(targets) > 1 and args.output:
        logger.error("--output is only valid for a single tune target; use --dump-dir for multi-target artifacts")
        sys.exit(2)
    validate_trace_adapter_args(args)

    from emmy.compiler.pipeline.search import SearchDB

    setup_pipeline_runtime(args)
    # tune compiles at -Xcicc -O1 by default to dodge a cicc/LLVM blowup on big
    # unrolled register-tile kernels (up to ~200x faster compile). The trade-off:
    # -O1 latencies are a RANKING signal, NOT -O3-optimal — reduction / attention
    # kernels can run 1.5-3x slower. Re-bench the winner at -O3 (``tune --bench``,
    # or ``emmy run --bench``) for deployable numbers.
    nvcc_flags = apply_nvcc_flags(args, default="-Xcicc -O1")
    if "-O1" in nvcc_flags or "-O0" in nvcc_flags:
        logger.info(
            "tune compiling at cicc %s — latencies are a RANKING signal, not -O3-optimal",
            "-O1" if "-O1" in nvcc_flags else "-O0",
        )

    db_path = resolve_tune_db()  # ``EMMY_TUNE_DB`` env overrides the default path
    if args.clean:  # one shape or many: a fresh sweep clears once, then accumulates
        _clean_caches(db_path)
    db = SearchDB(path=db_path)
    logger.info("Tuning DB: %s", db_path)
    # One device-pinned bench worker per GPU (subprocess-isolated) + one prior shared
    # across every target — benching can't dirty the parent, so a single long-lived
    # process loops cleanly. ``[None]`` (default) = one unpinned worker = serial.
    devices = _resolve_devices(args)
    backends = [_tune_backend(device_id=d) for d in devices]
    if len(backends) > 1:
        sys.stderr.write(f"[tune] per-kernel parallel across {len(backends)} GPUs: {[d for d in devices]}\n")
    ctx = _context_for_device(devices[0], target=getattr(args, "target", None))
    if working_document is not None:
        try:
            validate_working_gpu(working_document, ctx)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
    # One session id per CLI invocation (a golden sweep = one collection session) —
    # stamped on every node row this run writes, so cross-run keep-min drift in the
    # node store is traceable to its tune session.
    from emmy.compiler.pipeline.search.two_level import _mint_run_id

    run_id = _mint_run_id()

    one_pin_set = len({tuple(sorted(target.pins.items())) for target in targets}) == 1
    if working_document is not None and len(backends) > 1 and len(targets) > 1 and one_pin_set:
        sys.stderr.write(f"[tune] target-parallel working-golden sweep: {len(targets)} target(s) across {len(backends)} GPUs\n")
        try:
            from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

            with pinned_knobs(targets[0].pins):
                _tune_working_multi(
                    args,
                    targets,
                    working_document,
                    backends=backends,
                    db=db,
                    ctx=ctx,
                    run_id=run_id,
                )
        except KeyboardInterrupt:
            sys.stderr.write("\n[tune] interrupted — partial measured results are preserved in the DB\n")
            _exit_flushed(0)
        except RuntimeError as exc:
            if isinstance(exc, NotImplementedError):
                raise
            sys.stderr.write(f"\n[tune] aborted: {exc}\n")
            _exit_flushed(1)
        _exit_flushed(0)

    multi = len(targets) > 1
    if multi:
        sys.stderr.write(f"[tune] {len(targets)} shape(s) into {db_path}{' (--clean)' if args.clean else ''}\n")
    done = 0
    for i, target in enumerate(targets):
        label, code = target.label, target.code
        _select_tune_target(args, target)
        if multi:
            sys.stderr.write(f"\n[tune] === {i + 1}/{len(targets)}: {label} → {code} ===\n")
        target_dir = _target_artifact_name(i, label) if multi else None
        dump, tmp_dump = _bench_dump(args, target_dir=target_dir)
        try:
            from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

            regime = pinned_knobs(target.pins) if working_document is not None else nullcontext()
            with regime:
                result, bench_bundle = _tune_one(
                    args,
                    backends=backends,
                    db=db,
                    ctx=ctx,
                    dump=dump,
                    run_id=run_id,
                    proposals=target.proposals,
                    proposal_ranking_callback=(
                        (lambda measured, target=target: persist_proposal_rankings(args.golden_file, working_document, target, measured))
                        if working_document is not None and target.proposals
                        else None
                    ),
                )
        except KeyboardInterrupt:
            # Per-op bests already landed in the DB as they were measured, so a re-run resumes.
            sys.stderr.write(f"\n[tune] interrupted{f' at {label}' if multi else ''} — partial per-op results are in the DB\n")
            _cleanup_temp_dump(tmp_dump)
            _exit_flushed(0)
        except RuntimeError as exc:
            # A NotImplementedError is never the watchdog signal — it's a
            # compiler contract bug (e.g. an unconsumed AtomTile reaching
            # render); re-raise so the traceback isn't swallowed.
            if isinstance(exc, NotImplementedError):
                _cleanup_temp_dump(tmp_dump)
                raise
            # Bench watchdog couldn't bail (GPU queue saturated) → the parent CUDA stream
            # is dirty, so the rest of the sweep can't run reliably here. Abort (the DB has
            # the per-op bests; a re-run resumes). os._exit bypasses the cupy atexit deadlock.
            sys.stderr.write(f"\n[tune] aborted{f' at {label}' if multi else ''}: {exc}\n")
            _cleanup_temp_dump(tmp_dump)
            _exit_flushed(1)

        try:
            if result.best_reward is None:
                if not multi:
                    sys.stderr.write("[tune] no kernels tuned — exiting without output\n")
            else:
                # Only write the assembled CUDA when ``--output`` is given (a multi-kB dump
                # to stdout after a long tune is noise; ``-o`` or ``compile`` replays it).
                if args.output and result.assembled is not None:
                    Path(args.output).write_text(format_stage(result.assembled, "cuda"))
                    logger.info("Saved cuda IR: %s", args.output)
                if args.bench and result.assembled is not None:
                    _run_bench(
                        args,
                        bench_bundle,
                        result.assembled,
                        dump,
                        html_dir=(dump.dir if dump and tmp_dump is None else None),
                        device_id=backends[0].device_id,
                    )
            if working_document is not None:
                winner = result.best_reward.searched_winner() if result.best_reward is not None else None
                persist_tune_winner(
                    args.golden_file,
                    working_document,
                    target,
                    winner,
                    compile_flags=config.nvcc_flags(),
                    replay_plan=getattr(result, "replay_plan", None),
                )
                sys.stderr.write(f"[tune] updated working golden rankings: {args.golden_file}\n")
            done += 1
        finally:
            _cleanup_temp_dump(tmp_dump)

    if multi:
        sys.stderr.write(f"\n[tune] done: {done}/{len(targets)} shape(s)\n")
    _exit_flushed(0)


def _clean_caches(db_path) -> None:
    """``--clean``: nuke the tuning DB (+ WAL/SHM sidecars) and the kernel
    caches (emmy's cubin cache + cupy's NVRTC cache) for a fresh sweep."""
    import shutil

    from emmy.compiler.backend.cuda import nvcc

    removed = []
    if db_path is not None:
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{db_path}{suffix}")
            if p.exists():
                p.unlink()
                removed.append(str(p))
    # The online-prior checkpoint file (a fresh sweep should start cold).
    for p in (config.online_path(), config.online_path().with_suffix(config.online_path().suffix + ".tmp")):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    nvcc.clear_cubin_cache()
    removed.append(str(nvcc.cubin_cache_dir()))
    try:
        import cupy as cp

        shutil.rmtree(cp.cuda.compiler.get_cache_dir(), ignore_errors=True)
        removed.append(cp.cuda.compiler.get_cache_dir())
    except Exception:  # noqa: BLE001 — cupy cache clear is best-effort
        pass
    sys.stderr.write(f"[tune] --clean: removed tuning DB + kernel caches ({', '.join(removed)})\n")


# Parent wall-clock caps for the isolated deployable benches: on overrun the worker is SIGKILLed
# (frees the device, parent stays clean). Generous over real cold-start — a hung kernel is caught
# far sooner by the worker's own 1 s per-launch watchdog, which exits the child promptly. Full-model
# is larger: it reloads the HF module + traces + JITs torch.compile in the child.
_FULL_MODEL_BENCH_WALL_S = 300.0
_PER_KERNEL_BENCH_WALL_S = 120.0


def _run_bench(args, bench_bundle, assembled, dump, *, html_dir, device_id: int | None = None) -> None:
    """``tune --bench``: re-bench the tuned winner at -O3 (deployable numbers, NOT the
    -O1 ranking pass) — full model **against the real torch module** (eager /
    torch.compile / Emmy) and each in-memory per-kernel frontend slice against
    its torch-ref reconstruction, each in the SIGKILL-able bench worker so a hung kernel
    can't wedge the run. Prints both tables and (when ``html_dir`` is set) writes an HTML
    per-kernel chart. ``bench_bundle = (module, args, kwargs) | None``; when ``None`` (an
    ``--ir`` JSON tune with no module) the full-model bench is skipped and only the
    per-kernel table runs."""
    from emmy.commands.run import _collect_sym_env, _print_table, _symbolic_bench_note
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from emmy.compiler.pipeline.search.db import SearchDB

    # Re-bench at -O3 (deployable) unless the user explicitly pinned --nvcc-flags;
    # tune searched at -O1, which is a ranking signal only. The lowering-fork
    # selection that tuning recorded is keyed by Op.cache_key (opt-level independent),
    # so the -O1-tuned winners are still picked when re-benching here at -O3.
    bench_flags = args.nvcc_flags if args.nvcc_flags is not None else ""
    os.environ[config.NVCC_FLAGS] = bench_flags
    bench_ctx = _context_for_device(device_id, target=getattr(args, "target", None))
    sys.stderr.write(f"\n[tune] --bench: re-benching at -O3 ({bench_flags or 'nvcc default -O3'}) — deployable numbers\n")

    # Build the bench backend with EMMY_DUMP_DIR cleared: CudaBackend defaults its
    # dump to CompilerDump.from_env(), whose __post_init__ would replace the dump dir.
    # Clearing it also avoids per-launch dump noise during benching.
    saved_dump_env = os.environ.pop(config.DUMP_DIR, None)
    # ``backend`` here is only the handle to resolve the tune DB (for the per-kernel re-lowering);
    # the benches themselves run in the SIGKILL-able worker (``benchmark_compare_isolated_async``), which
    # builds its own backend. DUMP_DIR is cleared so CudaBackend's default CompilerDump doesn't
    # replace the artifact dir while the bench is using its in-memory slices.
    backend = CudaBackend(tune_db="auto")
    if saved_dump_env is not None:
        os.environ[config.DUMP_DIR] = saved_dump_env
    db = SearchDB(path=backend.tune_db) if (backend.tune_db is not None and backend.tune_db.exists()) else None

    if bench_bundle is not None:
        sys.stderr.write("\n[tune] full-model bench (eager / torch.compile / emmy):\n")
        # The worker rebuilds the real module from these args via ``load_or_trace`` (no live module
        # crosses the pipe) and runs the comparison in-child — a hung emmy kernel hangs the
        # child, which the parent SIGKILLs, instead of wedging the run.
        trace_args = {
            "code": args.code,
            "input": args.input,
            "adapter": getattr(args, "adapter", "causal-lm"),
            "layer": args.layer,
            "seq_len": args.seq_len,
            "dynamic": getattr(args, "dynamic", None),
        }
        try:
            full, _, _, full_captured, _ = asyncio.run(
                benchmark_compare_isolated_async(
                    lowered=assembled,
                    torch_spec=("trace_args", trace_args),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_FULL_MODEL_BENCH_WALL_S,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    nvcc_flags=bench_flags,
                    device_id=device_id,
                )
            )
            # The worker tiled the torch inputs to the hint for a symbolic graph
            # (``_hint_sized_inputs`` inside ``bench_full_model_real``); label the
            # table so the reader knows the numbers are hint-shaped.
            capture_note = None if full_captured else "(graph-capture fallback: timings include host launch overhead)"
            sym_env = _collect_sym_env([assembled] if assembled is not None else [])
            notes = [n for n in (_symbolic_bench_note(sym_env), capture_note) if n]
            _print_table(full, note="\n".join(notes) if notes else None)
        except RuntimeError as exc:
            # Any worker failure (incl. a SIGKILL on a hung kernel) surfaces as RuntimeError. The
            # parent device stays clean — per-kernel runs in its own worker — so continue.
            sys.stderr.write(f"[tune] full-model bench failed ({exc}); continuing to per-kernel\n")
    else:
        sys.stderr.write("\n[tune] full-model bench skipped (the embedded program has no runnable eager module)\n")

    rows, fallback = _bench_per_kernel(args, dump, db, device_id=device_id, ctx=bench_ctx)
    if rows:
        _print_per_kernel_table(rows)
        if fallback:
            print(f"note: timed without CUDA graph capture (host dispatch included): {', '.join(fallback)}")
        if html_dir is not None:
            render_kernel_chart(rows, Path(html_dir) / "kernels.html")


def _bench_per_kernel(args, dump, db, *, device_id: int | None = None, ctx=None):
    """Bench each kernel's frontend provenance slice (re-lowered greedily so the tuned
    DB-best forks are picked) vs eager / torch.compile / emmy at -O3 — each in the SIGKILL-able
    worker (``benchmark_compare_isolated_async``). Re-lowering runs in the parent (CPU; greedy forks read
    the DB); only the GPU bench is isolated, so a hung / failed kernel skips just that reproducer and
    the sweep continues. Returns ``(rows, fallback)`` — ``rows`` is ``[(label, {backend: us})]``,
    ``fallback`` the labels that benched without CUDA graph capture (dispatch-inclusive timings)."""
    from emmy.commands.run import _detect_stage, _passes_after_stage
    from emmy.compiler.backend import torch_ref
    from emmy.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from emmy.compiler.pipeline import Pipeline

    repros = dump.frontend_reproducers()
    if not repros:
        return [], []
    bench_flags = os.environ.get(config.NVCC_FLAGS, "")
    sys.stderr.write(f"\n[tune] per-kernel bench: {len(repros)} reproducer(s) at -O3\n")
    rows: list[tuple[str, dict]] = []
    fallback: list[str] = []
    records: list[dict] = []  # persisted as 62_kernel_bench.json — the `emmy compare` input
    for name, frontend in sorted(repros.items()):
        label = _short_kernel(name)
        try:
            g = frontend.copy()
            fe = g.copy() if torch_ref.is_runnable(g) else None
            tail = _passes_after_stage(_detect_stage(g))
            # No dump here — re-creating a CompilerDump on the repro dir would rmtree it.
            lowered = Pipeline.build(tail).run(g, ctx=ctx, db=db) if tail else g
            results, _, reference_available, captured, accuracy_error = asyncio.run(
                benchmark_compare_isolated_async(
                    lowered=lowered,
                    torch_spec=("frontend_graph", fe),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_PER_KERNEL_BENCH_WALL_S,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    nvcc_flags=bench_flags,
                    device_id=device_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 — isolated, so a hung / failed kernel skips just this one
            sys.stderr.write(f"[tune]   {label}: skipped ({exc})\n")
            continue
        rows.append((label, results))
        records.append(
            {
                "kernel": name,
                "label": label,
                "captured": captured,
                "reference_available": reference_available,
                "accuracy_error": accuracy_error,
                "backends": results,
            }
        )
        if not captured:
            fallback.append(label)
        if accuracy_error is not None:
            sys.stderr.write(f"[tune]   {label}: accuracy failed ({accuracy_error})\n")
        dp = results.get("Emmy")
        sys.stderr.write(f"[tune]   {label}: emmy={dp:.0f}us\n" if dp is not None else f"[tune]   {label}: (no result)\n")
    if records:
        # Per-kernel -O3 bench results in machine-readable form, beside the table /
        # kernels.html — the per-kernel input `emmy compare <dumpA> <dumpB>` diffs.
        import json

        (dump.dir / "62_kernel_bench.json").write_text(json.dumps(records, indent=2, default=str))
    return rows, fallback


def _short_kernel(name: str) -> str:
    """Readable kernel label with its trailing structural hash removed."""
    import re

    return re.sub(r"_[0-9a-f]{6}$", "", name)


def _fmt_us(us) -> str:
    return f"{us:.0f}" if us is not None else "-"


def _print_per_kernel_table(rows) -> None:
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    cols = [Col("Kernel"), Col("eager", "r"), Col("tcompile", "r"), Col("emmy", "r"), Col("vs eager", "r")]
    data = []
    for label, res in sorted(rows, key=lambda kv: kv[1].get("Emmy") or 0.0, reverse=True):
        eager = res.get("Eager PyTorch")
        dp = res.get("Emmy")
        spd = f"{eager / dp:.2f}x" if (eager and dp) else "-"
        data.append([label, _fmt_us(eager), _fmt_us(res.get("torch.compile")), _fmt_us(dp), spd])
    print()
    for line in render_table(cols, data, rule=True):
        print(line)


def render_kernel_chart(rows, out_html) -> None:
    """Render the per-kernel latency comparison as a horizontal bar chart (HTML + a
    best-effort PNG) via :mod:`emmy.visualize`."""
    from emmy.visualize import Bar, BarChart, render_bar_chart

    rows = sorted(rows, key=lambda kv: kv[1].get("Emmy") or 0.0, reverse=True)
    n_vs = sum("Eager PyTorch" in res for _, res in rows)
    chart = BarChart(
        categories=[label for label, _ in rows],
        bars=[
            Bar("Emmy", [res.get("Emmy") for _, res in rows], color="#4dabf7"),
            Bar("Eager PyTorch", [res.get("Eager PyTorch") for _, res in rows], color="#999999"),
            Bar("torch.compile", [res.get("torch.compile") for _, res in rows], color="#ffd166"),
        ],
        value_name="latency (µs) — lower is faster",
        title="tune --bench — per-kernel latency (-O3)",
        subtitle=f"{len(rows)} kernels benched from frontend provenance slices ({n_vs} torch-comparable, rest emmy-only).",
        orientation="horizontal",
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_bar_chart(chart, theme="dark", transparent=True))
    sys.stderr.write(f"[tune]   chart → {out_html}\n")
    try:
        from emmy.visualize import render_image

        png = out_html.with_suffix(".png")
        render_image(out_html.read_text(), png, height=max(300, 40 * len(rows)))
        sys.stderr.write(f"[tune]   png   → {png}\n")
    except Exception as exc:  # noqa: BLE001 — PNG needs the [visualize] extra (playwright)
        sys.stderr.write(f"[tune]   png skipped: {exc}\n")
