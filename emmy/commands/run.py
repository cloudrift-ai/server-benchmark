"""Run an inline torch expression through the emmy CUDA pipeline.

Compiles ``--code`` to CUDA, executes it on real input data, and verifies
correctness against eager PyTorch. With ``--bench``, also benchmarks all
backends (eager, torch.compile, emmy) and prints a comparison table —
the same shape as ``scripts/bench_block.py`` but for arbitrary inline ops.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import uuid
from collections import namedtuple
from pathlib import Path

from emmy import config
from emmy.compiler.pipeline.search.pins import pinned_knobs, unreproducible_pin_flag

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

    parser = subparsers.add_parser("run", help="Compile + run a model / inline torch expression on the CUDA backend")
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "HuggingFace model ID or .json IR file. A model ID is traced + compiled + executed and "
            "(with --bench) timed end-to-end against the real torch module; a .json IR file behaves "
            "like --ir. Mutually exclusive with --code / --ir."
        ),
    )
    parser.add_argument(
        "--code",
        "-c",
        help=(
            "Inline Python expression whose last statement is a call (same grammar as "
            "``compile --code``). Example: 'torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))'. "
            "Mutually exclusive with the positional input / --ir."
        ),
    )
    parser.add_argument(
        "--adapter",
        choices=["causal-lm", "dit"],
        default="causal-lm",
        help=(
            "Model trace adapter. ``causal-lm`` preserves the existing HuggingFace text-model path; "
            "``dit`` traces one fixed-shape FP16 Diffusers DiT transformer block (requires --layer)."
        ),
    )
    parser.add_argument(
        "--ir",
        help=(
            "Path to a JSON IR dump (any stage: torch / tensor / loop / tile / kernel / cuda). "
            "The remaining lowering passes are run, then the kernel(s) are executed with random "
            "inputs and benchmarked. Skips eager accuracy check (no reference model available). "
            "Equivalent to passing the same .json path as the positional input."
        ),
    )
    parser.add_argument(
        "--golden",
        metavar="PATH",
        help="Run every embedded target in this working golden YAML. Mutually exclusive with other inputs.",
    )
    parser.add_argument(
        "--target",
        dest="golden_target",
        metavar="NAME",
        help="Run only the matching target from --golden instead of every target.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (when the positional input is a model ID). Omit to run the whole model.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_HINT,
        help="Sequence length for full-model tracing when the input is a model ID (default: 512, matching ``DEFAULT_SEQ_HINT``).",
    )
    parser.add_argument("--bench", "-b", action="store_true", help="Benchmark eager / torch.compile / emmy and print a comparison table.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Re-launch each kernel under ``ncu`` to collect hardware counters "
        "(SM-active %%, FMA pipe util, L1/DRAM bandwidth, smem bank-conflict %%) and print a "
        "side-by-side table of the emmy kernels vs the torch/cuBLAS reference kernels. "
        "Implies --bench. Skipped if ncu is not on PATH or the user lacks performance-counter permissions.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument(
        "--bench-backends",
        default=None,
        help=(
            "Comma-separated subset of backends to time under --bench: any of "
            "``eager``, ``tcompile`` (a.k.a. ``torch.compile`` / ``compile``), "
            "or ``emmy``. Falls back to ``EMMY_BENCH_BACKENDS`` env var, "
            "then to the default ``eager,emmy`` (drops the ~0.8 s "
            "torch.compile JIT from the per-case cost). ``emmy`` is "
            "implicit even if omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --ir random inputs (default: 0).")
    parser.add_argument(
        "--ab",
        action="append",
        default=None,
        metavar="KNOBS",
        help=(
            "Compile + bench an extra variant with these knobs pinned (``K1=V1,K2=V2`` — the "
            "``EMMY_KNOBS`` grammar) and show it as a live A/B row beneath the matching greedy "
            "kernel in the --bench kernel table, knob diffs red. Repeatable. Requires --bench and a "
            "re-lowerable input (--code / --golden / --ir)."
        ),
    )
    parser.add_argument(
        "--dynamic",
        action="append",
        default=None,
        metavar="NAME@INPUT:AXIS",
        help=(
            "Make a tensor dim symbolic. Form: ``NAME@INPUT:AXIS`` — axis ``AXIS`` of the "
            "traced input named ``INPUT`` becomes ``Dim(NAME)``. Repeatable. Forwards to "
            "``torch.export(..., dynamic_shapes={...})``; the compiled CUDA kernel signature "
            "gains an ``int <NAME>`` runtime arg per dim. Example: ``--dynamic seq_len@x:1``."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help=(
            "With --bench: also write the whole comparison as machine-readable JSON to PATH. When --golden "
            "runs multiple targets, PATH is an output directory containing one JSON file per target. Records include the "
            "backend table (eager / torch.compile / emmy), the per-kernel greedy rows (plus, when "
            "pinned rows benched, the ``greedy.isolated`` emmy-only re-bench — the pinned-comparable "
            "greedy baseline), and every --golden / --ab A/B row with its integrity flags "
            "(realized-vs-pinned knob check, arithmetic-intensity floor, wrong-answer check). "
            "Each pinned row and the greedy block carry a ``lane`` (fm/std) so the sweep filters to "
            "the greedy's lane — comparing a pinned [fm] latency against a std greedy is a phantom "
            "regression. Retires ad-hoc table parsing in the golden-sweep workflow."
        ),
    )
    parser.add_argument(
        "--strict",
        dest="strict_correctness",
        action="store_true",
        help=(
            "With --bench on runnable frontend IR or an embedded golden, fail unless every requested backend, "
            "captured timing, exact pin, and direct Emmy-vs-eager comparison is valid."
        ),
    )
    parser.add_argument(
        "--no-record-nodes",
        action="store_true",
        help=(
            "Skip the default bench-to-node recording. When pinned rows bench (--golden / --ab) at tune-standard "
            "quality (warmup >= 5, iters >= 20), each clean row AND the greedy pick's isolated re-bench are recorded "
            "as leaf rows in the tune DB's node store — the training-data feed for the offline prior. Flagged rows "
            "(pin mismatch, wrong answer, intensity floor) and the --ir path never record."
        ),
    )
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts.")
    parser.add_argument("--debug", action="store_true", help="Per-launch tensor dumps in the emmy backend.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase verbosity. Default (no flag): only the accuracy verdict and (with --bench) "
            "the timing tables are printed. -v: also tracer messages and pass / per-rule timings. "
            "-vv: also per-rule application snapshots."
        ),
    )
    from emmy.commands.compile import add_nvcc_args
    from emmy.compiler.target import add_target_arg

    add_nvcc_args(parser)
    add_target_arg(parser, dest="gpu_arch", option="--gpu-arch")
    parser.set_defaults(func=handle_run)


def handle_run(args):
    from emmy.commands.compile import apply_nvcc_flags
    from emmy.compiler.target import apply_target_arg

    apply_nvcc_flags(args, default="")  # run uses nvcc default -O3 (representative codegen)
    apply_target_arg(args, dest="gpu_arch")
    if args.profile:
        args.bench = True  # --profile re-launches under ncu via the bench path; profiling implies benching
    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.golden_target and not args.golden:
        logger.error("--target requires --golden PATH")
        sys.exit(2)
    if args.golden:
        _run_golden_targets(args)
        return

    _handle_run_once(args)


def _handle_run_once(args):
    try:
        import torch
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from emmy.commands.compile import load_or_trace, resolve_golden_arg, validate_trace_adapter_args
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.dump import CompilerDump

    if getattr(args, "golden_file", None):
        resolve_golden_arg(args)
    else:
        args.golden_configs = []
    if sum(x is not None for x in (args.input, args.code, args.ir)) > 1:
        logger.error("input / --code / --ir are mutually exclusive")
        sys.exit(1)
    validate_trace_adapter_args(args)

    # A .json input (via --ir or as the positional) takes the IR path: finish
    # lowering an arbitrary-stage dump, no traced module to bench against torch.
    ir_path = args.ir
    if ir_path is None and args.input is not None and Path(args.input).suffix == ".json" and Path(args.input).exists():
        ir_path = args.input
    if args.ab:
        if not args.bench:
            logger.error("--ab requires --bench (the A/B rows render in the kernel table)")
            sys.exit(2)
        if args.code is None and ir_path is None and not hasattr(args, "_golden_graph"):
            logger.error("--ab requires a re-lowerable input: --code, --golden, or --ir (each config re-lowers a fresh graph)")
            sys.exit(2)
        try:
            _ab_samples(args.ab)  # fail fast on a malformed KNOBS spec
        except ValueError as exc:
            logger.error("--ab: %s", exc)
            sys.exit(2)
    if args.strict_correctness and not args.bench:
        logger.error("--strict requires --bench")
        sys.exit(2)
    if args.strict_correctness and ir_path is None and not hasattr(args, "_golden_graph"):
        logger.error("--strict currently requires runnable frontend IR or an embedded --golden")
        sys.exit(2)
    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    if ir_path is not None:
        args.ir = ir_path
        _handle_run_ir(args, CudaBackend, CompilerDump)
        return

    if hasattr(args, "_golden_graph"):
        _handle_run_ir(args, CudaBackend, CompilerDump)
        return

    if args.input is None and args.code is None:
        logger.error("Either a model ID / .json input, --code, --golden PATH, or --ir is required")
        sys.exit(1)

    # Model ID or --code: trace to a frontend graph + keep the runnable module
    # (+ example inputs) so accuracy / --bench compare against real torch.
    graph, _base_name, bundle = load_or_trace(args)
    module, example_args, example_kwargs = bundle

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Backend auto-resolves ``EMMY_TUNE_DB`` env →
    # ``~/.cache/emmy/autotune.db`` (opens if the file exists,
    # silent fall-back to rule defaults otherwise).
    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    if backend.tune_db is not None and backend.tune_db.exists():
        logger.info("Using tuning DB: %s", backend.tune_db)
    compiled = backend.compile(graph)

    # The ncu child of a ``--profile`` invocation skips the accuracy *check*
    # (the parent already verified it) but still launches both sides once —
    # the emmy program AND the eager reference — because the comparison
    # table needs the cuBLAS / aten kernel rows in the captured CSV beside
    # the ``k_*`` rows.
    skip_accuracy = config.ncu_child()

    if not args.bench:
        # No bench: the accuracy probe is the command's whole GPU action. It stays
        # in-process — the ``--debug`` per-launch dumps and the ncu child's profiled
        # launches live here — so a hung kernel still poisons this process's stream.
        try:
            input_data = _bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=args.input)
        except RuntimeError as exc:
            logger.error(exc)
            sys.exit(1)
        try:
            if not skip_accuracy:
                run_result, _ = backend.run(compiled, input_data=input_data)
                if dump and backend.last_debug_result is not None:
                    dump.dump_per_launch_values(backend.last_debug_result.per_launch)
                err = _check_accuracy(run_result.outputs, _eager_output(module, example_args, example_kwargs))
                if err is not None:
                    logger.error(err)
                    sys.exit(1)
            else:
                # ncu child: one emmy launch (our metrics) + one eager forward
                # (the reference rows for the comparison table); no accuracy diff.
                backend.run(compiled, input_data=input_data)
                _eager_output(module, example_args, example_kwargs)
        except RuntimeError as exc:
            # Per-launch watchdog fired in ``run_program`` (kernel >1 s).
            # The CUDA context is dirty — bypass Python cleanup so cupy's
            # atexit doesn't block on the still-running kernel.
            sys.stderr.write(f"accuracy check failed: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return

    # --bench: the parent does NO GPU work. The greedy comparison (eager / torch.compile /
    # emmy, torch rebuilt in-child from ``trace_args``) and every pinned golden / ``--ab``
    # row run as jobs on ONE persistent SIGKILL-able bench worker — the same isolation and
    # measurement path for every row. A hung kernel dies with the child (the parent stays
    # clean, the row records bench_fail, the next job respawns a fresh context), so no
    # ``os._exit`` and no per-row special-casing. The greedy job also carries the accuracy
    # check (in-child, on the rebuilt module's real inputs) and — when pinned rows exist —
    # returns that run's (inputs, outputs) as their wrong-answer reference. When pinned
    # rows exist the greedy graph is ALSO re-benched emmy-only through the pinned-row path
    # (``_bench_greedy_isolated``): the interleaved greedy number is torch-comparable but
    # NOT pinned-comparable (the documented ~7% environment skew), so pinned speedups read
    # against the isolated row.
    _resolve_backends(args.bench_backends)  # validate the spelling before spending a worker job
    pinned = list(getattr(args, "golden_configs", None) or []) + (_ab_samples(args.ab, dynamic=args.dynamic) if args.ab else [])
    trace_payload = {
        "code": args.code,
        "input": args.input,
        "adapter": getattr(args, "adapter", "causal-lm"),
        "layer": args.layer,
        "seq_len": args.seq_len,
        "dynamic": list(args.dynamic) if getattr(args, "dynamic", None) else None,
    }

    async def _bench_session():
        greedy_fail = results = bench = accuracy_error = ab_ref = golden_benches = greedy_iso = None
        captured = True
        try:
            try:
                resp = await backend.benchmark_compare_async(
                    compiled,
                    torch_spec=("trace_args", trace_payload),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_compare_wall_s(compiled, backend, base_s=_GREEDY_COMPARE_BASE_S),
                    warmup=args.warmup,
                    iters=args.iters,
                    accuracy=not skip_accuracy,
                    want_ref=bool(pinned),
                )
            except RuntimeError as exc:
                # Worker SIGKILL (hung kernel), in-child bench budget, EOF — the greedy
                # pick failing to bench is a *finding*: record it, keep going.
                greedy_fail = f"greedy run/bench failed: {exc}"
            else:
                results, bench, captured = resp["results"], resp["result"], resp["captured"]
                accuracy_error, ab_ref = resp["accuracy_error"], resp["run_io"]
            if pinned and accuracy_error is None:
                if greedy_fail:
                    logger.error("%s — greedy row marked bench_fail; pinned rows still bench in the worker", greedy_fail)
                greedy_iso = await _bench_greedy_isolated(backend, compiled, warmup=args.warmup, iters=args.iters)
                golden_benches = await _bench_golden_variants(backend, args.code, pinned, warmup=args.warmup, iters=args.iters, ref=ab_ref)
        finally:
            await backend.aclose_async_worker()
        return greedy_fail, results, bench, captured, accuracy_error, golden_benches, greedy_iso

    greedy_fail, results, bench, captured, accuracy_error, golden_benches, greedy_iso = asyncio.run(_bench_session())

    if accuracy_error is not None:
        # Correctness gate: the deployed program computes the wrong answer, so no latency
        # comparison is meaningful (the child skipped the bench; pinned rows were skipped).
        logger.error(accuracy_error)
        sys.exit(1)

    if dump and bench is not None:
        dump.dump_benchmark(bench)
        _dump_bench_compare(dump.dir, results, args.warmup, args.iters)

    if results is not None:
        capture_note = None if captured else "(graph-capture fallback: timings include host launch overhead)"
        notes = [n for n in (_symbolic_bench_note(_collect_sym_env([compiled])), capture_note) if n]
        _print_table(results, note="\n".join(notes) if notes else None)
    _print_kernel_stats(compiled, bench, golden_benches=golden_benches, greedy_fail=greedy_fail, greedy_iso=greedy_iso)
    if getattr(args, "json", None):
        _write_ab_json(args, results or {}, compiled, bench, golden_benches, greedy_fail=greedy_fail, greedy_iso=greedy_iso)
    _record_bench_nodes(args, golden_benches, greedy_iso)
    if args.profile and greedy_fail is None:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)
    if (
        greedy_fail is not None
        or (greedy_iso is not None and greedy_iso.status != "ok")
        or any(gb.status != "ok" for gb in golden_benches or [])
    ):
        sys.exit(1)  # every row is reported above; any failed row (greedy or pinned) exits non-zero


def _run_golden_targets(args) -> None:
    """Run a working golden's selected targets sequentially in this process."""
    from copy import copy  # noqa: PLC0415

    from emmy.compiler.pipeline.search.golden import load_golden_file, load_golden_records  # noqa: PLC0415

    if args.input or args.code or args.ir:
        logger.error("--golden is mutually exclusive with positional input / --code / --ir")
        sys.exit(2)
    try:
        records = load_golden_records(load_golden_file(args.golden))
    except (OSError, ValueError) as exc:
        logger.error("cannot load --golden %s: %s", args.golden, exc)
        sys.exit(2)
    names = [args.golden_target] if args.golden_target else list(dict.fromkeys(record.name for record in records))
    if not names:
        logger.error("--golden contains no targets: %s", args.golden)
        sys.exit(2)

    output_dir = None
    if len(names) > 1 and args.json:
        output_dir = Path(args.json)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            logger.error("multi-target --json directory must be empty: %s", output_dir)
            sys.exit(2)
        output_dir.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(names):
        target_args = copy(args)
        target_args.golden_file = args.golden
        target_args.golden = name
        target_args.golden_target = None
        if output_dir is not None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "target"
            target_args.json = str(output_dir / f"{index:03d}-{safe_name}.json")
        _handle_run_once(target_args)


def _recordable_bench_leaves(golden_benches, greedy_iso) -> list:
    """The benched rows honest enough to record into the node store: every ``ok`` row
    with NO integrity flag (a flagged ok row measured something untrue — wrong answer,
    intensity floor), plus ``bench_fail`` rows whose config actually realized (a
    genuine "doesn't bench here" negative). ``pin_unmatched`` rows and compile
    failures (``graph is None``) never record — the claimed config never ran, and
    "not offered" is not "doesn't launch". Pure over the ``_GoldenBench`` duck type,
    so tests drive it with stubs."""
    from emmy.compiler.pipeline.search.bench_record import bench_leaves  # noqa: PLC0415

    leaves = []
    for gb in ([greedy_iso] if greedy_iso is not None else []) + list(golden_benches or []):
        if gb.graph is None or gb.status == "pin_unmatched":
            continue
        if gb.status == "ok" and not gb.flags:
            leaves += bench_leaves(gb.graph, gb.bench)
        elif gb.status == "bench_fail":
            leaves += bench_leaves(gb.graph, None, status="bench_fail")
    return leaves


def _record_bench_nodes(args, golden_benches, greedy_iso) -> None:
    """Default-on bench-to-node recording (``--no-record-nodes`` opts out): the pinned
    A/B rows and the greedy isolated re-bench become node-store leaves — the training
    data the tune-only write path let every manual sweep evaporate from. Records only
    at tune-standard measurement quality; ``record_nodes``' plausibility gate and
    quality-aware leaf replacement still judge every row."""
    if getattr(args, "no_record_nodes", False) or (greedy_iso is None and not golden_benches):
        return
    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import meets_quality_bar, record_bench_leaves  # noqa: PLC0415

    # print, not logger.info: `emmy run` gates the root logger to WARNING at default
    # verbosity, and a default-on WRITE to the user's tune DB must announce itself
    # (the bench table around it prints too — same CLI-output surface).
    if not meets_quality_bar(args.warmup, args.iters):
        print(
            f"[record-nodes] --warmup {args.warmup} / --iters {args.iters} below the tune bench standard — "
            f"measurements NOT recorded into the node store (raise them or pass --no-record-nodes to silence)"
        )
        return
    leaves = _recordable_bench_leaves(golden_benches, greedy_iso)
    if not leaves:
        return
    db_path = resolve_tune_db()
    n = record_bench_leaves(db_path, Context.probe(), leaves)
    print(f"[record-nodes] {n} bench row(s) recorded into the node store ({db_path}) — opt out with --no-record-nodes")


def _reset_persisting_l2_cache() -> None:
    """Drop every persisting-L2 carveout on the current CUDA context.

    Wraps ``cudaCtxResetPersistingL2Cache`` (CUDA 11+). cuBLAS, cuDNN,
    and other libraries can install ``cudaAccessPolicyWindow`` hints
    that pin specific gmem regions in L2 across kernel launches; once
    set, the carveout persists until the context tears down (it is NOT
    released by ``torch.cuda.empty_cache`` or stream sync). That eats
    L2 bandwidth from later cp.async-staged kernels — the article
    matmul drops from 1660 µs to 330 µs after the reset. Best-effort:
    silently skipped if the runtime symbol isn't resolvable (CUDA < 11,
    non-Linux, etc.); a non-zero return is logged at DEBUG so the
    bench path doesn't fail loud on driver quirks.

    Loads ``libcudart`` from the already-mapped image so the call
    targets the SAME runtime torch + cupy are using (the system
    ``libcudart.so`` may belong to a different CUDA install and would
    operate on a different driver context — its reset returns success
    but doesn't touch our context's L2 carveout). Walks
    ``/proc/self/maps`` for the first mapped ``libcudart.so.*`` path.
    """
    import ctypes

    libpath = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.rstrip().split(None, 5)
                if len(parts) < 6:
                    continue
                p = parts[-1]
                if "libcudart.so" in p and p.startswith("/"):
                    libpath = p
                    break
    except OSError:
        pass

    try:
        cudart = ctypes.CDLL(libpath) if libpath else ctypes.CDLL("libcudart.so")
        cudart.cudaCtxResetPersistingL2Cache.restype = ctypes.c_int
        cudart.cudaCtxResetPersistingL2Cache.argtypes = []
        err = cudart.cudaCtxResetPersistingL2Cache()
        if err != 0:
            logger.debug("cudaCtxResetPersistingL2Cache returned %d (continuing)", err)
    except (OSError, AttributeError) as e:
        logger.debug("cudaCtxResetPersistingL2Cache unavailable: %s", e)


def _dump_bench_compare(dump_dir, results: dict, warmup: int, iters: int) -> None:
    """Persist the eager / torch.compile / emmy comparison table
    so downstream tooling (``make bench-kernels``) can parse one file
    per case instead of grepping kernel stdout."""
    import json as _json
    from pathlib import Path as _Path

    eager_us = results.get("Eager PyTorch")
    payload = {
        "warmup": warmup,
        "iters": iters,
        "backends": {name: {"latency_us": us} for name, us in results.items()},
    }
    if eager_us:
        for name, us in results.items():
            payload["backends"][name]["speedup_vs_eager"] = (eager_us / us) if us else 0.0
    out = _Path(dump_dir) / "60_bench_compare.json"
    out.write_text(_json.dumps(payload, indent=2, default=str))


# Base setup budgets (seconds) for the ``run --bench`` comparison jobs' SIGKILL wall caps —
# what the child pays BEFORE any per-kernel work: the ``trace_args`` greedy job reloads the
# model / re-traces the --code expression and may run a torch.compile JIT; the ``--ir`` job
# only rebuilds a torch reference from the shipped frontend snapshot.
_GREEDY_COMPARE_BASE_S = 240.0
_IR_COMPARE_BASE_S = 60.0


def _compare_wall_s(graph, backend, *, base_s: float) -> float:
    """Workload-scaled SIGKILL cap for a worker comparison job. A fixed cap false-fails
    legitimate big runs: the child pays a first nvcc compile of EVERY kernel (the in-child
    compile budget bounds each) plus the bench itself, on top of ``base_s`` for its own
    setup — a dozen-kernel model layer legitimately needs several hundred seconds where a
    single --code kernel needs a fraction."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    n_kernels = sum(1 for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    return base_s + max(1, n_kernels) * backend.bench_compile_timeout_s + backend.bench_run_timeout_s


# One recorded golden config compiled + benched with its knobs pinned this run.
# ``flags`` are the integrity-gate verdicts (empty = clean): the arithmetic-intensity
# floor and the output-correctness check against the greedy or eager reference. ``bench`` is ``None``
# for a row that never benched — ``status`` says why: ``"pin_unmatched"`` (the pinned config
# didn't realize, so benching would measure the planner's own pick under the pin's name) or
# ``"bench_fail"`` (compile/bench of the pinned config failed); ``"ok"`` rows carry a bench.
_GoldenBench = namedtuple("_GoldenBench", "sample graph bench flags status correctness", defaults=("ok", None))


def _lane(knobs: dict) -> str:
    """The precision REGIME a knob dict realizes: ``"fm"`` (an f16-accumulate mma atom or
    ``FAST_EXP``, :func:`~emmy.compiler.pipeline.search.golden.fast_math_knobs`) else ``"std"``.
    Derived from the knobs, never a stored flag, so it can't drift. A working-golden target
    pins BOTH the std and the ``[fm]`` config recorded under one name; comparing a pinned ``[fm]``
    latency against a ``"std"`` greedy manufactures a phantom regression, so every A/B row (and the
    greedy it's compared to) carries its lane and the parser filters to matching lanes."""
    from emmy.compiler.pipeline.search.golden import fast_math_knobs  # noqa: PLC0415

    return "fm" if fast_math_knobs(knobs) else "std"


def _graph_lane(graph) -> str:
    """The lane the greedy graph deployed in — ``"fm"`` when ANY realized CUDA kernel's knobs are
    fast-math (any kernel on an f16acc atom ⇒ the run was under the fast-math gate). Judged
    per-kernel, never on a dict union: two kernels both realizing ``TILE`` collide on the key,
    the union keeps only the last kernel's value, and the reported lane becomes
    launch-order-dependent — the phantom-regression trap this feature exists to prevent."""
    return "fm" if any(_lane(d) == "fm" for d in _cuda_knob_dicts(graph)) else "std"


def _intensity_floor_flag(sample, total_us: float) -> str | None:
    """The arithmetic-intensity floor gate: the FLOP/s a benched row implies from its
    shape must stay below the device's peak dense throughput — a row above it is a wrong
    bench (skipped finalize, silent failure), not a fast kernel (the 8.2 µs "2 PFLOP/s"
    2048³ golden row of the sixth sweep). Returns a flag string, or ``None`` when clean /
    ungateable (no shape, no FLOPs, unknown device peak)."""
    # Use a FLOP count only when the sample source can derive one without guessing. Never reconstruct it from
    # the ShapeKey: the join key excludes symbolic axes on the matmul side but includes them on
    # the reduce-tier/attention side, so the old hint-multiplier formula overcounted reduce-tier
    # ``.dynM`` replays 512× and flagged every one "impossible" at its correct recorded value.
    flops = getattr(sample, "flops", None)
    if flops is None or flops <= 0 or total_us <= 0:
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(gpu.live_name() or "")
    peak = spec.peak_tflops(getattr(sample, "dtype", None) or "fp32") if spec else None
    if peak is None:
        return None
    implied = flops / total_us / 1e6  # FLOP / µs → TFLOP/s
    if implied > peak:
        return f"impossible: implies {implied:.0f} TFLOP/s > {peak:.0f} device peak"
    return None


def _wrong_answer_flag(outputs: dict, ref_outputs: dict) -> str | None:
    """Output-correctness gate for a pinned A/B row: compare the pinned kernel's outputs
    against the greedy run's on the SAME inputs. Both sides are emmy kernels over one
    graph, so they agree to reduction-reorder noise — a large deviation means the pinned
    config computed the wrong answer (the ``g2a`` atomic-split re-bench class: a skipped
    zero-init / finalize benches fast and silently wrong). Returns a flag string or
    ``None``; loose 5% relative tolerance so split-K / atomic reorders never trip it."""
    import numpy as np  # noqa: PLC0415

    worst = 0.0
    for nid, ref in ref_outputs.items():
        got = outputs.get(nid)
        if got is None:
            return f"wrong-answer: output {nid!r} missing"
        a = np.asarray(got, dtype=np.float64)
        b = np.asarray(ref, dtype=np.float64)
        if a.shape != b.shape:
            return f"wrong-answer: output {nid!r} shape {a.shape} != greedy {b.shape}"
        denom = float(np.abs(b).max()) or 1.0
        worst = max(worst, float(np.abs(a - b).max()) / denom)
    if worst > 0.05:
        return f"wrong-answer: rel err {worst:.3f} vs greedy output"
    return None


def _strict_correctness_proof(outputs: dict, eager_out, *, rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """Return a direct Emmy-vs-eager tolerance verdict with reproducible error statistics.

    The pass rule is the same elementwise rule used by ``torch.testing.assert_close`` for
    compiler baselines: ``abs(actual - expected) <= atol + rtol * abs(expected)``. Eager
    outputs may be tensors, a positional tensor sequence, or an output-name mapping (the
    latter is what the embedded-golden worker returns to pinned-row replay).
    """
    import numpy as np  # noqa: PLC0415

    def _array(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    names = list(outputs)
    if isinstance(eager_out, dict):
        missing = [name for name in names if name not in eager_out]
        extra = [name for name in eager_out if name not in outputs]
        if missing or extra:
            return {
                "status": "fail",
                "reference": "eager",
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output names differ: missing={missing}, extra={extra}",
            }
        refs = [eager_out[name] for name in names]
    else:
        refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
        if len(refs) != len(names):
            return {
                "status": "fail",
                "reference": "eager",
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output count differs: Emmy={len(names)}, eager={len(refs)}",
            }

    max_abs = 0.0
    max_rel = 0.0
    abs_sum = 0.0
    count = 0
    failure = None
    for name, ref in zip(names, refs, strict=True):
        actual = _array(outputs[name])
        expected = _array(ref)
        if actual.shape != expected.shape:
            failure = f"output {name!r} shape {actual.shape} != eager {expected.shape}"
            break
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            failure = f"output {name!r} contains non-finite values"
            break
        absolute = np.abs(actual - expected)
        tolerance = atol + rtol * np.abs(expected)
        if absolute.size:
            max_abs = max(max_abs, float(absolute.max()))
            max_rel = max(max_rel, float((absolute / np.maximum(np.abs(expected), atol)).max()))
            abs_sum += float(absolute.sum())
            count += int(absolute.size)
            if failure is None and not np.all(absolute <= tolerance):
                failure = f"output {name!r} exceeds rtol={rtol:g}, atol={atol:g}"

    proof = {
        "status": "fail" if failure else "pass",
        "reference": "eager",
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / count if count else 0.0,
        "max_rel_error": max_rel,
    }
    if failure:
        proof["error"] = failure
    return proof


def _eager_outputs_by_name(outputs: dict, eager_out) -> dict:
    """Map positional eager outputs to the lowered graph's stable output names."""
    refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    if len(refs) != len(outputs):
        return {}
    return {name: ref.detach().float().cpu().numpy() if hasattr(ref, "detach") else ref for name, ref in zip(outputs, refs, strict=True)}


def _cuda_knob_dicts(graph) -> list[dict]:
    """Raw ``op.knobs`` per ``CudaOp`` of a compiled pinned graph, in launch order —
    the realized side of the pin gate."""
    return [dict(n.op.knobs or {}) for n in _launch_order_cuda_nodes(graph)]


def _ab_samples(specs, dynamic=None):
    """One shapeless pseudo-sample per ``--ab "K1=V1,K2=V2"`` spec: ``.knobs`` to pin
    (the ``EMMY_KNOBS`` grammar), ``.name`` the table label, ``.shape None`` —
    the marker :func:`_print_kernel_stats` uses to nest the row by the benched
    kernel's own ``S_*`` signature instead of a golden's matmul shape. ``dynamic``
    stamps the run's own ``--dynamic`` specs on each pseudo-sample so the A/B
    re-trace builds the same symbolic graph as the greedy run."""
    from types import SimpleNamespace  # noqa: PLC0415

    from emmy.compiler.pipeline.knob import parse_knob_spec  # noqa: PLC0415

    dyn = tuple(dynamic) if dynamic else None
    return [SimpleNamespace(name=f"ab {raw}", knobs=parse_knob_spec(raw), shape=None, dynamic=dyn) for raw in specs]


def _sample_replay_knobs(sample) -> dict:
    """All knob pins needed to reproduce a golden winner or explicit A/B row."""
    return {**getattr(sample, "pins", {}), **sample.knobs}


async def _bench_golden_variants(backend, source, golden_configs, *, warmup, iters, ref=None, strict_correctness=False):
    """Compile + bench each recorded golden config with its knobs pinned — one
    ``_GoldenBench`` per config so :func:`_print_kernel_stats` can show each as a measured
    row beside the greedy pick. ``golden_configs`` are
    :class:`~emmy.compiler.pipeline.search.data.Sample`s, whose ``pins`` hold the
    input regime and whose ``knobs`` hold the tunable-only measured winner
    (``S_*`` / ``H_*`` features are not knobs) —
    or the shapeless ``--ab`` pseudo-samples from :func:`_ab_samples` (same duck
    type). Each config re-traces a **fresh** graph from ``code`` — a frontend graph
    can't be re-compiled (the first lowering mutates it in place, so a reused graph
    would yield the first config's kernel every time). A sample carrying ``dynamic``
    specs (a dynamic golden, or an ``--ab`` row of a ``--dynamic`` run) re-traces
    symbolically, so the pinned kernel is the same masked-tile artifact the greedy
    run deployed and benches at the same hint. A config that fails to compile / bench
    is kept as a ``bench_fail`` row (never dropped — the table / ``--json`` show why).

    Compiling (trace + lowering) is parent CPU work; every GPU touch is one job on the
    backend's persistent SIGKILL-able worker (``bench_pinned_async``) — the same path the
    greedy row measured through, so all rows share one timing semantics, and a pinned
    config that hangs dies with the child: the row records ``bench_fail``, the parent's
    CUDA context stays clean, and the next row's job respawns a fresh child.

    **A pin matching no offered row fails the row loudly, before any GPU time**: the
    realized-vs-pinned check (:func:`unreproducible_pin_flag`) runs right after the pinned
    compile, and a miss marks the row ``pin_unmatched`` / NOT benched. Benching the fallback
    realization would measure the planner's own pick under the pin's name — the silent-degrade
    class that misled the hd256 flash sweep (a misspelled pin read as a form refusal).

    Two more integrity gates flag (never drop) each benched row: the arithmetic-intensity
    floor (:func:`_intensity_floor_flag`) and — when ``ref`` carries reference inputs and
    outputs — an output-correctness check. The default check compares against the greedy
    Emmy output with :func:`_wrong_answer_flag`; ``strict_correctness`` compares every pinned
    row directly against eager under rtol=atol=1e-3 and records error statistics. Flags
    render as a ``!`` marker in the table and ride the ``--json`` record."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs  # noqa: PLC0415

    out = []
    ref_inputs, ref_outputs = ref if ref is not None else (None, None)
    if strict_correctness and (ref_inputs is None or ref_outputs is None):
        raise ValueError("strict pinned correctness requires same-input eager reference outputs")
    # Session-unique cache key: the (potentially hundreds-of-MB) reference inputs cross
    # the worker pipe once per child, not once per row (see benchmark_pinned_isolated_async).
    ref_key = uuid.uuid4().hex if ref_inputs is not None else None
    for sample in golden_configs or []:
        replay_knobs = _sample_replay_knobs(sample)
        # String sources must be retraced at the recorded symbolic shape. An
        # embedded stable program already carries its symbolic dimensions.
        dyn = getattr(sample, "dynamic", None) if isinstance(source, str) else None
        flags = []
        try:
            dynamic_shapes = build_torch_dynamic_shapes(parse_position_specs(list(dyn))) if dyn else None
            with pinned_knobs(replay_knobs):
                # Fresh graph; lowering mutates it and bakes the pins into the kernel.
                if isinstance(source, str):
                    graph, _, _ = graph_from_code(source, dynamic_shapes=dynamic_shapes)
                else:
                    graph = source.copy()
                g_compiled = backend.compile(graph)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
            logger.warning("[golden] %s: compile of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, None, None, [f"compile failed: {exc}"], "bench_fail"))
            continue
        flag = unreproducible_pin_flag(replay_knobs, _cuda_knob_dicts(g_compiled))
        if flag:
            flags.append(f"{flag} — row NOT benched")
            logger.error(
                "[golden] %s: %s — the pinned config did not realize, so benching it would measure the planner's "
                "own pick under the pin's name; fix the pin spelling (row kept unbenched in the table / --json)",
                sample.name,
                flag,
            )
            out.append(_GoldenBench(sample, g_compiled, None, flags, "pin_unmatched"))
            continue
        try:
            g_bench, run_outputs = await backend.bench_pinned_async(
                g_compiled, run_inputs=ref_inputs, run_inputs_key=ref_key, warmup=warmup, num_iters=iters
            )
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own bench table
            logger.warning("[golden] %s: bench of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, g_compiled, None, [f"bench_fail: {exc}"], "bench_fail"))
            continue
        correctness = None
        if run_outputs is not None and ref_outputs is not None:
            if strict_correctness:
                correctness = _strict_correctness_proof(run_outputs, ref_outputs)
                if correctness["status"] != "pass":
                    flags.append(f"strict eager correctness failed: {correctness.get('error', 'tolerance exceeded')}")
            else:
                flag = _wrong_answer_flag(run_outputs, ref_outputs)
                if flag:
                    flags.append(flag)
        total_us = (g_bench.min_ms if g_bench.min_ms is not None else g_bench.time_ms) * 1000
        flag = _intensity_floor_flag(sample, total_us)
        if flag:
            flags.append(flag)
        for f in flags:
            logger.warning("[golden] %s: %s — row flagged (marked ! in the table, flagged in --json)", sample.name, f)
        out.append(_GoldenBench(sample, g_compiled, g_bench, flags, "ok", correctness))
    return out


async def _bench_greedy_isolated(backend, compiled, *, warmup, iters):
    """Re-bench the greedy deploy's compiled graph emmy-only through the pinned-row worker
    path (``bench_pinned_async``) — the pinned-comparable greedy baseline. The greedy
    comparison row times emmy interleaved with the live torch closures (same warm clocks /
    caches for the eager / torch.compile table — deliberate), which leaves torch allocator
    state and cuBLAS L2 carveouts active while emmy is measured; a pinned row never runs
    torch, so the same config reads ~7% apart between the two positions (worst on split-K
    pairs, whose finalize re-reads the partials workspace). One number can't be both
    torch-comparable and pinned-comparable, so the greedy config benches twice: this row is
    the one pinned golden / ``--ab`` speedups read against. Reuses the already-compiled
    greedy graph — one extra worker job, no recompile. Returns a ``_GoldenBench`` (status
    ``ok`` / ``bench_fail``); a failure never blocks the pinned rows."""
    from types import SimpleNamespace  # noqa: PLC0415

    sample = SimpleNamespace(name="greedy (isolated)", knobs={}, shape=None, dynamic=None)
    try:
        g_bench, _ = await backend.bench_pinned_async(compiled, warmup=warmup, num_iters=iters)
    except Exception as exc:  # noqa: BLE001 — an iso-bench failure must not abort the pinned rows
        logger.warning("greedy isolated re-bench failed (%s) — row kept as bench_fail; pinned rows still bench", exc)
        return _GoldenBench(sample, compiled, None, [f"bench_fail: {exc}"], "bench_fail")
    return _GoldenBench(sample, compiled, g_bench, [], "ok")


def _launch_order_cuda_nodes(graph):
    """CudaOp nodes in the backend's launch order (``graph.topological_order()`` — the order
    ``bench.per_launch`` indexes). Graph dict order diverges from launch order after
    node-splitting passes (a split partial lands after its consumer in dict order), which
    cross-labels the per-kernel timings if used for pairing."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    return [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]


def _print_kernel_stats(graph, bench, golden_benches=None, greedy_fail=None, greedy_iso=None):
    """Per-kernel breakdown. Pulls structural stats off each ``CudaOp``
    (block / grid / smem), per-launch timings from ``bench.per_launch``,
    and per-kernel hardware attributes from the compiled cupy RawKernels
    (register count, achieved theoretical occupancy). One row per kernel
    — quick at-a-glance for spotting which kernel dominates, whether
    register pressure is killing occupancy, etc.

    ``golden_benches`` (from a selected working-golden target, see :func:`_bench_golden_variants`)
    are each their own live compile+bench of a recorded golden config's pinned
    knobs; their kernels print beneath the greedy kernel of matching shape, labeled
    ``golden NAME [fm|std]`` in the Kernel column (the lane tag so an ``[fm]``-vs-``std``
    row is never misread against a ``std`` greedy) — a real A/B, not the recorded number. Their
    ``%`` column is ``--`` (they're not part of the emmy TOTAL), and their knob
    cells are colored red where they differ from the greedy pick (like ``eval``).
    A pinned row that never benched (unmatched pin / bench_fail — ``gb.bench is None``)
    still prints its kernels with ``--`` timings so the failure is visible, never dropped.

    ``greedy_iso`` (a ``_GoldenBench`` from :func:`_bench_greedy_isolated`, present when
    pinned rows benched) is the greedy graph re-benched emmy-only through the pinned-row
    path: each greedy kernel row gets a ``greedy (isolated)`` twin right beneath it (same
    kernel, pinned-row timing semantics) — the number to compare golden / ``--ab`` rows
    against; the greedy row's own µs is torch-comparable, not pinned-comparable (the
    documented ~7% skew). A failed iso re-bench prints as a ``!`` note, its rows skipped
    (they would duplicate the greedy geometry with ``--`` timings).

    ``bench=None`` + ``greedy_fail`` is the degraded mode: the greedy deploy itself failed
    (hung / blew the bench budget) in its worker child, so its rows print with ``--``
    timings and the TOTAL row reads ``bench_fail``. Kernel attributes (regs / occupancy)
    still collect — every bench runs in a child process, so THIS process's CUDA context is
    clean regardless of row failures, and regs/occ are exactly the diagnostic a hung deploy
    needs."""
    from emmy.commands.table import Col, knob_columns, render_table  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp, resolve_dim
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data import ShapeKey  # noqa: PLC0415

    cuda_nodes = _launch_order_cuda_nodes(graph)
    if not cuda_nodes:
        return

    # Per-kernel best-case (min) to match the min-based comparison table.
    times_by_idx = (
        {} if bench is None else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (bench.per_launch or [])}
    )
    total_us = None if bench is None else (bench.min_ms if bench.min_ms is not None else bench.time_ms) * 1000
    attrs_by_kname = _collect_kernel_attrs(graph)
    occ_limits = _occupancy_limits()

    # Symbolic grids (ceil-div over a dynamic axis) need a concrete env to
    # resolve for display — use each symbolic input dim's hint, so the printed
    # geometry reflects the hint-sized tile the kernel was tuned for.
    sym_env: dict[str, int] = {}
    for nid in graph.inputs:
        for dim in graph.buffer(nid).shape:
            if isinstance(dim.expr, Var) and dim.hint is not None:
                sym_env.setdefault(dim.expr.name, dim.hint)

    def _geom(op, attrs: dict):
        block_dims = [resolve_dim(d, sym_env) for d in op.block]
        grid_dims = [resolve_dim(d, sym_env) for d in op.grid]
        block_threads = block_dims[0] * block_dims[1] * block_dims[2]
        grid_total = grid_dims[0] * grid_dims[1] * grid_dims[2]
        regs = (attrs.get(op.kernel_name) or {}).get("num_regs", 0)
        occ_pct = _theoretical_occupancy(regs, op.smem_bytes, block_threads, occ_limits)
        occ_str = f"{occ_pct:>3.0f}%" if occ_pct is not None else "  --"
        return grid_total, block_threads, op.smem_bytes / 1024, regs, occ_str

    def _op_sig(op):
        return ShapeKey.from_s_features(getattr(op, "knobs", {}) or {})

    used_ab: set[int] = set()
    matched_golden: set[int] = set()

    def _matching(op):
        """Benched pinned variants whose shape matches this kernel — keyed via
        ``ShapeKey.from_s_features`` over the op's stamped ``S_*`` features, the
        same join key the prior diagnostics match goldens on (so the dtype flag
        splits fp32/fp16 twins here too). A golden carries its ``ShapeKey``
        on ``sample.shape``; a shapeless ``--ab`` entry matches through its own
        benched kernels' signatures and nests under the first greedy kernel it
        matches (``used_ab`` / ``matched_golden`` track placement; every unmatched
        entry — including a golden whose shape matches NO greedy kernel because
        greedy deployed a split partial+finalize pair — is appended after the
        greedy rows, so a row is never silently dropped)."""
        sig = _op_sig(op)
        out = []
        for gb in golden_benches or []:
            if gb.sample.shape is not None:
                if gb.sample.shape == sig:
                    matched_golden.add(id(gb))
                    out.append(gb)
            elif (
                gb.graph is not None
                and id(gb) not in used_ab
                and any(_op_sig(n.op) == sig for n in gb.graph.nodes.values() if isinstance(n.op, CudaOp))
            ):
                used_ab.add(id(gb))
                out.append(gb)
        return out

    # Build row records first (the knob columns are aligned across all rows, so we
    # can't stream): (name, t_us, pct_cell, geom, knobs, ref_knobs_or_None). A
    # golden row's ``ref`` is its matching greedy kernel's knobs — cells that differ
    # are colored red.
    records: list[tuple] = []

    def _gb_rows(gb, ref):
        """Append the rows of one benched pinned variant (golden / --ab), each
        kernel's knobs diffed red against ``ref`` (the greedy pick's knobs). A
        flagged row (unmatched pin / bench_fail / intensity floor / wrong-answer)
        gets a ``!`` marker; an unbenched row (``gb.bench is None``) prints its
        kernels with ``--`` timings (a compile failure has no kernels — one bare
        label row keeps the failure visible)."""
        label = f"golden {gb.sample.name}" if gb.sample.shape is not None else gb.sample.name
        label = f"{label} [{_lane(gb.sample.knobs)}]"  # so an [fm]-vs-std A/B can't be misread
        if gb.flags:
            label = f"! {label}"
        if gb.graph is None:
            records.append((label, None, "--", (0, 0, 0.0, 0, "  --"), {}, ref))
            return
        g_times = (
            {}
            if gb.bench is None
            else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (gb.bench.per_launch or [])}
        )
        g_attrs = {} if gb.bench is None else _collect_kernel_attrs(gb.graph)
        g_nodes = _launch_order_cuda_nodes(gb.graph)
        for gidx, gnode in enumerate(g_nodes):
            gk = dict(tuning_knob_items(gnode.op.knobs or {}))
            records.append(
                (label, g_times.get(gidx) if gb.bench is None else g_times.get(gidx, 0.0), "--", _geom(gnode.op, g_attrs), gk, ref)
            )

    # Per-launch times of the isolated greedy re-bench — same graph, so indexes align
    # 1:1 with ``cuda_nodes`` and each greedy kernel row gets its iso twin beneath it.
    iso_times = (
        None
        if greedy_iso is None or greedy_iso.bench is None
        else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (greedy_iso.bench.per_launch or [])}
    )
    for idx, node in enumerate(cuda_nodes):
        op = node.op
        t_us = None if bench is None else times_by_idx.get(idx, 0.0)
        pct_cell = "--" if total_us is None else f"{((t_us / total_us * 100) if total_us > 0 else 0.0):.1f}%"
        gknobs = dict(tuning_knob_items(op.knobs or {}))
        records.append((op.kernel_name, t_us, pct_cell, _geom(op, attrs_by_kname), gknobs, None))
        if iso_times is not None:
            records.append(("greedy (isolated)", iso_times.get(idx, 0.0), "--", _geom(op, attrs_by_kname), gknobs, gknobs))
        for gb in _matching(op):
            _gb_rows(gb, gknobs)
    # Catch-all: pinned rows that matched no greedy kernel still print — the golden
    # rows attach to the SHAPE (the run), not to a kernel node, so a greedy split
    # partial+finalize deploy can no longer silently drop the shape's golden A/B rows.
    for gb in golden_benches or []:
        if id(gb) not in used_ab and id(gb) not in matched_golden:
            _gb_rows(gb, None)

    kcols, kcells = knob_columns(
        [{k: (str(v), ref is not None and str(ref.get(k)) != str(v)) for k, v in knobs.items()} for *_, knobs, ref in records]
    )
    columns = [
        Col("Kernel"),
        Col("us", "r"),
        Col("%", "r"),
        Col("grid", "r"),
        Col("block", "r"),
        Col("smem", "r"),
        Col("regs", "r"),
        Col("occ", "r"),
        *kcols,
    ]
    data = []
    for rec, kc in zip(records, kcells, strict=True):
        name, t_us, pct_cell, (grid_total, block_threads, smem_kb, regs, occ_str) = rec[:4]
        us_cell = "--" if t_us is None else f"{t_us:.1f}"
        data.append([name, us_cell, pct_cell, str(grid_total), str(block_threads), f"{smem_kb:.1f}K", str(regs), occ_str.strip(), *kc])
    data.append(["TOTAL", "bench_fail" if total_us is None else f"{total_us:.1f}", *[""] * (len(columns) - 2)])
    # TOTAL sums per-launch solo windows (each kernel replayed back-to-back in
    # its own event window); the whole-program row is one window around the
    # full launch sequence — the number the backend comparison table reports.
    if bench is not None and bench.e2e_min_ms is not None:
        data.append(["whole-program (e2e)", f"{bench.e2e_min_ms * 1000:.1f}", *[""] * (len(columns) - 2)])

    print()
    print("knobs (greedy pick); golden / ab rows are red where they differ from the greedy pick:")
    for line in render_table(columns, data):
        print(line)
    if greedy_fail is not None:
        print(f"! greedy: bench_fail — {greedy_fail}")
    for gb in ([greedy_iso] if greedy_iso is not None else []) + list(golden_benches or []):
        for flag in gb.flags:
            print(f"! {gb.sample.name}: {flag}")


def _write_ab_json(
    args,
    results: dict,
    graph,
    bench,
    golden_benches,
    greedy_fail=None,
    greedy_iso=None,
    greedy_reference_us=None,
) -> None:
    """``--json PATH``: the whole ``--bench`` comparison as one machine-readable record —
    the backend table (eager / torch.compile / emmy), the per-kernel greedy rows, and every
    ``--golden`` / ``--ab`` pinned row with its recorded reference latencies and integrity
    flags. ``pinned_knobs`` is the exact input-regime-plus-winner map used for replay. This is
    the golden-sweep workflow's parse target (it retires the ad-hoc stdout
    table parsers) and where the intensity-floor / wrong-answer verdicts become fields —
    the confirm-twice rule diffs two of these files instead of two terminal scrollbacks.

    Each kernel row carries ``record_knobs`` — the realized tuning knobs with EVERY schedule
    family explicitly stamped (:func:`~emmy.compiler.pipeline.knob.stamp_schedule_families`,
    OFF spelling included) — the map to copy verbatim into a golden YAML ``knobs:`` entry, so
    a new recording never leaves a family to the planner's replay-time fill (the recurring
    unpinned-``REDUCE`` drift class). Failure states are fields, not absences: the greedy
    block carries ``status`` (``"bench_fail"`` + ``error`` when the deploy failed) and each
    pinned row its ``status`` (``ok`` / ``pin_unmatched`` / ``bench_fail``) with ``us`` /
    ``total_us`` null where nothing was measured.

    ``greedy_iso`` (present when pinned rows benched) lands as ``greedy.isolated`` —
    the same graph re-benched emmy-only through the pinned-row path, shaped like a pinned
    row (``status`` / ``total_us`` / ``e2e_us`` / ``kernels`` / ``flags``). Sweep tooling
    compares pinned ``e2e_us`` when a multi-kernel program exposes it, otherwise ``total_us``,
    against THIS block; the greedy block's own ``total_us`` is the
    interleaved (torch-comparable) number, ~7% apart from pinned-row semantics.

    Every pinned row and the greedy block carry a ``lane`` (``"fm"`` / ``"std"``, :func:`_lane`):
    a working-golden target pins BOTH lanes recorded under one name, and comparing a pinned
    ``[fm]`` latency against a ``"std"`` greedy is the phantom-regression trap the sweep must not
    fall into — a parser filters ``pinned`` to the rows whose ``lane`` matches ``greedy.lane``
    (which ``greedy.isolated`` shares, being the same graph)."""
    import json as _json  # noqa: PLC0415

    from emmy import gpu  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import stamp_schedule_families, tuning_knob_items  # noqa: PLC0415

    def _kernel_rows(g, b) -> list[dict]:
        import hashlib  # noqa: PLC0415

        if g is None:
            return []
        times = {} if b is None else {lt.idx: (min(lt.samples) if lt.samples else lt.time_ms) * 1000 for lt in (b.per_launch or [])}
        rows = []
        for idx, node in enumerate(_launch_order_cuda_nodes(g)):
            op = node.op
            source = getattr(op, "kernel_source", None) or ""
            rows.append(
                {
                    "kernel": op.kernel_name,
                    "us": None if b is None else times.get(idx, 0.0),
                    "smem_bytes": op.smem_bytes,
                    "knobs": {k: str(v) for k, v in tuning_knob_items(op.knobs or {})},
                    "record_knobs": stamp_schedule_families(op.knobs or {}),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
            )
        return rows

    def _timing(b) -> dict:
        if b is None:
            return {"total_us": None, "timing_semantics": None, "captured": None, "num_launches": 0}
        per_launch = b.per_launch or []
        num_launches = getattr(b, "num_launches", 0) or len(per_launch)
        e2e_min_ms = getattr(b, "e2e_min_ms", None)
        if e2e_min_ms is not None:
            total_ms = e2e_min_ms
            semantics = "whole_program_e2e"
        else:
            total_ms = b.min_ms if b.min_ms is not None else b.time_ms
            semantics = "single_launch" if num_launches <= 1 else "per_launch_sum"
        return {
            "total_us": total_ms * 1000,
            "timing_semantics": semantics,
            "captured": bool(getattr(b, "captured", False)),
            "num_launches": num_launches,
        }

    def _e2e_us(b) -> float | None:
        return None if b is None or b.e2e_min_ms is None else b.e2e_min_ms * 1000

    pinned = []
    for gb in golden_benches or []:
        sample = gb.sample
        pinned.append(
            {
                "name": sample.name,
                "kind": "golden" if sample.shape is not None else "ab",
                "lane": _lane(sample.knobs),
                "status": gb.status,
                "pinned_knobs": {k: str(v) for k, v in _sample_replay_knobs(sample).items()},
                "total_us": _total_us(gb.bench),
                "e2e_us": _e2e_us(gb.bench),
                "kernels": _kernel_rows(gb.graph, gb.bench),
                "flags": list(gb.flags),
                "correctness": gb.correctness,
                "recorded_emmy_us": getattr(sample, "latency_us", None),
                "recorded_ref_us": getattr(sample, "ref_us", None),
            }
        )
    greedy = {
        "status": "ok" if greedy_fail is None else "bench_fail",
        "lane": _graph_lane(graph),
        **_timing(bench),
        "kernels": _kernel_rows(graph, bench),
    }
    if greedy_fail is not None:
        greedy["error"] = greedy_fail
    if greedy_reference_us is not None:
        greedy["reference_run_us"] = greedy_reference_us
    if greedy_iso is not None:
        greedy["isolated"] = {
            "status": greedy_iso.status,
            "total_us": _total_us(greedy_iso.bench),
            "e2e_us": _e2e_us(greedy_iso.bench),
            "kernels": _kernel_rows(greedy_iso.graph, greedy_iso.bench),
            "flags": list(greedy_iso.flags),
        }
    captured = bool(getattr(bench, "captured", False))
    backend_semantics = "captured_whole_forward" if captured else "uncaptured_forward"
    backend_rows = {
        name: {
            "latency_us": us,
            "captured": captured,
            "timing_semantics": backend_semantics,
            **({"correctness": {"status": "pass", "rtol": 1e-3, "atol": 1e-3, "fullgraph": True}} if name == "torch.compile" else {}),
        }
        for name, us in (results or {}).items()
    }
    if correctness is not None and "Emmy" in backend_rows:
        backend_rows["Emmy"]["correctness"] = correctness
    eager_us = (results or {}).get("Eager PyTorch")
    if eager_us:
        for name, us in (results or {}).items():
            backend_rows[name]["speedup_vs_eager"] = eager_us / us if us else 0.0

    payload = {
        "input": args.code or args.input or getattr(args, "ir", None),
        "golden": getattr(args, "golden", None),
        "dynamic": list(args.dynamic) if getattr(args, "dynamic", None) else [],
        "gpu": gpu.live_name(),
        "warmup": args.warmup,
        "iters": args.iters,
        "backends": backend_rows,
        "greedy": greedy,
        "pinned": pinned,
    }
    if strict_errors is not None:
        payload["strict"] = {"status": "fail" if strict_errors else "pass", "errors": list(strict_errors)}
    out = Path(args.json)
    out.write_text(_json.dumps(payload, indent=2, default=str))
    print(f"A/B record → {out}")


def _collect_kernel_attrs(graph) -> dict[str, dict]:
    """Compile each kernel via ``cupy.RawKernel`` (cached by source) to
    pull post-PTXAS hardware attributes — register count, static smem,
    spill bytes. Returns ``{kernel_name: attrs_dict}``."""
    from emmy.compiler.ir.cuda.ir import CudaOp

    try:
        import cupy as cp
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for _, node in graph.nodes.items():
        if not isinstance(node.op, CudaOp):
            continue
        try:
            k = cp.RawKernel(node.op.kernel_source, node.op.kernel_name, options=("--use_fast_math",))
            out[node.op.kernel_name] = dict(k.attributes)
        except Exception:  # pragma: no cover — environment-dependent
            continue
    return out


def _occupancy_limits() -> dict | None:
    """Per-device limits used to estimate theoretical occupancy. ``None``
    when cupy / CUDA aren't available."""
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        a = dev.attributes
        return {
            "max_threads_per_sm": a.get("MaxThreadsPerMultiProcessor", 0),
            "max_blocks_per_sm": a.get("MaxBlocksPerMultiprocessor", 0),
            "max_regs_per_sm": a.get("MaxRegistersPerMultiprocessor", 0),
            "max_smem_per_sm": a.get("MaxSharedMemoryPerMultiprocessor", 0),
            "warp_size": a.get("WarpSize", 32),
        }
    except Exception:
        return None


def _theoretical_occupancy(regs_per_thread: int, smem_per_block: int, threads_per_block: int, limits: dict | None) -> float | None:
    """Active-warps-per-SM ÷ peak-warps-per-SM × 100. Computed from the
    static-occupancy limits: register file, shared memory, and per-SM
    block / thread caps. Doesn't account for the dynamic-only spill +
    stack overhead but is enough to flag occupancy cliffs (regs > 64
    drops most consumer GPUs from 100% → 50%, smem > 49KB likewise)."""
    if not limits or threads_per_block <= 0 or regs_per_thread <= 0:
        return None
    warp_size = limits["warp_size"]
    max_warps = limits["max_threads_per_sm"] // warp_size
    if max_warps <= 0:
        return None

    blocks_by_threads = limits["max_threads_per_sm"] // threads_per_block
    blocks_by_blocks = limits["max_blocks_per_sm"]
    blocks_by_regs = limits["max_regs_per_sm"] // max(regs_per_thread * threads_per_block, 1)
    blocks_by_smem = limits["max_smem_per_sm"] // max(smem_per_block, 1) if smem_per_block > 0 else blocks_by_threads
    active_blocks = max(0, min(blocks_by_threads, blocks_by_blocks, blocks_by_regs, blocks_by_smem))
    active_warps = active_blocks * (threads_per_block // warp_size)
    return min(100.0, 100.0 * active_warps / max_warps)


_NCU_RECURSE_GUARD = config.NCU_CHILD

# Curated ncu metric set — verified to populate on RTX 5090 (sm_120) +
# TMA kernels. ``--set detailed`` is broken there (``SpeedOfLight_Roofline``
# divides by zero, ``SourceCounters`` / ``PCSamplingData`` need missing
# ``smsp__pcsamp_*`` metrics) so we enumerate explicit metrics instead.
# Add to this list to surface new columns in the perf summary; the
# downstream parser keys on metric names directly.
_NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_lsu.sum",
    "launch__registers_per_thread",
)


def _run_ncu_profile(args, *, dump_dir=None):
    """Re-launch the same ``emmy run`` invocation under ``ncu`` to
    collect a curated set of hardware counters (occupancy, bank
    conflicts, SM/DRAM/FMA throughput, register pressure). Output is
    captured in CSV form; when ``EMMY_DUMP_DIR`` (or ``--dump-dir``
    propagated as ``dump_dir``) is set, the raw CSV and a parsed
    per-kernel JSON are written there. Otherwise the counters print to
    stdout in the same CSV form for one-shot inspection.

    Spawns one extra subprocess at minimal iter count — ncu's per-launch
    overhead is huge (10-100×). The ``EMMY_NCU_CHILD`` env var
    prevents the profiled child from re-spawning ncu recursively.

    Skipped silently when ``ncu`` is not on PATH. ncu's own stderr is
    relayed when it fails (typical failure: NVIDIA's perf-counter
    permission gate)."""
    import json as _json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path as _Path

    if config.ncu_child():
        return

    ncu = shutil.which("ncu")
    if ncu is None:
        logger.info("ncu not found on PATH; skipping --profile output")
        return

    env = dict(os.environ)
    env[_NCU_RECURSE_GUARD] = "1"
    # The ncu child process re-runs trace + compile, which would
    # ``shutil.rmtree`` the parent's dump dir from ``CompilerDump``'s
    # ``__post_init__``. Drop the env var for the child so the parent's
    # ``60_*.json`` (bench results) survive — ncu output is captured via
    # stdout and saved by the parent below.
    env.pop(config.DUMP_DIR, None)

    cmd: list[str] = [
        ncu,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(_NCU_METRICS),
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
    ]
    if getattr(args, "golden_file", None):
        cmd.extend(["--golden", args.golden_file, "--target", args.golden])
    elif args.code is not None:
        cmd.extend(["--code", args.code])
    elif args.ir is not None:
        cmd.extend(["--ir", args.ir])
    else:
        # Positional model ID / .json path. Forward the trace shape flags so the
        # child profiles the same graph the parent benched.
        cmd.append(args.input)
        if args.layer is not None:
            cmd.extend(["--layer", str(args.layer)])
        cmd.extend(["--seq-len", str(args.seq_len)])
    if getattr(args, "gpu_arch", None) is not None:
        cmd.extend(["--gpu-arch", args.gpu_arch])
    # Forward the symbolic-dim specs so the child re-traces the SAME (masked-tile)
    # graph the parent benched. ``args.dynamic`` is the ``NAME@INPUT:AXIS`` CLI form
    # (a dynamic ``--golden`` sets it too, via ``resolve_golden_arg``); without this
    # the child re-compiles the static twin and ncu profiles the wrong kernel.
    for spec in getattr(args, "dynamic", None) or []:
        cmd.extend(["--dynamic", spec])
    # ncu's per-launch overhead means we want one or two launches per
    # kernel — enough for the counters to populate, not so many that
    # the run drags out. Match ``emmy run --bench``'s minimal
    # warmup so the profiled launches see a realistic-ish steady state.
    cmd.extend(["--warmup", "2", "--iters", "3"])

    print()
    print("=" * 80)
    print("ncu --csv (curated hardware metrics)")
    print("=" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("ncu profiling timed out")
        return

    # ncu writes ``==PROF==`` status lines first, then a CSV table
    # starting at the ``"ID"`` header. Split the two so we can save just
    # the CSV part as a file and surface the status lines for
    # diagnostics.
    stdout = result.stdout
    lines = stdout.splitlines()
    csv_start = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            csv_start = i
            break
    if csv_start is None:
        if stdout.strip():
            print(stdout)
        if result.returncode != 0:
            logger.warning("ncu exit=%d", result.returncode)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)
        return

    csv_text = "\n".join(lines[csv_start:]) + "\n"
    status_text = "\n".join(lines[:csv_start])

    parsed = _parse_ncu_csv(csv_text)
    _print_ncu_compare(parsed, _ncu_units(csv_text))

    if dump_dir is not None:
        out_dir = _Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "61_ncu_metrics.csv").write_text(csv_text)
        (out_dir / "61_ncu_metrics.json").write_text(_json.dumps(parsed, indent=2, default=str))
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.csv'}")
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.json'}")
    elif status_text.strip():
        # No dump dir: the comparison table above is the product; relay the
        # ==PROF== status lines for diagnostics (the raw CSV lands in the dump
        # dir when one is set).
        print(status_text)

    if result.returncode != 0:
        logger.warning("ncu exit=%d", result.returncode)
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)


# Display order + short labels for the per-kernel comparison table — keys are
# entries of the curated ``_NCU_METRICS`` set.
_NCU_COMPARE_COLS = (
    ("gpu__time_duration.sum", "dur"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "occ%"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm%"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram%"),
    ("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "fma%"),
    ("smsp__inst_executed_pipe_lsu.sum", "lsu.inst"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "ld.cnflct"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "st.cnflct"),
    ("launch__registers_per_thread", "regs"),
)

_NCU_REF_ROWS_MAX = 12  # eager forwards can launch dozens of tiny aten kernels


def _print_ncu_compare(parsed: dict, units: dict[str, str] | None = None) -> None:
    """One aligned table over the parsed ncu metrics: the emmy ``k_*``
    kernels first, then the reference backend's kernels (cuBLAS / cutlass /
    aten) — so the counter deltas (occupancy, SM/DRAM/FMA utilization, smem
    bank conflicts, register pressure) read straight down a column instead of
    across two CSV dumps. Each side sorts by duration; reference rows truncate
    to the :data:`_NCU_REF_ROWS_MAX` slowest with a count of the rest."""
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    if not parsed:
        return
    dur_key = _NCU_COMPARE_COLS[0][0]
    dep = sorted((k for k in parsed if k.startswith("k_")), key=lambda k: -parsed[k].get(dur_key, 0.0))
    ref = sorted((k for k in parsed if not k.startswith("k_")), key=lambda k: -parsed[k].get(dur_key, 0.0))
    hidden_ref = max(0, len(ref) - _NCU_REF_ROWS_MAX)
    ref = ref[:_NCU_REF_ROWS_MAX]

    def fmt(v) -> str:
        if v is None:
            return "-"
        return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:.1f}"

    def row(side: str, kname: str) -> list[str]:
        shown = kname if len(kname) <= 60 else kname[:57] + "..."
        return [side, shown, *(fmt(parsed[kname].get(metric)) for metric, _ in _NCU_COMPARE_COLS)]

    unit = (units or {}).get(dur_key, "")
    cols = [Col("side"), Col("kernel")] + [
        Col(f"dur ({unit})" if label == "dur" and unit else label, "r") for _, label in _NCU_COMPARE_COLS
    ]
    data = [row("dep", k) for k in dep] + [row("ref", k) for k in ref]
    print()
    print("ncu compare — emmy kernels vs the torch/cuBLAS reference (same counters, one table):")
    for line in render_table(cols, data, rule=True):
        print(line)
    if hidden_ref:
        print(f"  … {hidden_ref} more reference kernel(s) below the duration cut")


def _ncu_units(csv_text: str) -> dict[str, str]:
    """``{metric: unit}`` from the CSV's ``Metric Unit`` column (first seen wins)."""
    import csv as _csv  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    units: dict[str, str] = {}
    for row in _csv.DictReader(_io.StringIO(csv_text)):
        metric, unit = row.get("Metric Name", ""), row.get("Metric Unit", "")
        if metric and unit and metric not in units:
            units[metric] = unit
    return units


def _parse_ncu_csv(csv_text: str) -> dict:
    """Reduce ncu's launch-by-launch CSV into per-kernel metric dicts.

    Each row in the CSV is one (kernel, metric) datum for one launch;
    multi-launch profiling produces many rows per (kernel, metric) which
    we aggregate: ``.sum`` and ``smsp__*`` counters get summed, every
    other metric (percentages, per-thread regs) gets averaged. Returns
    ``{kernel_name: {metric_name: numeric_value}}`` ready to be merged
    with the bench-comparison JSON downstream.

    Keeps every kernel — the emmy ``k_*`` rows AND the reference
    backend's (cuBLAS / cutlass / aten) rows the child's eager forward
    contributes — so :func:`_print_ncu_compare` can put the two sides in
    one table. Consumers split the sides on the ``k_*`` naming convention.
    """
    import csv as _csv
    import io as _io

    reader = _csv.DictReader(_io.StringIO(csv_text))
    per_kernel: dict[str, dict[str, list[float]]] = {}
    for row in reader:
        kname = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        raw = row.get("Metric Value", "").replace(",", "")
        if not (kname and metric and raw):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        per_kernel.setdefault(kname, {}).setdefault(metric, []).append(val)

    out: dict[str, dict[str, float]] = {}
    for kname, metrics in per_kernel.items():
        reduced: dict[str, float] = {}
        for metric, vals in metrics.items():
            if metric.endswith(".sum") or metric.startswith("smsp__"):
                reduced[metric] = sum(vals)
            else:
                reduced[metric] = sum(vals) / len(vals)
        out[kname] = reduced
    return out


def _detect_stage(graph) -> str:
    """Identify the IR stage by scanning op type names. Returns one of
    ``torch | tensor | loop | tile | kernel | cuda`` — the highest-stage
    op present in the graph wins, since lowering produces a graph mixed
    only briefly during a pass and stable in the post-pass form."""
    stage_by_op: dict[str, str] = {
        "CudaOp": "cuda",
        "KernelOp": "kernel",
        "TileOp": "tile",
        "LoopOp": "loop",
    }
    order = ["torch", "tensor", "loop", "tile", "kernel", "cuda"]
    best = "torch"
    for node in graph.nodes.values():
        s = stage_by_op.get(type(node.op).__name__)
        if s and order.index(s) > order.index(best):
            best = s
    # Anything that's not Loop/Tile/Kernel/Cuda but is a frontend/tensor
    # op stays at "torch" — they get rewritten by the frontend passes.
    return best


def _passes_after_stage(stage: str) -> list[str]:
    """Pipeline tail to run after a graph has reached ``stage``."""
    from emmy.compiler.pipeline import (
        CUDA_PASSES,
        KERNEL_PASSES,
        LOOP_PASSES,
        TENSOR_PASSES,
        TILE_PASSES,
    )

    completed = {
        "torch": [],
        "tensor": TENSOR_PASSES,
        "loop": LOOP_PASSES,
        "tile": TILE_PASSES,
        "kernel": KERNEL_PASSES,
        "cuda": CUDA_PASSES,
    }[stage]
    return [p for p in CUDA_PASSES if p not in completed]


def _replay_stage_and_passes(graph, *, embedded_golden: bool) -> tuple[str, list[str]]:
    """The input label and pass list for an IR replay.

    A persisted golden Loop target stores stable algebra, not the derived ``LoopOp.knobs`` from
    the structural stamp. Replay it through the full pipeline so deploy evidence can see those
    features. A direct ``--ir`` input keeps its declared-stage tail semantics.
    """
    if embedded_golden:
        from emmy.compiler.pipeline import CUDA_PASSES  # noqa: PLC0415

        return "golden Loop", CUDA_PASSES
    stage = _detect_stage(graph)
    return stage, _passes_after_stage(stage)


async def bench_lowered_vs_torch(
    frontend,
    lowered,
    backend,
    *,
    seed,
    do_bench,
    warmup,
    iters,
    bench_backends,
    capture_graphs=True,
    ref_out=None,
    ref_us_out=None,
):
    """Run + (optionally) benchmark a lowered graph against its torch reference on
    shared random inputs. The common bench primitive behind ``run --ir`` and
    ``tune --bench``.

    ``frontend`` is the pristine frontend-dialect snapshot (must be
    ``torch_ref``-runnable); pass ``None`` for a non-frontend / unsupported graph to
    bench emmy-only. One random source is drawn per distinct constant
    ``source_path`` and each side replays its own ``load_ops`` (so a weight lowering
    transposed + renamed stays the same underlying tensor on both sides). The lowered
    graph runs once for a non-fatal accuracy check vs the torch eager reference; then,
    when ``do_bench``, the selected backends are timed — interleaved when a torch ref
    exists (full ``warmup``/``iters``), else emmy-only at reduced iters.

    ``capture_graphs`` (default on — this function's callers are the per-kernel
    reproducer paths, where the torch side replays the frontend graph op-by-op and
    would otherwise be dispatch-bound) wraps every timed backend in a CUDA graph so
    the event windows measure pure GPU time. All-or-nothing: if any torch backend
    or the emmy launch loop fails to capture, the whole bench retries
    uncaptured (warning logged) so one table never mixes timing semantics. The
    accuracy check always runs uncaptured, before any capture.

    Returns ``(results, bench, torch_available, captured, accuracy_error)``: ``results``
    is the ``{backend: latency_us}`` dict (``None`` when ``do_bench`` is False), ``bench``
    the emmy ``BenchmarkResult`` (``None`` when ``do_bench`` is False),
    ``torch_available`` whether an eager/torch.compile reference was built, ``captured``
    whether the timings came from graph-captured (pure-GPU) windows, and
    ``accuracy_error`` the non-fatal accuracy verdict (``None`` = passed or no reference;
    also logged here — returned so a worker-side run can ship it back to the parent, whose
    child logs are invisible). When ``ref_out`` is a list, the greedy run's same-input
    ``(input_data, outputs)`` pair is appended for pinned A/B verification; ``ref_us_out``
    receives that execution's event timing, which remains valid reference metadata even when
    the later repeated greedy benchmark fails. Does no printing / dumping — callers own that."""
    import numpy as np
    import torch

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.loader.binder import bind_constants

    rng = np.random.default_rng(seed)

    # Symbolic reproducers bench at each dim's hint (``DEFAULT_SEQ_HINT`` for a
    # bare seq axis) — the same size the backend resolves a symbolic graph to
    # when no inputs are supplied — so the random inputs built here get concrete
    # hint-sized shapes. Atomic dims carry their own hint; composite exprs
    # (e.g. ``S * 2``) eval over the collected env.
    from emmy.compiler.dim import Dim

    sym_env = _collect_sym_env(([frontend] if frontend is not None else []) + [lowered])

    def _static(shape):
        return tuple((d.as_static() if d.is_static else int(d.expr.eval(sym_env))) if isinstance(d, Dim) else int(d) for d in shape)

    # One random source array per distinct constant ``source_path`` — shared by
    # emmy (lowered graph) and the torch ref (frontend graph). ``bind_constants``
    # replays each constant's ``load_ops`` on it, so a weight that lowering transposed +
    # renamed (linear: out×in → in×out) stays the same underlying tensor on both sides.
    sources: dict[str, np.ndarray] = {}
    for gph in ([frontend] if frontend is not None else []) + [lowered]:
        for node in gph.nodes.values():
            op = node.op
            if not (isinstance(op, ConstantOp) and op.value is None):
                continue
            if op.source_path and op.source_path not in sources:
                shp = _static(op.source_shape or node.output.shape)
                sources[op.source_path] = _random_source_values(rng, shp, op.source_dtype)
            # A merged (source_parts) constant draws one random source PER PART, keyed by the
            # part path — the same tensors the pre-merge frontend reference binds its separate
            # weights from, so emmy's concat and the torch ref stay numerically aligned.
            for path, shp in op.source_parts:
                if path not in sources:
                    sources[path] = _random_source_values(rng, _static(shp), op.source_dtype)

    input_data: dict[str, object] = {}
    input_tensors: dict[str, object] = {}
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, InputOp):
            arr = _random_input_values(rng, _static(node.output.shape), node.output.dtype)
            # Keep the ndarray shape (no flatten) — a symbolic graph's launch
            # reads the runtime seq_len off the input array's shape.
            input_data[nid] = arr
            input_tensors[nid] = _to_cuda_tensor(arr, node.output.dtype)
        elif isinstance(node.op, ConstantOp) and node.op.value is not None:
            input_data[nid] = [float(node.op.value)]

    input_data.update(bind_constants(lowered, sources))
    if frontend is not None:
        for fid, arr in bind_constants(frontend, sources).items():
            input_tensors[fid] = _to_cuda_tensor(arr, frontend.nodes[fid].output.dtype)

    # Fallback for any value-None constant with no source_path (synthetic).
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.value is None and nid not in input_data:
            arr = rng.standard_normal(_static(node.output.shape), dtype=np.float32) * 0.02
            input_data[nid] = arr.flatten().tolist()

    result, _ = backend.run(lowered, input_data=input_data)
    if ref_out is not None:
        ref_out.append((input_data, result.outputs))
    if ref_us_out is not None:
        ref_us_out.append(result.time_ms * 1000)
    for nid, arr in result.outputs.items():
        finite = np.isfinite(arr).all()
        logger.info("Output %s: shape=%s finite=%s mean=%.4f", nid, arr.shape, bool(finite), float(arr.mean()))

    # Build the torch reference from the frontend snapshot, fed the same inputs.
    torch_fn = torch_inputs = accuracy_error = correctness = reference = None
    if frontend is not None:
        try:
            torch_fn, torch_inputs = torch_ref.build_callable(frontend, input_tensors)
            with torch.no_grad():
                eager_out = torch_fn(*torch_inputs)
            if strict_accuracy:
                correctness = _strict_correctness_proof(result.outputs, eager_out)
                if correctness["status"] != "pass":
                    accuracy_error = f"strict eager correctness failed: {correctness.get('error', 'tolerance exceeded')}"
            else:
                accuracy_error = _check_accuracy(result.outputs, eager_out)
            reference = (input_data, _eager_outputs_by_name(result.outputs, eager_out))
            if accuracy_error is not None:
                qualifier = "fatal when strict correctness is requested" if strict_accuracy else "non-fatal (random-input reproducer)"
                logger.warning("%s — %s; benching anyway", accuracy_error, qualifier)
        except Exception as exc:  # noqa: BLE001 — torch ref is best-effort
            logger.warning("torch reference unavailable (%s) — skipping vs-torch comparison", exc)
            torch_fn = None
            if strict_accuracy:
                accuracy_error = f"strict eager correctness unavailable: {exc}"

    if strict_accuracy and frontend is None:
        accuracy_error = "strict eager correctness unavailable: frontend IR is not runnable"
    torch_available = torch_fn is not None
    if not do_bench:
        base = (None, None, torch_available, False, accuracy_error)
        return (*base, correctness, reference) if return_reference else base

    if torch_available:
        backends = _resolve_backends(bench_backends)
        torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends=backends)
        if capture_graphs:
            results, bench, captured = await _bench_interleaved_captured(
                torch_fn, torch_inputs, {}, backend, lowered, warmup, iters, torch_fns=torch_fns
            )
        else:
            results, bench = await _bench_interleaved(
                torch_fn, torch_inputs, {}, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=False
            )
            captured = False
        base = (results, bench, True, captured, accuracy_error)
        return (*base, correctness, reference) if return_reference else base
    # Emmy-only: a capture failure falls back inside ``benchmark_program``
    # (warned + reported via ``bench.captured``) — nothing to de-mix.
    bench = await backend.benchmark_async(lowered, warmup=max(3, warmup // 5), num_iters=max(10, iters // 5), capture_graphs=capture_graphs)
    base = ({"Emmy": bench.time_ms * 1000}, bench, False, bench.captured, accuracy_error)
    return (*base, correctness, reference) if return_reference else base


async def bench_full_model_real(module, args_t, kwargs, lowered, backend, *, warmup, iters, bench_backends):
    """End-to-end full-model bench against the **real torch module** — eager /
    ``torch.compile`` / Emmy — using the all-or-nothing CUDA-graph-captured
    interleaved bench (real modules occasionally resist capture; those fall back
    to uncaptured wall timing, flagged in ``captured``). The module + its
    trace-time inputs come from ``load_or_trace``'s bundle; for a symbolic
    graph the torch closures run on hint-tiled inputs (``_hint_sized_inputs``)
    so both sides bench the hint shape. Skips the accuracy check (emmy's
    bench uses synthetic activations vs torch's bound inputs, so only latency
    is comparable here — accuracy lives in the per-kernel path).
    Returns ``(results, bench, captured)``."""
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args_t)
    cuda_kwargs = _to_cuda_kwargs(kwargs)
    cuda_args, cuda_kwargs, _ = _hint_sized_inputs(lowered, cuda_args, cuda_kwargs)
    backends = _resolve_backends(bench_backends)
    torch_fns = _build_torch_fns(cuda_module, cuda_args, cuda_kwargs, warmup, backends=backends)
    if not torch_fns:
        # Emmy-only bench (``--bench-backends emmy``): any prior torch activity in this
        # process (the accuracy probe's eager forward, the warmups above) left
        # cuBLAS-installed ``cudaAccessPolicyWindow`` carveouts pinned in L2, which steal
        # L2 bandwidth from cp.async-staged kernels (~4x on the article tile — 1660 µs
        # with the window live, 330 µs after the reset). With torch backends in the table
        # the state must stay alive and the comparison absorbs it intrinsically; emmy-only,
        # reset it. (Caller-held module refs keep allocator blocks alive — the persisting-L2
        # window is the dominant term and resets regardless.)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _reset_persisting_l2_cache()
    return await _bench_interleaved_captured(cuda_module, cuda_args, cuda_kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns)


def _pinned_samples_for_ir(args, embedded):
    """Automatic verified pins plus explicit ``--ab`` pins for an embedded golden target."""
    pinned = list(getattr(args, "golden_configs", None) or [])
    if embedded is not None and (specs := getattr(args, "ab", None)):
        pinned.extend(_ab_samples(specs, dynamic=getattr(args, "dynamic", None)))
    return pinned


def _strict_benchmark_errors(args, results, bench, captured, correctness, pinned_rows) -> list[str]:
    """Return every strict verification failure in one ordinary run result."""

    def valid_proof(proof) -> bool:
        if not isinstance(proof, dict):
            return False
        required = {"status": "pass", "reference": "eager", "rtol": 1e-3, "atol": 1e-3}
        if any(proof.get(key) != value for key, value in required.items()):
            return False
        return all(
            not isinstance(proof.get(metric), bool) and isinstance(proof.get(metric), int | float) and proof[metric] >= 0
            for metric in ("max_abs_error", "mean_abs_error", "max_rel_error")
        )

    errors = []
    display_names = {"eager": "Eager PyTorch", "tcompile": "torch.compile", "emmy": "Emmy"}
    for backend in _resolve_backends(args.bench_backends):
        name = display_names[backend]
        latency = (results or {}).get(name)
        if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
            errors.append(f"requested backend {name} has no positive latency")
    if captured is not True:
        errors.append("backend comparison did not use CUDA graph capture")
    if bench is None or not bool(getattr(bench, "captured", False)):
        errors.append("deployed Emmy timing was not captured")
    if not valid_proof(correctness):
        errors.append("deployed Emmy row lacks direct strict eager correctness")

    ab_rows = [row for row in pinned_rows or [] if getattr(row.sample, "shape", None) is None]
    expected_ab_rows = len(args.ab or [])
    if len(ab_rows) != expected_ab_rows:
        errors.append(f"expected {expected_ab_rows} exact --ab row(s), got {len(ab_rows)}")
    exact_rows = ab_rows if expected_ab_rows else list(pinned_rows or [])
    for row in exact_rows:
        label = row.sample.name
        if row.status != "ok" or row.flags:
            errors.append(f"{label} failed exact-pin integrity: status={row.status}, flags={list(row.flags)}")
            continue
        kernels = list(_launch_order_cuda_nodes(row.graph)) if row.graph is not None else []
        if not kernels:
            errors.append(f"{label} has no generated kernel inventory")
        if row.bench is None or not bool(getattr(row.bench, "captured", False)):
            errors.append(f"{label} timing was not captured")
        else:
            num_launches = getattr(row.bench, "num_launches", 0) or len(row.bench.per_launch or [])
            if num_launches <= 0:
                errors.append(f"{label} has no measured launches")
            e2e_ms = getattr(row.bench, "e2e_min_ms", None)
            if num_launches > 1 and e2e_ms is None:
                errors.append(f"{label} multi-launch timing is not whole-program end-to-end")
            total_ms = e2e_ms if e2e_ms is not None else getattr(row.bench, "min_ms", None)
            if total_ms is None:
                total_ms = getattr(row.bench, "time_ms", None)
            if isinstance(total_ms, bool) or not isinstance(total_ms, int | float) or total_ms <= 0:
                errors.append(f"{label} has no positive whole-program timing")
        if not valid_proof(row.correctness):
            errors.append(f"{label} lacks direct strict eager correctness")

    return errors


def _handle_run_ir(args, CudaBackend, CompilerDump):
    """Run path: load JSON IR (any stage), finish lowering, execute, bench."""
    import json

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline import Pipeline

    strict_correctness = bool(getattr(args, "strict_correctness", False))
    embedded = getattr(args, "_golden_graph", None)
    path = Path(args.ir) if embedded is None else None
    if embedded is None:
        with open(path) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
    else:
        graph = embedded.copy()
    if getattr(args, "dynamic", None):
        logger.error("--dynamic is incompatible with --ir (the trace is already complete)")
        sys.exit(2)

    stage, tail = _replay_stage_and_passes(graph, embedded_golden=embedded is not None)
    logger.info("Loaded %s IR; running tail passes: %s", stage, tail or "(none)")

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Snapshot the pre-lowering frontend graph so we can build a torch
    # reference (eager + torch.compile) and compare accuracy/latency vs torch —
    # the same table the --code path produces for a debug Graph IR input.
    # Non-frontend IR (loop/tile/…) has no torch twin → emmy-only bench.
    frontend = graph.copy() if torch_ref.is_runnable(graph) else None

    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    db = None
    if backend.tune_db is not None and backend.tune_db.exists():
        from emmy.compiler.pipeline.search.db import SearchDB

        db = SearchDB(path=backend.tune_db)
        logger.info("Using tuning DB: %s", backend.tune_db)
    if tail:
        # Finish the tail lowering. NOTE: the single-shot ``Pipeline.run`` has no
        # prior (uniform PUCT → emission-order, option-0) and does not replay tuned
        # variants from the DB; ``db=`` is kept for perf recording only. Wiring a
        # warm-started prior into single-shot compile is a deferred follow-up.
        graph = Pipeline.build(tail).run(graph, db=db, dump=dump)

    if not args.bench:
        # No bench: one in-process run + non-fatal accuracy vs the torch reference
        # (random boundary inputs — informational, logged by ``bench_lowered_vs_torch``).
        try:
            asyncio.run(
                bench_lowered_vs_torch(
                    frontend,
                    graph,
                    backend,
                    seed=args.seed,
                    do_bench=False,
                    warmup=args.warmup,
                    iters=args.iters,
                    bench_backends=args.bench_backends,
                )
            )
        except RuntimeError as exc:
            # Per-launch watchdog fired — the CUDA context is dirty; bypass cupy's atexit.
            sys.stderr.write(f"run failed: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return

    # --bench: same uniform worker path as ``handle_run`` — the greedy comparison
    # (``bench_lowered_vs_torch`` rebuilt in-child from the frontend snapshot) and every
    # ``--ab`` row are jobs on one persistent SIGKILL-able worker; a hung kernel dies with
    # the child and the row records bench_fail.
    if args.ab and not tail:
        logger.warning("--ab ignored: %s IR is fully lowered (no forks left to pin)", stage)

    pinned = _pinned_samples_for_ir(args, embedded)

    async def _bench_session():
        greedy_fail = results = bench = accuracy_error = ab_ref = reference_error = ab_benches = greedy_iso = None
        greedy_reference_us = None
        torch_available = captured = False
        pinned = _pinned_samples_for_ir(args, embedded)
        try:
            try:
                resp = await backend.benchmark_compare_async(
                    graph,
                    torch_spec=("frontend_graph", frontend),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_compare_wall_s(graph, backend, base_s=_IR_COMPARE_BASE_S),
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    want_ref=bool(pinned),
                )
            except RuntimeError as exc:
                greedy_fail = f"greedy run/bench failed: {exc}"
            else:
                results, bench, captured = resp["results"], resp["result"], resp["captured"]
                torch_available, accuracy_error = resp["torch_available"], resp["accuracy_error"]
                ab_ref = resp["run_io"]
                greedy_reference_us = resp.get("reference_run_us")
                if resp.get("greedy_error"):
                    greedy_fail = f"greedy timing failed after reference execution: {resp['greedy_error']}"
            if pinned and tail:
                if ab_ref is None:
                    reason = greedy_fail or "the greedy worker returned no run outputs"
                    missing = "pinned embedded-Loop verification requires same-input greedy outputs, but none were returned"
                    reference_error = f"{missing}: {reason}"
                else:
                    if greedy_fail:
                        logger.error("%s — untimed greedy is ineligible; pinned rows still bench", greedy_fail)
                    else:
                        greedy_iso = await _bench_greedy_isolated(backend, graph, warmup=args.warmup, iters=args.iters)
                    ab_benches = await _bench_golden_variants(
                        backend,
                        embedded,
                        pinned,
                        warmup=args.warmup,
                        iters=args.iters,
                        ref=ab_ref,
                    )
            elif args.ab and tail:
                if greedy_fail:
                    logger.error("%s — greedy row marked bench_fail; --ab rows still bench in the worker", greedy_fail)
                greedy_iso = await _bench_greedy_isolated(backend, graph, warmup=args.warmup, iters=args.iters)
                ab_benches = await _bench_ab_variants_ir(backend, path, tail, args.ab, warmup=args.warmup, iters=args.iters, db=db)
        finally:
            await backend.aclose_async_worker()
        return (
            greedy_fail,
            results,
            bench,
            torch_available,
            captured,
            accuracy_error,
            reference_error,
            ab_benches,
            greedy_iso,
            greedy_reference_us,
        )

    (
        greedy_fail,
        results,
        bench,
        torch_available,
        captured,
        accuracy_error,
        reference_error,
        ab_benches,
        greedy_iso,
        greedy_reference_us,
    ) = asyncio.run(_bench_session())

    if reference_error is not None:
        logger.error(reference_error)
        sys.exit(1)

    if accuracy_error is not None:
        # Random boundary-input reproducers keep their historical informational check, while
        # strict verification aborts after preserving JSON.
        if strict_correctness:
            logger.error(accuracy_error)
        else:
            logger.warning("%s — non-fatal (random-input reproducer); benched anyway", accuracy_error)
    if dump and bench is not None:
        dump.dump_benchmark(bench)
    if torch_available:
        note = None if captured else "(graph-capture fallback: timings include host launch overhead)"
        _print_table(results, note=note)
    elif bench is not None:
        # Non-frontend IR (loop/tile/…): no torch twin → emmy-only.
        from emmy.commands.table import Col, render_table  # noqa: PLC0415

        print()
        for line in render_table([Col("Backend"), Col("Latency (us)", "r")], [["Emmy", f"{bench.time_ms * 1000:.0f}"]], rule=True):
            print(line)
    _print_kernel_stats(graph, bench, golden_benches=ab_benches, greedy_fail=greedy_fail, greedy_iso=greedy_iso)
    strict_errors = _strict_benchmark_errors(args, results, bench, captured, correctness, ab_benches) if strict_correctness else None
    if getattr(args, "json", None):
        _write_ab_json(
            args,
            results or {},
            graph,
            bench,
            ab_benches,
            greedy_fail=greedy_fail,
            greedy_iso=greedy_iso,
            greedy_reference_us=greedy_reference_us,
        )
    if args.profile and greedy_fail is None:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)
    if (
        (strict_correctness and accuracy_error is not None)
        or bool(strict_errors)
        or greedy_fail is not None
        or (greedy_iso is not None and greedy_iso.status != "ok")
        or any(gb.status != "ok" for gb in ab_benches or [])
    ):
        sys.exit(1)  # every row is reported above; any failed row (greedy or --ab) exits non-zero


async def _bench_ab_variants_ir(backend, ir_path, tail, specs, *, warmup, iters, db=None):
    """The ``--ab`` counterpart of :func:`_bench_golden_variants` for the ``--ir``
    path: each config reloads the IR file fresh (the tail lowering mutates the graph
    in place) and re-lowers it with the knobs pinned, so the pin collapses every
    remaining fork. Same row semantics as the golden path: a config that fails to
    compile / bench is kept as a ``bench_fail`` row, a pin the re-lowering didn't
    realize fails its row loudly BEFORE benching (``pin_unmatched`` — benching the
    fallback would measure the planner's own pick under the pin's name), and every
    bench is one job on the backend's persistent SIGKILL-able worker, so a hung row
    dies with the child and the remaining rows continue on a fresh one. Serialized
    ops drop ``knobs``, so only tail-lowered kernels carry realized values — a pinned
    family the tail never re-decides has no stamp to check and is skipped
    (ungateable), while a family the tail did re-decide still verifies."""
    import json as _json  # noqa: PLC0415

    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    out = []
    for sample in _ab_samples(specs):
        try:
            with pinned_knobs(sample.knobs):
                g = Graph.from_dict(_json.loads(Path(ir_path).read_text()))
                if tail:
                    g = Pipeline.build(tail).run(g, db=db)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own table
            logger.warning("[ab] %s: compile of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, None, None, [f"compile failed: {exc}"], "bench_fail"))
            continue
        flag = unreproducible_pin_flag(sample.knobs, _cuda_knob_dicts(g))
        if flag:
            logger.error("[ab] %s: %s — the pinned config did not realize; fix the pin spelling (row kept unbenched)", sample.name, flag)
            out.append(_GoldenBench(sample, g, None, [f"{flag} — row NOT benched"], "pin_unmatched"))
            continue
        try:
            g_bench, _ = await backend.bench_pinned_async(g, warmup=warmup, num_iters=iters)
        except Exception as exc:  # noqa: BLE001 — a bad pin must not abort the run's own table
            logger.warning("[ab] %s: bench of the pinned config failed (%s) — row kept as bench_fail", sample.name, exc)
            out.append(_GoldenBench(sample, g, None, [f"bench_fail: {exc}"], "bench_fail"))
            continue
        out.append(_GoldenBench(sample, g, g_bench, []))
    return out


def _bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=None):
    """Match graph inputs and constants to tensors from ``module`` / call args.

    Activations come from the call's positional/keyword tensors. Constants
    come from ``module.named_parameters()`` / ``named_buffers()`` keyed
    by each ``ConstantOp.source_path`` recorded at trace time. Each
    constant's ``load_ops`` chain is replayed via the NumPy backend
    (see ``compiler.loader.binder``), so any compile-time-folded
    transpose / reshape is honored uniformly.

    ``checkpoint`` (the model id / path, when the caller has one) covers the
    constants a live module cannot supply: an fp8 checkpoint's weights bind as
    raw bits + scale sources (or a folded ``source_graph`` record over them)
    that exist only in the safetensors shards, so any still-unbound constant is
    re-tried through the safetensors loader before the hard error below.
    """
    import numpy as np
    import torch

    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.binder import bind_constants

    flat_inputs: list[torch.Tensor] = []
    for v in example_args:
        flat_inputs.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat_inputs.extend(_flatten_tensors(v))

    input_ids = list(compiled.inputs)
    if len(input_ids) != len(flat_inputs):
        # Raise, don't sys.exit: this also runs inside the bench worker child, where an
        # exit(1) surfaces to the parent as an opaque ``SystemExit(1)`` with the real
        # cause stranded in the child's log stream.
        raise RuntimeError(f"Input arity mismatch: graph has {len(input_ids)} inputs, code provided {len(flat_inputs)}")

    input_data: dict[str, np.ndarray] = {}
    torch_f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    for nid, tensor in zip(input_ids, flat_inputs, strict=True):
        np_dtype = compiled.nodes[nid].output.dtype.np
        tensor = tensor.detach().cpu()
        # An fp8 activation/weight INPUT binds the raw bit pattern on its uint8 carrier —
        # numpy has no fp8 scalar type, and a value cast would decode the bits (the same
        # rule the constant side and the serving expert feed already follow).
        if tensor.dtype in torch_f8:
            tensor = tensor.view(torch.uint8)
        input_data[nid] = tensor.numpy().astype(np_dtype, copy=False)

    # ``remove_duplicate=False`` so tied weights (e.g. a model whose lm_head
    # shares the embedding matrix) are surfaced under *every* name, including
    # the ``source_path`` the tracer recorded — otherwise the alias the trace
    # picked may be the one torch dedups away and the constant won't bind.
    sources: dict[str, np.ndarray] = {}
    for path, tensor in module.named_parameters(remove_duplicate=False):
        sources[path] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, tensor in module.named_buffers(remove_duplicate=False):
        sources[path] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    input_data.update(bind_constants(compiled, sources))

    if checkpoint is not None and any(nid not in input_data for nid, _op in compiled.loadable_constants()):
        from emmy.compiler.trace.huggingface import quantized_checkpoint_dir  # noqa: PLC0415

        quant_dir = quantized_checkpoint_dir(checkpoint)
        if quant_dir is not None:
            from emmy.compiler.loader.safetensors import load_constants_from_safetensors  # noqa: PLC0415

            for nid, arr in load_constants_from_safetensors(compiled, str(quant_dir)).items():
                input_data.setdefault(nid, arr)

    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp) or nid in input_data:
            continue
        if node.op.value is not None:
            continue  # backend materializes scalars from node.op.value
        raise RuntimeError(f"Could not bind constant {nid} (source_path={node.op.source_path!r})")
    return input_data


def _flatten_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_tensors(v))
        return out
    return []


def _collect_sym_env(graphs) -> dict[str, int]:
    """Map every symbolic dim var appearing in ``graphs`` to its hint
    (``DEFAULT_SEQ_HINT`` for a bare seq axis) — the size the backend resolves
    a symbolic graph to when benching without supplied inputs."""
    from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim

    sym_env: dict[str, int] = {}
    for gph in graphs:
        for node in gph.nodes.values():
            for d in node.output.shape:
                if isinstance(d, Dim) and not d.is_static:
                    hint = d.hint or DEFAULT_SEQ_HINT
                    for v in d.expr.free_vars():
                        sym_env.setdefault(v, hint)
    return sym_env


def _tile_to(tensor, axis: int, size: int):
    """Resize ``tensor`` along ``axis`` to ``size`` by repeating its values
    (ceil) and slicing — keeps every element drawn from the original (token
    ids stay in-vocab, masks keep their value range), unlike fresh randoms."""
    old = tensor.shape[axis]
    if old == size:
        return tensor
    reps = [1] * tensor.dim()
    reps[axis] = -(-size // old)
    return tensor.repeat(*reps).narrow(axis, 0, size)


def _map_tensors(value, fn):
    """Rebuild ``value``'s nested list/tuple structure with ``fn`` applied to
    each tensor leaf, in :func:`_flatten_tensors` order."""
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors(v, fn) for v in value)
    return value


def _hint_sized_inputs(lowered, example_args, example_kwargs):
    """Tile a symbolic graph's example inputs out to its ``Dim`` hints.

    The emmy side of the full-model bench runs a symbolic graph at each
    dim's hint (``backend.benchmark`` builds hint-sized synthetic inputs when
    none are supplied), while the torch closures close over the trace-time
    example tensors — different shapes, so the table would compare e.g. a
    seq-512 emmy program against seq-32 eager. Grow every symbolic input
    axis to its hint (positional ``graph.inputs`` ↔ flattened example tensors
    pairing, same as :func:`_bind_inputs`) so both sides run the hint shape.
    Values are tiled repeats of the trace inputs — valid for latency, which is
    all this table compares (accuracy runs on the original trace inputs).

    Returns ``(args, kwargs, sym_env)`` — the inputs unchanged and ``{}`` for
    a static graph.
    """
    sym_env = _collect_sym_env([lowered])
    if not sym_env:
        return example_args, example_kwargs, {}
    from emmy.compiler.dim import Dim

    flat: list = []
    for v in example_args:
        flat.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat.extend(_flatten_tensors(v))
    input_ids = list(lowered.inputs)
    if len(input_ids) != len(flat):
        logger.warning("Input arity mismatch (graph %d vs example %d) — benching torch on trace-sized inputs", len(input_ids), len(flat))
        return example_args, example_kwargs, sym_env
    resized = []
    for nid, tensor in zip(input_ids, flat, strict=True):
        for axis, d in enumerate(lowered.buffer(nid).shape):
            if isinstance(d, Dim) and not d.is_static:
                tensor = _tile_to(tensor, axis, int(d.expr.eval(sym_env)))
        resized.append(tensor)
    it = iter(resized)
    args = tuple(_map_tensors(v, lambda _: next(it)) for v in example_args)
    kwargs = {k: _map_tensors(v, lambda _: next(it)) for k, v in example_kwargs.items()}
    return args, kwargs, sym_env


def _symbolic_bench_note(sym_env: dict[str, int]) -> str | None:
    """The full-model table's shape label for a symbolic graph (``None`` for
    static): both sides bench at the hint, and the reader should know the
    number is hint-shaped, not trace-shaped."""
    if not sym_env:
        return None
    dims = ", ".join(f"{name}={size}" for name, size in sorted(sym_env.items()))
    return f"benched at {dims} (symbolic hint; torch inputs tiled to match)"


def _eager_output(module, args, kwargs):
    """Eager reference forward. Returns the module's output — a Tensor, or a
    TUPLE of Tensors for a multi-output module (``_check_accuracy`` compares
    each graph output buffer against the tuple positionally)."""
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args)
    cuda_kwargs = _to_cuda_kwargs(kwargs)
    with torch.no_grad():
        out = cuda_module(*cuda_args, **cuda_kwargs)
    if isinstance(out, tuple):
        tensors = tuple(t for t in out if isinstance(t, torch.Tensor))
        return tensors if len(tensors) > 1 else tensors[0] if tensors else out[0]
    return out


def _to_cuda_tensor(arr, dtype):
    """numpy array → CUDA torch tensor in the node's dtype (default fp32)."""
    import torch

    from emmy.compiler.backend.torch_ref import torch_dtype

    return torch.from_numpy(arr).to("cuda").to(torch_dtype(dtype) or torch.float32)


def _to_cuda_kwargs(kwargs):
    import torch

    cuda_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            cuda_kwargs[k] = v.to("cuda")
        elif isinstance(v, tuple):
            cuda_kwargs[k] = tuple(t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v)
        else:
            cuda_kwargs[k] = v
    return cuda_kwargs


def _check_accuracy(outputs, eager_out) -> str | None:
    """Compare backend ``outputs`` against the eager reference — a **pure verdict**:
    ``None`` when every output passes, else a message naming what failed. No printing
    and no exit here: the check runs inside the bench worker child, whose stdout is the
    pickle protocol channel, so the verdict must travel as a return value and fatality
    is the caller's call — the ``--code`` / model path aborts on it, while ``run --ir``
    treats it as informational (a sliced reproducer is fed random *boundary* inputs
    that can be out-of-domain for the op, e.g. a kernel expecting a mean-of-squares
    gets random signed data → NaN from a downstream rsqrt)."""
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    # Multi-output: the eager reference may be a tuple — compare each backend
    # output against its positional eager counterpart (graph-output order).
    # Historic single-output behavior (every output vs THE eager tensor) is the
    # degenerate case of the fallback-to-first pairing below.
    eager_refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    eager_flats = [t.detach().cpu().flatten().tolist() for t in eager_refs]
    if any(e != e for flat in eager_flats for e in flat):
        return "eager reference contains NaN (reproducer inputs out of domain)"
    failures: list[str] = []
    for pos, (buf_name, arr) in enumerate(outputs.items()):
        eager_flat = eager_flats[pos] if len(eager_flats) == len(outputs) else eager_flats[0]
        values = arr.flatten().tolist()
        if any(v != v for v in values):
            return f"CORRECTNESS FAIL: output {buf_name} contains NaN"
        if len(values) == len(eager_flat):
            diffs = [abs(a - e) for a, e in zip(values, eager_flat, strict=True)]
            max_diff = max(diffs)
            mean_diff = sum(diffs) / len(diffs)
            # Scale tolerance by max|eager| and by output dtype.
            #
            # fp32: matmul reduction-order drift grows with both K and
            # output magnitude. A fixed threshold flags benign drift on
            # randn×randn at large K as a failure (cp.async + split-K
            # atomic-add ordering vs eager's pairwise sum). Dual check:
            # ``max_diff <= 8% of peak`` (tight ceiling for the typical
            # case) OR ``mean_diff <= 0.5% of peak`` (escape hatch for
            # the long tail — splitK atomic-reduce on randn×randn
            # produces a handful of outliers at K=1024 / 2048 even when
            # the bulk of the output is accurate to 4+ decimals).
            # Codegen bugs that systematically corrupt the output (e.g.
            # the matmul_add fusion adding the residual per-K_s CTA)
            # fail both clauses: mean_diff lifts to ~2-3% of peak.
            #
            # fp16: every step has ~3 fewer decimal digits than fp32. The
            # split-K matmul path is dominated by atomicAdd into an
            # ``__half*`` buffer — each per-CTA partial converts to fp16
            # at the atomic boundary and loses ~11 bits per write. After
            # 1024 K-partials that's RMS error on the order of
            # ``peak * 0.3``. The proper fix (f32 scratch for split-K +
            # separate cast pass) is a future architectural change; for
            # now ``emmy run --bench`` needs to remain usable on
            # legitimate fp16 graphs, so the rtol budget tracks the
            # achievable accuracy of the current path. Bugs that
            # actually corrupt outputs still fail (whole-row mismatch /
            # NaN / order-of-magnitude wrong).
            is_fp16 = arr.dtype == np.float16
            # Three-part verdict — each clause with ONE job:
            #
            #   PASS iff (max_ok AND mean_diff ≤ mean_tol) OR escape_ok
            #   max_ok    = max_diff ≤ tol                                  (fp32)
            #             = max_diff ≤ 4·tol AND count(diff > tol) ≤ budget (fp16; budget
            #               EARNED by a heavy-tailed output — 0 on a gaussian one)
            #   escape_ok = mean_diff ≤ escape_tol [fp16: AND max_diff ≤ 4·tol AND count ≤ budget]
            #
            # - ``tol`` (the MAX clause) is the loose per-cell OUTLIER ceiling: fp16
            #   atomic-reduce accumulation can drift a cell up to ``peak`` in pathological
            #   cancellation cases (random-signed partials), fp32 stays at 8% of peak.
            # - ``mean_tol`` is the SYSTEMATIC-CORRUPTION gate and must stay tight: a
            #   permuted/scrambled output lands at mean_diff ≈ E|a−e| of independent
            #   same-distribution draws (≈ 15% of peak on a gaussian matmul output), while a
            #   correct kernel — including the half-atomic fp16 split-K path — measures
            #   ≤ ~0.1% of peak. The historical fp16 form (``max ≤ peak OR mean ≤ peak``)
            #   PASSED a matmul whose A slab was scrambled by a fill/drain swizzle mismatch
            #   (mean_diff 52 under tol 340, values of scale ~60 — the PR #354 sync-fill
            #   incident); 3% of peak keeps ~300× headroom over measured correct runs and
            #   fails any permutation-class bug REGARDLESS of its max (the old OR let any
            #   output with max_diff ≤ peak pass without ever consulting the mean).
            #   On HEAVY-TAILED outputs 3% of peak over-scales (a gemma-4 layer output:
            #   peak ≈ 24·RMS — a permutation scores mean_diff ≈ 1.04× that gate, a coin
            #   flip), so the gate is additionally floored by the output's own
            #   PERMUTATION SCORE: ``floor = mean|e_i − e_perm(i)|`` over a fixed shuffle
            #   of the eager values — what a scramble of THIS draw would measure. The
            #   ``min(3%·peak, 0.7·floor)`` keeps gaussian outputs on the 3%-of-peak gate
            #   (floor ≈ 1.13·σ ≫ it) and gives heavy-tailed ones ≥ 1.4× margin on BOTH
            #   sides (measured gemma layer-0: correct 0.52×, scramble 1.43× the gate).
            # - ``escape_tol`` (0.5% of peak) preserves the split-K outlier escape hatch: a
            #   handful of atomic-reorder outliers may exceed the max ceiling on randn×randn
            #   at K=1024/2048 while the bulk stays accurate to 4+ decimals — a near-zero
            #   mean vouches for them.
            #
            # fp16 MAX clause = outlier BUDGET + hard garbage ceiling, not a single-cell
            # max. HEAVY-TAILED fp16 outputs (a gemma-4 layer output: peak ≈ 24·RMS, the
            # layer_scalar / outlier-channel dynamic range) put single-cell diffs at
            # O(peak) IN-DISTRIBUTION on the current legitimate fp16 path (residual-
            # cancellation cells + the half-atomic split-K partials): measured correct
            # layer-0 runs show max_diff 0.7–0.9× peak with a tail draw occasionally
            # crossing 1.0× — a binary ``max ≤ peak`` intermittently aborts the bench on
            # a rerun-passes coin flip. The budget separates "a handful of tail cells"
            # (≤ ~0.0015% of cells over ``tol``, each under the 4× garbage ceiling) from
            # the real bug classes: a garbage/saturated cell (uninitialized read, Inf
            # math) blows the HARD ceiling regardless of count; a corrupt tile / row /
            # tail-guard window (the historical incidents: thousands of cells) blows the
            # BUDGET on either pass path (the escape hatch consults it too — a near-zero
            # mean must not vouch for a low-amplitude mass corruption); a permutation
            # blows the mean gate. The budget itself is EARNED by the output's tail shape
            # (``peak > 8·RMS`` — gemma layers sit at ~24×, a gaussian at ~5×): only there
            # do in-distribution cells legitimately land past ``tol``. A gaussian output
            # keeps the strict pre-budget max clause, and its handful of split-K reorder
            # outliers pass only through the escape hatch (near-zero mean AND ≤ budget).
            # Residual blind spot, accepted: ≤ budget cells at ≤ 4·tol with a near-zero
            # mean is statistically indistinguishable from the legit reorder class.
            # NOTE the mean gate's scramble
            # margin THINS on heavy-tailed outputs (a permuted gemma layer output scores
            # ≈ 1.04× mean_tol — 4% clear, vs ~5× on gaussian matmuls): that margin is
            # bounded by the current fp16 path's honest noise (mean_diff ≈ 0.4× mean_tol
            # on correct gemma runs), and widens only when split-K gets its f32 scratch.
            rel_tol = 1.0 if is_fp16 else 0.08
            abs_tol = 1e-1 if is_fp16 else 1e-3
            peak = max((abs(e) for e in eager_flat), default=0.0)
            tol = max(abs_tol, rel_tol * peak)
            perm = list(eager_flat)
            random.Random(0).shuffle(perm)
            perm_floor = sum(abs(a - e) for a, e in zip(perm, eager_flat, strict=True)) / max(1, len(perm))
            mean_tol = max(abs_tol, min(0.03 * peak, 0.7 * perm_floor))
            escape_tol = max(abs_tol, 0.005 * peak)
            outliers = sum(1 for d in diffs if d > tol)
            budget = max(4, len(diffs) // 65536)
            rms = (sum(e * e for e in eager_flat) / max(1, len(eager_flat))) ** 0.5
            heavy_tailed = peak > 8.0 * rms
            if is_fp16:
                # The hard 4·tol garbage ceiling bounds BOTH pass paths: the escape hatch
                # vouches for reorder outliers (peak-bounded by construction), never for a
                # garbage/saturated cell a near-zero mean would otherwise hide.
                max_ok = max_diff <= 4 * tol and outliers <= (budget if heavy_tailed else 0)
                escape_ok = mean_diff <= escape_tol and max_diff <= 4 * tol and outliers <= budget
            else:
                max_ok = max_diff <= tol
                escape_ok = mean_diff <= escape_tol
            verdict = "PASS" if (max_ok and mean_diff <= mean_tol) or escape_ok else "FAIL"
            if verdict == "FAIL":
                failures.append(
                    f"output {buf_name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol:.6f}"
                    + (f" outliers={outliers}/budget {budget}" if is_fp16 else "")
                )
            else:
                logger.info(
                    "Accuracy vs eager: max_diff=%.6f mean_diff=%.6f tol=%.6f outliers=%d/%d PASS",
                    max_diff,
                    mean_diff,
                    tol,
                    outliers,
                    budget,
                )
        else:
            logger.warning("Output size %d does not match eager %d; skipping accuracy", len(values), len(eager_flat))
    return f"accuracy check failed vs eager: {'; '.join(failures)}" if failures else None


_BACKEND_ALIASES = {
    "eager": "eager",
    "emmy": "emmy",
    "tcompile": "tcompile",
    "torch.compile": "tcompile",
    "compile": "tcompile",
}


def _resolve_backends(cli_value: str | None) -> set[str]:
    """Pick which bench backends to time. Precedence:

    1. ``--bench-backends`` CLI arg (comma-separated).
    2. ``EMMY_BENCH_BACKENDS`` env var (same syntax).
    3. Default ``eager,emmy`` — torch.compile is excluded so the
       per-case wall time isn't dominated by a ~0.8 s Inductor JIT
       that most users don't need on every run.

    ``emmy`` is always included even if omitted (the kernel under
    test is the point of the bench). Returns the canonical backend
    keys ``{"eager", "tcompile", "emmy"}``.
    """
    raw = config.bench_backends_raw(cli_value)
    selected: set[str] = {"emmy"}
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        canonical = _BACKEND_ALIASES.get(tok)
        if canonical is None:
            logger.error("unknown bench backend %r — choose from %s", tok, sorted(set(_BACKEND_ALIASES.values())))
            sys.exit(1)
        selected.add(canonical)
    return selected


def _build_torch_fns(module, args, kwargs, warmup, *, backends: set[str]):
    """Pre-build the per-backend ``torch_fns`` dict, including the
    ``torch.compile`` JIT step when requested. The JIT (mostly
    Inductor CPU work plus a few warmup launches) sits *outside* the
    GPU lock in ``handle_run`` so parallel workers can compile
    concurrently — the lock then only wraps the actual measurement
    iters.

    Returns only the torch-side closures; emmy's bench loop is
    driven separately by ``backend.benchmark`` in ``_bench_interleaved``.
    """
    import torch

    torch_fns: dict[str, callable] = {}
    if "eager" in backends:
        torch_fns["Eager PyTorch"] = lambda: module(*args, **kwargs)
    if "tcompile" in backends:
        try:
            # Persistent/repeated bench workers compile a fresh closure of
            # the SAME ``torch_ref`` ``fn`` code object per row; dynamo's
            # recompile limit is keyed per code object, so after ~8 rows it
            # silently stops compiling and the tcompile column quietly measures
            # eager (observed: 8 µs rows reading 55-80 µs). Reset dynamo so
            # every bench compiles fresh.
            import torch._dynamo  # noqa: PLC0415

            torch._dynamo.reset()
            compiled_torch_module = torch.compile(module, fullgraph=True, mode="max-autotune")
            for _ in range(warmup + 5):
                with torch.no_grad():
                    compiled_torch_module(*args, **kwargs)
            with torch.no_grad():
                eager_output = module(*args, **kwargs)
                compiled_output = compiled_torch_module(*args, **kwargs)
            torch.testing.assert_close(compiled_output, eager_output, rtol=1e-3, atol=1e-3)
            torch_fns["torch.compile"] = lambda: compiled_torch_module(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("torch.compile failed: %s", e)
    return torch_fns


def _capture_torch_fn(fn):
    """Warm ``fn`` on a side stream, capture one call into a CUDA graph, and
    return the (host-cheap) replay closure.

    The replay slots straight into ``_bench_interleaved``'s ``on_iter`` window:
    each ``g.replay()`` is one host call that enqueues every captured kernel, so
    the stream stays dense and the CUDA events measure pure GPU time instead of
    per-op Python dispatch gaps. Requires static inputs/outputs — true for the
    bench closures, which close over fixed CUDA tensors. The returned bound
    method keeps the graph (and its private memory pool, holding any tensors the
    fn allocates per call) alive for the bench's lifetime."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    # Default ``capture_error_mode`` (global): the bench is single-threaded and
    # no cupy call happens between begin/end. If CI ever flakes on concurrent
    # CUDA activity, ``capture_error_mode="thread_local"`` is the one-line knob.
    with torch.no_grad(), torch.cuda.graph(g):
        fn()
    return g.replay


def _capture_torch_fns(torch_fns: dict) -> dict | None:
    """Capture every backend closure into a CUDA graph; ``None`` if ANY fails.

    All-or-nothing by design: a comparison table must not mix captured
    (pure-GPU) and uncaptured (dispatch-inclusive) timings, so a single
    backend resisting capture (e.g. torch.compile guard machinery) falls the
    whole invocation back to uncaptured timing."""
    captured: dict = {}
    for name, fn in torch_fns.items():
        try:
            captured[name] = _capture_torch_fn(fn)
        except Exception as exc:  # noqa: BLE001 — capture is best-effort, fallback covers
            logger.warning("CUDA graph capture failed for %s (%s) — benching all backends uncaptured", name, exc)
            return None
    return captured


async def _bench_interleaved_captured(module, args, kwargs, backend, lowered, warmup, iters, *, torch_fns):
    """All-or-nothing CUDA-graph-captured interleaved bench.

    Captures every torch closure (``_capture_torch_fns``) and runs the
    interleaved loop with the emmy side captured too. If ANY side fails —
    a torch backend resists capture, or the emmy launch loop fell back
    (``bench.captured`` False) — the whole bench re-runs uncaptured with the
    original closures, so one table never mixes timing semantics. Returns
    ``(results, bench, captured)``."""
    captured_fns = _capture_torch_fns(torch_fns)
    if captured_fns is not None:
        results, bench = await _bench_interleaved(
            module, args, kwargs, backend, lowered, warmup, iters, torch_fns=captured_fns, capture_graphs=True
        )
        if bench.captured:
            return results, bench, True
        if not torch_fns:
            return results, bench, False  # emmy-only: nothing to de-mix
        # benchmark_program already logged the capture failure; re-run only to de-mix the table.
        logger.warning("emmy side fell back to uncaptured timing — re-benching all backends uncaptured")
    results, bench = await _bench_interleaved(
        module, args, kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=False
    )
    return results, bench, False


async def _bench_interleaved(module, args, kwargs, backend, compiled_graph, warmup, iters, *, torch_fns, capture_graphs=False):
    """Time the selected backends by alternating one iter of each per
    loop step. All backends see the same warm GPU state across the
    measurement window — same clocks, same caches, same thermal drift
    — instead of running in sequential phases that each get a
    different steady state.

    Driven by ``backend.benchmark_async(on_iter=...)``: emmy is the
    backbone, ``on_iter`` runs each torch closure and records its
    cuda events, and the same call returns per-launch emmy
    timings — so the kernel-stats breakdown shares the same warm
    state as the comparison numbers.

    Per-iter ``torch.cuda.Event``s queue on the (legacy) default
    stream; cupy's default stream is the same NULL stream, so events
    from both libraries see all preceding work.

    ``torch_fns`` is the pre-built backend closure dict from
    :func:`_build_torch_fns` (``handle_run`` builds it outside the
    GPU lock so the slow ``torch.compile`` JIT runs concurrently with
    peer workers; the lock then wraps only this measurement loop).

    ``capture_graphs`` forwards to the emmy side only; the torch
    closures must be pre-captured by the caller to keep one timing
    semantics per table — use :func:`_bench_interleaved_captured`,
    which owns that all-or-nothing pairing.
    """
    import torch

    # Each entry: (start_event, stop_event, batch_size_used). The
    # batch size is propagated by ``benchmark_program``'s ``on_iter``
    # so peer torch backends time the same number of back-to-back
    # calls emmy does per CUDA event window — both sides then
    # measure sustained per-call latency, no warm-vs-cold asymmetry.
    torch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]] = {name: [] for name in torch_fns}

    def on_iter(batch_size: int = 1) -> None:
        for name, fn in torch_fns.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                for _ in range(batch_size):
                    fn()
                stop.record()
            torch_events[name].append((start, stop, batch_size))

    bench = await backend.benchmark_async(compiled_graph, warmup=warmup, num_iters=iters, on_iter=on_iter, capture_graphs=capture_graphs)
    torch.cuda.synchronize()

    results: dict[str, float] = {}
    for name, evt in torch_events.items():
        measured = evt[warmup:]
        if measured:
            # ``elapsed_time`` is in ms across the whole batch; divide
            # by the batch size and multiply by 1000 to get per-call us.
            # ``min`` (best-case iter) for both torch and emmy — the
            # least-noise latency, matching tune's min-over-variants reporting
            # so tune and run numbers are comparable.
            per_iter_us = [s.elapsed_time(e) * 1000.0 / b for s, e, b in measured]
            results[name] = min(per_iter_us)
    # Whole-program (e2e) time when available — windows around replays of one
    # all-launches CUDA graph, the same semantics the captured torch closures
    # above get. The fallback sums per-launch solo windows, which is not an
    # end-to-end number (no cross-kernel cache effects).
    dep_ms = bench.e2e_min_ms if bench.e2e_min_ms is not None else (bench.min_ms if bench.min_ms is not None else bench.time_ms)
    results["Emmy"] = dep_ms * 1000
    return results, bench


def _print_table(results, note: str | None = None):
    from emmy.commands.table import Col, render_table  # noqa: PLC0415

    eager_us = results.get("Eager PyTorch", 0)
    cols = [Col("Backend"), Col("Latency (us)", "r"), Col("vs Eager", "r")]
    rows = [[name, f"{us:.0f}", f"{eager_us / us:.2f}x" if us > 0 else "-"] for name, us in results.items()]
    print()
    for line in render_table(cols, rows, rule=True):
        print(line)
    if note:
        print(note)
