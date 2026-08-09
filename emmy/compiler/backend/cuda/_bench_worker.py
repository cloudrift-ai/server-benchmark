"""Persistent benchmark worker — runs bench / comparison jobs (:func:`_run_job`) in
an isolated subprocess so the parent can SIGKILL on wall-timeout and
the dirty CUDA stream (or a non-terminating kernel) dies with the child.

Protocol (length-prefixed pickle on stdin/stdout):

- Parent writes ``<8-byte little-endian length><pickled request>``.
- One request shape, handled by :func:`_run_job`: ``{"graph": Graph, "nvcc_flags": str|None,
  "torch_spec": None | (...), ...}``. ``torch_spec`` selects the work — ``None`` is a pure
  emmy bench (the autotune sweep and ``run --bench``'s pinned golden / ``--ab`` rows; ``kwargs``
  carries warmup / num_iters / timeouts, an optional ``run_inputs`` ndarray dict adds one
  pre-bench execution on those inputs with the outputs returned — the wrong-answer gate's
  measurement side), otherwise the deployable eager / torch.compile / emmy comparison
  (``tune --bench`` / ``run --bench``'s greedy row), rebuilt and run *here* so a hung kernel
  hangs this child (SIGKILL-recoverable). A ``trace_args`` comparison honors two extra flags:
  ``accuracy`` (bind the rebuilt module's real inputs, run the emmy program on them, compare vs
  eager — the verdict rides back as ``accuracy_error``, and a numeric failure skips the bench)
  and ``want_ref`` (return that run's ``(input_data, outputs)`` as ``run_io`` — the reference
  the parent feeds each pinned row's wrong-answer check).
- Response: ``{"ok": True, "result": BenchmarkResult, "results": dict|None, "torch_available": bool,
  "captured": bool, "run_outputs": dict|None, "accuracy_error": str|None, "run_io": tuple|None}``
  or ``{"ok": False, "error": str, "traceback": str}`` on exception.
- Worker imports cupy / torch lazily on first request, writes ``<8-byte length><pickled response>``.
- A ``worker_warmup`` request initializes CuPy and the CUDA context without consuming a candidate's wall budget.
- EOF on stdin (or parent SIGKILL) terminates the worker.

Errors raised inside ``benchmark_program`` (bench_compile_timeout_s,
bench_run_timeout_s, per-launch ``_KERNEL_TIMEOUT_MS``) propagate back
as ``ok: False`` and the parent surfaces them as ``RuntimeError``.

A failing bench may have left the CUDA context in a sticky-error state: an
illegal / misaligned memory access corrupts the context so that *every*
subsequent CUDA call returns the same error until the context is destroyed.
Reusing the worker would then cascade identical false bench_fails across all
later configs. So after any error we probe the context (:func:`_context_dirty`)
and, if it's poisoned, send the response and exit — the parent's next request
sees a dead process and respawns a clean context. Benign failures (NVRTC
compile errors, OOM that's cleaned up) leave the context healthy and keep the
worker alive, so they don't pay the respawn cost.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
import traceback


def _read_n(fd: int, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk:
            return bytes(out)  # short read → caller treats as EOF
        out.extend(chunk)
    return bytes(out)


class InputsCacheMissError(RuntimeError):
    """A job referenced ``run_inputs_key`` but this worker holds no cached inputs for it —
    the worker respawned since the inputs were shipped. The parent retries once with the
    inputs included (``benchmark_pinned_isolated_async``)."""


# Reference inputs cached across jobs of one run session (the pinned-row wrong-answer
# gate re-executes every row on the SAME greedy inputs — ~hundreds of MB on the big
# --code shapes, shipped once instead of per row). One entry: a new key evicts the old.
_RUN_INPUTS_CACHE: dict[str, dict] = {}


def _hung(exc: BaseException) -> bool:
    """``True`` iff ``exc`` is the per-launch hung-kernel watchdog. A hung kernel is still
    resident on the device, so :func:`_context_dirty`'s ``deviceSynchronize`` probe would block
    on it forever — treat it as dirty without probing, so the worker exits promptly and process
    teardown kills the kernel."""
    from emmy.compiler.backend.cuda.program import HungKernelError

    return isinstance(exc, HungKernelError)


async def _run_job(req: dict) -> dict:
    """Run one bench job; return ``{"result": BenchmarkResult, "results": dict|None,
    "torch_available": bool, "captured": bool}`` (``captured``: timings came from
    CUDA-graph-captured windows; False = the all-or-nothing uncaptured fallback ran).

    ``req["torch_spec"]`` picks the work — a pure emmy bench is just the comparison with a
    no-op torch request:

    - ``None`` → the emmy-only autotune bench (``benchmark_program``), no torch comparison.
    - ``("trace_args", {code/input/adapter/layer/seq_len/dynamic})`` → ``load_or_trace`` rebuilds the real
      module here (HF model id or ``--code`` expr) → ``bench_full_model_real``.
    - ``("frontend_graph", Graph | None)`` → ``bench_lowered_vs_torch`` (per-kernel reproducer).

    Rebuilding the torch side **here** (not pickling a live module) means a hung emmy kernel
    hangs *this* child, which the parent SIGKILLs — recovering the device. ``nvcc_flags`` re-points
    the compile at a given opt level (the cubin cache key folds it in)."""
    if req.get("worker_warmup"):
        import cupy as cp

        cp.cuda.Device().use()
        cp.cuda.runtime.free(0)
        cp.cuda.runtime.deviceSynchronize()
        return {"warmed": True}

    from emmy import config

    with config.nvcc_flags_override(req.get("nvcc_flags")):
        spec = req.get("torch_spec")
        if spec is None:
            from emmy.compiler.backend.cuda.program import benchmark_program

            run_outputs = None
            run_inputs, key = req.get("run_inputs"), req.get("run_inputs_key")
            if key is not None and run_inputs is not None:
                _RUN_INPUTS_CACHE.clear()
                _RUN_INPUTS_CACHE[key] = run_inputs
            elif key is not None:
                run_inputs = _RUN_INPUTS_CACHE.get(key)
                if run_inputs is None:
                    raise InputsCacheMissError(f"run_inputs {key!r} not cached in this worker (respawned?) — resend with the inputs")
            if run_inputs is not None:
                # One pre-bench execution on the caller's inputs — the pinned-row
                # wrong-answer gate's measurement side (outputs cross back as ndarrays;
                # ``CudaBackend.run`` resolves symbolic output shapes from the inputs).
                from emmy.compiler.backend.cuda.backend import CudaBackend

                run_result, _ = CudaBackend(bench_compile_timeout_s=60.0, bench_run_timeout_s=60.0).run(req["graph"], input_data=run_inputs)
                run_outputs = run_result.outputs
            result = benchmark_program(req["graph"], **req["kwargs"])
            return {"result": result, "results": None, "torch_available": False, "captured": result.captured, "run_outputs": run_outputs}

        from emmy.compiler.backend.cuda.backend import CudaBackend

        # In-process within this child — the parent's SIGKILL is the wall-timeout backstop.
        backend = CudaBackend(bench_compile_timeout_s=60.0, bench_run_timeout_s=60.0)
        kind, payload = spec
        accuracy_error = run_io = None
        if kind == "frontend_graph":
            from emmy.commands.run import bench_lowered_vs_torch

            results, bench, avail, captured, accuracy_error = await bench_lowered_vs_torch(
                payload,
                req["graph"],
                backend,
                seed=req["seed"],
                do_bench=True,
                warmup=req["warmup"],
                iters=req["iters"],
                bench_backends=req["bench_backends"],
            )
        elif kind == "trace_args":
            import types

            from emmy.commands.compile import load_or_trace
            from emmy.commands.run import bench_full_model_real

            _, _, bundle = load_or_trace(types.SimpleNamespace(**payload))
            if bundle is None:
                raise RuntimeError("trace_args produced no runnable module (--ir JSON path has none)")
            module, args_t, kwargs = bundle
            if req.get("accuracy"):
                # The run path's correctness gate, in-child: bind the rebuilt module's real
                # inputs, run the emmy program on them, compare vs the eager forward. A
                # numeric failure skips the bench — the parent aborts on the verdict, so
                # benching a miscompiling program would be wasted GPU time. ``want_ref``
                # ships this run's (inputs, outputs) back as the pinned rows' wrong-answer
                # reference (bounded: pinned rows only exist for --code inputs).
                from emmy.commands.run import _bind_inputs, _check_accuracy, _eager_output

                input_data = _bind_inputs(req["graph"], module, args_t, kwargs, checkpoint=payload.get("input"))
                run_result, _ = backend.run(req["graph"], input_data=input_data)
                eager_out = _eager_output(module, args_t, kwargs)
                accuracy_error = _check_accuracy(run_result.outputs, eager_out)
                if accuracy_error is not None:
                    return {
                        "result": None,
                        "results": None,
                        "torch_available": True,
                        "captured": False,
                        "accuracy_error": accuracy_error,
                        "run_io": None,
                    }
                if req.get("want_ref"):
                    run_io = (input_data, run_result.outputs)
            results, bench, captured = await bench_full_model_real(
                module,
                args_t,
                kwargs,
                req["graph"],
                backend,
                warmup=req["warmup"],
                iters=req["iters"],
                bench_backends=req["bench_backends"],
            )
            avail = True
        else:
            raise ValueError(f"unknown torch_spec kind: {kind!r}")
        return {
            "result": bench,
            "results": results,
            "torch_available": avail,
            "captured": captured,
            "accuracy_error": accuracy_error,
            "run_io": run_io,
        }


def _context_dirty() -> bool:
    """``True`` iff the live CUDA context is in a sticky-error state.

    A cheap ``deviceSynchronize`` surfaces a context-wide sticky error (e.g.
    ``CUDA_ERROR_MISALIGNED_ADDRESS`` / ``CUDA_ERROR_ILLEGAL_ADDRESS``) left by
    a prior illegal access — those keep returning the same status on every call
    until the context is torn down. Returns ``False`` when cupy was never
    imported / no context exists (a compile-only failure touches no context),
    or when the sync succeeds (context healthy)."""
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return False  # no CUDA context was ever created in this worker
    try:
        cupy.cuda.runtime.deviceSynchronize()
        return False
    except Exception:  # noqa: BLE001 — any CUDA error here means the context is unusable
        return True


def main() -> None:
    # A bench worker never dumps compiler artifacts — but a child-built ``CudaBackend()``
    # defaults its dump to ``CompilerDump.from_env()``, whose ``__post_init__`` rmtrees the
    # directory. With ``EMMY_DUMP_DIR`` inherited from the parent that would wipe the
    # parent's dump (the ``.torch.json`` reproducers a ``tune --bench`` is about to read).
    from emmy import config

    os.environ.pop(config.DUMP_DIR, None)
    in_fd = sys.stdin.buffer.fileno()
    out_fd = sys.stdout.buffer.fileno()
    while True:
        header = _read_n(in_fd, 8)
        if len(header) < 8:
            return
        n = int.from_bytes(header, "little")
        body = _read_n(in_fd, n)
        if len(body) < n:
            return
        dirty = False
        try:
            resp = {"ok": True, **asyncio.run(_run_job(pickle.loads(body)))}
        except BaseException as exc:  # noqa: BLE001 — surface every failure mode to the parent
            # A bare ``SystemExit`` repr hides the cause (a CLI-style ``sys.exit`` after a
            # logged error) — point the parent at the traceback + stderr, where it lives.
            error = (
                f"child exited via sys.exit({exc.code}) — the cause is in the traceback / child stderr"
                if isinstance(exc, SystemExit)
                else repr(exc)
            )
            resp = {
                "ok": False,
                "error": error,
                "traceback": traceback.format_exc(),
                "cache_miss": isinstance(exc, InputsCacheMissError),
            }
            dirty = _hung(exc) or _context_dirty()
        payload = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        os.write(out_fd, len(payload).to_bytes(8, "little"))
        os.write(out_fd, payload)
        if dirty:
            # Corrupted context — don't serve more requests from it. Exit so the
            # parent respawns a fresh context on its next bench (program.py
            # ``_AsyncBenchWorker.run_job`` re-spawns when the proc is dead).
            return


if __name__ == "__main__":
    main()
