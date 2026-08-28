"""Loading, regeneration and the four realization oracles for the corpus cases.

A case file is a working golden document carrying exactly one config with exactly one
realization, plus the authored ``pins`` / ``knobs`` that name the schedule the compiler is
expected to realize. Everything here is GPU-free except :func:`built` and :func:`correct`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from emmy.compiler.context import Context
from emmy.compiler.pipeline.knob import canon_family_value, family_of
from emmy.compiler.pipeline.search.golden import (
    GoldenRecord,
    dump_golden_file,
    golden_record_from_entry,
    kernel_identity,
    load_golden_file,
)
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pins import pinned_knobs, unreproducible_pin_flag

CASES_DIR = Path(__file__).parent / "cases"

#: The four assertions a case walks, in order. A case's filename may name one of them as the
#: stage it is expected to fail at; the walker stops there.
STAGES = ("offered", "realized", "built", "correct")

_XFAIL = re.compile(r"_xfail_(?P<stage>[a-z_]+)$")

#: Families whose pin is consumed structurally rather than stamped on a kernel, so "the pinned
#: family is stamped" cannot be asked of them. ``PLACE`` is consumed by a splice; a ``REDUCE``
#: cross-CTA split replaces the kernel outright. :func:`unreproducible_pin_flag` already reads
#: both the same way — this is the same rule for the complementary stamping check.
_UNSTAMPABLE = ("PLACE", "REDUCE")


class CaseError(Exception):
    """A case file is not a usable corpus case — a hard error, never a skip."""


@dataclass(frozen=True)
class Case:
    """One corpus case: its file, its document, its single record, and its expectation."""

    path: Path
    document: dict
    record: GoldenRecord
    #: The stage this case is expected to fail at, or ``None`` when every stage must pass.
    xfail_stage: str | None

    @property
    def id(self) -> str:
        """The pytest parameter id — the case's path relative to ``cases/``, which is its identity."""
        return self.path.relative_to(CASES_DIR).as_posix()

    @property
    def compute_cap(self) -> tuple[int, int]:
        return tuple(self.document["compute_cap"])

    @property
    def pinned(self) -> dict:
        """The full pin the oracles publish: input pins plus the authored schedule row."""
        return {**self.record.pin_map, **self.record.knobs}

    def context(self) -> Context:
        """The case's own context — its declared capability, never the live card's. This is what
        makes stages 1 and 2 machine-independent, so an sm_70 lockout is exercised on any box."""
        return Context.from_target(self.compute_cap)


def case_files() -> list[Path]:
    return sorted(CASES_DIR.rglob("*.yaml"))


def expectation(path: Path) -> str | None:
    """The stage named by the filename suffix, or ``None`` for a closed case.

    An ``_xfail``-shaped token that is not one of the four stages is a hard error: the filename is
    semantic, so a typo would otherwise quietly strengthen the assertion into a closed case.
    """
    stem = path.stem
    if "_xfail" not in stem:
        return None
    match = _XFAIL.search(stem)
    if match is None or match["stage"] not in STAGES:
        raise CaseError(f"{path.name}: expected a _xfail_<stage> suffix naming one of {', '.join(STAGES)}")
    return match["stage"]


def load_case(path: Path) -> Case:
    """Load one case, enforcing the one-config / one-realization invariant the harness relies on."""
    try:
        document = load_golden_file(path)
    except ValueError as exc:
        raise CaseError(str(exc)) from exc
    configs = document["configs"]
    if len(configs) != 1 or len(configs[0]["realizations"]) != 1:
        raise CaseError(f"{path.name}: a case holds exactly one config with exactly one realization")
    realization = configs[0]["realizations"][0]
    if "knobs" not in realization:
        raise CaseError(f"{path.name}: a case must carry a knobs mapping, empty only for a forkless kernel")
    stage = expectation(path)
    if stage is not None and not evidence_line(path):
        raise CaseError(f"{path.name}: an open case must carry a leading '# evidence:' comment naming why it should realize")
    return Case(path=path, document=document, record=golden_record_from_entry(document, configs[0], realization), xfail_stage=stage)


def evidence_line(path: Path) -> str | None:
    """The case's ``# evidence:`` citation, read out of its leading comment block."""
    for line in leading_comment(path).splitlines():
        body = line.lstrip("#").strip()
        if body.lower().startswith("evidence:"):
            return body
    return None


def leading_comment(path: Path) -> str:
    """The file's leading ``#`` block. ``dump_golden_file`` is a plain YAML dump and drops
    comments, so regeneration captures this and re-prepends it."""
    lines: list[str] = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.startswith("#"):
            break
        lines.append(line)
    return "".join(lines)


# --- the derived half -------------------------------------------------------------------------
#
# Program wire, target, realization name and identity are all *derived* from the stored program by
# the compiler in front of you; the authored pins and knobs are not, and regeneration structurally
# cannot produce them. Recomputing the first group and comparing is what keeps a stored case from
# rotting into a phantom lockout when a kernel identity or a schedule codec changes.


def regenerate(document: dict) -> dict:
    """The case document as the current compiler would derive it, authored fields preserved.

    Runs the inventory writer through the library under an explicit ``Context.from_target`` — not
    through ``emmy trace``, which stamps ``gpu_name`` from the live card and needs torch. The
    result is machine-independent, so this check fires and its fix works on any box.
    """
    import tempfile  # noqa: PLC0415

    from emmy.compiler.pipeline.search.working_golden import write_trace_inventory  # noqa: PLC0415
    from emmy.compiler.torch_wire import graph_from_wire  # noqa: PLC0415

    entry = document["configs"][0]
    realization = entry["realizations"][0]
    ctx = Context.from_target(tuple(document["compute_cap"]))
    graph = graph_from_wire(document["programs"][entry["program"]])
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "regenerated.yaml"
        write_trace_inventory(graph, destination, ctx=ctx, model=document.get("model"), force_loop_targets="loop" in entry["target"])
        fresh = yaml.safe_load(destination.read_text())

    matched = _matching_entry(fresh, entry)
    rebuilt = dict(fresh)
    rebuilt["configs"] = [matched]
    row = dict(matched["realizations"][0])
    row["bindings"] = dict(realization.get("bindings") or {})
    row["pins"] = dict(realization.get("pins") or {})
    row["knobs"] = canonical_knobs(realization["knobs"])
    identity = kernel_identity(golden_record_from_entry(rebuilt, matched, row))
    if identity is not None:
        row["identity"] = identity
    if realization.get("latency") is not None:
        # Measured on a card, never derived from the program: a regeneration on a CPU box must
        # not erase a 4090's recorded timings.
        row["latency"] = dict(realization["latency"])
    matched["realizations"] = [row]
    if document.get("model") is not None:
        rebuilt["model"] = document["model"]
    return rebuilt


def _matching_entry(fresh: dict, entry: dict) -> dict:
    """The regenerated config that selects the same target as the stored one."""
    for candidate in fresh["configs"]:
        if candidate["target"] == entry["target"]:
            return dict(candidate)
    targets = ", ".join(repr(candidate["target"]) for candidate in fresh["configs"])
    raise CaseError(f"the stored target {entry['target']!r} no longer resolves; the program now offers {targets}")


def canonical_knobs(knobs: dict) -> dict:
    """The authored knobs in their codec's normal form, strictly.

    Strictness is the point: ``canon_family_value`` normally swallows a ``ValueError`` and returns
    the raw string, which is exactly how a retired spelling survives to match no candidate and
    report as a compiler lockout.
    """
    canonical = {}
    for name, value in knobs.items():
        try:
            canonical[name] = canon_family_value(name, value, strict=True)
        except ValueError as exc:
            raise CaseError(f"knob {name}={value!r} is not a spelling this compiler's codec accepts: {exc}") from exc
    return canonical


def write_case(path: Path, document: dict) -> None:
    """Persist a regenerated case, restoring the leading comment block the YAML dump drops."""
    comment = leading_comment(path)
    dump_golden_file(document, path, overwrite=True)
    if comment:
        path.write_text(comment + path.read_text())


# --- the four oracles -------------------------------------------------------------------------


def offered(case: Case) -> str | None:
    """Stage 1 — under the case's pin, does the planner still enumerate its schedule?

    Pinned-enumeration membership is the primary oracle, not ``unreproducible_pin_flag`` alone:
    the flag answers ``None`` for a registered family that nothing stamped, so a pin that cannot
    be offered at all would read as satisfied. Membership is asked per row, *through* the flag, so
    the structural families it already reads correctly stay correctly read here.
    """
    pinned = case.pinned
    try:
        with pinned_knobs(pinned):
            rows = enumerate_graph(case.record.target_program.copy(), case.context()).rows
    except Exception as exc:  # noqa: BLE001 — a pin the enumeration refuses outright is not offered
        return f"{type(exc).__name__}: {exc}"
    if any(unreproducible_pin_flag(pinned, [row]) is None for row in rows):
        return None
    if not case.record.knobs:
        # A FORKLESS kernel: its schedule space collapsed to one row, so it opens no fork and the
        # enumeration has nothing to return. There is no schedule to be denied, so nothing here can
        # fail — `realized` still proves it lowers, and the later stages still prove it runs. This
        # mirrors how `_candidate_row_keys` reads a forkless kernel's row off the resolved op.
        return None
    return f"no enumerated row carries the pin ({len(rows)} rows offered at sm_{''.join(map(str, case.compute_cap))})"


def realized(case: Case) -> str | None:
    """Stage 2 — does the graph lower to CUDA under the pin, with the pin actually realized?

    Three questions, because each catches a different way a pin is lost: the lowering itself may
    refuse; a kernel may realize a *different* value; or the pinned family may reach no kernel at
    all, which ``unreproducible_pin_flag`` deliberately treats as ungateable.
    """
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415

    pinned = case.pinned
    try:
        with pinned_knobs(pinned):
            lowered = Pipeline.build(CUDA_PASSES).run(case.record.target_program.copy(), ctx=case.context())
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        return f"{type(exc).__name__}: {exc}"
    rows = [dict(node.op.knobs or {}) for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    if not rows:
        return "lowering produced no CUDA kernel"
    flag = unreproducible_pin_flag(pinned, rows)
    if flag is not None:
        return flag
    unstamped = _unstamped_families(case.record.knobs, rows)
    if unstamped:
        return f"pinned but unstamped: {', '.join(sorted(unstamped))}"
    return None


def _unstamped_families(knobs: dict, rows: list[dict]) -> set[str]:
    """The authored schedule families no kernel carries a key for.

    Only the authored ``knobs`` are asked, never ``pins``: an input pin like ``FAST_MATH`` is an
    umbrella that gates which forks are offered and is never a stamped kernel property itself.
    """
    stamped = {family_of(key) for row in rows for key in row}
    wanted = {family_of(name) for name in knobs} - set(_UNSTAMPABLE)
    return wanted - stamped


def built(case: Case):
    """Stage 3 — nvcc accepts the pinned kernel. Returns the compiled graph, raising on refusal."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    with pinned_knobs(case.pinned):
        return CudaBackend().compile(case.record.target_program.copy())


def correct(case: Case, compiled) -> None:
    """Stage 4 — the pinned kernel computes the reference answer.

    The reference is derived from the target, the way ``emmy run`` already derives it: a frontend
    program (``target: {origins: …}``) has a numpy twin; an exact Loop target has none, so it
    compares against the same-input greedy execution of the same program.
    """
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

    program = case.record.target_program
    feed = seeded_inputs(program)
    with pinned_knobs(case.pinned):
        result, _ = CudaBackend().run(compiled, input_data=dict(feed))
    if case.record.loop_wire is None:
        reference = NumpyBackend()
        want, _ = reference.run(reference.compile(program.copy()), input_data=dict(feed))
    else:
        greedy = CudaBackend()
        want, _ = greedy.run(greedy.compile(program.copy()), input_data=dict(feed))
    narrow = _has_narrow_operand(program)
    for name in program.outputs:
        reference = np.asarray(want.outputs[name])
        np.testing.assert_allclose(
            np.asarray(result.outputs[name]), reference, err_msg=f"{case.id}: output {name}", **_tolerance(narrow, reference)
        )


def seeded_inputs(program) -> dict[str, np.ndarray]:
    """Deterministic inputs for the target's declared shapes, scaled so an fp16 reduction of a
    model-sized K does not saturate.

    A symbolic axis resolves to its own ``Dim`` hint — the size ``emmy run`` already resolves a
    symbolic reproducer to. The corpus therefore exercises a symbolic kernel at its hint; it has no
    spelling for "compile at the hint, run at some other size", because binding a symbol in a case
    file SPECIALIZES the program rather than sizing a run of it.
    """
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415

    rng = np.random.default_rng(0)
    feed: dict[str, np.ndarray] = {}
    for name in program.inputs:
        shape = tuple(dim.as_static() if dim.is_static else (dim.hint or DEFAULT_SEQ_HINT) for dim in program.nodes[name].output.shape)
        feed[name] = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    for node_id, node in program.nodes.items():
        if isinstance(node.op, ConstantOp) and node_id not in feed and node.op.value is not None:
            feed[node_id] = np.array([node.op.value], dtype=np.float32)
    return feed


#: Dtypes narrow enough that rounding the OPERANDS dominates the comparison.
_NARROW_DTYPES = ("f16", "bf16", "e4m3", "e5m2")


def _has_narrow_operand(program) -> bool:
    """Whether any buffer in the target is narrow enough to set the drift bound.

    Read across the whole program, not off the output. An f16-by-f16 contraction that STORES f32
    still carries a full f16 ulp of operand rounding, and a bound picked from the store alone sits
    below it — no amount of correct codegen can pass that.
    """
    for node in program.nodes.values():
        tensor = getattr(node, "output", None)
        if tensor is not None and tensor.dtype.name in _NARROW_DTYPES:
            return True
    return False


def _tolerance(narrow: bool, reference: np.ndarray) -> dict[str, float]:
    """Comparison bounds, scaled by the reference's own peak when an operand is narrow.

    A fixed absolute bound cannot serve both a 128-long f16 reduction and a 4096-long one: the
    drift bound is roughly K times the peak times the operand epsilon, so a constant either fails
    the long case or stops asserting anything about the short one. This is the same peak-relative
    form the e2e coverage matrices this corpus replaces already use.
    """
    if not narrow:
        return {"rtol": 1e-4, "atol": 1e-5}
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    return {"rtol": 0.05, "atol": max(5e-3, 0.05 * peak)}


# --- stage 5: latency ---------------------------------------------------------------------------
#
# Stages 1 to 4 answer "can this schedule be realized and is it correct". This answers "and is it
# still as fast as it was" — the failure a lockout leaves behind when the compiler quietly stops
# selecting a schedule and falls back to a slower tier.
#
# It cannot ride the correctness walker. `make test` compiles at `-Xcicc -O1`, which by the
# glossary's own definition is not a measurement lane, so a latency assertion there would measure
# the wrong regime entirely.

#: Where a stored latency stops being a match. MEASURED, not guessed: ten estimates per case over
#: four repeats on an idle RTX 5090, spanning 1.5 us to 579 us, put the best-of-three estimator's
#: own run-to-run spread at a median of 0.17% and a maximum of 0.74%. Five percent is roughly seven
#: times that worst case — enough headroom for a card that throttles under a sustained sweep, and
#: still tight enough to see a five-percent regression rather than only a cliff.
#:
#: The ~7% gap `run --json` documents between two timing semantics does not apply here: a stored
#: latency and a fresh one are the same measurement of the same pinned row, compared like with
#: like. And a case outside the band only REPORTS, so a false positive costs a line of output
#: rather than a red build — which is itself an argument for the tighter bound.
LATENCY_BAND = 0.05

#: Interference is one-sided — a busy machine makes a kernel slower, never faster than the hardware
#: can go — so the minimum of several runs is the honest estimator, and requiring every run to be
#: slow is what keeps a developer box that is also compiling something from crying wolf.
#:
#: Taken lazily: a run inside the band ends the case, because the minimum can only fall and no
#: later run could lift it back out. Only a case that looks slow pays for the extra runs, which is
#: exactly the case where interference is the question. On an idle card, where the measured spread
#: is under 1%, that makes the lane three times cheaper at identical strength.
LATENCY_REPEATS = 3


def live_hardware_id() -> str:
    return Context.probe().hardware_id()


def recorded_latency(case: Case, hardware_id: str) -> dict | None:
    return (case.record.latency or {}).get(hardware_id)


def bench_command(case: Case, output: Path) -> list[str]:
    """The one way to bench a case: `emmy run`, pinned to the schedule the case authors.

    A corpus case is deliberately not `VERIFIED` — filling `measurements` would auto-pin it in
    `emmy run --golden` and fold test data into the replay tooling's trusted evidence — so the
    authored row is pinned with `--ab`, which is exactly "bench this row beside the greedy pick".
    """
    import sys  # noqa: PLC0415

    row = ",".join(f"{name}={value}" for name, value in case.record.knobs.items())
    return [
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
        "--golden-file",
        str(case.path),
        "--golden",
        case.record.name,
        "--bench",
        "--bench-backends",
        "eager,tcompile,emmy",
        "--ab",
        row,
        "--json",
        str(output),
    ]


def measure(case: Case, *, within: float | None = None) -> tuple[list[float], float]:
    """Best-of-N emmy microseconds for the case's own schedule, and the torch.compile number.

    Stops early once a sample lands at or below ``within``: the minimum can only fall, so a run
    already inside the band settles the case and the remaining runs would change nothing.

    Deployable optimization is forced: the correctness lane's `-O1` changes runtime performance,
    and a timing measured there is not one a deploy would ever see.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    environment = {**os.environ, "EMMY_NVCC_FLAGS": ""}
    samples: list[float] = []
    tcompile = 0.0
    with tempfile.TemporaryDirectory(prefix="emmy_corpus_bench_") as directory:
        for repeat in range(LATENCY_REPEATS):
            output = Path(directory) / f"{repeat}.json"
            result = subprocess.run(bench_command(case, output), capture_output=True, text=True, env=environment, timeout=1800)
            if result.returncode != 0 or not output.exists():
                raise AssertionError(f"{case.id}: bench failed (exit {result.returncode})\n{result.stderr[-2000:]}")
            record = json.loads(output.read_text())
            rows = [row for row in record.get("pinned", []) if row.get("status") == "ok" and row.get("total_us")]
            if not rows:
                raise AssertionError(f"{case.id}: the pinned row measured nothing — {record.get('pinned')}")
            samples.append(float(rows[0]["total_us"]))
            tcompile = float(record["backends"].get("torch.compile", {}).get("latency_us") or tcompile)
            if within is not None and samples[-1] <= within:
                break
    return samples, tcompile


def describe(case: Case) -> dict[str, str]:
    """What a case is, for a report: its operations, its input shapes and its dtype.

    Derived from the stored program rather than carried as metadata. A curated case list has to
    keep these in sync by hand — which is how the list it replaces came to claim a dtype the
    compiler had outgrown.
    """
    program = case.record.program_wire
    by_id = {node["id"]: node for node in program["nodes"]}
    ops = [by_id[origin]["op"] for origin in case.record.origins if origin in by_id]
    inputs = [by_id[name]["outputs"][0] for name in program["inputs"] if name in by_id]
    shapes = " x ".join("(" + ",".join(str(dim) for dim in spec[2]) + ")" for spec in inputs)
    dtypes = {spec[1] for spec in inputs} or {"?"}
    return {
        "op": "+".join(dict.fromkeys(op.rsplit(".", 1)[-1] for op in ops)) or "loop",
        "shape": shapes,
        "dtype": "/".join(sorted(dtypes)),
        "family": case.path.parent.name,
    }
