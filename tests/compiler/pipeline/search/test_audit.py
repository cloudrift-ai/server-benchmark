"""The verified-tier drift audit — one verdict per consultation, over the strict identity tier.

A consultation is a SCHEDULE fork the tier was asked about. MATCH means a record carrying the
fork's ``deploy_identity`` had a spelled row equal to exactly one enumerated leaf; DRIFT means
records carry the identity but no offered leaf equals any of their rows; GAP means no record
carries it. Nothing here classifies a shape — the verdict key is the identity digest itself.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from emmy import config
from emmy.compiler.context import Context
from emmy.compiler.pipeline.knob import stamp_schedule_families
from emmy.compiler.pipeline.search import audit
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import kernel_identity, load_golden_records
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph

_GPU = "NVIDIA GeForce RTX 5090"
_CAP = (12, 0)


@lru_cache(maxsize=1)
def _fixture() -> tuple:
    """The audited graph plus one enumerated schedule row of its kernel — the recording a MATCH
    replays and a DRIFT mutates away from. Cached: enumerating the pool is the expensive half."""
    from emmy.commands.trace import trace_inline_code

    graph = trace_inline_code("torch.matmul(torch.randn(64,128, dtype=torch.float16), torch.randn(128,64, dtype=torch.float16))")["graph"]
    with config.nvcc_flags_override(""):  # the deployable -O3 regime the tier is gated on
        ctx = Context.from_target(_CAP, gpu_name=_GPU)
    rows = enumerate_graph(graph.copy(), ctx)
    row = next(r for r in rows if str(r.get("WORK", "")).startswith("w") and r.get("STAGE") == "d2/smem-async")
    return graph, stamp_schedule_families(row)


def _records(knobs: dict, *, name: str = "probe.m64", us: float = 7.5) -> list:
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.torch_wire import graph_to_wire

    graph, _ = _fixture()
    origins = [nid for nid, node in graph.nodes.items() if not isinstance(node.op, InputOp)]
    return load_golden_records(
        {
            "gpu_name": _GPU,
            "compute_cap": list(_CAP),
            "model": "org/model",
            "programs": [graph_to_wire(graph)],
            "configs": [
                {
                    "program": 0,
                    "target": {"origins": origins},
                    "realizations": [
                        {
                            "name": name,
                            "bindings": {},
                            "pins": {"FAST_MATH": False},
                            "knobs": knobs,
                            "measurements": {"emmy_us": us, "reference_us": 9.0, "reference_backend": "cublas"},
                        }
                    ],
                }
            ],
        }
    )


def _audit(goldens: list | None) -> list[dict]:
    graph, _ = _fixture()
    return audit.audit_card({"probe": graph.copy()}, _GPU, _CAP, goldens=goldens)["probe"]


def test_recorded_row_audits_as_match() -> None:
    """A record whose spelled row equals an enumerated leaf of the audited graph MATCHes, keyed by
    the fork's structural identity and naming the record that decided it."""
    _, knobs = _fixture()
    records = _records(knobs)
    verdicts = _audit(records)

    matched = [r for r in verdicts if r["verdict"] == "MATCH"]
    assert matched, f"expected a MATCH, got {Counter(r['verdict'] for r in verdicts)}"
    assert {r["key"] for r in matched} == {kernel_identity(records[0])}
    assert matched[0]["golden"] == "probe.m64"
    assert matched[0]["us"] == 7.5
    assert matched[0]["n_rows"] > 1  # the fork really enumerated alternatives
    assert matched[0]["unrealized"] == []  # the one record realizes
    assert not [r for r in verdicts if r["verdict"] == "DRIFT"]


def test_row_that_no_longer_enumerates_audits_as_drift() -> None:
    """The regression the gate exists for: the identity still matches, but the recorded row equals
    no leaf the enumeration offers, so the deploy falls through — DRIFT, with the unrealizable
    record named under ``unrealized``."""
    _, knobs = _fixture()
    records = _records({**knobs, "TILE": "mma_m16n8k16_f16_f32/f9x9"})  # a fragment nothing offers
    verdicts = _audit(records)

    drifted = [r for r in verdicts if r["verdict"] == "DRIFT"]
    assert drifted, f"expected a DRIFT, got {Counter(r['verdict'] for r in verdicts)}"
    assert {r["key"] for r in drifted} == {kernel_identity(records[0])}
    assert drifted[0]["golden"] == "probe.m64"
    assert drifted[0]["us"] is None
    assert drifted[0]["unrealized"] == records
    assert not [r for r in verdicts if r["verdict"] == "MATCH"]


def test_uncovered_fork_audits_as_gap() -> None:
    """With no record carrying the fork's identity every consultation is a GAP, and the identity
    shows up in the coverage set the release gate ratchets on."""
    verdicts = _audit([])

    assert verdicts and all(r["verdict"] == "GAP" for r in verdicts)
    assert all(r["golden"] is None and r["us"] is None and r["unrealized"] is None for r in verdicts)
    assert audit.gap_keys({"probe": verdicts}) == {r["key"] for r in verdicts}
    assert audit.consultation_counts({"probe": verdicts}) == {"probe": len(verdicts)}


def test_scoped_records_are_restored_after_the_audit() -> None:
    """``goldens`` scopes the tier's own loader for the audit and nothing beyond it — a release
    gate that leaked its lane scoping would silently rescope every later compile in the process."""
    assert golden_mod.RECORDS_OVERRIDE is None
    _audit([])
    assert golden_mod.RECORDS_OVERRIDE is None


def test_verdict_helpers_summarize_and_collect_gaps() -> None:
    records = {
        "pre": [
            {"verdict": "MATCH", "key": "id-a"},
            {"verdict": "GAP", "key": "id-b"},
            {"verdict": "GAP", "key": "id-c"},
        ],
        "post": [
            {"verdict": "DRIFT", "key": "id-a"},
            {"verdict": audit.COMPILE_FAIL, "key": None},
        ],
    }

    assert audit.summarize(records) == Counter({"GAP": 2, "MATCH": 1, "DRIFT": 1, audit.COMPILE_FAIL: 1})
    assert audit.gap_keys(records) == {"id-b", "id-c"}
    # COMPILE_FAIL is not a consultation; an unconsulted graph still counts as 0.
    assert audit.consultation_counts({**records, "empty": []}) == {"pre": 3, "post": 1, "empty": 0}


def test_audit_card_keeps_graph_failures_and_restores_target(monkeypatch) -> None:
    from emmy.compiler import target

    context = object()
    calls = []
    monkeypatch.setattr(Context, "from_target", staticmethod(lambda cap, *, gpu_name: context))

    def fake_audit_graph(graph, ctx):
        calls.append((graph, ctx))
        if graph == "bad":
            raise RuntimeError("cannot compile")
        return [{"verdict": "MATCH"}]

    monkeypatch.setattr(audit, "audit_graph", fake_audit_graph)
    previous_target = target._OVERRIDE

    result = audit.audit_card({"pre": "good", "post": "bad"}, "Test GPU", (8, 9))

    assert calls == [("good", context), ("bad", context)]
    assert result["pre"] == [{"verdict": "MATCH"}]
    assert result["post"][0]["verdict"] == audit.COMPILE_FAIL
    assert result["post"][0]["error"] == "cannot compile"
    assert target._OVERRIDE == previous_target
