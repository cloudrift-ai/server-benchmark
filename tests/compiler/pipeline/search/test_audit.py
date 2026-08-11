from collections import Counter

from emmy.compiler.pipeline.search import audit
from emmy.compiler.pipeline.search.data.shape import ShapeKey


def test_verdict_helpers_summarize_and_classify_gaps() -> None:
    minor = ShapeKey(free_prod=64, reduce_max=0, is_warp=True)
    major = ShapeKey(free_prod=128, reduce_max=64, is_warp=True)
    records = {
        "pre": [
            {"verdict": "MATCH", "key": None},
            {"verdict": "GAP", "key": minor},
            {"verdict": "GAP", "key": major},
        ],
        "post": [
            {"verdict": "DRIFT", "key": None},
            {"verdict": audit.COMPILE_FAIL, "key": None},
        ],
    }

    assert audit.summarize(records) == Counter({"GAP": 2, "MATCH": 1, "DRIFT": 1, audit.COMPILE_FAIL: 1})
    assert audit.gap_keys(records) == {minor, major}
    assert audit.major_gap_keys(records) == {major}


def test_audit_card_keeps_graph_failures_and_restores_target(monkeypatch) -> None:
    from emmy.compiler import target
    from emmy.compiler.context import Context

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
