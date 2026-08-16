"""Recipe query parsing, row expansion, and ordering."""

import pytest

import emmy.recipe.query as recipe_query
from emmy.recipe.query import (
    build_query_rows,
    parse_predicate,
    parse_sort,
    query_document,
    query_rows,
    referenced_fields,
)


def _record(model_id, tags, deployments, path=None):
    name = model_id.split("/", 1)[1]
    return {
        "name": name,
        "path": path or f"recipes/{name}/recipe.yaml",
        "model_id": model_id,
        "tags": tags,
        "task": "generate",
        "runnable": "onboarding" not in tags,
        "deployments": deployments,
        "rationale": f"Qualify {model_id}.",
    }


def _deployment(gpu, count=1):
    return {"gpu": gpu, "gpu_count": count, "context_length": 8192}


def test_query_expands_deployments_and_preserves_declaration_order():
    inventory = [
        _record(
            "org/model",
            ["onboarding", "untested"],
            [_deployment("GPU B", 2), _deployment("GPU A")],
        )
    ]

    rows = build_query_rows(inventory, expand_deployments=True)

    assert [row["deployment"]["index"] for row in rows] == [0, 1]
    assert [row["deployment"]["gpu"] for row in rows] == ["GPU B", "GPU A"]
    assert all(row["operation"] == "onboarding" for row in rows)
    assert all(row["expected_lifecycle"] == "best-effort" for row in rows)


def test_query_filters_and_sorts_lifecycle_then_oldest_results_then_model():
    inventory = [
        _record("org/maintained-new", ["maintained"], [_deployment("GPU")]),
        _record("org/onboarding-b", ["onboarding", "untested"], [_deployment("GPU")]),
        _record(
            "org/onboarding-a",
            ["onboarding", "untested"],
            [_deployment("Unavailable GPU"), _deployment("Available GPU")],
        ),
        _record("org/best-effort", ["best-effort"], [_deployment("GPU")]),
    ]
    rows = build_query_rows(inventory, expand_deployments=True)
    timestamps = {
        "org/maintained-new": "2026-08-10T00:00:00Z",
        "org/onboarding-b": "2026-08-01T00:00:00Z",
        "org/onboarding-a": None,
        "org/best-effort": None,
    }
    for row in rows:
        row["results"]["last_run_at"] = timestamps[row["model_id"]]
        row["deployment"]["availability"]["cloudrift"] = row["deployment"]["gpu"] != "Unavailable GPU"

    selected = query_rows(
        rows,
        filters=[
            parse_predicate('lifecycle in ["onboarding", "maintained"]'),
            parse_predicate("deployment.availability.cloudrift == true"),
        ],
        requirements=[],
        sorts=[
            parse_sort('lifecycle order ["onboarding", "maintained"]'),
            parse_sort("results.last_run_at asc nulls-first"),
            parse_sort("model_id asc"),
            parse_sort("deployment.index asc"),
        ],
        limit=1,
    )

    assert [row["model_id"] for row in selected] == ["org/onboarding-a"]
    assert selected[0]["deployment"]["index"] == 1


def test_query_manual_missing_model_is_onboarding_candidate():
    rows = build_query_rows(
        [],
        model_id="org/new-model",
        allow_missing_model=True,
        gpu="NVIDIA H200 141GB",
        gpu_count=1,
        expand_deployments=True,
    )

    assert len(rows) == 1
    assert rows[0]["recipe_path"] is None
    assert rows[0]["lifecycle"] is None
    assert rows[0]["operation"] == "onboarding"
    assert rows[0]["expected_lifecycle"] == "best-effort"
    assert rows[0]["deployment"]["gpu"] == "NVIDIA H200 141GB"


def test_query_manual_existing_model_uses_explicit_deployment():
    inventory = [_record("org/model", ["maintained"], [_deployment("Declared GPU", 8)])]

    rows = build_query_rows(
        inventory,
        model_id="org/model",
        gpu="Requested GPU",
        gpu_count=2,
        expand_deployments=True,
    )

    assert rows[0]["operation"] == "verification"
    assert rows[0]["expected_lifecycle"] == "maintained"
    assert rows[0]["deployment"]["gpu"] == "Requested GPU"
    assert rows[0]["deployment"]["gpu_count"] == 2


def test_query_requirement_fails_instead_of_discarding_candidate():
    rows = build_query_rows(
        [_record("org/model", ["obsolete"], [_deployment("GPU")])],
        expand_deployments=True,
    )

    with pytest.raises(ValueError, match="requirement failed.*lifecycle !="):
        query_rows(
            rows,
            filters=[],
            requirements=[parse_predicate('lifecycle != "obsolete"')],
            sorts=[],
            limit=1,
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('unknown == "value"', "Unknown recipe query field"),
        ('lifecycle in "maintained"', "requires a JSON array"),
        ('model_id matches "["', "Invalid regular expression"),
        ("lifecycle == maintained", "valid JSON"),
    ],
)
def test_query_rejects_invalid_predicates(source, message):
    with pytest.raises(ValueError, match=message):
        parse_predicate(source)


@pytest.mark.parametrize(
    "source",
    [
        "model_id newest",
        "unknown asc",
        "lifecycle order []",
        'lifecycle order ["maintained", "maintained"]',
    ],
)
def test_query_rejects_invalid_sorts(source):
    with pytest.raises(ValueError):
        parse_sort(source)


def test_query_referenced_fields_drive_lazy_enrichment():
    filters = [parse_predicate("deployment.availability.cloudrift == true")]
    requirements = [parse_predicate("provider.cloudrift.team_access == true")]
    sorts = [parse_sort("results.last_run_at asc nulls-first")]

    assert referenced_fields(filters, requirements, sorts) == {
        "deployment.availability.cloudrift",
        "provider.cloudrift.team_access",
        "results.last_run_at",
    }


async def test_query_resolves_team_access_before_cloudrift_availability(monkeypatch):
    calls = []

    async def validate_team(_api_key, _team_id):
        calls.append("team")
        return "normalized-team-id"

    async def list_available(_api_key):
        calls.append("availability")
        return {"rtx49-10c-kn.1"}

    monkeypatch.setattr(recipe_query, "validate_team_id_access", validate_team)
    monkeypatch.setattr(recipe_query, "list_available_instance_types", list_available)
    rows = build_query_rows(
        [],
        model_id="org/new-model",
        allow_missing_model=True,
        gpu="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        expand_deployments=True,
    )

    await recipe_query.enrich_query_rows(
        rows,
        {"provider.cloudrift.team_access", "deployment.availability.cloudrift"},
        cloudrift_api_key="test-key",
        cloudrift_team_id="test-team",
    )

    assert calls == ["team", "availability"]
    assert rows[0]["provider"]["cloudrift"]["team_access"] is True
    assert rows[0]["deployment"]["availability"]["cloudrift"] is True


def test_query_document_has_independent_versioned_schema():
    assert query_document([]) == {"schema_version": 1, "rows": []}
