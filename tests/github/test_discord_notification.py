import importlib.util
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "discord_notification.py"
SPEC = importlib.util.spec_from_file_location("discord_notification", MODULE_PATH)
discord_notification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discord_notification)

BASE_ENVIRONMENT = {
    "GITHUB_REPOSITORY": "cloudrift-ai/emmy",
    "GITHUB_RUN_ID": "12345",
    "GITHUB_SERVER_URL": "https://github.com",
}


def test_onboarding_success_payload_has_target_pr_and_no_mentions():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "onboard",
        "WORKFLOW_RESULT": "success",
        "SELECTED": "true",
        "MODEL_ID": "Qwen/Qwen3-Embedding-8B",
        "TARGET_GPU": "NVIDIA GeForce RTX 4090",
        "TARGET_GPU_COUNT": "1",
        "ONBOARD_MODE": "onboarding",
        "DEPLOYMENT_SUMMARY": "vLLM 0.22.1, 32K context, concurrency 8",
        "PERFORMANCE_SUMMARY": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
        "PR_NUMBER": "487",
    }

    payload = discord_notification.build_payload(environment, now=datetime(2026, 8, 16, tzinfo=UTC))
    embed = payload["embeds"][0]

    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Model onboarding completed"
    assert embed["url"] == "https://github.com/cloudrift-ai/emmy/actions/runs/12345"
    assert embed["timestamp"] == "2026-08-16T00:00:00+00:00"
    assert embed["fields"] == [
        {"name": "Model", "value": "`Qwen/Qwen3-Embedding-8B`", "inline": False},
        {"name": "Target", "value": "`NVIDIA GeForce RTX 4090 x1`", "inline": False},
        {"name": "Deployment", "value": "vLLM 0.22.1, 32K context, concurrency 8", "inline": False},
        {"name": "Performance", "value": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures", "inline": False},
        {"name": "Mode", "value": "`onboarding`", "inline": True},
        {
            "name": "Rolling PR",
            "value": "[#487](https://github.com/cloudrift-ai/emmy/pull/487)",
            "inline": True,
        },
    ]


def test_discovery_success_payload_groups_only_modified_models():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "discover",
        "WORKFLOW_RESULT": "success",
        "MODIFIED_MODELS": json.dumps(
            [
                {"model_id": "org/maintained", "lifecycle": "maintained"},
                {"model_id": "org/best-effort", "lifecycle": "best-effort"},
                {"model_id": "org/new", "lifecycle": "onboarding"},
            ]
        ),
    }

    payload = discord_notification.build_payload(environment)

    assert payload["embeds"][0]["fields"] == [
        {"name": "Maintained", "value": "• `org/maintained`", "inline": False},
        {"name": "Best effort", "value": "• `org/best-effort`", "inline": False},
        {"name": "Onboarding", "value": "• `org/new`", "inline": False},
    ]


def test_discovery_success_payload_reports_no_recipe_changes():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "discover",
        "WORKFLOW_RESULT": "success",
        "MODIFIED_MODELS": "[]",
    }

    payload = discord_notification.build_payload(environment)

    assert payload["embeds"][0]["fields"] == [
        {"name": "Modified models", "value": "None; the lifecycle review produced no recipe changes.", "inline": False}
    ]


def test_discovery_failure_payload_is_noticeable_without_model_fields():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "discover",
        "WORKFLOW_RESULT": "failure",
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Model discovery failed"
    assert embed["color"] == discord_notification.FAILURE_COLOR
    assert embed["fields"] == []


def test_no_eligible_onboarding_is_a_neutral_summary():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "onboard",
        "WORKFLOW_RESULT": "success",
        "SELECTED": "false",
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert embed["title"] == "No eligible model deployment"
    assert embed["color"] == discord_notification.NEUTRAL_COLOR


def test_delivery_retries_and_requests_a_confirmed_discord_response():
    calls = []
    sleeps = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"{}"

    def opener(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) < 3:
            raise URLError("temporary failure")
        return Response()

    payload = {"allowed_mentions": {"parse": []}}

    delivered = discord_notification.send_notification(
        "https://discord.com/api/webhooks/id/token?thread_id=7",
        payload,
        opener=opener,
        sleeper=sleeps.append,
    )

    assert delivered is True
    assert len(calls) == 3
    assert calls[-1][0].full_url.endswith("?thread_id=7&wait=true")
    assert calls[-1][1] == 15
    assert json.loads(calls[-1][0].data) == payload
    assert sleeps == [1, 2]


def test_missing_webhook_is_non_fatal(caplog):
    with caplog.at_level(logging.WARNING, logger="discord_notification"):
        result = discord_notification.main({})

    assert result == 0
    assert "DISCORD_EMMY_ROBOTS_WEBHOOK_URL is not configured" in caplog.text
