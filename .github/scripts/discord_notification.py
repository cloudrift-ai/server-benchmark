"""Post a compact, non-pinging model lifecycle summary to Discord."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

SUCCESS_COLOR = 5_763_719
FAILURE_COLOR = 15_548_997
CANCELLED_COLOR = 16_705_372
NEUTRAL_COLOR = 9_807_270


def _run_url(environment: Mapping[str, str]) -> str:
    return f"{environment['GITHUB_SERVER_URL']}/{environment['GITHUB_REPOSITORY']}/actions/runs/{environment['GITHUB_RUN_ID']}"


def _pull_request_field(environment: Mapping[str, str]) -> dict[str, Any] | None:
    number = environment.get("PR_NUMBER", "")
    if not number.isdigit():
        return None
    url = f"{environment['GITHUB_SERVER_URL']}/{environment['GITHUB_REPOSITORY']}/pull/{number}"
    return {"name": "Rolling PR", "value": f"[#{number}]({url})", "inline": True}


def _onboard_summary(environment: Mapping[str, str]) -> tuple[str, str, int, list[dict[str, Any]]]:
    result = environment.get("WORKFLOW_RESULT", "failure")
    selected = environment.get("SELECTED", "")
    mode = environment.get("ONBOARD_MODE", "")
    model_id = environment.get("MODEL_ID") or "Automatic selection"
    gpu = environment.get("TARGET_GPU") or "Automatic selection"
    gpu_count = environment.get("TARGET_GPU_COUNT", "")
    target = f"{gpu} x{gpu_count}" if gpu_count else gpu

    if result == "success" and selected == "false":
        title = "No eligible model deployment"
        description = "The nightly selector found no onboarding or maintained recipe with available hardware."
        color = NEUTRAL_COLOR
    elif result == "success":
        operation = "verification" if mode == "verification" else "onboarding"
        title = f"Model {operation} completed"
        description = "The model lifecycle artifacts were pushed and the rolling pull request was updated."
        color = SUCCESS_COLOR
    elif result == "cancelled":
        title = "Model onboarding workflow cancelled"
        description = "The model lifecycle workflow was cancelled before it completed."
        color = CANCELLED_COLOR
    else:
        operation = f" {mode}" if mode else ""
        title = f"Model{operation} failed"
        description = "The model lifecycle workflow failed. Open the run for the failing step and logs."
        color = FAILURE_COLOR

    fields: list[dict[str, Any]] = [
        {"name": "Model", "value": f"`{model_id[:1000]}`", "inline": False},
        {"name": "Target", "value": f"`{target[:1000]}`", "inline": False},
    ]
    if mode:
        fields.append({"name": "Mode", "value": f"`{mode[:1000]}`", "inline": True})
    return title, description, color, fields


def _discover_summary(environment: Mapping[str, str]) -> tuple[str, str, int, list[dict[str, Any]]]:
    result = environment.get("WORKFLOW_RESULT", "failure")
    if result == "success":
        return (
            "Model discovery completed",
            "The maintained recipe set was reviewed and the rolling model lifecycle pull request was refreshed.",
            SUCCESS_COLOR,
            [],
        )
    if result == "cancelled":
        return (
            "Model discovery workflow cancelled",
            "The model discovery workflow was cancelled before it completed.",
            CANCELLED_COLOR,
            [],
        )
    return (
        "Model discovery failed",
        "The model discovery workflow failed. Open the run for the failing step and logs.",
        FAILURE_COLOR,
        [],
    )


def build_payload(environment: Mapping[str, str], *, now: datetime | None = None) -> dict[str, Any]:
    """Build the Discord webhook body from a GitHub Actions environment."""
    workflow_kind = environment.get("WORKFLOW_KIND", "")
    if workflow_kind == "onboard":
        title, description, color, fields = _onboard_summary(environment)
    elif workflow_kind == "discover":
        title, description, color, fields = _discover_summary(environment)
    else:
        raise ValueError(f"Unsupported WORKFLOW_KIND: {workflow_kind!r}")

    pull_request = _pull_request_field(environment)
    if pull_request is not None:
        fields.append(pull_request)
    timestamp = now or datetime.now(UTC)
    return {
        "username": "Emmy Robots",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": title,
                "url": _run_url(environment),
                "description": description,
                "color": color,
                "fields": fields,
                "footer": {"text": f"{environment['GITHUB_REPOSITORY']} · run {environment['GITHUB_RUN_ID']}"},
                "timestamp": timestamp.astimezone(UTC).isoformat(),
            }
        ],
    }


def send_notification(
    webhook_url: str,
    payload: Mapping[str, Any],
    *,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    """Send a webhook payload, retrying transient delivery failures three times."""
    separator = "&" if "?" in webhook_url else "?"
    request = Request(
        f"{webhook_url}{separator}wait=true",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "emmy-github-actions"},
        method="POST",
    )
    for attempt in range(1, 4):
        try:
            with opener(request, timeout=15) as response:
                response.read()
                if response.status != 200:
                    raise RuntimeError(f"unexpected HTTP status {response.status}")
            return True
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            if attempt == 3:
                LOGGER.warning(
                    "::warning::Discord notification failed after three attempts: %s",
                    type(error).__name__,
                )
                return False
            sleeper(attempt)
    return False


def main(environment: Mapping[str, str] | None = None) -> int:
    """Send the configured notification without making its workflow fail."""
    values = os.environ if environment is None else environment
    webhook_url = values.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        LOGGER.warning("::warning::DISCORD_EMMY_ROBOTS_WEBHOOK_URL is not configured; skipping Discord notification")
        return 0
    send_notification(webhook_url, build_payload(values))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
