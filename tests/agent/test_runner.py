import asyncio
import json
import os

import httpx
import pytest

from emmy.agent import runner


def test_tool_definitions_are_json_and_detached():
    first = runner.tool_definitions()
    first[0]["function"]["name"] = "changed"

    assert runner.tool_definitions()[0]["function"]["name"] == "shell"
    json.dumps(runner.tool_definitions())


def test_agent_tools_cli_prints_definitions(run_cli):
    returncode, stdout, _ = run_cli("agent", "tools")

    assert returncode == 0
    assert [tool["function"]["name"] for tool in json.loads(stdout)] == [
        "shell",
        "read_file",
        "write_file",
        "replace_in_file",
        "web_search",
        "fetch_url",
    ]


async def test_run_executes_tracked_skill_and_writes_final(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "result.txt"
    key = tmp_path / "key"
    skill.write_text("# Test skill\n")
    prompt.write_text("Do the test.\n")
    key.write_text("secret\n")
    key.chmod(0o600)
    seen = {}

    async def complete(_client, endpoint, api_key, payload):
        seen.update(endpoint=endpoint, api_key=api_key, payload=payload)
        return {"role": "assistant", "content": "complete"}

    monkeypatch.setattr(runner, "_completion", complete)

    final = await runner.run(
        runner.AgentRun(
            skill=skill,
            prompt=prompt,
            model="test-model",
            output=output,
            workspace=tmp_path,
            max_output_tokens=1024,
            api_key_file=key,
        )
    )

    assert final == "complete"
    assert output.read_text() == "complete\n"
    assert not key.exists()
    assert seen["api_key"] == "secret"
    assert seen["payload"]["max_tokens"] == 1024
    assert seen["payload"]["tools"] == runner.tool_definitions()


async def test_run_retries_an_empty_final_response(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "result.txt"
    key = tmp_path / "key"
    skill.write_text("# Test skill\n")
    prompt.write_text("Return JSON.\n")
    key.write_text("secret\n")
    key.chmod(0o600)
    responses = iter(
        [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": '{"result": "complete"}'},
        ]
    )
    payloads = []

    async def complete(_client, _endpoint, _api_key, payload):
        payloads.append(list(payload["messages"]))
        return next(responses)

    monkeypatch.setattr(runner, "_completion", complete)

    final = await runner.run(
        runner.AgentRun(
            skill=skill,
            prompt=prompt,
            model="test-model",
            output=output,
            workspace=tmp_path,
            api_key_file=key,
        )
    )

    assert final == '{"result": "complete"}'
    assert output.read_text() == '{"result": "complete"}\n'
    assert payloads[1][-1]["content"].startswith("Your previous response was empty")


async def test_run_can_force_the_final_response(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "result.txt"
    key = tmp_path / "key"
    source = tmp_path / "source.txt"
    skill.write_text("# Test skill\n")
    prompt.write_text("Write the result.\n")
    key.write_text("secret\n")
    key.chmod(0o600)
    source.write_text("evidence\n")
    responses = iter(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-1",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "source.txt"})},
                    }
                ],
            },
            {"role": "assistant", "content": '{"complete": true}'},
        ]
    )
    payloads = []

    async def complete(_client, _endpoint, _api_key, payload):
        payloads.append(json.loads(json.dumps(payload)))
        return next(responses)

    monkeypatch.setattr(runner, "_completion", complete)

    await runner.run(
        runner.AgentRun(
            skill=skill,
            prompt=prompt,
            model="test-model",
            output=output,
            workspace=tmp_path,
            max_turns=2,
            force_final_turn=2,
            api_key_file=key,
        )
    )

    assert payloads[1]["messages"][-1] == {"role": "user", "content": runner.FORCE_FINAL_REMINDER}
    assert payloads[1]["tool_choice"] == "none"
    assert output.read_text() == '{"complete": true}\n'


async def test_completion_retries_a_transient_server_error(monkeypatch):
    responses = iter(
        [
            httpx.Response(502, json={"error": {"message": "upstream unavailable"}}),
            httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "complete"}}]}),
        ]
    )
    sleeps = []

    async def handler(_request):
        return next(responses)

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        message = await runner._completion(client, "https://example.com/v1", "key", {"model": "test"})

    assert message["content"] == "complete"
    assert sleeps == [1]


async def test_run_enforces_request_timeout_as_a_wall_clock_deadline(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    prompt = tmp_path / "prompt.md"
    output = tmp_path / "result.txt"
    key = tmp_path / "key"
    skill.write_text("# Test skill\n")
    prompt.write_text("Finish.\n")
    key.write_text("secret\n")
    key.chmod(0o600)

    async def complete(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_completion", complete)

    with pytest.raises(RuntimeError, match="did not complete within 0.01 seconds"):
        await runner.run(
            runner.AgentRun(
                skill=skill,
                prompt=prompt,
                model="test-model",
                output=output,
                workspace=tmp_path,
                request_timeout=0.01,
                api_key_file=key,
            )
        )


def test_tool_environment_removes_cloud_and_github_credentials(monkeypatch):
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "cloud-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp-secret.json")
    monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "actions-secret")
    monkeypatch.setenv("GCP_KEY_FILE", "/tmp/gcp-key.json")
    monkeypatch.setenv("HF_TOKEN", "hf-needed-by-onboarding")

    environment = runner._tool_environment()

    assert "CLOUDRIFT_API_KEY" not in environment
    assert "GH_TOKEN" not in environment
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment
    assert "ACTIONS_RUNTIME_TOKEN" not in environment
    assert "GCP_KEY_FILE" not in environment
    assert environment["HF_TOKEN"] == "hf-needed-by-onboarding"


def test_path_resolution_allows_only_workspace_and_explicit_output(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    summary = tmp_path / "summary.json"

    assert runner._resolve_path(workspace, "recipes/model.yaml", {summary}, writing=True) == workspace / "recipes/model.yaml"
    assert runner._resolve_path(workspace, str(summary), {summary}, writing=True) == summary
    with pytest.raises(ValueError, match="outside the repository"):
        runner._resolve_path(workspace, str(tmp_path / "unexpected"), {summary}, writing=True)


async def test_shell_tool_does_not_receive_cloudrift_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "cloud-secret")
    result = await runner._run_tool(
        "shell",
        {"command": 'test -z "${CLOUDRIFT_API_KEY:-}"'},
        tmp_path,
        set(),
    )

    assert "exit_code=0" in result


def test_take_api_key_reads_and_unlinks_file(tmp_path):
    path = tmp_path / "agent-key"
    path.write_text("secret-key\n")
    path.chmod(0o600)
    args = type("Args", (), {"api_key_file": path, "api_key_fd": None})()

    assert runner._take_api_key(args) == "secret-key"
    assert not path.exists()


def test_take_api_key_rejects_permissive_file_and_still_unlinks(tmp_path):
    path = tmp_path / "agent-key"
    path.write_text("secret-key\n")
    path.chmod(0o644)
    args = type("Args", (), {"api_key_file": path, "api_key_fd": None})()

    with pytest.raises(RuntimeError, match="permissions must be 0600"):
        runner._take_api_key(args)
    assert not path.exists()


def test_take_api_key_closes_inherited_descriptor():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"secret-key\n")
    os.close(write_fd)
    args = type("Args", (), {"api_key_file": None, "api_key_fd": read_fd})()

    assert runner._take_api_key(args) == "secret-key"
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_compact_messages_keeps_assistant_tool_groups_intact(monkeypatch):
    monkeypatch.setattr(runner, "MAX_TRANSCRIPT_CHARS", 100)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 120},
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "recent"},
    ]

    compacted = runner._compact_messages(messages)

    assert all(message.get("tool_call_id") != "old" for message in compacted)
    assert any(message.get("tool_call_id") == "new" for message in compacted)
    assert "Earlier tool transcript omitted" in compacted[1]["content"]
    assert [message["role"] for message in compacted].count("system") == 1
    assert compacted[0]["role"] == "system"


def test_search_parser_returns_bounded_target_fields():
    parser = runner._SearchParser()
    parser.feed(
        """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fmodel">Model release</a>
        <a class="result__snippet">A current model announcement and benchmark.</a>
        """
    )
    parser.close()

    assert parser.results == [
        {
            "title": "Model release",
            "url": "https://example.com/model",
            "snippet": "A current model announcement and benchmark.",
        }
    ]


def test_text_parser_omits_script_and_style_content():
    parser = runner._TextParser()
    parser.feed("<h1>Model</h1><script>secret()</script><p>Public benchmark</p><style>hidden</style>")
    parser.close()

    assert "Model" in "".join(parser.parts)
    assert "Public benchmark" in "".join(parser.parts)
    assert "secret" not in "".join(parser.parts)
    assert "hidden" not in "".join(parser.parts)


async def test_validate_public_url_rejects_private_literal():
    with pytest.raises(ValueError, match="non-public"):
        await runner._validate_public_url("http://169.254.169.254/latest/meta-data")


async def test_search_query_is_bounded_without_network():
    with pytest.raises(ValueError, match="1-256"):
        await runner._search_web("x" * 257)


async def test_search_results_and_tool_output_are_bounded(monkeypatch):
    body = "".join(
        f'<a class="result__a" href="https://example.com/{index}">Model {index}</a><a class="result__snippet">{"x" * 800}</a>'
        for index in range(12)
    )

    async def bounded_get(*_args, **_kwargs):
        return body, "unused"

    monkeypatch.setattr(runner, "_bounded_get", bounded_get)

    value = await runner._search_web("new model", results=99)

    assert len(value) <= runner.MAX_TOOL_OUTPUT
    assert len(runner.json.loads(value)) == 8
    assert all(len(result["snippet"]) == 600 for result in runner.json.loads(value))


def test_trim_preserves_bounded_head_and_tail():
    value = runner._trim("a" * (runner.MAX_TOOL_OUTPUT + 100))

    assert value.startswith("a" * 100)
    assert value.endswith("a" * 100)
    assert "characters omitted" in value
