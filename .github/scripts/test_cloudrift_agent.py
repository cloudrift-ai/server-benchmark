import importlib.util
import os
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("cloudrift_agent.py")
SPEC = importlib.util.spec_from_file_location("cloudrift_agent", MODULE_PATH)
cloudrift_agent = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cloudrift_agent)


def test_tool_environment_removes_cloud_and_github_credentials(monkeypatch):
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "cloud-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp-secret.json")
    monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "actions-secret")
    monkeypatch.setenv("GCP_KEY_FILE", "/tmp/gcp-key.json")
    monkeypatch.setenv("HF_TOKEN", "hf-needed-by-onboarding")

    environment = cloudrift_agent._tool_environment()

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

    assert cloudrift_agent._resolve_path(workspace, "recipes/model.yaml", {summary}, writing=True) == workspace / "recipes/model.yaml"
    assert cloudrift_agent._resolve_path(workspace, str(summary), {summary}, writing=True) == summary
    with pytest.raises(ValueError, match="outside the repository"):
        cloudrift_agent._resolve_path(workspace, str(tmp_path / "unexpected"), {summary}, writing=True)


def test_shell_tool_does_not_receive_cloudrift_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "cloud-secret")
    result = cloudrift_agent._run_tool(
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

    assert cloudrift_agent._take_api_key(args) == "secret-key"
    assert not path.exists()


def test_take_api_key_rejects_permissive_file_and_still_unlinks(tmp_path):
    path = tmp_path / "agent-key"
    path.write_text("secret-key\n")
    path.chmod(0o644)
    args = type("Args", (), {"api_key_file": path, "api_key_fd": None})()

    with pytest.raises(RuntimeError, match="permissions must be 0600"):
        cloudrift_agent._take_api_key(args)
    assert not path.exists()


def test_take_api_key_closes_inherited_descriptor():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"secret-key\n")
    os.close(write_fd)
    args = type("Args", (), {"api_key_file": None, "api_key_fd": read_fd})()

    assert cloudrift_agent._take_api_key(args) == "secret-key"
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_compact_messages_keeps_assistant_tool_groups_intact(monkeypatch):
    monkeypatch.setattr(cloudrift_agent, "MAX_TRANSCRIPT_CHARS", 100)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 120},
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "recent"},
    ]

    compacted = cloudrift_agent._compact_messages(messages)

    assert all(message.get("tool_call_id") != "old" for message in compacted)
    assert any(message.get("tool_call_id") == "new" for message in compacted)
    assert any(message.get("content", "").startswith("Earlier tool transcript") for message in compacted)


def test_search_parser_returns_bounded_target_fields():
    parser = cloudrift_agent._SearchParser()
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
    parser = cloudrift_agent._TextParser()
    parser.feed("<h1>Model</h1><script>secret()</script><p>Public benchmark</p><style>hidden</style>")
    parser.close()

    assert "Model" in "".join(parser.parts)
    assert "Public benchmark" in "".join(parser.parts)
    assert "secret" not in "".join(parser.parts)
    assert "hidden" not in "".join(parser.parts)


def test_validate_public_url_rejects_private_literal():
    with pytest.raises(ValueError, match="non-public"):
        cloudrift_agent._validate_public_url("http://169.254.169.254/latest/meta-data")


def test_search_query_is_bounded_without_network():
    with pytest.raises(ValueError, match="1-256"):
        cloudrift_agent._search_web("x" * 257)


def test_search_results_and_tool_output_are_bounded(monkeypatch):
    body = "".join(
        f'<a class="result__a" href="https://example.com/{index}">Model {index}</a><a class="result__snippet">{"x" * 800}</a>'
        for index in range(12)
    )
    monkeypatch.setattr(cloudrift_agent, "_bounded_get", lambda *_args, **_kwargs: (body, "unused"))

    value = cloudrift_agent._search_web("new model", results=99)

    assert len(value) <= cloudrift_agent.MAX_TOOL_OUTPUT
    assert len(cloudrift_agent.json.loads(value)) == 8
    assert all(len(result["snippet"]) == 600 for result in cloudrift_agent.json.loads(value))


def test_trim_preserves_bounded_head_and_tail():
    value = cloudrift_agent._trim("a" * (cloudrift_agent.MAX_TOOL_OUTPUT + 100))

    assert value.startswith("a" * 100)
    assert value.endswith("a" * 100)
    assert "characters omitted" in value
