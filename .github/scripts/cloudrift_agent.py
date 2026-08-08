#!/usr/bin/env python3
"""Run a tracked repository skill through CloudRift's Chat Completions API."""

from __future__ import annotations

import argparse
import html.parser
import ipaddress
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

DEFAULT_ENDPOINT = "https://inference.cloudrift.ai/v1"
MAX_TOOL_OUTPUT = 12_000
MAX_TRANSCRIPT_CHARS = 160_000
MAX_FETCH_BYTES = 256_000
SECRET_ENV_NAMES = {
    "AGENT_KEY_FILE",
    "APP_KEY_FILE",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
    "ACTIONS_RUNTIME_TOKEN",
    "CLOUDRIFT_API_KEY",
    "CLOUDRIFT_API_URL",
    "EXPERIMENT_APP_PRIVATE_KEY",
    "GCP_SERVICE_ACCOUNT",
    "GCP_CONFIG_DIR",
    "GCP_KEY_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GITHUB_TOKEN",
    "GH_TOKEN",
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command in the repository. CLOUDRIFT_API_KEY and GitHub credentials are never inherited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 2700},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Atomically write a UTF-8 file in the repository, or an explicitly allowed output path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace one exact, unique string in a UTF-8 repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web without credentials and return bounded titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 256},
                    "results": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch bounded readable text from one public HTTP(S) URL; private and metadata addresses are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "maxLength": 2048},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 12000},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


class _SearchParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None
        self.capture_tag: str | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if "result__a" in classes:
            self._flush_current()
            self.current = {"title": "", "url": _duckduckgo_target(values.get("href") or ""), "snippet": ""}
            self.capture, self.capture_tag, self.parts = "title", tag, []
        elif "result__snippet" in classes and self.current is not None:
            self.capture, self.capture_tag, self.parts = "snippet", tag, []

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.capture and tag == self.capture_tag and self.current is not None:
            self.current[self.capture] = " ".join("".join(self.parts).split())
            self.capture = self.capture_tag = None
            self.parts = []

    def close(self) -> None:
        super().close()
        self._flush_current()

    def _flush_current(self) -> None:
        if self.current and self.current["title"] and self.current["url"]:
            self.current["title"] = self.current["title"][:300]
            self.current["url"] = self.current["url"][:2048]
            self.current["snippet"] = self.current["snippet"][:600]
            self.results.append(self.current)
        self.current = None


class _TextParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in {"br", "div", "h1", "h2", "h3", "li", "p", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in {"div", "h1", "h2", "h3", "li", "p", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _duckduckgo_target(raw_url: str) -> str:
    absolute = urljoin("https://duckduckgo.com", raw_url)
    parsed = urlparse(absolute)
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else absolute


def _validate_public_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must be public HTTP(S) without embedded credentials")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"URL host resolves to a non-public address: {address}")
    return raw_url


def _bounded_get(raw_url: str, *, params: dict | None = None) -> tuple[str, str]:
    headers = {"User-Agent": "emmy-model-discovery/1.0"}
    current = raw_url
    with httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=False, headers=headers) as client:
        for _ in range(4):
            _validate_public_url(current)
            with client.stream("GET", current, params=params) as response:
                params = None
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Redirect response omitted its location")
                    current = urljoin(str(response.url), location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not any(kind in content_type for kind in ("text/", "json", "xml")):
                    raise ValueError(f"URL returned unsupported content type: {content_type or 'unknown'}")
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_FETCH_BYTES:
                        raise ValueError(f"URL response exceeded {MAX_FETCH_BYTES} bytes")
                return bytes(chunks).decode(response.encoding or "utf-8", errors="replace"), str(response.url)
    raise ValueError("URL exceeded the redirect limit")


def _search_web(query: str, results: int = 5) -> str:
    query = query.strip()
    if not query or len(query) > 256:
        raise ValueError("Search query must contain 1-256 characters")
    results = max(1, min(int(results), 8))
    body, _ = _bounded_get("https://html.duckduckgo.com/html/", params={"q": query})
    parser = _SearchParser()
    parser.feed(body)
    parser.close()
    return _trim(json.dumps(parser.results[:results], ensure_ascii=False))


def _fetch_url(raw_url: str, max_chars: int = 8000) -> str:
    max_chars = max(500, min(int(max_chars), 12000))
    body, final_url = _bounded_get(raw_url)
    parser = _TextParser()
    parser.feed(body)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    return _trim(json.dumps({"url": final_url, "text": text[:max_chars]}, ensure_ascii=False))


def _trim(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT:
        return value
    half = MAX_TOOL_OUTPUT // 2
    return f"{value[:half]}\n... {len(value) - MAX_TOOL_OUTPUT} characters omitted ...\n{value[-half:]}"


def _compact_messages(messages: list[dict]) -> list[dict]:
    """Keep complete recent assistant/tool groups within a bounded transcript."""
    if len(messages) <= 2:
        return messages
    groups: list[list[dict]] = []
    for message in messages[2:]:
        if message.get("role") == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)

    kept: list[list[dict]] = []
    total = 0
    for group in reversed(groups):
        size = len(json.dumps(group, ensure_ascii=False))
        if kept and total + size > MAX_TRANSCRIPT_CHARS:
            break
        kept.append(group)
        total += size
    kept.reverse()
    if len(kept) == len(groups):
        return messages
    notice = {"role": "system", "content": "Earlier tool transcript omitted to stay within the context budget."}
    return [*messages[:2], notice, *(message for group in kept for message in group)]


def _tool_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in SECRET_ENV_NAMES}


def _take_api_key(args: argparse.Namespace) -> str:
    if args.api_key_file is not None:
        path = args.api_key_file.resolve()
        try:
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise RuntimeError(f"API key file permissions must be 0600: {path}")
            value = path.read_text().strip()
        finally:
            path.unlink(missing_ok=True)
        return value
    with os.fdopen(args.api_key_fd, "r") as source:
        return source.read().strip()


def _resolve_path(workspace: Path, raw_path: str, allowed_writes: set[Path], *, writing: bool) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    if candidate == workspace or workspace in candidate.parents:
        return candidate
    if writing and candidate in allowed_writes:
        return candidate
    raise ValueError(f"Path is outside the repository: {raw_path}")


def _run_tool(
    name: str,
    arguments: dict,
    workspace: Path,
    allowed_writes: set[Path],
) -> str:
    if name == "web_search":
        return _search_web(arguments["query"], arguments.get("results", 5))
    if name == "fetch_url":
        return _fetch_url(arguments["url"], arguments.get("max_chars", 8000))
    if name == "shell":
        timeout = min(int(arguments.get("timeout_seconds", 600)), 2700)
        try:
            result = subprocess.run(
                arguments["command"],
                cwd=workspace,
                env=_tool_environment(),
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return _trim(f"exit_code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        except subprocess.TimeoutExpired as exc:
            return _trim(f"timeout after {timeout}s\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}")

    path = _resolve_path(workspace, arguments["path"], allowed_writes, writing=name != "read_file")
    if name == "read_file":
        return _trim(path.read_text())
    if name == "write_file":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(arguments["content"])
        temporary.replace(path)
        return f"wrote {path}"
    if name == "replace_in_file":
        current = path.read_text()
        count = current.count(arguments["old"])
        if count != 1:
            raise ValueError(f"Expected exactly one match in {path}, found {count}")
        path.write_text(current.replace(arguments["old"], arguments["new"], 1))
        return f"updated {path}"
    raise ValueError(f"Unknown tool: {name}")


def _completion(client: httpx.Client, endpoint: str, api_key: str, payload: dict) -> dict:
    response = client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("CloudRift returned an invalid Chat Completions response") from exc


def run(args: argparse.Namespace) -> int:
    api_key = _take_api_key(args)
    if not api_key:
        raise RuntimeError("CloudRift API key file or descriptor was empty")

    workspace = args.workspace.resolve()
    skill = args.skill.resolve()
    if workspace not in skill.parents:
        raise RuntimeError("Skill must be tracked inside the repository")
    allowed_writes = {path.resolve() for path in args.allow_write}
    system = f"""You are an autonomous repository agent running non-interactively in GitHub Actions.

Follow the tracked skill below exactly. Inspect linked repository documentation and skills before acting. Work until
the requested outcome is complete or a real gate fails. Never rent or delete a VM, commit, push, open a pull request,
or print credentials; the workflow owns those operations. Use the supplied SSH target only. Keep changes scoped to
the request, clean exploratory output, and leave the worktree ready for automated validation and submission.

<tracked-skill path={skill.relative_to(workspace)}>
{skill.read_text()}
</tracked-skill>
"""
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": args.prompt.read_text()},
    ]
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    with httpx.Client(timeout=httpx.Timeout(args.request_timeout, connect=30)) as client:
        for _ in range(args.max_turns):
            messages = _compact_messages(messages)
            payload["messages"] = messages
            message = _completion(client, args.endpoint, api_key, payload)
            tool_calls = message.get("tool_calls") or []
            messages.append(message)
            if not tool_calls:
                final = message.get("content") or ""
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(final.rstrip() + "\n")
                print(final)
                return 0
            for tool_call in tool_calls:
                try:
                    function = tool_call["function"]
                    arguments = json.loads(function.get("arguments") or "{}")
                    result = _run_tool(function["name"], arguments, workspace, allowed_writes)
                except Exception as exc:
                    result = f"tool_error: {exc}"
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
    raise RuntimeError(f"Agent exceeded {args.max_turns} model turns")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-write", type=Path, action="append", default=[])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--endpoint", default=os.environ.get("CLOUDRIFT_INFERENCE_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument("--request-timeout", type=float, default=600)
    api_key = parser.add_mutually_exclusive_group(required=True)
    api_key.add_argument("--api-key-file", type=Path)
    api_key.add_argument("--api-key-fd", type=int)
    return parser


def main() -> int:
    try:
        return run(_parser().parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
