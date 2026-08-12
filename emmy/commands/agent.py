"""CLI registration for the repository-skill agent runner."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from emmy.agent import AgentRun, run, tool_definitions
from emmy.agent.runner import DEFAULT_ENDPOINT

logger = logging.getLogger(__name__)


def _handle_run(args) -> None:
    try:
        final = asyncio.run(
            run(
                AgentRun(
                    skill=args.skill,
                    prompt=args.prompt,
                    model=args.model,
                    output=args.output,
                    workspace=args.workspace,
                    endpoint=args.endpoint,
                    allow_write=tuple(args.allow_write),
                    max_turns=args.max_turns,
                    force_final_turn=args.force_final_turn,
                    max_output_tokens=args.max_output_tokens,
                    request_timeout=args.request_timeout,
                    api_key_file=args.api_key_file,
                    api_key_fd=args.api_key_fd,
                )
            )
        )
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info(final)


def _handle_tools(args) -> None:
    payload = json.dumps(tool_definitions(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        logger.info(payload.rstrip())
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)


def register_agent_command(subparsers) -> None:
    """Register `emmy agent run/tools`."""
    parser = subparsers.add_parser("agent", help="Run tracked repository skills with an OpenAI-compatible model")
    actions = parser.add_subparsers(dest="agent_action", required=True)

    run_parser = actions.add_parser("run", help="Run one tracked skill non-interactively")
    run_parser.add_argument("--skill", type=Path, required=True)
    run_parser.add_argument("--prompt", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--allow-write", type=Path, action="append", default=[])
    run_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    run_parser.add_argument("--endpoint", default=os.environ.get("CLOUDRIFT_INFERENCE_URL", DEFAULT_ENDPOINT))
    run_parser.add_argument("--max-turns", type=int, default=160)
    run_parser.add_argument("--force-final-turn", type=int)
    run_parser.add_argument("--max-output-tokens", type=int, default=8192)
    run_parser.add_argument("--request-timeout", type=float, default=600)
    api_key = run_parser.add_mutually_exclusive_group(required=True)
    api_key.add_argument("--api-key-file", type=Path)
    api_key.add_argument("--api-key-fd", type=int)
    run_parser.set_defaults(func=_handle_run)

    tools_parser = actions.add_parser("tools", help="Print the model tool definitions as JSON")
    tools_parser.add_argument("--output", type=Path)
    tools_parser.set_defaults(func=_handle_tools)
