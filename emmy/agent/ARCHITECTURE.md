# Agent runner

`emmy agent run` executes one tracked repository skill through an OpenAI-compatible Chat Completions endpoint. The
runner is independent of GitHub Actions: local automation, CI workflows, and other skill-driven jobs use the same
implementation in `runner.py`.

The caller supplies a skill, a task prompt, model, endpoint, final-output path, and an API key through a mode-`0600`
one-use file or inherited file descriptor. The key is consumed before the first model request. Tool subprocesses get
a scrubbed environment without CloudRift, GitHub, GCP, or runner credential variables; explicitly required workload
credentials such as `HF_TOKEN` remain available.

The runner exposes bounded repository, shell, and public-web tools. Repository reads stay inside the selected
workspace. Writes stay inside that workspace except for paths explicitly named with `--allow-write`. Shell output,
web responses, redirects, extracted text, individual tool results, and retained message history all have fixed
bounds. Public fetches reject credentials in URLs and every address that is not globally routable. An empty terminal
assistant message is not a successful result; the runner asks for the requested final response again within the same
bounded turn budget. Retained history stays below the response endpoint's practical context ceiling; older complete
assistant/tool groups are dropped together, so a tool result is never separated from the call that produced it. The
context notice is folded into the original user prompt, preserving the endpoint contract that the one system message
must come first. The caller can lower the per-turn output reservation with `--max-output-tokens` when a workflow has a
small atomic result. A caller with exactly one `--allow-write` path can use `--force-write-turn` to append one user
message and require the existing `write_file` tool on that turn; later turns return to automatic tool selection and
the maximum turn count remains the hard limit. HTTP failures include a bounded response detail in the runner error so
endpoint validation failures remain actionable.

Tool descriptions and JSON schemas live beside their handlers in `runner.py`, which is the single source of truth.
`emmy agent tools` serializes a detached copy of those definitions as JSON for inspection or integration; there is no
second checked-in JSON file that could drift from executable behavior.

The runner enforces mechanics and credential isolation, not workflow authorization. A workflow remains responsible
for selecting the skill, limiting repository permissions, defining allowed external actions, validating the resulting
worktree, and owning VM, Git, publication, and pull-request operations.
