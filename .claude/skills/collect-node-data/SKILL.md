---
name: collect-node-data
description: Use this skill when the user wants to populate or refresh the autotune DB's node table (the cross-hardware search-tree node store) with data measured on a SPECIFIC GPU — e.g. "collect node data for an H200", "run the node sweep on a rented <GPU> and merge the nodes back", "gather cross-hardware node-store data", "populate / update the node table from <hardware>". Rents a fresh single-GPU server (via start-remote-server), rsyncs + sets up emmy there, runs the budgeted three-slice golden sweep (`remote_node_collect.py` → `golden_neighbor_bench.py`: every golden kind's candidate pool, sliced own-neighborhood / cross-card exchange / uniform tail at 60/25/15 budget shares, paired -O1/-O3 pinned benches, ledger-resumed), merges the remote node rows into the local `~/.cache/emmy/autotune.db` (nodes table only, GPU-keyed so cards never collide), backs the DB up locally, and tears the server down.
version: 0.3.0
---

# Collect node-store data from specific hardware

The autotune `node` table (`SearchDB`, default `~/.cache/emmy/autotune.db`) is a **cross-hardware** dataset of
search-tree value-of-position rows — the leaf measurements the future offline prior trains on, with the full feature
dict the prior sees. It is read by `emmy eval prior --dataset nodes` (per-card fork sibling-ranking + leaf
reachability) and feeds prior diagnostics. Because your dev box (and most of the fleet) has no local CUDA GPU, the
data for any given card must be **measured on that card** and brought back.

This skill does exactly that for one GPU, in a single budgeted run on the rented box: rent it → set up emmy →
the golden sweep (`scripts/remote_node_collect.py`, driving `scripts/golden_neighbor_bench.py`) → merge the new card's
node rows into the local DB → back the DB up → tear the server down.

**Why a budgeted sweep (and not a search-driven tune).** The old flow ran an ε-greedy `emmy tune --dataset golden`
first: its wall time grew with the golden set (patience-stopped search per shape, no budget knob), and even at
ε=0.25 three quarters of its benches followed the incumbent prior — the collected rows over-sampled the branches the
prior already liked (a censoring feedback loop). The sweep replaces it: a fixed `--budget-s` wall clock, and points
drawn from the enumeration itself rather than a search trajectory, so the future offline prior gets clean leaf rows
whose selection didn't depend on the incumbent's opinions. `emmy tune --explore-eps` still exists for interactive
tuning; it is just no longer the collection vehicle.

**Why three slices.** Every golden shape's candidate pool (all kinds — matmul, reduce, rms_norm, softmax, attention,
the fused norm_linear / mlp_geglu, …) is split by distance to the recorded golden anchors, and the budget is spent
across the slices at configurable shares (default 60/25/15):

- **own** — rows within `--max-dist` of the live card's OWN goldens: dense label support (and fine ranking
  resolution) exactly where this card's deploys land.
- **cross** — rows near OTHER cards' goldens that realize on this card: every such point is verified-excellent
  somewhere, so measuring it here yields either transfer signal or an arch-disagreement row — the data the prior's
  arch × knob interactions (and a leave-one-card-out evaluation) need. An anchor that doesn't enumerate here is
  skipped, which is itself the realizability filter.
- **tail** — a capped, hash-ordered (seed-independent, so stable across sessions) subsample of the rest of the
  enumeration: landscape support so the prior also learns what bad looks like.

**Why paired -O1/-O3.** The offline prior trains on deployable (-O3) records, and the historical tune data is mostly
-O1 — the dataset holds few points measured at BOTH opt levels. The sweep benches every point at both `-Xcicc -O1`
and `-Xcicc -O3` (`emmy run --bench --ab`, node recording on): the twins share the knob set, so they join later on
`op_sig` + tunables — the dataset a future "how well does -O1 approximate -O3 on this shape" model fits on. Both
legs of a pair are measured on the same box in the same session.

**Why merging into the single local DB is safe (read this once).** The node store is keyed by GPU:
`node_key = digest(context_key, gpu, op_sig, tunable-knobs)` and the `node` table carries a `gpu` column
(`Context.hardware_id()` — the canonicalized PCIe product name). Different cards therefore **never collide** — the
upsert (branch keep-min / leaf newest-measurement) only collapses rows *within one card*. So accumulating many GPUs'
node rows in the one canonical `~/.cache/emmy/autotune.db` is the intended design (that is what makes it a
cross-hardware dataset). Do **not** keep per-GPU DB files. After each merge the driver also snapshots the DB to
`~/.cache/emmy/backups/` (newest 5 kept) — the local DB is the sole copy of data that cost rented-GPU hours, and
cache wipes happen.

**Scope: nodes table only.** This skill copies back the `node` table and nothing else (not `perf`, `cuda_op`,
`lowering`, or the online prior checkpoint). Those are needed only for `--dataset db` / greedy replay, out of scope
here.

The sweep neither needs **`HF_TOKEN` nor downloads models** (golden snippets are pure `torch.randn`) — but it **does**
need `nvcc` (it compiles CUDA kernels). Budget **~4.5–5.5 h total** for the default 4-hour sweep including env build;
confirm with the user before starting if they expected a quick run. The budget samples the pool, it doesn't finish
it — repeated rentals of the same card model resume from the fetched ledger and keep extending coverage.

## Inputs to confirm

Ask only for what the user hasn't already given:

1. **GPU model** — must map to a key in `emmy/hardware.py::GPU_INSTANCE_TYPES` (e.g. "H200" → `"NVIDIA H200
   141GB"`). If it isn't in the table, stop and say so — don't guess. This card's identity is what the node rows are
   keyed by.
2. **Provider** — only ask if the GPU is offered by more than one (e.g. H200 is on CloudRift and GCP). A user-named
   provider is **binding** (never silently substitute).
3. **Env file** (CloudRift creds) — default `.env`; if the user named an overlay or wants a non-default cluster, follow
   the `start-remote-server` sourcing rules (base first, overlay second, same Bash call). H200 on CloudRift often needs
   a non-default cluster.
4. **GPU count is fixed at 1.** Node data is per-card; one GPU is all the sweep needs. Don't rent more.

Don't pass `--billing-exempt` for any rentals — all rentals bill normally (the flag is admin-only and not used by
any skill flow).

## Step 1 — Provision the server (delegate to `start-remote-server`)

Provision exactly one GPU using the orchestrator, following the full `start-remote-server` skill (credential sourcing,
candidate fallback, capacity handling, the binding `--provider` rule). The command shape:

```bash
[ -f .env ] && set -a && . ./.env && set +a && \
emmy vm create gpu --gpu "<full GPU name>" --gpu-count 1 [--provider cloudrift|gcp]
```

Capture from the final `VM ready at <user@host[:port]>` line:

- `REMOTE` — `user@host` (and the port, if any)
- the teardown handle (`--instance-id <id>` for CloudRift, `--instance <name> --zone <zone>` for GCP)

Do not wrap the command in a retry loop — the orchestrator handles fallback itself.

## Step 2 — Set up, sweep, merge, and back up on the remote (one backgrounded run of one script)

`scripts/remote_node_collect.py` does the whole core in **one process**: ensures the Python 3.12 venv/dev packages +
`nvcc`, rsyncs your working tree (exact local code, incl. uncommitted changes) to `~/.local/share/emmy/node-collect/`
(the repo's `REMOTE_DEPLOY_DIR` layout), runs `make setup` (output to `setup.log` in that dir — only a tail returns
on failure), pushes the resume ledger, launches the sweep detached, **polls the remote log internally** until it
finishes, then **fetches the ledger and node rows back and merges them** into the local `~/.cache/emmy/autotune.db`
(`node` table only, GPU-keyed so other cards are untouched), snapshots the DB to `~/.cache/emmy/backups/`, and prints
the per-card receipt. The robustness traps are baked in: argv-list ssh (no zsh word-split), bracket-pgrep liveness
(no self-match), one short ssh per poll (no broken-pipe), venv/dev always installed, and a non-tty-safe detached
launch.

Run it in the **background** (Bash `run_in_background: true`) — ~4.5 h, past any foreground tool timeout, and you
want only the final summary in context, not ~20 ssh polls. The harness re-invokes you with the summary when it
exits, so **do not poll manually** — wait for the completion notification.

```bash
./venv/bin/python scripts/remote_node_collect.py --remote "<user@host>" --ssh-key ~/.ssh/id_ed25519 [--port <PORT>] \
    --budget-s 14400
```

(`--budget-s` caps the sweep's wall time at 4 h. Leave `--timeout` alone — it derives as budget + 3900 s, covering the
two in-flight 1800 s invocations an O1+O3 batch can add past the budget. Reaching `--timeout` is NOT a failure: the
driver stops the sweep and harvests everything measured so far (ledger + merge + backup still run); a thinner explicit
`--timeout` just kills the in-flight batch, whose points retry next run. Slice shares default to 60/25/15
(`--share-own/--share-cross/--share-tail`; `--tail-cap` bounds the per-shape tail at 200). The resume ledger defaults
to `~/.cache/emmy/neighbor_bench/ledger.json` locally — pushed before the run, fetched back after, so a later session
on the same card model continues instead of repeating.)

**On a card's first run**, consider a quick pool sanity pass before the budgeted sweep: ssh in and run
`./venv/bin/python scripts/golden_neighbor_bench.py --dry-run` from the remote repo dir (with the CUDA PATH export) —
eyeball the per-shape `[pool]` lines: own/cross anchor counts per kind, and that the fused kinds (norm_linear,
mlp_geglu) and static attention shapes report non-empty pools with matched anchors. A local (no-GPU) `--dry-run` only
validates imports and the trace path — off-GPU every anchor labels cross and pools are not meaningful.

When the run exits you get one compact result:

- **success** → `status: ok`, a `points: neighbor-bench done: <done>/<total> points (<new> new this run)` line plus
  `ledger: fetched -> …` — **followed by the merge receipt** (the rented card appears as its own line in `node rows
  per card now: …`; cards already present are unchanged), a `backup: <path>` line, and `status: COMPLETE (sweep +
  merge done)`. Merge and backup are folded in, so there is **no separate step to launch** — the local DB is already
  updated and snapshotted. A run stopped at `--timeout` reports the same way with `sweep stopped at --timeout`
  markers — that is still success (partial harvest), not a failure to retry.
- **failure** → `status: FAILED (<why>)` or `merge FAILED: …` plus the last 40 lines of the relevant remote log
  (`setup.log` / `neighbors.log` in that dir). Fix and re-run — the script is idempotent (rsync + `make setup` no-op
  when already done; the merge is safe to repeat — a re-merged snapshot's rows are equal-timestamped, so they never
  churn values; a re-run resumes from the ledger). `--no-merge` runs the collection only.
- `<done>` well short of `<total>` is **normal** — the pool is a long-horizon dataset; the budget takes a
  distribution-preserving sample of it and later runs keep going. Don't extend the run past the budget without
  asking the user.

**Re-merge / manual fallback.** If the merge step failed (or you used `--no-merge`), merge the rented card's rows without
re-sweeping: `./venv/bin/python scripts/merge_node_db.py --remote "<user@host>" [--ssh-key … --port …]` (or `--src
<snapshot.db>` for an already-fetched file; `--db <path>` to override the destination). Same `node`-only, per-kind
upsert semantics.

**Manual debugging** (only if the script fails and you need to poke the box): the harness shell is zsh, so pass ssh
options as an **array** — `SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -i
"$HOME/.ssh/id_ed25519")` (a plain string var fails: zsh doesn't word-split it). Then `ssh "${SSH_OPTS[@]}" "$REMOTE"
'tail -n 40 ~/.local/share/emmy/node-collect/neighbors.log'`; check liveness with `pgrep -f "[g]olden_neighbor_bench"`
(bracket trick — a plain pattern self-matches the poll's own argv); never run a multi-minute loop inside one ssh
session.

## Step 3 — Verify

```bash
./venv/bin/emmy eval prior --dataset nodes
```

`node_report` groups by card — confirm a block headed with the rented GPU's name appears (`[<gpu>] <n> nodes`, with fork
sibling-ranking + leaf reachability for that card). `--kernel matmul` / `reduce` / `pointwise` narrows to one op family.

## Step 4 — Tear down the server

The VM bills until deleted, and this skill's job is done once the merge verifies. **Confirm with the user, then delete**
(never delete a VM without an explicit go-ahead):

```bash
emmy vm delete cloudrift --instance-id <id>           # CloudRift
emmy vm delete gcp --instance <name> --zone <zone>    # GCP
```

If the user wants to keep the box for more sweeping, leave it up and report the SSH target + the exact teardown command.

## Verification checklist

Before reporting success:

- [ ] `vm create gpu` exited 0 and an SSH target was captured.
- [ ] The run ended with `status: COMPLETE (sweep + merge done)`, a `neighbor-bench done: <done>/<total> points`
      line, and `ledger: fetched -> …` (a short `<done>` count is fine — the budget samples the pool, it doesn't
      finish it).
- [ ] The startup log shows per-slice pool counts (`own <o> / cross <c> / tail <t>`) and per-kind `[pool]` lines
      without wholesale anchor-skip storms on a kind (a few `anchor skipped` lines are normal — cross-card configs
      that don't realize here).
- [ ] The merge receipt shows the rented card as its own per-card line; counts for other cards are unchanged.
- [ ] A `backup: <path>` line appeared (or the log explains why the backup was skipped).
- [ ] `eval prior --dataset nodes` shows a block for the rented GPU.
- [ ] The VM was deleted (or the user explicitly chose to keep it, and has the teardown command).

If any check fails, report the failure + raw output instead of claiming success.

## Common mistakes to avoid

- **Don't keep per-GPU DB files.** The node key includes `gpu`, so the single `~/.cache/emmy/autotune.db` is the
  correct cross-hardware accumulator. Splitting it defeats the design and breaks the per-card `eval` views.
- **Don't `scp` over the local DB / merge the whole DB.** That clobbers other cards' node rows (and the unrelated
  `perf`/`cuda_op`/`lowering` tables). Use `scripts/merge_node_db.py` — the per-kind upsert, `node` table only.
- **Don't forget the `nvcc` PATH/CUDA_HOME export** in both `make setup` and any manual invocation — without it
  kernel compiles hard-fail (there is no NVRTC fallback).
- **Don't set `HF_TOKEN` or expect model downloads** — the golden snippets are pure torch expressions.
- **Don't rent more than 1 GPU** — node data is per-card; extra GPUs just burn money.
- **Don't auto-delete the VM** — confirm teardown with the user first (and never modify a CloudRift server beyond the
  sweep we explicitly started).
- **Don't hand-roll the remote setup/poll loop** — `scripts/remote_node_collect.py` (Step 2) already handles it correctly:
  argv-list ssh (zsh doesn't word-split a string var), the `[g]olden_neighbor_bench` bracket-pgrep (a plain pattern
  self-matches the poll's own argv and reports the sweep alive forever), one short ssh per poll, and venv/dev install.
  Only drop to manual ssh for debugging — and then keep the same precautions (array ssh opts; never name a var
  `status`/`path`, which are read-only in zsh).
