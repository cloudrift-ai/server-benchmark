---
name: collect-node-data
description: Use this skill when the user wants to populate or refresh the autotune DB's node table (the cross-hardware search-tree node store) with data measured on a SPECIFIC GPU — e.g. "collect node data for an H200", "tune the goldens on a rented <GPU> and merge the nodes back", "gather cross-hardware node-store data", "populate / update the node table from <hardware>", "run the golden node sweep on a remote <GPU>". Rents a fresh single-GPU server (via start-remote-server), rsyncs + sets up emmy there, runs `emmy tune --dataset golden --explore-eps 0.25 --patience 25` (ε-greedy so the node data covers siblings the incumbent prior would skip; lowered patience since exploration keeps resetting the no-new-best streak), then a 4-hour golden-neighborhood sweep (`remote_node_tune.py --mode neighbors` — paired -O1/-O3 pinned benches around the recorded goldens, ledger-resumed), merges the remote node rows into the local `~/.cache/emmy/autotune.db` (nodes table only, GPU-keyed so cards never collide), and tears the server down.
version: 0.2.0
---

# Collect node-store data from specific hardware

The autotune `node` table (`SearchDB`, default `~/.cache/emmy/autotune.db`) is a **cross-hardware** dataset of
search-tree value-of-position rows — every partial branch + leaf of each per-kernel search, with the full feature dict
the prior sees. It is read by `emmy eval prior --dataset nodes` (per-card fork sibling-ranking + leaf reachability)
and feeds prior diagnostics. Because your dev box (and most of the fleet) has no local CUDA GPU, the data for any given
card must be **measured on that card** and brought back.

This skill does exactly that for one GPU, in two phases on the same box: rent it → set up emmy →
**(A)** `emmy tune --dataset golden --explore-eps 0.25 --patience 25` → **(B)** a 4-hour golden-neighborhood sweep
(`remote_node_tune.py --mode neighbors`) → merge the new card's node rows into the local DB → tear the server down.

**Why the tune runs with ε-greedy exploration (`--explore-eps 0.25`).** The node store's labels are
value-of-position minimums over the branches the search actually benched, and deterministic PUCT (eps 0) only benches
the branches the incumbent prior already prefers — so eps-0 data just confirms the current prior (a censoring feedback
loop) and leaves most forks with a single explored child, useless for the fork sibling-ranking dataset. ε-greedy
collection visits the other siblings too, so a *better* prior can be trained and evaluated on them. This default lives
in `scripts/remote_node_tune.py` (its `--explore-eps` flag overrides it); plain `emmy tune` elsewhere keeps its
deterministic eps-0 default.

**Why lowered patience (`--patience 25` vs the tune default 50).** Patience stops an op's search after N consecutive
benches with no new best. Under ε-greedy the random exploration keeps resetting that streak, so default patience
overspends benches per op relative to the sibling coverage gained; halving it frees that time for phase B, which
spends it on the golden-neighborhood points instead. This is a collection-quality trade, not a tuning one — the
winners themselves are protected by the goldens-first deploy evidence tier.

**Why the neighborhood sweep (phase B).** The tune's rows are almost all -O1; only winners near the -O1 best get a
deployable -O3 re-bench. Phase B benches the candidate rows within a small knob distance of every recorded golden at
BOTH `-Xcicc -O1` and `-Xcicc -O3` (`emmy run --bench --ab`, node recording on), growing the dataset's
opt-level-paired slice around the goldens. Points are sampled proportionally to the remaining pool, so 4 hours yields
a near-uniform sample of it; a resume ledger (pushed/fetched by the script) makes later runs on the same card model
continue instead of repeating.

**Why merging into the single local DB is safe (read this once).** The node store is keyed by GPU:
`node_key = digest(context_key, gpu, op_sig, tunable-knobs)` and the `node` table carries a `gpu` column
(`Context.hardware_id()` — the canonicalized PCIe product name). Different cards therefore **never collide** — the
upsert (branch keep-min / leaf newest-measurement) only collapses rows *within one card*. So accumulating many GPUs'
node rows in the one canonical
`~/.cache/emmy/autotune.db` is the intended design (that is what makes it a cross-hardware dataset). Do **not** keep
per-GPU DB files.

**Scope: nodes table only.** This skill copies back the `node` table and nothing else (not `perf`, `cuda_op`,
`lowering`, or the online prior checkpoint). Those are needed only for `--dataset db` / greedy replay, out of scope here.

The golden tune is ~30–45 min (every recorded golden shape; the matmul/reduce/pointwise snippets are pure `torch.randn`,
hardware-independent, so they yield valid node rows for whatever card is rented) and the neighborhood sweep is a fixed
4 h budget. Neither needs **`HF_TOKEN` or downloads models** — but both **do** need `nvcc` (they compile CUDA kernels).
Budget **~5–6 h total** including env build; confirm with the user before starting if they expected a quick run.

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
4. **GPU count is fixed at 1.** Node data is per-card; one GPU is all the tune needs. Don't rent more.

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

## Step 2 — Set up, collect, and merge on the remote (two backgrounded runs of one script)

`scripts/remote_node_tune.py` does each phase's whole core in **one process**: ensures the Python 3.12 venv/dev
packages + `nvcc`, rsyncs your working tree (exact local code, incl. uncommitted changes) to
`~/.local/share/emmy/node-tune/` (the repo's `REMOTE_DEPLOY_DIR` layout), runs `make setup` (output to `setup.log` in
that dir — only a tail returns on failure; both no-op on the second phase), launches the phase's collection detached,
**polls the remote log internally** until it finishes, then **fetches the node rows back and merges them** into the
local `~/.cache/emmy/autotune.db` (`node` table only, GPU-keyed so other cards are untouched) and prints the per-card
receipt. The robustness traps are baked in: argv-list ssh (no zsh word-split), bracket-pgrep liveness (no self-match),
one short ssh per poll (no broken-pipe), venv/dev always installed, and a non-tty-safe detached launch.

Run each phase in the **background** (Bash `run_in_background: true`) — 30–60 min and ~4 h respectively, past any
foreground tool timeout, and you want only the final summary in context, not ~20 ssh polls. The harness re-invokes you
with the summary when each exits, so **do not poll manually** — wait for the completion notification, check phase A's
summary, then launch phase B.

**Phase A — ε-greedy golden tune (lowered patience):**

```bash
./venv/bin/python scripts/remote_node_tune.py --remote "<user@host>" --ssh-key ~/.ssh/id_ed25519 [--port <PORT>] \
    --patience 25
```

**Phase B — golden-neighborhood sweep, 4-hour budget (after A's summary arrives):**

```bash
./venv/bin/python scripts/remote_node_tune.py --remote "<user@host>" --ssh-key ~/.ssh/id_ed25519 [--port <PORT>] \
    --mode neighbors --budget-s 14400 --timeout 16200
```

(`--budget-s` caps the driver's wall time at 4 h; `--timeout` = budget + the driver's per-invocation cap of 1800 s, so
the wait never false-fails on an in-flight batch finishing past the budget. The resume ledger defaults to
`~/.cache/emmy/neighbor_bench/ledger.json` locally — pushed before the run, fetched back after, so a later session on
the same card model continues coverage instead of repeating it.)

When each phase exits you get one compact result:
- **success** → the phase summary — A: `status: ok`, `shapes: N/N`, `bench_fails: K` (a `bench_fail` on a big shape
  like `square.4096` is expected, an 8 s bench-wall guard); B: `points: neighbor-bench done: <done>/<total> points
  (<new> new this run)` plus `ledger: fetched -> …` — **followed by the merge receipt** and `status: COMPLETE
  (<mode> + merge done)`. The rented card appears as its own line in `node rows per card now: …`; cards already
  present are unchanged. Merge is folded in, so there is **no separate merge step to launch** — the local DB is
  already updated.
- **failure** → `status: FAILED (<phase>)` or `merge FAILED: …` plus the last 40 lines of the relevant remote log
  (`setup.log` / `tune.log` / `neighbors.log` in that dir). Fix and re-run — the script is idempotent (rsync + `make
  setup` no-op when already done; the merge is safe to repeat — a re-merged snapshot's rows are equal-timestamped, so
  they never churn values; a re-run of B resumes from the ledger). `--no-merge` runs the collection only.
- B finishing with `<done>` well short of `<total>` is **normal** — the dist-2 pool is a long-horizon dataset; the
  4-hour budget takes a near-uniform sample of it and later runs keep going. Don't extend the run past the budget
  without asking the user.

**Re-merge / manual fallback.** If the merge step failed (or you used `--no-merge`), merge the rented card's rows without
re-tuning: `./venv/bin/python scripts/merge_node_db.py --remote "<user@host>" [--ssh-key … --port …]` (or `--src
<snapshot.db>` for an already-fetched file; `--db <path>` to override the destination). Same `node`-only, per-kind
upsert semantics.

**Manual debugging** (only if the script fails and you need to poke the box): the harness shell is zsh, so pass ssh
options as an **array** — `SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -i
"$HOME/.ssh/id_ed25519")` (a plain string var fails: zsh doesn't word-split it). Then `ssh "${SSH_OPTS[@]}" "$REMOTE"
'tail -n 40 ~/.local/share/emmy/node-tune/tune.log'` (phase B logs to `neighbors.log` in the same dir); check
liveness with `pgrep -f "[e]mmy tune"` for phase A or `pgrep -f "[g]olden_neighbor_bench"` for phase B (bracket
trick — a plain pattern self-matches the poll's own argv); never run a multi-minute loop inside one ssh session.

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

If the user wants to keep the box for more tuning, leave it up and report the SSH target + the exact teardown command.

## Verification checklist

Before reporting success:

- [ ] `vm create gpu` exited 0 and an SSH target was captured.
- [ ] Phase A ended with `status: COMPLETE (tune + merge done)`, `shapes: N/N` (not `FAILED`).
- [ ] Phase B ended with `status: COMPLETE (neighbors + merge done)`, a `neighbor-bench done: <done>/<total> points`
      line, and `ledger: fetched -> …` (a short `<done>` count is fine — the budget samples the pool, it doesn't
      finish it).
- [ ] Each merge receipt shows the rented card as its own per-card line; counts for other cards are unchanged.
- [ ] `eval prior --dataset nodes` shows a block for the rented GPU.
- [ ] The VM was deleted (or the user explicitly chose to keep it, and has the teardown command).

If any check fails, report the failure + raw output instead of claiming success.

## Common mistakes to avoid

- **Don't keep per-GPU DB files.** The node key includes `gpu`, so the single `~/.cache/emmy/autotune.db` is the
  correct cross-hardware accumulator. Splitting it defeats the design and breaks the per-card `eval` views.
- **Don't `scp` over the local DB / merge the whole DB.** That clobbers other cards' node rows (and the unrelated
  `perf`/`cuda_op`/`lowering` tables). Use `scripts/merge_node_db.py` — the per-kind upsert, `node` table only.
- **Don't forget the `nvcc` PATH/CUDA_HOME export** in both `make setup` and the tune invocation — without it kernel
  compiles hard-fail (there is no NVRTC fallback).
- **Don't set `HF_TOKEN` or expect model downloads** — the golden tune is pure torch snippets.
- **Don't rent more than 1 GPU** — node data is per-card; extra GPUs just burn money.
- **Don't auto-delete the VM** — confirm teardown with the user first (and never modify a CloudRift server beyond the
  tune we explicitly started).
- **Don't hand-roll the remote setup/poll loop** — `scripts/remote_node_tune.py` (Step 2) already handles it correctly:
  argv-list ssh (zsh doesn't word-split a string var), the `[e]mmy tune` bracket-pgrep (a plain pattern self-matches
  the poll's own argv and reports the tune alive forever), one short ssh per poll, and venv/dev install. Only drop to
  manual ssh for debugging — and then keep the same precautions (array ssh opts; never name a var `status`/`path`, which
  are read-only in zsh).
