#!/usr/bin/env python
"""Drive a node-data collection sweep on a remote GPU host, then wait for it — the
token-heavy middle of the ``collect-node-data`` skill, extracted so the agent makes
ONE backgrounded tool call instead of ~20 verbose ssh polls.

Given an SSH target it: ensures the Python 3.12 venv/dev packages + ``nvcc``, rsyncs
the local working tree, runs ``make setup`` (output to a remote logfile — only a tail
comes back on failure), launches the collection detached, then **polls the remote log
internally** until it finishes — a run that outlives ``--timeout`` is STOPPED and
harvested, never abandoned — merges the node rows home, and backs up the local DB.
All the per-poll ssh chatter happens inside this process, so it never reaches the
agent's context.

The collection is ``scripts/golden_neighbor_bench.py`` — the budgeted three-slice sweep
(own-golden neighborhood / cross-card golden exchange / uniform tail, kind-stratified
within each slice) of every golden
kind's candidate pool, paired -O1/-O3, under a wall-time budget (``--budget-s``; the
wait ``--timeout`` derives from it, see the margin note on ``_WAIT_MARGIN``). It
replaced the ε-greedy ``emmy tune --dataset golden`` mode:
search-driven collection over-samples the branches the incumbent prior already likes
and its wall time grows with the golden set, while the sweep samples the enumeration at
a fixed budget (``emmy tune --explore-eps`` itself survives for interactive use).
Resume state lives in a ledger JSON: the local copy (``--ledger``) is pushed to the box
before the run and fetched back after, so successive runs — on this box or a freshly
rented one of the same card — keep covering new points until the pool is exhausted.

Run it from the agent in the BACKGROUND (Bash ``run_in_background: true``) — a run
takes hours, well past a foreground tool timeout; the harness re-invokes the agent
with just the final summary when this exits.

    ./venv/bin/python scripts/remote_node_collect.py --remote user@host \
        [--ssh-key ~/.ssh/id_ed25519] [--port 57006] [--repo /path/to/emmy] \
        [--poll 60] [--budget-s 14400] [--timeout SEC] [--filter SUB]

Then (separately, if ``--no-merge``) merge the node rows home with
``scripts/merge_node_db.py --remote``.

Robustness baked in (the four traps the first run hit): all ssh goes through argv
lists (no shell word-splitting / zsh quirks); liveness uses the
``[g]olden_neighbor_bench`` bracket-pgrep so it never self-matches; each poll is its
own short ssh (no long-held session that broken-pipes); and venv/dev are always
installed before ``make setup``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from emmy.provisioning.ssh_transport import REMOTE_DEPLOY_DIR

# Matches emmy/provisioning/ssh_transport.py::ssh_base_args.
_SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=20",
]

_CUDA_EXPORT = "export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda"
# Nest under the repo's established remote layout (REMOTE_DEPLOY_DIR = ~/.local/share/emmy) —
# the same base the deploy/bench paths use (they already clone a repo under it).
_BASE = f"{REMOTE_DEPLOY_DIR}/node-collect"
_REMOTE_DIR = f"{_BASE}/repo"  # rsync target — the emmy checkout the sweep runs from
_SETUP_LOG = f"{_BASE}/setup.log"
_NEIGHBOR_LOG = f"{_BASE}/neighbors.log"
_REMOTE_LEDGER = f"{_BASE}/neighbor_ledger.json"
_REMOTE_RECEIPTS = f"{_BASE}/receipts"
# The driver's final summary line: ``neighbor-bench done: <done>/<total> points (<new> new this run)``.
# The poll's ``grep -oE`` extracts only the prefix up to ``points`` — match that.
_NEIGHBOR_DONE_RE = re.compile(r"neighbor-bench done: (\d+)/(\d+) points")

# The sweep can outlive ``--budget-s`` by up to TWO in-flight ``emmy run`` invocations
# (its ``--run-timeout``, 1800 s each — one budget check guards an O1+O3 pair), so the
# derived wait extends that far past the budget. Reaching --timeout is not a failure —
# the driver stops the sweep and harvests what was measured — so a thinner wait only
# kills the in-flight batch early (those points retry next run); letting it finish is
# the better default.
_WAIT_MARGIN = 2 * 1800 + 300  # worst-case in-flight overshoot + launch/poll slack


def _opts(ssh_key: str | None, port: int | None) -> list[str]:
    out = list(_SSH_OPTS)
    if ssh_key:
        out += ["-i", ssh_key]
    if port:
        out += ["-p", str(port)]  # rsync/ssh use -p (scp would use -P, not relevant here)
    return out


def _run(remote: str, ssh_key: str | None, port: int | None, command: str, *, timeout: int = 600, detach: bool = False) -> tuple[int, str]:
    """Run one remote command over a fresh ssh connection; return (rc, combined output).

    ``detach=True`` adds ssh ``-n`` (stdin from /dev/null) — needed when launching a
    remote background process: without it (plus a ``< /dev/null`` redirect on the
    command itself) ssh holds the channel open past a ``nohup … &``, so this call
    times out (rc 124) long *after* the process already started successfully."""
    argv = ["ssh", *(["-n"] if detach else []), *_opts(ssh_key, port), remote, command]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "(ssh command timed out)"


def _log(msg: str) -> None:
    print(f"[remote_node_collect] {msg}", flush=True)


def _fail(remote: str, ssh_key: str | None, port: int | None, why: str, logfile: str) -> int:
    _, tail = _run(remote, ssh_key, port, f"tail -n 40 {logfile} 2>/dev/null")
    print("\n=== remote_node_collect summary ===", flush=True)
    print(f"status: FAILED ({why})", flush=True)
    print(f"--- last 40 lines of {remote}:{logfile} ---", flush=True)
    print(tail.strip() or "(empty / not found)", flush=True)
    return 1


def _backup_local_db(keep: int = 5) -> Path | None:
    """Snapshot the local autotune DB (the SOLE copy of rented-GPU node data) to a
    timestamped file under ``~/.cache/emmy/backups/``, pruning to the newest ``keep``.
    ``VACUUM INTO`` is the WAL-safe copy (same pattern as the remote snapshot fetch).
    Best-effort: a backup failure warns and returns None — the merge already landed."""
    import sqlite3  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    from emmy.commands.compile import resolve_tune_db  # noqa: PLC0415

    try:
        src = resolve_tune_db()
        if not Path(src).exists():
            _log(f"backup skipped: {src} does not exist")
            return None
        backups = Path.home() / ".cache/emmy/backups"
        backups.mkdir(parents=True, exist_ok=True)
        dst = backups / f"autotune-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
        with sqlite3.connect(src) as conn:
            conn.execute("VACUUM INTO ?", (str(dst),))
        for old in sorted(backups.glob("autotune-*.db"))[:-keep]:
            old.unlink()
        return dst
    except Exception as exc:  # noqa: BLE001 — never fail the run over a backup
        _log(f"backup FAILED: {exc!r}")
        return None


def _stop_sweep(remote: str, ssh_key: str | None, port: int | None) -> bool:
    """Kill the detached sweep (and its in-flight ``emmy run`` bench child) so the harvest
    reads a quiet node store. The killed batch's points stay non-terminal in the ledger and
    simply retry next run. Returns True once nothing sweep-related remains."""
    for attempt in range(6):
        sig = "-9 " if attempt >= 2 else ""
        kill = f"pkill {sig}-f '[g]olden_neighbor_bench' 2>/dev/null; pkill {sig}-f '[e]mmy run --code' 2>/dev/null; true"
        _run(remote, ssh_key, port, kill)
        time.sleep(5)
        _, out = _run(
            remote,
            ssh_key,
            port,
            "pgrep -f '[g]olden_neighbor_bench' >/dev/null 2>&1 || pgrep -f '[e]mmy run --code' >/dev/null 2>&1 && echo ALIVE || echo DEAD",
        )
        if "DEAD" in out:
            return True
        _log(f"sweep still alive after kill (attempt {attempt + 1}/6) — retrying")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", required=True, help="SSH target, user@host")
    ap.add_argument("--ssh-key", help="SSH private key (default: ssh's own default)")
    ap.add_argument("--port", type=int, help="SSH port (default 22)")
    ap.add_argument("--repo", help="Local repo root to rsync (default: git toplevel)")
    ap.add_argument("--poll", type=int, default=60, help="Seconds between completion polls (default 60)")
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Seconds after launch before the sweep is stopped and the partial run harvested "
        "(default: --budget-s + 3900 — two in-flight 1800 s invocations + slack)",
    )
    ap.add_argument("--no-merge", action="store_true", help="Skip the folded-in node-row merge after a successful run")
    ap.add_argument(
        "--budget-s",
        type=float,
        default=None,
        help="Sweep wall-time budget in seconds (default 3600, or --timeout minus the 3900 s margin when only --timeout is given)",
    )
    ap.add_argument("--max-dist", type=int, default=2, help="Max knob-component distance from a golden anchor (default 2)")
    ap.add_argument("--batch", type=int, default=6, help="Pinned rows per emmy run invocation (default 6)")
    ap.add_argument("--share-own", type=float, default=None, help="Budget share for the own-neighborhood slice (default: sweep's own, 60)")
    ap.add_argument("--share-cross", type=float, default=None, help="Budget share for the cross-card slice (default: sweep's own, 25)")
    ap.add_argument("--share-tail", type=float, default=None, help="Budget share for the uniform-tail slice (default: sweep's own, 15)")
    ap.add_argument("--tail-cap", type=int, default=None, help="Max tail points per shape (default: sweep's own, 200)")
    ap.add_argument("--filter", help="Only shapes with a golden name containing this substring")
    ap.add_argument(
        "--ledger",
        default=str(Path.home() / ".cache/emmy/neighbor_bench/ledger.json"),
        help="LOCAL resume-ledger path — pushed to the box before the run, fetched back after",
    )
    args = ap.parse_args()

    # Resolve the budget/timeout pair so the wait outlives the sweep (see _WAIT_MARGIN):
    # either may be given and the other derives; both given only warns, it never shrinks
    # an explicit budget.
    if args.budget_s is None:
        args.budget_s = max(600, (args.timeout or 7500) - _WAIT_MARGIN)
    if args.timeout is None:
        args.timeout = int(args.budget_s + _WAIT_MARGIN)
    elif args.timeout < args.budget_s + _WAIT_MARGIN:
        _log(
            f"WARNING: --timeout {args.timeout}s leaves <{_WAIT_MARGIN}s past the {args.budget_s:.0f}s budget — "
            "an in-flight batch may get killed at the wait cap and re-benched next run"
        )

    remote, key, port = args.remote, args.ssh_key, args.port
    repo = args.repo
    if not repo:
        gp = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if gp.returncode != 0:
            print("error: not inside a git repo and --repo not given; pass --repo <path>", flush=True)
            return 2
        repo = gp.stdout.strip()

    # 1. make the remote base dir (so the logs + rsync target exist), then deps: always
    #    install the venv/dev packages (the image ships 3.12 without them); install the CUDA
    #    toolkit only if nvcc is genuinely absent. All noisy output → the setup log.
    _log(f"ensuring deps on {remote} (apt; output to {_SETUP_LOG}) ...")
    deps = (
        f"mkdir -p {_BASE}; "
        f"sudo apt-get update -qq >> {_SETUP_LOG} 2>&1; "
        f"sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >> {_SETUP_LOG} 2>&1; "
        "if ! command -v nvcc >/dev/null 2>&1 && ! ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; then "
        f"sudo apt-get install -y -qq cuda-toolkit-12-9 >> {_SETUP_LOG} 2>&1; fi; "
        "if command -v nvcc >/dev/null 2>&1 || ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; "
        "then echo NVCC_OK; else echo NVCC_MISSING; fi"
    )
    rc, out = _run(remote, key, port, deps, timeout=1800)
    if rc != 0 or "NVCC_OK" not in out:
        # rc==0 with NVCC_OK absent means the apt steps ran but nvcc is still missing
        # (the chain ends in an echo, so rc reflects only that last command); rc!=0 means
        # the ssh/apt command itself failed. The setup-log tail below carries the detail.
        reason = "nvcc not found after toolkit install" if rc == 0 else f"ssh/apt rc={rc}"
        return _fail(remote, key, port, f"deps/nvcc ({reason})", _SETUP_LOG)

    # 2. rsync the working tree (exact local code, incl. uncommitted changes).
    _log(f"rsyncing {repo} -> {remote}:{_REMOTE_DIR} ...")
    e = "ssh " + " ".join(_opts(key, port))
    rsync = [
        "rsync",
        "-az",
        "-e",
        e,
        "--exclude",
        "venv",
        "--exclude",
        ".git",
        "--exclude",
        "recipes",
        "--exclude",
        "_tune",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.pyc",
        f"{repo}/",
        f"{remote}:{_REMOTE_DIR}/",
    ]
    rp = subprocess.run(rsync, capture_output=True, text=True)
    if rp.returncode != 0:
        _log(f"rsync failed: {(rp.stderr or rp.stdout).strip()}")
        return 1

    # 3. make setup (output → the setup log); on failure rebuild a clean venv once.
    _log(f"running make setup (output to {_SETUP_LOG}) ...")
    setup = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && make setup >> {_SETUP_LOG} 2>&1 && echo SETUP_OK || echo SETUP_FAIL"
    rc, out = _run(remote, key, port, setup, timeout=2400)
    if "SETUP_OK" not in out:
        _log("make setup failed; wiping venv and retrying once ...")
        setup2 = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && rm -rf venv && make setup >> {_SETUP_LOG} 2>&1 && echo SETUP_OK || echo SETUP_FAIL"
        rc, out = _run(remote, key, port, setup2, timeout=2400)
        if "SETUP_OK" not in out:
            return _fail(remote, key, port, "make setup", _SETUP_LOG)

    # 4. build the sweep's launch command, then launch it detached. Redirect ALL three
    #    std streams away from the ssh channel (`< /dev/null` + `> <log> 2>&1`) and use
    #    ssh -n (detach=True), else ssh holds the channel open past the `&` and this
    #    call times out *after* a successful launch. The bracket-pgrep pattern stops
    #    the liveness check from matching the poll command's own argv.
    # Push the local resume ledger (if any) so the run continues prior coverage.
    local_ledger = Path(args.ledger).expanduser()
    if local_ledger.exists():
        _log(f"pushing resume ledger {local_ledger} -> {remote}:{_REMOTE_LEDGER} ...")
        e = "ssh " + " ".join(_opts(key, port))
        lp = subprocess.run(["rsync", "-az", "-e", e, str(local_ledger), f"{remote}:{_REMOTE_LEDGER}"], capture_output=True, text=True)
        if lp.returncode != 0:
            _log(f"ledger push failed ({(lp.stderr or lp.stdout).strip()}) — starting without resume state")
    budget = args.budget_s
    cmd = (
        f"./venv/bin/python scripts/golden_neighbor_bench.py --ledger {_REMOTE_LEDGER} "
        f"--receipts {_REMOTE_RECEIPTS} --budget-s {budget:.0f} --max-dist {args.max_dist} --batch {args.batch}"
    )
    for flag, val in (
        ("--share-own", args.share_own),
        ("--share-cross", args.share_cross),
        ("--share-tail", args.share_tail),
        ("--tail-cap", args.tail_cap),
    ):
        if val is not None:
            cmd += f" {flag} {val}"
    if args.filter:
        cmd += f" --filter {args.filter}"
    log_file, proc_pat = _NEIGHBOR_LOG, "[g]olden_neighbor_bench"
    done_grep, done_re = "neighbor-bench done: [0-9]+/[0-9]+ points", _NEIGHBOR_DONE_RE
    _log(f"launching `{cmd}` (detached -> {log_file}) ...")
    launch = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && nohup {cmd} > {log_file} 2>&1 < /dev/null & echo run_launched"
    rc, out = _run(remote, key, port, launch, timeout=60, detach=True)
    if "run_launched" not in out:
        # Belt-and-suspenders: if the launch ssh still didn't confirm, the run may
        # have started anyway — verify the process before declaring failure.
        time.sleep(5)
        _, chk = _run(remote, key, port, f"pgrep -f '{proc_pat}' >/dev/null 2>&1 && echo ALIVE || echo DEAD")
        if "ALIVE" not in chk:
            return _fail(remote, key, port, f"launch (rc={rc})", log_file)
        _log("launch ssh did not confirm but the process is alive — continuing")

    # 5. poll the remote log internally until done / dead / --timeout. Hitting --timeout
    #    is NOT a failure: it stops the sweep and falls through to the same harvest.
    poll = (
        f"echo \"DONE_MARK=$(grep -aoE '{done_grep}' {log_file} 2>/dev/null | tail -1)\"; "
        f"pgrep -f '{proc_pat}' >/dev/null 2>&1 && echo PROC=ALIVE || echo PROC=DEAD"
    )
    start = time.monotonic()
    dead_streak = 0
    stopped = False  # True → the driver stopped the sweep at --timeout (partial harvest)
    while True:
        if time.monotonic() - start > args.timeout:
            # --timeout is a harvest trigger, not an abandon: rented-GPU data must never
            # be stranded behind a wait cap. Stop the sweep, then collect what it made.
            _log(f"--timeout {args.timeout}s reached — stopping the sweep, then harvesting the partial run")
            if not _stop_sweep(remote, key, port):
                return _fail(remote, key, port, f"timeout after {args.timeout}s and the sweep would not stop", log_file)
            stopped = True
            break
        time.sleep(args.poll)
        rc, out = _run(remote, key, port, poll)
        # The poll always echoes a ``PROC=`` marker when it actually ran on the remote;
        # its absence means the ssh itself failed (timeout / connection reset), which is
        # NOT evidence the run died — only a confirmed ``PROC=DEAD`` is.
        reachable = "PROC=" in out
        done_mark = next((ln[len("DONE_MARK=") :] for ln in out.splitlines() if ln.startswith("DONE_MARK=")), "")
        alive = "PROC=ALIVE" in out
        if done_re.search(done_mark):
            break
        if not reachable:
            # Transient ssh/network failure this cycle — keep waiting (still bounded by
            # --timeout); don't mistake an unreachable host for a crashed run.
            _log(f"poll ssh failed (rc={rc}) — host unreachable this cycle, retrying")
            continue
        if not alive:
            dead_streak += 1
            if dead_streak >= 2:  # two consecutive confirmed DEADs without a done marker ⇒ it crashed
                return _fail(remote, key, port, "sweep process died without a done marker", log_file)
        else:
            dead_streak = 0
            _log(f"collecting... ({int(time.monotonic() - start)}s elapsed)")

    # 6. harvest — identical for a finished sweep and one stopped at --timeout: the ledger
    #    and remote node store hold every terminal point either way.
    elapsed = int(time.monotonic() - start)
    _, bf = _run(remote, key, port, f"grep -aEc 'bench_fail|bench worker exceeded' {log_file} 2>/dev/null || echo 0")
    bench_fails = (bf.strip().splitlines() or ["0"])[0]
    print("\n=== remote_node_collect summary ===", flush=True)
    print(f"status: ok{' (sweep stopped at --timeout — partial run harvested)' if stopped else ''}", flush=True)
    _, done_line = _run(remote, key, port, f"grep -a 'neighbor-bench done:' {log_file} 2>/dev/null | tail -1")
    print(f"points: {done_line.strip() or '(sweep stopped before its done line — coverage lives in the fetched ledger)'}", flush=True)
    # Fetch the resume ledger home so the next run (any box, same card)
    # continues the coverage instead of redoing it. Best-effort: a lost
    # ledger only costs re-benched points (the node store upsert dedups).
    local_ledger = Path(args.ledger).expanduser()
    local_ledger.parent.mkdir(parents=True, exist_ok=True)
    e = "ssh " + " ".join(_opts(key, port))
    fp = subprocess.run(["rsync", "-az", "-e", e, f"{remote}:{_REMOTE_LEDGER}", str(local_ledger)], capture_output=True, text=True)
    if fp.returncode == 0:
        print(f"ledger: fetched -> {local_ledger}", flush=True)
    else:
        print(f"ledger: fetch FAILED ({(fp.stderr or fp.stdout).strip()}) — next run redoes this run's points", flush=True)
    print(f"bench_fails: {bench_fails}", flush=True)
    print(f"elapsed (wait only): {elapsed}s", flush=True)
    print(f"remote log: {remote}:{log_file}", flush=True)
    manual = (
        f"./venv/bin/python scripts/merge_node_db.py --remote {remote}"
        + (f" --ssh-key {key}" if key else "")
        + (f" --port {port}" if port else "")
    )
    if args.no_merge:
        print(f"merge: skipped (--no-merge); run `{manual}`", flush=True)
        return 0
    # Fold the merge in so the single backgrounded call ends with the node rows
    # already in the local DB and the per-card receipt printed — no separate step
    # for the caller to remember to launch or to report on.
    print("\n--- merging node rows into the local DB ---", flush=True)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import merge_node_db  # noqa: PLC0415  (sibling script in scripts/)

        merged = merge_node_db.fetch_and_merge(remote, ssh_key=key, port=port)
    except Exception as exc:  # noqa: BLE001 — report any merge failure, don't crash the run
        print(f"merge FAILED: {exc!r}; the collected data is safe on {remote} — re-run `{manual}`", flush=True)
        return 1
    if merged == 0:
        # The run finished but nothing came home. After a --timeout stop that can be
        # legitimate (nothing measured yet); otherwise the remote wrote its node store
        # somewhere other than the snapshot's default path (e.g. a remote EMMY_TUNE_DB),
        # so don't claim success.
        if stopped:
            print("merge brought back 0 node rows — the stopped sweep had not recorded any measurement yet", flush=True)
            return 0
        print(
            f"merge brought back 0 node rows — the remote node store wasn't at the expected "
            f"path; the collected data is on {remote}. Check the remote EMMY_TUNE_DB and re-run "
            f"`{manual} --remote-db <path>`.",
            flush=True,
        )
        return 1
    # The local DB is now the sole copy of everything the rental paid for —
    # snapshot it so a cache wipe can't erase the data.
    backup = _backup_local_db()
    if backup is not None:
        print(f"backup: {backup}", flush=True)
    print(f"\nstatus: COMPLETE (sweep {'stopped at --timeout' if stopped else 'done'} + merge done)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
