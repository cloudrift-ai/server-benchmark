#!/usr/bin/env python
"""Drive a node-data collection sweep on a remote GPU host, then wait for it — the
token-heavy middle of the ``collect-node-data`` skill, extracted so the agent makes
ONE backgrounded tool call instead of ~20 verbose ssh polls.

Given an SSH target (a freshly rented box or any existing server the user provides) it:
rsyncs the local working tree, ensures the Python 3.12 venv/dev packages + ``nvcc`` and
runs ``make setup`` (inside the ``emmy-node-setup`` tmux session; output to a remote
logfile — only a tail comes back on failure), launches the collection inside the
``emmy-node-sweep`` tmux session, then **polls the remote log internally** until it
finishes — a run that outlives ``--timeout`` is STOPPED and harvested, never abandoned —
merges the node rows home, and backs up the local DB. All the per-poll ssh chatter
happens inside this process, so it never reaches the agent's context.

Every long-running remote process lives in a named tmux session, so a dropped ssh
connection never kills it and a human can watch or debug with
``tmux attach -t emmy-node-sweep`` (or ``emmy-node-setup``) on the box.

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

Robustness baked in (the traps earlier runs hit): all ssh goes through argv lists (no
shell word-splitting / zsh quirks); liveness reads ``tmux has-session`` (no pgrep
self-match trap, no ssh channel held open across a ``nohup … &``); each poll is its own
short ssh (no long-held session that broken-pipes); venv/dev are always installed
before ``make setup``; and a launch refuses to start when a previous sweep session is
still alive on the box (relevant for user-provided servers).
"""

from __future__ import annotations

import argparse
import re
import shlex
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
_SETUP_STATUS = f"{_BASE}/setup.status"  # one status word (SETUP_OK / SETUP_FAIL / NVCC_MISSING) per setup attempt
_NEIGHBOR_LOG = f"{_BASE}/neighbors.log"
_REMOTE_LEDGER = f"{_BASE}/neighbor_ledger.json"
_REMOTE_RECEIPTS = f"{_BASE}/receipts"
# Every long-running remote process runs inside one of these named tmux sessions —
# attachable for debugging, and immune to the launching ssh connection dropping.
_SETUP_SESSION = "emmy-node-setup"
_SWEEP_SESSION = "emmy-node-sweep"
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


def _run(remote: str, ssh_key: str | None, port: int | None, command: str, *, timeout: int = 600) -> tuple[int, str]:
    """Run one remote command over a fresh ssh connection; return (rc, combined output).

    Long-running work never goes through here directly — it is launched into a tmux
    session (``_tmux_launch``), so no call holds an ssh channel open for the duration
    of a build or sweep."""
    argv = ["ssh", *_opts(ssh_key, port), remote, command]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "(ssh command timed out)"


def _log(msg: str) -> None:
    print(f"[remote_node_collect] {msg}", flush=True)


def _tmux_launch(remote: str, ssh_key: str | None, port: int | None, session: str, body: str, *, pre: str = "") -> bool:
    """Start ``body`` inside a fresh detached tmux session named ``session`` (killing any
    stale same-named session first). ``tmux new-session -d`` returns immediately, so the
    launching ssh never holds a channel open — the old ``nohup``/``ssh -n`` detach dance
    is gone. Returns True once the launch is confirmed (echo, or the session visible)."""
    launch = f"{pre}tmux kill-session -t {session} 2>/dev/null; tmux new-session -d -s {session} {shlex.quote(body)} && echo tmux_launched"
    _, out = _run(remote, ssh_key, port, launch, timeout=60)
    if "tmux_launched" in out:
        return True
    # Belt-and-suspenders: the launch ssh may fail to report after the session started.
    time.sleep(5)
    _, chk = _run(remote, ssh_key, port, f"tmux has-session -t {session} 2>/dev/null && echo ALIVE || echo DEAD")
    if "ALIVE" in chk:
        _log("launch ssh did not confirm but the tmux session is alive — continuing")
        return True
    return False


def _run_setup_session(remote: str, ssh_key: str | None, port: int | None, *, wipe_venv: bool) -> str:
    """Run deps + ``make setup`` inside the ``emmy-node-setup`` tmux session and wait for
    the status word the session writes to ``_SETUP_STATUS`` (``SETUP_OK`` / ``SETUP_FAIL``
    / ``NVCC_MISSING``). Returns that word, or '' when the session died without writing
    one (or the wait timed out). Always (re)installs the venv/dev packages; installs the
    CUDA toolkit only if ``nvcc`` is genuinely absent. All noisy output → the setup log."""
    nvcc_probe = "command -v nvcc >/dev/null 2>&1 || ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1"
    body = (
        f"sudo apt-get update -qq >> {_SETUP_LOG} 2>&1; "
        f"sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >> {_SETUP_LOG} 2>&1; "
        f"if ! ({nvcc_probe}); then sudo apt-get install -y -qq cuda-toolkit-12-9 >> {_SETUP_LOG} 2>&1; fi; "
        f"if ! ({nvcc_probe}); then echo NVCC_MISSING > {_SETUP_STATUS}; exit 0; fi; "
        f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && "
        + ("rm -rf venv && " if wipe_venv else "")
        + f"make setup >> {_SETUP_LOG} 2>&1 && echo SETUP_OK > {_SETUP_STATUS} || echo SETUP_FAIL > {_SETUP_STATUS}"
    )
    if not _tmux_launch(remote, ssh_key, port, _SETUP_SESSION, body, pre=f"rm -f {_SETUP_STATUS}; "):
        return ""
    deadline = time.monotonic() + 4500  # apt (incl. a possible CUDA toolkit install) + make setup
    dead_streak = 0
    while time.monotonic() < deadline:
        time.sleep(20)
        _, out = _run(
            remote,
            ssh_key,
            port,
            f"cat {_SETUP_STATUS} 2>/dev/null; tmux has-session -t {_SETUP_SESSION} 2>/dev/null && echo SESS=ALIVE || echo SESS=DEAD",
        )
        word = next((w for w in ("SETUP_OK", "SETUP_FAIL", "NVCC_MISSING") if w in out), "")
        if word:
            return word
        if "SESS=" not in out:
            continue  # ssh blip this cycle — not evidence the session died
        if "SESS=DEAD" in out:
            dead_streak += 1
            if dead_streak >= 2:  # two confirmed DEADs without a status word ⇒ the session crashed
                return ""
        else:
            dead_streak = 0
    return ""


def _fail(remote: str, ssh_key: str | None, port: int | None, why: str, logfile: str, *, harvest_ledger: str | None = None) -> int:
    """Print the FAILED summary (+ remote log tail) and return 1. ``harvest_ledger``
    (the local ledger path) is passed by post-launch failure paths: the sweep already
    spent rented-GPU time, so its coverage is harvested best-effort even on failure —
    the remote ledger is fetched (else the next run would push the stale local copy
    over it, re-benching everything) and the node rows merged home."""
    _, tail = _run(remote, ssh_key, port, f"tail -n 40 {logfile} 2>/dev/null")
    print("\n=== remote_node_collect summary ===", flush=True)
    print(f"status: FAILED ({why})", flush=True)
    print(f"--- last 40 lines of {remote}:{logfile} ---", flush=True)
    print(tail.strip() or "(empty / not found)", flush=True)
    if harvest_ledger:
        local_ledger = Path(harvest_ledger).expanduser()
        local_ledger.parent.mkdir(parents=True, exist_ok=True)
        e = "ssh " + " ".join(_opts(ssh_key, port))
        fp = subprocess.run(["rsync", "-az", "-e", e, f"{remote}:{_REMOTE_LEDGER}", str(local_ledger)], capture_output=True, text=True)
        if fp.returncode == 0:
            print(f"ledger: fetched -> {local_ledger} (despite the failure — coverage is preserved)", flush=True)
        else:
            print(f"ledger: fetch FAILED ({(fp.stderr or fp.stdout).strip()}) — next run redoes this run's points", flush=True)
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import merge_node_db  # noqa: PLC0415  (sibling script in scripts/)

            merged = merge_node_db.fetch_and_merge(remote, ssh_key=ssh_key, port=port)
            print(f"merge (best-effort): {merged} node rows brought home", flush=True)
        except Exception as exc:  # noqa: BLE001 — the harvest must never mask the original failure
            print(f"merge (best-effort) FAILED: {exc!r} — the collected data is still on {remote}", flush=True)
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
    """Kill the sweep's tmux session (and its in-flight ``emmy run`` bench child) so the
    harvest reads a quiet node store. The killed batch's points stay non-terminal in the
    ledger and simply retry next run. Returns True once nothing sweep-related remains."""
    for attempt in range(6):
        sig = "-9 " if attempt >= 2 else ""
        kill = (
            f"tmux kill-session -t {_SWEEP_SESSION} 2>/dev/null; "
            f"pkill {sig}-f '[g]olden_neighbor_bench' 2>/dev/null; pkill {sig}-f '[e]mmy run --code' 2>/dev/null; true"
        )
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

    # 1. bootstrap: the remote base dir (so the logs + rsync target exist) and tmux itself —
    #    the one dependency that can't be installed from inside a tmux session. Everything
    #    long-running after this point (apt deps, make setup, the sweep) runs inside tmux.
    _log(f"ensuring tmux on {remote} ...")
    boot = (
        f"mkdir -p {_BASE}; "
        "if ! command -v tmux >/dev/null 2>&1; then "
        f"sudo apt-get update -qq >> {_SETUP_LOG} 2>&1; sudo apt-get install -y -qq tmux >> {_SETUP_LOG} 2>&1; fi; "
        "command -v tmux >/dev/null 2>&1 && echo TMUX_OK || echo TMUX_MISSING"
    )
    rc, out = _run(remote, key, port, boot, timeout=900)
    if "TMUX_OK" not in out:
        return _fail(remote, key, port, f"tmux bootstrap (rc={rc})", _SETUP_LOG)
    # Refuse to pile a second sweep onto a box that is still running one (an existing /
    # user-provided server may have leftovers) — the ledger makes reruns cheap, collisions
    # on the node store mid-sweep are not.
    _, out = _run(remote, key, port, f"tmux has-session -t {_SWEEP_SESSION} 2>/dev/null && echo SWEEP=ALIVE || echo SWEEP=DEAD")
    if "SWEEP=ALIVE" in out:
        return _fail(
            remote,
            key,
            port,
            f"a sweep is already running in tmux session {_SWEEP_SESSION} on {remote} — "
            f"attach with `tmux attach -t {_SWEEP_SESSION}` or kill it first",
            _NEIGHBOR_LOG,
        )

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

    # 3. deps + make setup, inside the setup tmux session (a dropped ssh no longer kills a
    #    long apt/CUDA-toolkit install or venv build); on SETUP_FAIL rebuild a clean venv once.
    _log(f"running deps + make setup in tmux session {_SETUP_SESSION} (output to {_SETUP_LOG}) ...")
    word = _run_setup_session(remote, key, port, wipe_venv=False)
    if word != "SETUP_OK" and word != "NVCC_MISSING":
        _log("make setup failed; wiping venv and retrying once ...")
        word = _run_setup_session(remote, key, port, wipe_venv=True)
    if word == "NVCC_MISSING":
        return _fail(remote, key, port, "deps/nvcc (nvcc not found after toolkit install)", _SETUP_LOG)
    if word != "SETUP_OK":
        return _fail(remote, key, port, "make setup", _SETUP_LOG)

    # 4. build the sweep's launch command, then launch it inside the sweep tmux session
    #    (stdout/stderr → the log; the session outlives any ssh drop and is attachable).
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
        cmd += f" --filter {shlex.quote(args.filter)}"
    log_file = _NEIGHBOR_LOG
    done_grep, done_re = "neighbor-bench done: [0-9]+/[0-9]+ points", _NEIGHBOR_DONE_RE
    _log(f"launching `{cmd}` in tmux session {_SWEEP_SESSION} (log -> {log_file}) ...")
    body = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && {cmd} > {log_file} 2>&1"
    # rm the old log at launch: on a reused box a stale ``neighbor-bench done:`` line
    # would otherwise satisfy the poll's done-grep if the new session dies before
    # truncating the file.
    if not _tmux_launch(remote, key, port, _SWEEP_SESSION, body, pre=f"rm -f {log_file}; "):
        return _fail(remote, key, port, "launch (tmux session did not start)", log_file)

    # 5. poll the remote log internally until done / dead / --timeout. Hitting --timeout
    #    is NOT a failure: it stops the sweep and falls through to the same harvest.
    poll = (
        f"echo \"DONE_MARK=$(grep -aoE '{done_grep}' {log_file} 2>/dev/null | tail -1)\"; "
        f"tmux has-session -t {_SWEEP_SESSION} 2>/dev/null && echo PROC=ALIVE || echo PROC=DEAD"
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
                return _fail(
                    remote, key, port, f"timeout after {args.timeout}s and the sweep would not stop", log_file, harvest_ledger=args.ledger
                )
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
                return _fail(remote, key, port, "sweep process died without a done marker", log_file, harvest_ledger=args.ledger)
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
