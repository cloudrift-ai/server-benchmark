"""Stage local repo files to a remote VM via git ls-files + tar over SSH.

Used by command-workload recipes that declare a non-empty `command.stage` list.
The local repo's `git ls-files --cached --others --exclude-standard <paths>`
output is the source of truth — tracked + untracked files are included, but
gitignored files are excluded. This lets a user iterate on a script and run
it on the remote without committing first.
"""

import asyncio
import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path

from emmy.provisioning.ssh_transport import ssh_base_args

logger = logging.getLogger(__name__)


def _source_id(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


async def enumerate_staged_files(repo_root: Path, stage_paths: list[str]) -> list[str]:
    """Run `git ls-files --cached --others --exclude-standard -- <paths>`.

    Returns a sorted, deduplicated list of repo-relative file paths.
    Empty `stage_paths` returns an empty list.
    """
    if not stage_paths:
        return []
    args = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *stage_paths,
    ]
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(repo_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {stderr.decode().strip()}")
    files = sorted({line for line in stdout.decode().splitlines() if line})
    return files


def build_stage_tar(repo_root: Path, files: list[str]) -> bytes:
    """Build an in-memory gzipped tar containing the given repo-relative files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in files:
            full = repo_root / rel
            if not full.exists():
                continue
            tar.add(str(full), arcname=rel)
    return buf.getvalue()


async def _git_output(repo_root: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(repo_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.decode().strip()}")
    return stdout.decode().rstrip()


async def build_stage_manifest(repo_root: Path, stage_paths: list[str], files: list[str] | None = None) -> dict:
    """Describe the exact content staged for a command workload."""
    candidates = files if files is not None else await enumerate_staged_files(repo_root, stage_paths)
    selected = [relative for relative in candidates if (repo_root / relative).exists()]
    digests = {relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest() for relative in selected}
    revision = await _git_output(repo_root, "rev-parse", "HEAD")
    status = await _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all", "--", *stage_paths)
    dirty = [line for line in status.splitlines() if line]
    return {
        "schema_version": 1,
        "source_id": _source_id(digests),
        "git_revision": revision,
        "clean": not dirty,
        "dirty": dirty,
        "stage_paths": list(stage_paths),
        "files": digests,
    }


async def stage_to_remote(
    repo_root: Path,
    stage_paths: list[str],
    server: str,
    ssh_key: str,
    ssh_port: int,
    remote_dir: str,
    dry_run: bool = False,
    require_clean: bool = False,
) -> dict | None:
    """Stream a tar of staged files into `remote_dir` on the remote VM.

    No-op when `stage_paths` is empty.
    """
    if not stage_paths:
        return None

    files = await enumerate_staged_files(repo_root, stage_paths)
    if not files:
        logger.warning(f"stage_to_remote: no files matched {stage_paths}")
        return None
    manifest = await build_stage_manifest(repo_root, stage_paths, files)
    if require_clean and not manifest["clean"]:
        raise RuntimeError(f"command staging requires a clean source tree: {manifest['dirty']}")

    if dry_run:
        logger.info(f"[dry-run] stage {len(files)} files ({manifest['source_id']}) to {server}:{remote_dir}")
        return manifest

    tar_bytes = build_stage_tar(repo_root, files)

    ssh_args = ssh_base_args(server, ssh_key, ssh_port)
    # Clean before extract: remove stale files from previous runs,
    # then extract fresh snapshot. Preserves venv/ (not in the tar).
    ssh_args.append(
        f"mkdir -p {remote_dir}"
        f" && find {remote_dir} -maxdepth 1 -mindepth 1 -not -name venv -exec rm -rf {{}} +"
        f" && tar xzf - -C {remote_dir}"
    )

    proc = await asyncio.create_subprocess_exec(
        *ssh_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate(input=tar_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"stage_to_remote failed (rc={proc.returncode}): {stderr.decode().strip()}")
    logger.info(f"Staged {len(files)} files ({manifest['source_id']}) to {server}:{remote_dir}")
    return manifest
