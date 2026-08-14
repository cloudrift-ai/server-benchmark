"""Typed, system-only record for one experiment row."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from emmy.redact import redact_secrets
from emmy.system_info import SoftwareInformation, SystemInformation

if TYPE_CHECKING:
    from emmy.planner import BenchmarkTask


def _repo_relative(path: str) -> str:
    resolved = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def _builtins(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _builtins(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtins(item) for item in value]
    return value


def _redacted(value: Any) -> Any:
    if isinstance(value, dict):
        return {_redacted(key): _redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


@dataclass
class ExperimentRow:
    task_id: str
    row_id: str
    directory: str
    kind: str
    variant: str
    parameters: dict[str, Any]


@dataclass
class Provenance:
    emmy_code_sha256: str
    git_revision: str | None
    git_dirty: bool | None
    source: dict[str, Any] | None = None


@dataclass
class Infrastructure:
    group: str
    requested_gpu: str
    requested_gpu_count: int
    address: str
    ssh_port: int
    provider: str | None = None
    instance_id: str | None = None
    zone: str | None = None
    state: str = "active"
    deleted_at: str | None = None


@dataclass
class ExecutionError:
    stage: str
    message: str


@dataclass
class Execution:
    run_id: str
    stage: str = "queued"
    started_at: str | None = None
    completed_at: str | None = None
    timing_seconds: dict[str, float] = field(default_factory=dict)
    infrastructure: Infrastructure | None = None
    error: ExecutionError | None = None
    cleanup_error: str | None = None


@dataclass
class Artifact:
    kind: str
    path: str
    status: str | None = None
    exit_code: int | None = None


@dataclass
class ExperimentRecord:
    """Serializable schema for generic facts about one experiment-row execution."""

    SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int
    timestamp: str
    status: str
    experiment: ExperimentRow
    provenance: Provenance
    system: SystemInformation | None
    execution: Execution
    artifacts: list[Artifact] = field(default_factory=list)

    @staticmethod
    def utc_timestamp() -> str:
        """Return an RFC 3339 UTC timestamp."""
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @classmethod
    def new_run_id(cls, code_hash: str) -> str:
        """Return a readable invocation identifier shared by all selected rows."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{code_hash[:8]}"

    @staticmethod
    def _git_provenance() -> tuple[str | None, bool | None]:
        root = Path(__file__).resolve().parents[2]

        def git(*args: str) -> str | None:
            try:
                result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                return None
            return result.stdout.strip() if result.returncode == 0 else None

        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=no")
        return revision, bool(status) if status is not None else None

    @classmethod
    def create(cls, task: BenchmarkTask, run_id: str, code_hash: str) -> ExperimentRecord:
        """Create the initial record for one expanded matrix row."""
        revision, dirty = cls._git_provenance()
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            timestamp=cls.utc_timestamp(),
            status="queued",
            experiment=ExperimentRow(
                task_id=task.task_id,
                row_id=task.row_id,
                directory=_repo_relative(task.recipe_dir),
                kind=task.recipe.kind,
                variant=str(task.variant),
                parameters=_builtins(task.variant.params),
            ),
            provenance=Provenance(
                emmy_code_sha256=code_hash,
                git_revision=revision,
                git_dirty=dirty,
            ),
            system=None,
            execution=Execution(run_id=run_id),
        )

    def start(self, stage: str) -> None:
        """Mark this row as running."""
        self.status = "running"
        self.execution.stage = stage
        self.execution.started_at = self.execution.started_at or self.utc_timestamp()

    def finish(self, *, success: bool, stage: str, timing: dict[str, float], error: str | None = None) -> None:
        """Finalize this row without interpreting workload output."""
        self.start(stage)
        self.status = "succeeded" if success else "failed"
        self.execution.stage = "complete" if success else stage
        self.execution.completed_at = self.utc_timestamp()
        self.execution.timing_seconds = timing
        self.execution.error = None if success else ExecutionError(stage=stage, message=error or f"{stage} failed")

    def artifact(self, task: BenchmarkTask, path: Path, kind: str, **metadata: Any) -> Artifact:
        """Describe a raw file relative to the experiment directory."""
        experiment_dir = Path(task.recipe_dir).resolve()
        try:
            rendered = str(path.resolve().relative_to(experiment_dir))
        except ValueError:
            rendered = str(path)
        return Artifact(kind=kind, path=rendered, **metadata)

    def missing_command_provenance(self) -> list[str]:
        """Return missing generic facts required by strict command rows."""
        missing = []
        source = self.provenance.source
        if not source or not source.get("source_id") or not source.get("files"):
            missing.append("staged source manifest")
        gpus = self.system.gpus if self.system else []
        if not gpus or not all(item.uuid or item.pci_bus_id for item in gpus):
            missing.append("GPU provenance")
        software = self.system.software if self.system else SoftwareInformation()
        if not software.cuda_compiler and not software.hip_compiler:
            missing.append("GPU compiler provenance")
        return missing

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the typed schema to YAML-safe built-in values."""
        return _redacted(asdict(self))

    def to_yaml(self) -> str:
        """Serialize the complete record as redacted YAML."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=False, allow_unicode=True, width=120)

    def write(self, path: Path) -> None:
        """Atomically serialize this record to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.to_yaml(), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ExperimentRecord:
        """Deserialize a record from YAML-safe built-in values."""
        version = value.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment record schema version: {version}")
        execution_value = value["execution"]
        infrastructure = execution_value.get("infrastructure")
        error = execution_value.get("error")
        return cls(
            schema_version=value["schema_version"],
            timestamp=value["timestamp"],
            status=value["status"],
            experiment=ExperimentRow(**value["experiment"]),
            provenance=Provenance(**value["provenance"]),
            system=SystemInformation.from_mapping(value["system"]) if value.get("system") else None,
            execution=Execution(
                run_id=execution_value["run_id"],
                stage=execution_value.get("stage", "queued"),
                started_at=execution_value.get("started_at"),
                completed_at=execution_value.get("completed_at"),
                timing_seconds=execution_value.get("timing_seconds", {}),
                infrastructure=Infrastructure(**infrastructure) if infrastructure else None,
                error=ExecutionError(**error) if error else None,
                cleanup_error=execution_value.get("cleanup_error"),
            ),
            artifacts=[Artifact(**item) for item in value.get("artifacts", [])],
        )

    @classmethod
    def read(cls, path: Path) -> ExperimentRecord:
        """Load and deserialize one YAML experiment record."""
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"experiment record must be a mapping: {path}")
        return cls.from_mapping(value)
