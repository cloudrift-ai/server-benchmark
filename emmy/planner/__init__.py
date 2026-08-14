"""Planner: group benchmark tasks into execution groups for VM allocation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.hardware import gpu_short_name
from emmy.planner.variant import Variant
from emmy.recipe.types import Recipe

if TYPE_CHECKING:
    from emmy.benchmark.experiment_record import ExperimentRecord


@dataclass
class BenchmarkTask:
    """One recipe+variant combination to benchmark."""

    recipe_dir: str
    variant: Variant
    recipe: Recipe
    run_dir: Path | None = None
    record: ExperimentRecord | None = field(default=None, repr=False, compare=False)

    @property
    def gpu_name(self) -> str:
        """Full GPU name from recipe deploy config."""
        return self.recipe.deploy.gpu

    @property
    def gpu_count(self) -> int:
        """GPU count from recipe deploy config."""
        return self.recipe.deploy.gpu_count

    @property
    def gpu_short(self) -> str:
        """Short GPU name from variant."""
        return self.variant.gpu_short

    @property
    def task_id(self) -> str:
        """Unique task identifier: {recipe_name}/{variant}."""
        return f"{self.recipe_name}/{self.variant}"

    @property
    def model_name(self) -> str:
        return self.recipe.model_name

    @property
    def recipe_name(self) -> str:
        """Basename of the recipe directory (e.g. 'Qwen3-Coder-30B-A3B-Instruct-AWQ')."""
        return os.path.basename(self.recipe_dir)

    @property
    def row_id(self) -> str:
        """Stable short identity for the raw matrix parameters."""
        payload = json.dumps(self.variant.params, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def file_stem(self) -> str:
        """Readable collision-safe stem shared by this row's artifacts."""
        return f"{self.variant}_{self.row_id}"

    def record_path(self) -> Path:
        """Path to the sole structured experiment-row output."""
        if self.run_dir is None:
            raise ValueError("experiment results directory has not been assigned")
        return self.run_dir / f"{self.file_stem}.experiment.yaml"

    def benchmark_log_path(self) -> Path:
        """Path to raw inference-client output."""
        if self.run_dir is None:
            raise ValueError("experiment results directory has not been assigned")
        return self.run_dir / f"{self.file_stem}.benchmark.log"

    def setup_results_dir(self, run_dir: Path) -> None:
        """Assign the fixed raw-results directory."""
        self.run_dir = run_dir

    @staticmethod
    def compute_code_hash() -> str:
        """SHA256 hash of all .py files under emmy/, sorted by relative path."""
        pkg_dir = Path(__file__).parent.parent
        hasher = hashlib.sha256()
        for py_file in sorted(pkg_dir.rglob("*.py")):
            rel = py_file.relative_to(pkg_dir)
            content = py_file.read_text(encoding="utf-8")
            hasher.update(f"{rel}\n{content}\n".encode())
        return hasher.hexdigest()

    @staticmethod
    def prepare_results_dir(base_dir: str, *, overwrite: bool) -> Path:
        """Create ``results/``, replacing the preceding run when requested."""
        experiment_dir = Path(base_dir).resolve()
        results_dir = experiment_dir / "results"
        if results_dir.is_symlink():
            raise ValueError(f"refusing to replace symlinked experiment results directory: {results_dir}")
        if overwrite and results_dir.exists():
            shutil.rmtree(results_dir)
        if overwrite:
            results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir


@dataclass
class ExecutionGroup:
    """Group of tasks sharing one VM."""

    gpu_name: str
    gpu_count: int
    tasks: list[BenchmarkTask] = field(default_factory=list)
    index: int | None = None

    @property
    def gpu_short(self) -> str:
        """Short GPU name (e.g. 'rtx5090')."""
        return gpu_short_name(self.gpu_name)

    @property
    def label(self) -> str:
        """Unique group label (e.g. 'rtx5090_x_8' or 'rtx5090_x_8_r01')."""
        base = f"{self.gpu_short}_x_{self.gpu_count}"
        if self.index is not None:
            return f"{base}_r{self.index:02d}"
        return base


class BenchmarkPlanner(ABC):
    """Abstract base for grouping benchmark tasks into execution groups."""

    @abstractmethod
    def plan(self, tasks: list[BenchmarkTask]) -> list[ExecutionGroup]:
        """Group benchmark tasks into execution groups."""
        ...
