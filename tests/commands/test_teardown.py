"""Tests for teardown state discovered from experiment records."""

from emmy.benchmark.experiment_record import Execution, ExperimentRecord, ExperimentRow, Infrastructure, Provenance
from emmy.commands.teardown import _load_active_instances


def _write_record(path, *, state, instance_id):
    record = ExperimentRecord(
        schema_version=1,
        timestamp="2026-08-14T12:00:00Z",
        status="succeeded",
        experiment=ExperimentRow(task_id="task", row_id="row", directory="experiment", kind="command", variant="v", parameters={}),
        provenance=Provenance(emmy_code_sha256="hash", git_revision="rev", git_dirty=False),
        system=None,
        execution=Execution(
            run_id="run",
            stage="complete",
            infrastructure=Infrastructure(
                group="group",
                requested_gpu="NVIDIA A100 80GB",
                requested_gpu_count=1,
                state=state,
                provider="gcp",
                instance_id=instance_id,
                zone="us-central1-b",
                address="user@host",
                ssh_port=22,
            ),
        ),
    )
    record.write(path)


def test_load_active_instances_scans_assembled_and_raw_records(tmp_path):
    _write_record(tmp_path / "first.experiment.yaml", state="active", instance_id="vm-1")
    _write_record(tmp_path / "results" / "second.experiment.yaml", state="active", instance_id="vm-1")
    _write_record(tmp_path / "old.experiment.yaml", state="deleted", instance_id="vm-old")

    instances = _load_active_instances(tmp_path)

    assert len(instances) == 1
    records = next(iter(instances.values()))
    assert {path.name for path, _ in records} == {"first.experiment.yaml", "second.experiment.yaml"}
