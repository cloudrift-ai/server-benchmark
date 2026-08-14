"""Tests for teardown state discovered from experiment records."""

import yaml

from emmy.commands.teardown import _load_active_instances


def _write_record(path, *, state, instance_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "infrastructure": {
                        "state": state,
                        "provider": "gcp",
                        "instance_id": instance_id,
                        "zone": "us-central1-b",
                        "address": "user@host",
                        "ssh_port": 22,
                    }
                }
            }
        )
    )


def test_load_active_instances_scans_assembled_and_raw_records(tmp_path):
    _write_record(tmp_path / "first.experiment.yaml", state="active", instance_id="vm-1")
    _write_record(tmp_path / "results" / "second.experiment.yaml", state="active", instance_id="vm-1")
    _write_record(tmp_path / "old.experiment.yaml", state="deleted", instance_id="vm-old")

    instances = _load_active_instances(tmp_path)

    assert len(instances) == 1
    records = next(iter(instances.values()))
    assert {path.name for path, _ in records} == {"first.experiment.yaml", "second.experiment.yaml"}
