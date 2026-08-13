"""The CLI entrypoint's own behaviour, ahead of any subcommand."""

from importlib.metadata import PackageNotFoundError

from emmy.emmy import _package_version


def test_version_flag_reports_the_installed_distribution(run_cli):
    """`--version` must answer without the required subcommand being supplied."""
    returncode, stdout, _ = run_cli("--version")

    assert returncode == 0
    assert stdout.startswith("emmy ")


def test_version_falls_back_when_the_package_is_not_installed(monkeypatch):
    """Running straight from a source checkout still has to produce a string."""

    def _missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr("emmy.emmy.version", _missing)

    assert _package_version() == "unknown"
