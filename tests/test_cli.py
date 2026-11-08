from typer.testing import CliRunner

from deltaaudit.cli import app


def test_info_runs():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "deltaaudit" in result.stdout
