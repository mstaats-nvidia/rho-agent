from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rho_agent.eval.harbor.cli import app

runner = CliRunner()


def test_list_configs_includes_bundled_templates() -> None:
    result = runner.invoke(app, ["list-configs"])

    assert result.exit_code == 0
    assert "terminal-bench-prelim-git.yaml" in result.stdout
    assert "terminal-bench-sample.yaml" in result.stdout
    assert "terminal-bench.yaml" in result.stdout


def test_write_config_supports_stem_name(tmp_path: Path) -> None:
    output = tmp_path / "sample.yaml"

    result = runner.invoke(
        app,
        ["write-config", "terminal-bench-prelim-git", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert "import_path: rho_agent.eval.harbor.agent:RhoAgent" in output.read_text()
    assert "install_source: git" in output.read_text()


def test_write_config_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    output = tmp_path / "terminal-bench.yaml"
    output.write_text("existing")

    result = runner.invoke(
        app,
        ["write-config", "terminal-bench", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert output.read_text() == "existing"
