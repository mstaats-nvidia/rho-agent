from __future__ import annotations

import importlib
import logging
import sys
import tarfile
import types
from pathlib import Path

import pytest


def _load_agent_module():
    harbor = types.ModuleType("harbor")
    harbor.__path__ = []

    agents = types.ModuleType("harbor.agents")
    agents.__path__ = []
    installed = types.ModuleType("harbor.agents.installed")
    installed.__path__ = []
    base = types.ModuleType("harbor.agents.installed.base")

    class BaseInstalledAgent:
        def __init__(
            self,
            logs_dir: Path,
            *args,
            model_name: str | None = None,
            logger=None,
            version: str | None = None,
            extra_env: dict[str, str] | None = None,
            **kwargs,
        ) -> None:
            self.logs_dir = logs_dir
            self.model_name = model_name
            self.logger = logger
            self._version = version
            self._extra_env = extra_env or {}

        def _get_env(self, key: str) -> str | None:
            import os

            return self._extra_env.get(key, os.environ.get(key))

    base.BaseInstalledAgent = BaseInstalledAgent

    environments = types.ModuleType("harbor.environments")
    environments.__path__ = []
    environment_base = types.ModuleType("harbor.environments.base")

    class BaseEnvironment:
        default_user = "root"

    environment_base.BaseEnvironment = BaseEnvironment

    models = types.ModuleType("harbor.models")
    models.__path__ = []
    agent_pkg = types.ModuleType("harbor.models.agent")
    agent_pkg.__path__ = []
    context_mod = types.ModuleType("harbor.models.agent.context")

    class AgentContext:
        pass

    context_mod.AgentContext = AgentContext

    sys.modules.update(
        {
            "harbor": harbor,
            "harbor.agents": agents,
            "harbor.agents.installed": installed,
            "harbor.agents.installed.base": base,
            "harbor.environments": environments,
            "harbor.environments.base": environment_base,
            "harbor.models": models,
            "harbor.models.agent": agent_pkg,
            "harbor.models.agent.context": context_mod,
        }
    )

    sys.modules.pop("rho_agent.eval.harbor.agent", None)
    return importlib.import_module("rho_agent.eval.harbor.agent")


@pytest.fixture
def harbor_agent_module():
    return _load_agent_module()


def test_default_template_variables_use_pypi_install(harbor_agent_module) -> None:
    agent = harbor_agent_module.RhoAgent(logs_dir=Path("/tmp/logs"))

    assert agent._template_variables == {
        "install_source": "pypi",
        "repo_url": "https://github.com/smith-nathanh/rho-agent.git",
        "venv_path": "/opt/rho-agent-venv",
    }


def test_git_install_template_variables_include_version(harbor_agent_module) -> None:
    agent = harbor_agent_module.RhoAgent(
        logs_dir=Path("/tmp/logs"),
        install_source="git",
        repo_url="https://example.com/custom/rho-agent.git",
        version="feature/harbor-fix",
    )

    assert agent._template_variables == {
        "install_source": "git",
        "repo_url": "https://example.com/custom/rho-agent.git",
        "venv_path": "/opt/rho-agent-venv",
        "version": "feature/harbor-fix",
    }


def test_invalid_install_source_raises(harbor_agent_module) -> None:
    with pytest.raises(ValueError, match="install_source"):
        harbor_agent_module.RhoAgent(logs_dir=Path("/tmp/logs"), install_source="wheel")


def test_local_install_uses_checkout_containing_adapter(harbor_agent_module) -> None:
    agent = harbor_agent_module.RhoAgent(
        logs_dir=Path("/tmp/logs"),
        install_source="local",
    )

    assert agent._source_dir == Path(__file__).resolve().parents[2]
    assert agent._template_variables["source_dir"] == str(agent._source_dir)


def test_local_source_archive_excludes_host_artifacts(
    harbor_agent_module, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    (source / "rho_agent").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='rho-agent'\n")
    (source / "rho_agent" / "kept.py").write_text("KEPT = True\n")
    (source / ".env").write_text("SECRET=do-not-package\n")
    (source / ".venv").mkdir()
    (source / ".venv" / "ignored.py").write_text("ignored\n")
    logs_dir = tmp_path / "logs"
    agent = harbor_agent_module.RhoAgent(
        logs_dir=logs_dir,
        install_source="local",
        source_dir=source,
    )

    archive_path = agent._build_local_source_archive()

    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
    assert "./rho_agent/kept.py" in names
    assert not any(".env" in name for name in names)
    assert not any(".venv" in name for name in names)


def test_run_command_uses_stable_container_venv(harbor_agent_module, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-5-mini")
    agent = harbor_agent_module.RhoAgent(
        logs_dir=Path("/tmp/logs"),
        bash_only=True,
        logger=logging.getLogger("test"),
    )

    command, env = agent._build_run_command("solve task")

    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["RHO_AGENT_MODEL"] == "openai/gpt-5-mini"
    assert "/opt/rho-agent-venv/bin/python -B -m rho_agent.eval.harbor.runner" in command
    assert "/rho-agent/.venv/bin/python" not in command
    assert "--bash-only" in command


def test_model_conflict_raises(harbor_agent_module, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-5.2")
    agent = harbor_agent_module.RhoAgent(
        logs_dir=Path("/tmp/logs"),
        model_name="openai/gpt-5.4",
        logger=logging.getLogger("test"),
    )

    with pytest.raises(ValueError, match="Model conflict"):
        agent._build_run_command("solve task")


def test_equivalent_model_names_do_not_conflict(harbor_agent_module, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    agent = harbor_agent_module.RhoAgent(
        logs_dir=Path("/tmp/logs"),
        model_name="openai/gpt-5.4",
        logger=logging.getLogger("test"),
    )

    _, env = agent._build_run_command("solve task")

    assert env["RHO_AGENT_MODEL"] == "gpt-5.4"
