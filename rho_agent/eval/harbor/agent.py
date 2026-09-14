"""Harbor BaseInstalledAgent wrapper for rho-agent."""

from __future__ import annotations

import json
import logging
import shlex
import tarfile
from importlib import metadata
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_CONTAINER_VENV = "/opt/rho-agent-venv"
_CONTAINER_SOURCE = "/installed-agent/rho-agent"
_CONTAINER_SOURCE_ARCHIVE = "/installed-agent/rho-agent-source.tar.gz"
_LOCAL_SOURCE_EXCLUDES = {
    ".env",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "jobs",
    "trials",
}


def _normalize_model_for_comparison(model: str | None) -> str | None:
    """Return a minimal normalized form for conflict detection."""
    if model is None:
        return None
    normalized = model.strip()
    if not normalized:
        return None
    if normalized.count("/") == 1:
        _, normalized = normalized.split("/", 1)
    return normalized


class RhoAgent(BaseInstalledAgent):
    """Runs rho-agent inside Harbor's container environment.

    This agent wrapper:
    1. Installs rho-agent in the container during setup
    2. Runs the rho_agent.eval.harbor.runner module with the task instruction
    3. Returns results for Harbor's verification system

    The container provides sandboxing, so rho-agent uses unrestricted
    eval-mode tools depending on the config settings.
    """

    # Harbor agent interface
    SUPPORTS_ATIF: bool = True

    DEFAULT_REPO_URL = "https://github.com/smith-nathanh/rho-agent.git"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        install_source: str = "pypi",
        repo_url: str | None = None,
        source_dir: Path | str | None = None,
        bash_only: bool = False,
        enable_reviewer: bool = False,
        reviewer_max_iterations: int = 1,
        enable_confirm_done: bool = True,
        confirm_done_max: int = 3,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        cost_ceiling_usd: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the agent.

        Args:
            logs_dir: Directory to write agent logs to.
            model_name: Model to use (e.g., "openai/gpt-5-mini").
            logger: Logger instance.
            install_source: How to install rho-agent in the task container: "pypi", "git",
                or "local". Local uploads the host checkout into the task container.
            repo_url: Optional git repository URL used when install_source="git".
            source_dir: Host checkout to upload when install_source="local". Defaults to the
                checkout containing this adapter.
            bash_only: If True, only provide bash tool (no Read, Grep, etc.).
            enable_reviewer: If True, run post-execution review after actor completes.
            reviewer_max_iterations: Max review-revise loops (0 = review only, no revision).
            enable_confirm_done: If True, require explicit CONFIRM_DONE after actor completes.
            confirm_done_max: Max confirm retries before proceeding (default: 3).
            temperature: Model temperature (default: None, uses API default).
            reasoning_effort: Reasoning effort level: "low", "medium", "high" (default: None).
            cost_ceiling_usd: Max cost per task in USD, 0 = disabled (default: 0.0).
        """
        normalized_install_source = install_source.strip().lower()
        if normalized_install_source not in {"pypi", "git", "local"}:
            raise ValueError(
                "install_source must be 'pypi', 'git', or 'local', "
                f"got {install_source!r}"
            )

        logger = logger or logging.getLogger(__name__)
        super().__init__(logs_dir, *args, model_name=model_name, logger=logger, **kwargs)
        self._install_source = normalized_install_source
        self._repo_url = repo_url or self.DEFAULT_REPO_URL
        self._source_dir = (
            Path(source_dir).expanduser().resolve()
            if source_dir is not None
            else Path(__file__).resolve().parents[3]
        )
        if self._install_source == "local":
            if not (self._source_dir / "pyproject.toml").is_file():
                raise ValueError(
                    f"local rho-agent source has no pyproject.toml: {self._source_dir}"
                )
            if not (self._source_dir / "rho_agent").is_dir():
                raise ValueError(
                    f"local rho-agent source has no rho_agent package: {self._source_dir}"
                )
        self._bash_only = bash_only
        self._enable_reviewer = enable_reviewer
        self._reviewer_max_iterations = reviewer_max_iterations
        self._enable_confirm_done = enable_confirm_done
        self._confirm_done_max = confirm_done_max
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._cost_ceiling_usd = cost_ceiling_usd

    @staticmethod
    def name() -> str:
        """Return the agent name for Harbor."""
        return "rho-agent"

    def version(self) -> str | None:
        """Return the agent version."""
        if self._version:
            return self._version
        try:
            return metadata.version("rho-agent")
        except metadata.PackageNotFoundError:
            return None

    def get_version_command(self) -> str:
        """Return a command that prints the installed rho-agent version in the container."""
        return (
            f"{_CONTAINER_VENV}/bin/python -c "
            "\"import importlib.metadata; print(importlib.metadata.version('rho-agent'))\""
        )

    @property
    def _install_agent_template_path(self) -> Path:
        """Path to the Jinja2 install script template."""
        return Path(__file__).parent / "install-rho-agent.sh.j2"

    @property
    def _template_variables(self) -> dict[str, str]:
        """Variables to pass to the install script template."""
        variables = {
            "install_source": self._install_source,
            "repo_url": self._repo_url,
            "venv_path": _CONTAINER_VENV,
        }
        if self._version:
            variables["version"] = self._version
        if self._install_source == "local":
            variables["source_dir"] = str(self._source_dir)
        return variables

    def _build_local_source_archive(self) -> Path:
        """Package local source while excluding credentials and host-only artifacts."""
        source_archive = self.logs_dir / "rho-agent-source.tar.gz"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        def exclude_local_artifacts(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if any(part in _LOCAL_SOURCE_EXCLUDES for part in Path(member.name).parts):
                return None
            return member

        with tarfile.open(source_archive, "w:gz") as archive:
            archive.add(
                self._source_dir,
                arcname=".",
                filter=exclude_local_artifacts,
            )
        return source_archive

    async def install(self, environment: BaseEnvironment) -> None:
        """Install the selected rho-agent source into the Harbor task container."""
        dependencies = ["ca-certificates", "curl"]
        if self._install_source == "git":
            dependencies.append("git")
        if self._install_source == "local":
            dependencies.append("tar")
        dependency_list = " ".join(dependencies)
        dependency_checks = " && ".join(
            f"command -v {dependency} >/dev/null 2>&1"
            for dependency in dependencies
            if dependency != "ca-certificates"
        )
        await self.exec_as_root(
            environment,
            command=(
                f"{{ {dependency_checks}; }} || "
                f"(apt-get update -qq && apt-get install -y -qq {dependency_list})"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=300,
        )

        path_setup = 'export PATH="$HOME/.local/bin:$PATH"'
        await self.exec_as_agent(
            environment,
            command=(
                f"{path_setup}; command -v uv >/dev/null 2>&1 || "
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            ),
            timeout_sec=120,
        )

        agent_user = str(environment.default_user or "root")
        quoted_user = shlex.quote(agent_user)
        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {shlex.quote(_CONTAINER_VENV)} "
                f"{shlex.quote(_CONTAINER_SOURCE)} && "
                f"chown -R {quoted_user}:{quoted_user} "
                f"{shlex.quote(_CONTAINER_VENV)} {shlex.quote(_CONTAINER_SOURCE)}"
            ),
        )

        if self._install_source == "local":
            source_archive = self._build_local_source_archive()
            await environment.upload_file(source_archive, _CONTAINER_SOURCE_ARCHIVE)
            await self.exec_as_root(
                environment,
                command=(
                    f"tar -xzf {shlex.quote(_CONTAINER_SOURCE_ARCHIVE)} "
                    f"-C {shlex.quote(_CONTAINER_SOURCE)} && "
                    f"chown -R {quoted_user}:{quoted_user} {shlex.quote(_CONTAINER_SOURCE)}"
                ),
            )
            package_spec = shlex.quote(f"{_CONTAINER_SOURCE}[evals]")
        elif self._install_source == "git":
            await self.exec_as_agent(
                environment,
                command=(
                    f"git clone {shlex.quote(self._repo_url)} "
                    f"{shlex.quote(_CONTAINER_SOURCE)}"
                ),
                timeout_sec=300,
            )
            if self._version:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"git -C {shlex.quote(_CONTAINER_SOURCE)} checkout "
                        f"{shlex.quote(self._version)}"
                    ),
                )
            package_spec = shlex.quote(f"{_CONTAINER_SOURCE}[evals]")
        else:
            version_suffix = f"=={self._version}" if self._version else ""
            package_spec = shlex.quote(f"rho-agent[evals]{version_suffix}")

        await self.exec_as_agent(
            environment,
            command=(
                f"{path_setup}; uv venv {shlex.quote(_CONTAINER_VENV)} --clear && "
                f"uv pip install --python {_CONTAINER_VENV}/bin/python {package_spec}"
            ),
            timeout_sec=300,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parse token usage and cost from telemetry, populate Harbor's AgentContext."""
        # Try incremental tokens file first (survives process kill)
        tokens_path = self.logs_dir / "tokens.json"
        if tokens_path.exists():
            try:
                data = json.loads(tokens_path.read_text())
                context.n_input_tokens = data.get("input", 0)
                context.n_output_tokens = data.get("output", 0)
                context.n_cache_tokens = data.get("cached", 0)
                if "cost_usd" in data:
                    context.cost_usd = data["cost_usd"]
                reasoning_tokens = data.get("reasoning", 0)
                self.logger.info(
                    f"Token usage: input={context.n_input_tokens}, "
                    f"output={context.n_output_tokens}, reasoning={reasoning_tokens}, "
                    f"cost=${context.cost_usd or 0:.4f}"
                )
                return
            except Exception as e:
                self.logger.warning(f"Failed to parse tokens.json: {e}")

        # Fall back to telemetry DB
        telemetry_path = self.logs_dir / "telemetry.db"
        if not telemetry_path.exists():
            return

        try:
            import sqlite3

            conn = sqlite3.connect(telemetry_path)
            cursor = conn.cursor()
            cursor.execute("SELECT total_input_tokens, total_output_tokens FROM sessions LIMIT 1")
            row = cursor.fetchone()
            conn.close()

            if row:
                context.n_input_tokens = row[0] or 0
                context.n_output_tokens = row[1] or 0
                self.logger.info(
                    f"Token usage (from DB): input={context.n_input_tokens}, "
                    f"output={context.n_output_tokens}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to parse telemetry DB: {e}")

    def _build_run_command(self, instruction: str) -> tuple[str, dict[str, str]]:
        """Build the in-container command and environment for a rho-agent run."""
        raw_rho_agent_model = self._get_env("RHO_AGENT_MODEL")
        raw_openai_model = self._get_env("OPENAI_MODEL")
        raw_config_model = self.model_name

        env_model = raw_rho_agent_model or raw_openai_model
        env_source = "RHO_AGENT_MODEL" if raw_rho_agent_model else "OPENAI_MODEL"

        normalized_env_model = _normalize_model_for_comparison(env_model)
        normalized_config_model = _normalize_model_for_comparison(raw_config_model)

        if (
            normalized_env_model is not None
            and normalized_config_model is not None
            and normalized_env_model != normalized_config_model
        ):
            raise ValueError(
                "Model conflict between Harbor config and environment: "
                f"config model_name={raw_config_model!r}, "
                f"{env_source}={env_model!r}. "
                "Remove one source or make them match."
            )

        if env_model:
            selected_model = env_model.strip()
            selected_source = env_source
        elif raw_config_model:
            selected_model = raw_config_model.strip()
            selected_source = "Harbor config"
        else:
            selected_model = "gpt-5-mini"
            selected_source = "default"

        effective_model = _normalize_model_for_comparison(selected_model)
        if effective_model is None:
            raise ValueError("Resolved model is empty after normalization")

        env = {
            "RHO_AGENT_MODEL": selected_model,
            "RHO_AGENT_TELEMETRY_DB": "/logs/agent/telemetry.db",
        }
        api_key = self._get_env("OPENAI_API_KEY")
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        # Add base URL if configured
        base_url = self._get_env("RHO_AGENT_BASE_URL") or self._get_env("OPENAI_BASE_URL")
        if base_url:
            env["RHO_AGENT_BASE_URL"] = base_url

        # Add service tier if configured (e.g., "flex" for lower cost)
        service_tier = self._get_env("RHO_AGENT_SERVICE_TIER")
        if service_tier:
            env["RHO_AGENT_SERVICE_TIER"] = service_tier

        # Add reviewer config if enabled
        if self._enable_reviewer:
            env["RHO_AGENT_ENABLE_REVIEWER"] = "1"
            env["RHO_AGENT_REVIEWER_MAX_ITERATIONS"] = str(self._reviewer_max_iterations)

        # Add completion confirmation config
        env["RHO_AGENT_CONFIRM_DONE"] = "1" if self._enable_confirm_done else "0"
        env["RHO_AGENT_CONFIRM_DONE_MAX"] = str(self._confirm_done_max)

        # Add temperature config (only if explicitly set)
        if self._temperature is not None:
            env["RHO_AGENT_TEMPERATURE"] = str(self._temperature)

        # Add reasoning effort config (only if explicitly set)
        if self._reasoning_effort:
            env["RHO_AGENT_REASONING_EFFORT"] = self._reasoning_effort

        # Add cost ceiling config (only if set > 0)
        if self._cost_ceiling_usd > 0:
            env["RHO_AGENT_COST_CEILING_USD"] = str(self._cost_ceiling_usd)

        self.logger.info(
            "Resolved Harbor model selection: "
            f"source={selected_source}, "
            f"selected={selected_model!r}, "
            f"effective={effective_model!r}, "
            f"config_model={raw_config_model!r}, "
            f"RHO_AGENT_MODEL={raw_rho_agent_model!r}, "
            f"OPENAI_MODEL={raw_openai_model!r}"
        )
        self.logger.info(
            f"Running rho-agent with model: {selected_model}, "
            f"effective_model: {effective_model}, "
            f"bash_only: {self._bash_only}, "
            f"reviewer: {self._enable_reviewer}, "
            f"confirm_done: {self._enable_confirm_done}, "
            f"temperature: {self._temperature}, "
            f"reasoning_effort: {self._reasoning_effort}, "
            f"cost_ceiling_usd: {self._cost_ceiling_usd}"
        )

        escaped = shlex.quote(instruction)
        bash_only_flag = " --bash-only" if self._bash_only else ""
        cmd = (
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f'{_CONTAINER_VENV}/bin/python -B -m rho_agent.eval.harbor.runner '
            f'{escaped} "$PWD"{bash_only_flag} '
            f"2>&1 | tee /logs/agent/stdout.txt"
        )
        return cmd, env

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run rho-agent in the task container using Harbor 0.22's agent API."""
        command, env = self._build_run_command(instruction)
        await self.exec_as_agent(environment, command=command, env=env)


# For Harbor's import_path to work
__all__ = ["RhoAgent"]
