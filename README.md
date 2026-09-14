# rho-agent

AI agents you can deploy, monitor, and control.

<!-- ![demo](assets/demo.gif) -->

rho-agent lets you deploy AI agents with predefined permissions into real environments — working directories, databases, production logs — and let them operate autonomously. Define a permission profile, hand off a task, and get back a result. Monitor live, review the full event trace, or resume from the CLI.

## Quickstart

```bash
git clone https://github.com/smith-nathanh/rho-agent.git
cd rho-agent
uv sync
uv tool install .              # adds rho-agent and rho-eval to PATH
uv tool install '.[daytona]'   # or include optional extras (db, daytona, all)
```

```bash
# Interactive debugging session
rho-agent --profile developer --working-dir ~/proj/myapp
> test_webhook has been flaky in CI — find the race condition and fix it
> now run the tests and make sure it passes

# Investigate a production database interactively
RHO_AGENT_DB_CONFIG=~/.config/rho-agent/databases.yaml rho-agent --profile readonly
> correlate the 3am latency spike with recent deployments and slow queries
> which tables are missing indexes?

# One-shot: migrate a module and run tests
rho-agent --profile developer --working-dir ~/proj/myapp \
  "migrate sync_client.py to async using aiohttp, update all callers, run tests"

# Triage a failed job using a prompt template
rho-agent --prompt examples/log_debugger/debug.md --var log_path=/mnt/logs/12345

# Use a saved agent config
rho-agent --config /path/to/agent.yaml "Analyze recent failures."
```

## Python API

Embed agents in services, workers, and batch systems:

```python
import asyncio
from rho_agent import Agent, AgentConfig, Session

async def main() -> None:
    config = AgentConfig(
        system_prompt="You are a research assistant.",
        profile="developer",
        working_dir="/tmp/work",
    )
    agent = Agent(config)
    session = Session(agent)
    result = await session.run(prompt="Analyze recent failures.")
    print(result.text, result.status, result.usage)

asyncio.run(main())
```

See the [Python SDK](docs/site/python-sdk.md) docs and [`examples/`](examples/) for more patterns including task-based parallelism and cancellation.

## Permission Profiles

Every agent runs under a profile that controls shell access, file write permissions, and database mutation behavior. The default is `readonly`.

| Profile | Shell | File Write | Database | Use Case |
|---------|-------|------------|----------|----------|
| `readonly` | Restricted (allowlist) | Off | SELECT only | Safe inspection of production systems |
| `developer` | Unrestricted | Full | SELECT only | Local development with file editing |
| `eval` | Unrestricted | Full | Full | Sandboxed benchmark execution |

```bash
rho-agent --profile readonly
rho-agent --profile developer
rho-agent --profile path/to/custom-profile.yaml
```

Custom profiles are defined in YAML. See [Profiles](docs/site/profiles.md) for the full schema.

## Highlights

- **Native tool handlers** — shell, file read/write/edit, grep, glob, and five database drivers (SQLite, PostgreSQL, MySQL, Oracle, Vertica) with no external plugins or MCP servers
- **Prompt templates** — Markdown with YAML frontmatter and Jinja2 variables for repeatable, parameterized agent tasks
- **Agent configs** — define reusable agent configurations in YAML with `AgentConfig`, load via `--config` on the CLI or `AgentConfig.from_file()` in Python
- **Multi-agent coordination** — delegate subtasks to child agents or interact with running agents through the monitor
- **Observability** — per-session `trace.jsonl` event logs with token usage, tool execution, and timing data; session directories at `~/.config/rho-agent/sessions/`
- **Session management** — monitor, pause, resume, and cancel running agents from another terminal (`rho-agent monitor <dir>`)
- **Remote sandboxing** — execute shell and file tools in a Daytona cloud sandbox with `--backend daytona`
- **ATIF export** — convert session traces to [Harbor](https://github.com/harbor-framework/harbor)'s Agent Trajectory Interchange Format for SFT/RL training data
- **Evaluation integrations** — [BIRD-Bench](rho_agent/eval/birdbench/) (text-to-SQL) and [TerminalBench](rho_agent/eval/harbor/) via Harbor

## Documentation

| | |
|---|---|
| [Quickstart](docs/site/quickstart.md) | Get running in minutes |
| [Installation](docs/site/installation.md) | Environment setup and install options |
| [Architecture](docs/site/architecture.md) | System design, agent loop, and session protocol |
| [CLI Reference](docs/site/cli-reference.md) | Commands, flags, and usage examples |
| [Python SDK](docs/site/python-sdk.md) | Create and run agents programmatically |
| [Tools](docs/site/tools.md) | Complete tool handler reference |
| [Profiles](docs/site/profiles.md) | Permission profiles and custom YAML |
| [Daytona](docs/site/daytona.md) | Remote sandbox execution via Daytona |
| [Observability](docs/site/observability.md) | Session traces, offline inspection, and observers |
| [Export](rho_agent/export/README.md) | Convert traces to ATIF for training data |
| [Monitor](docs/site/monitor.md) | Watch, control, and steer running agents |

## Development

```bash
uv sync --group dev          # install with dev dependencies
uv run python -m pytest      # run tests
```

## Release Binary Installer Notes

- `install.sh` looks for GitHub release assets named `rho-agent-<target>.tar.gz`
- The tarball should contain executables named `rho-agent` and `rho-eval`
- Supported targets in the installer today: macOS/Linux on `x86_64` and `arm64` (`aarch64`)

## Configuration

```bash
# .env
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5-mini                   # optional
OPENAI_BASE_URL=http://localhost:8000/v1   # optional
```

To make these available to child processes like `rho-agent` and `harbor`, export them before running commands:

```bash
set -a
source .env
set +a
```

Or export them directly:

```bash
export OPENAI_API_KEY=your-key
```

Database tools are loaded from `~/.config/rho-agent/databases.yaml` or the path in `RHO_AGENT_DB_CONFIG`. See [Tools](docs/site/tools.md) for the config format.
