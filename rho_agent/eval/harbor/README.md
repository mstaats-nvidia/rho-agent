# Harbor Integration

This package exposes `rho_agent.eval.harbor.agent:RhoAgent` for running `rho-agent` inside [Harbor](https://github.com/harbor-framework/harbor) jobs.

Harbor already supports loading installed agent classes by import path, so the supported integration point is:

```text
rho_agent.eval.harbor.agent:RhoAgent
```

## User Workflow

Use this when you want to run Harbor with a locally installed `rho-agent` adapter. Task
containers can install rho-agent from the same local checkout, from git, or from PyPI.

### Prerequisites

- Python 3.12+ for Harbor
- Docker
- Harbor 0.22.0
- API credentials exported in your shell or loaded from a `.env`

Harbor only forwards environment variables that exist in the `harbor run` process environment. A shell variable is not enough.

Before running Harbor, make sure your API key is exported:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5.2  # optional
printenv OPENAI_API_KEY | wc -c
```

If you keep credentials in a `.env` file, export them before running Harbor:

```bash
set -a
source .env
set +a
printenv OPENAI_API_KEY | wc -c
```

If `printenv OPENAI_API_KEY` is empty, Harbor will start but the agent inside the task will fail authentication.

Install Harbor and rho-agent into the same environment so Harbor can import
`rho_agent.eval.harbor.agent:RhoAgent`:

```bash
uv tool install 'harbor==0.22.0' \
  --with 'git+https://github.com/smith-nathanh/rho-agent.git#egg=rho-agent[evals]'
```

Or in a project environment:

```bash
uv add 'harbor==0.22.0' \
  'git+https://github.com/smith-nathanh/rho-agent.git#egg=rho-agent[evals]'
```

For local development, install this checkout into the same environment as Harbor and tell
the adapter to upload it into each task container:

```bash
uv venv
uv pip install 'harbor==0.22.0' -e '.[evals]'

harbor trial start \
  --path /path/to/harbor-task \
  --agent rho_agent.eval.harbor.agent:RhoAgent \
  --model openai/gpt-5-mini \
  --agent-kwarg install_source=local \
  --agent-kwarg source_dir="$PWD"
```

This mode includes uncommitted source changes, which makes it the appropriate path for
running locally generated ablations. Provider credentials and model configuration must be
exported before starting Harbor.

### Sample configs

The package ships Harbor config templates. List them with:

```bash
rho-eval harbor list-configs
```

Write one to your current directory:

```bash
rho-eval harbor write-config terminal-bench-prelim-git
rho-eval harbor write-config terminal-bench-sample
rho-eval harbor write-config terminal-bench
```

For a quick smoke test, use the git-backed hello-world config:

```bash
rho-eval harbor write-config terminal-bench-prelim-git
```

Edit `terminal-bench-prelim-git.yaml` and set:

- `version` to the branch, tag, or commit you want Harbor containers to install
- `kwargs.repo_url` to the repository Harbor should clone inside the task container

For a small real eval, use the 10-task sample dataset:

```bash
rho-eval harbor write-config terminal-bench-sample
```

This writes a config with:

```yaml
datasets:
  - name: terminal-bench-sample
    version: "2.0"
    registry: {}
    exclude_task_names:
      - "qemu-*"
```

The bundled sample config excludes `qemu-*` tasks because they are not reliable on ARM Macs.

Then run Harbor:

```bash
harbor run --config ./terminal-bench-prelim-git.yaml
harbor run --config ./terminal-bench-sample.yaml
RHO_AGENT_SERVICE_TIER=flex harbor run --config ./terminal-bench.yaml
```

For the `terminal-bench-prelim-git.yaml` smoke test, a good preflight is:

```bash
docker info >/dev/null
printenv OPENAI_API_KEY | wc -c
harbor run --config ./terminal-bench-prelim-git.yaml
```

### How installation works inside task containers

The Harbor host process imports the agent class from your local Python environment:

```yaml
agents:
  - import_path: rho_agent.eval.harbor.agent:RhoAgent
```

Inside each task container, `RhoAgent` can install `rho-agent` from a local checkout, git,
or PyPI. Local installation uploads `source_dir` through Harbor's environment API; it does
not require the checkout to be externally reachable. Git installation uses:

```yaml
agents:
  - import_path: rho_agent.eval.harbor.agent:RhoAgent
    version: main
    kwargs:
      install_source: git
      repo_url: https://github.com/smith-nathanh/rho-agent.git
```

`version` is the git ref to check out inside the task container.

### Important kwargs

Bundled configs use `kwargs` such as:

- `install_source`
- `repo_url`
- `source_dir`
- `bash_only`
- `enable_reviewer`
- `reviewer_max_iterations`
- `enable_confirm_done`
- `confirm_done_max`
- `temperature`
- `reasoning_effort`
- `cost_ceiling_usd`

## PyPI Workflow

If `rho-agent` is published to PyPI later, Harbor containers can install from PyPI instead of git:

```yaml
agents:
  - import_path: rho_agent.eval.harbor.agent:RhoAgent
    kwargs:
      install_source: pypi
```

For local iteration outside containers, editable installs are still fine:

```bash
cd ~/proj/harbor
uv pip install -e ~/proj/rho-agent
```

## Environment Variables

`RhoAgent` forwards provider configuration from the Harbor host process into the task
container through Harbor's environment API. The main variables are:

- `OPENAI_API_KEY`
- `RHO_AGENT_MODEL` or `OPENAI_MODEL`
- `RHO_AGENT_BASE_URL` or `OPENAI_BASE_URL`
- `RHO_AGENT_SERVICE_TIER`

These must be exported environment variables in the shell that launches `harbor run`.

Model selection is strict:

- If `RHO_AGENT_MODEL` or `OPENAI_MODEL` is set and `agents[].model_name` is also set, the run will fail fast if they disagree.
- Minimal comparison normalization is applied for conflict detection, so `openai/gpt-5.4` and `gpt-5.4` are treated as equivalent.
- The resolved model source and effective model are logged at agent startup.

## Bundled Configs

| File | Dataset | Tasks | Use case |
|------|---------|-------|----------|
| `terminal-bench-prelim-git.yaml` | `hello-world` | 1 | Git-backed smoke test |
| `terminal-bench-prelim.yaml` | `hello-world` | 1 | PyPI-backed smoke test |
| `terminal-bench-sample.yaml` | `terminal-bench-sample` | 10 | Short validation run |
| `terminal-bench.yaml` | `terminal-bench` | 89 | Full benchmark |
