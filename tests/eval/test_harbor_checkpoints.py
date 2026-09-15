"""No model calls: persist exact observed evidence before a turn can be killed."""

import asyncio
import json
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rho_agent.core.events import AgentEvent
from rho_agent.eval.harbor import runner
from rho_agent.eval.harbor.trajectory import TrajectoryBuilder


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_runner_checkpoints_before_turn_completion(tmp_path, monkeypatch, cancel):
    monkeypatch.setattr(runner, "_AGENT_LOGS", tmp_path)
    monkeypatch.setattr(runner, "LiteLLMClient", lambda **kwargs: object())
    monkeypatch.setenv("RHO_AGENT_CONFIRM_DONE", "0")
    monkeypatch.setenv("RHO_AGENT_ENABLE_REVIEWER", "0")
    monkeypatch.setenv("RHO_AGENT_COST_CEILING_USD", "0")
    path = tmp_path / "trajectory.json"
    observation = "full untruncated observation " * 100

    class FakeSession:
        def __init__(self, *args, **kwargs):
            self.state = SimpleNamespace(
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0,
                }
            )

        async def run(self, prompt, *, on_event):
            assert json.loads(path.read_text())["steps"] == [{"source": "user", "message": prompt}]
            await on_event(
                AgentEvent(
                    type="tool_start",
                    tool_name="bash",
                    tool_call_id="call-1",
                    tool_args={"command": "example"},
                )
            )
            await on_event(
                AgentEvent(
                    type="tool_end",
                    tool_name="bash",
                    tool_call_id="call-1",
                    tool_result=observation,
                )
            )
            checkpoint = json.loads(path.read_text())
            assert checkpoint["metadata"]["incomplete"] is True
            assert checkpoint["steps"][1]["observations"][0]["content"] == observation
            # Streaming text after the last checkpoint must survive graceful cancellation.
            await on_event(AgentEvent(type="text", content="latest partial response"))
            if cancel:
                raise asyncio.CancelledError

    monkeypatch.setattr(runner, "Session", FakeSession)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await runner.run_task("task", str(tmp_path))
    else:
        await runner.run_task("task", str(tmp_path))
    final = json.loads(path.read_text())
    assert final["metadata"].get("incomplete", False) is cancel
    assert len(final["steps"]) == 2
    assert final["steps"][1]["message"] == "latest partial response"
    assert final["steps"][1]["observations"][0]["content"] == observation


def test_checkpoint_keeps_completed_turns_without_duplicates(tmp_path):
    builder = TrajectoryBuilder(model="test")
    builder.build_from_events([AgentEvent(type="text", content="first response")], "first")
    path = tmp_path / "trajectory.json"
    events = [AgentEvent(type="text", content="second response")]
    for _ in range(2):
        builder.checkpoint(path, events, user_input="second")
    assert len(builder.to_trajectory()["steps"]) == 2
    assert len(json.loads(path.read_text())["steps"]) == 4
    builder.build_from_events(events, "second")
    builder.save(path)
    assert len(json.loads(path.read_text())["steps"]) == 4


def test_failed_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch):
    builder = TrajectoryBuilder(model="test")
    path = tmp_path / "trajectory.json"
    builder.add_user_step("original")
    builder.save(path)
    previous = path.read_bytes()
    builder.add_user_step("new")

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(type(path), "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        builder.save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_survives_process_kill_without_cleanup(tmp_path):
    path = tmp_path / "trajectory.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, signal, sys
from rho_agent.core.events import AgentEvent
from rho_agent.eval.harbor.trajectory import TrajectoryBuilder
builder = TrajectoryBuilder(model="test")
builder.checkpoint(sys.argv[1], [AgentEvent(type="text", content="observed")],
                   user_input="task")
os.kill(os.getpid(), signal.SIGKILL)
""",
            str(path),
        ],
        check=False,
    )
    assert result.returncode == -signal.SIGKILL
    assert json.loads(path.read_text())["metadata"]["incomplete"] is True
    assert json.loads(path.read_text())["steps"] == [
        {"source": "user", "message": "task"},
        {"source": "agent", "message": "observed"},
    ]
