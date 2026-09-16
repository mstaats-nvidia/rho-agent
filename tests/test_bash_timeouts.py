"""Real POSIX child processes must not keep a timed-out Bash invocation blocked."""

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys

import pytest

from rho_agent.tools.base import ToolInvocation
from rho_agent.tools.handlers.bash import BashHandler

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def running(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("shell_waits", [False, True])
@pytest.mark.asyncio
async def test_timeout_and_cancellation_stop_pipe_holding_child(tmp_path, cancel, shell_waits):
    pidfile = tmp_path / "child.pid"
    command = f"printf 'before timeout\\n'; sleep 60 & echo $! > {shlex.quote(str(pidfile))}"
    if shell_waits:
        command += "; wait"
    invocation = ToolInvocation(
        "test", "bash", {"command": command, "timeout": 60 if cancel else 0.2}
    )
    task = asyncio.create_task(BashHandler(restricted=False).handle(invocation))
    child = None
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.01)
        child = int(pidfile.read_text())
        assert os.getpgid(child) != os.getpgrp()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
        else:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=5)
            assert result.success is False
            assert result.metadata["timed_out"] is True
            assert "before timeout" in json.loads(result.content)["output"]
        for _ in range(100):
            if not running(child):
                break
            await asyncio.sleep(0.01)
        assert not running(child)
    finally:
        # This also cleans up safely if the regression reappears; never signal our own group.
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_escaped_child_cannot_block_cleanup(tmp_path, cancel):
    pidfile = tmp_path / "escaped.pid"
    script = (
        "import subprocess; from pathlib import Path; "
        "child = subprocess.Popen(['sleep', '60'], start_new_session=True); "
        f"Path({str(pidfile)!r}).write_text(str(child.pid)); "
        "child.wait()"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    task = asyncio.create_task(
        BashHandler(restricted=False).handle(
            ToolInvocation("test", "bash", {"command": command, "timeout": 60 if cancel else 0.5})
        )
    )
    child = None
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.01)
        child = int(pidfile.read_text())
        assert os.getpgid(child) == child
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=4)
        else:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=4)
            assert result.metadata["timed_out"] is True
        assert running(child)
    finally:
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Let EOF close the subprocess pipe transports after the escaped child exits.
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_normal_exit_preserves_output_and_status():
    result = await BashHandler(restricted=False).handle(
        ToolInvocation("test", "bash", {"command": "printf stdout; printf stderr >&2; exit 3"})
    )
    assert result.metadata["exit_code"] == 3
    assert json.loads(result.content)["output"] == "stdout\nstderr"
