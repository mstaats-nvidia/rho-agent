"""Bash execution handler with configurable restrictions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from typing import Any

from ..base import ToolHandler, ToolInvocation, ToolOutput

DEFAULT_TIMEOUT_RESTRICTED = 120  # seconds
DEFAULT_TIMEOUT_UNRESTRICTED = 300  # 5 minutes for complex builds


def _kill_shell_group(process: asyncio.subprocess.Process) -> None:
    """Kill this invocation's shell and same-session children on POSIX."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        # The process/group may exit between the deadline and cleanup.
        pass


# Allowlist of safe read-only commands (used in RESTRICTED mode)
ALLOWED_COMMANDS = {
    # File inspection
    "cat",
    "head",
    "tail",
    "less",
    "more",
    # Search
    "grep",
    "rg",
    "ag",
    "ack",
    "find",
    "locate",
    "which",
    "whereis",
    # Directory listing
    "ls",
    "tree",
    "du",
    "df",
    # File info
    "file",
    "stat",
    "wc",
    "md5",
    "sha256sum",
    "shasum",
    # Text processing (read-only)
    "awk",
    "sed",  # Note: sed -i is blocked by pipe check
    "cut",
    "sort",
    "uniq",
    "tr",
    "column",
    "fmt",
    "fold",
    "nl",
    "pr",
    "expand",
    "unexpand",
    # JSON/YAML/XML
    "jq",
    "yq",
    "xmllint",
    # Archive inspection (read-only)
    "tar",  # listing only, extraction blocked by write check
    "unzip",  # -l listing only
    "zipinfo",
    "zcat",
    "zless",
    "zgrep",
    "gzip",  # -l listing
    "gunzip",  # to stdout only
    # System info
    "pwd",
    "whoami",
    "hostname",
    "uname",
    "env",
    "printenv",
    "date",
    "uptime",
    "ps",
    "top",
    "free",
    # Networking (read-only)
    "ping",
    "curl",
    "wget",
    "dig",
    "nslookup",
    "host",
    "netstat",
    "ss",
    # Git (read-only)
    "git",
    # Misc
    "echo",
    "printf",
    "diff",
    "cmp",
    "comm",
    "hexdump",
    "xxd",
    "od",
    "strings",
}

# Patterns that indicate write operations (blocked in RESTRICTED mode)
DANGEROUS_SUBSTRINGS = [
    ">>",  # Redirect (append)
    ">",  # Redirect (overwrite)
]

DANGEROUS_COMMAND_WORDS = {
    "rm",
    "rmdir",
    "mv",
    "cp",
    "chmod",
    "chown",
    "chgrp",
    "mkdir",
    "touch",
    "truncate",
    "shred",
    "dd",
    "mkfs",
    "mount",
    "umount",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "systemctl",
    "service",
    "apt",
    "yum",
    "dnf",
    "brew",
    "pip",
    "npm",
    "yarn",
    "cargo",
    "sudo",
    "su",
    "doas",
}


def _strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc body text so safety checks only inspect executed shell syntax."""
    lines = command.splitlines()
    if not lines:
        return command

    result: list[str] = []
    active_delimiter: str | None = None

    for line in lines:
        if active_delimiter is not None:
            if line.strip() == active_delimiter:
                active_delimiter = None
            continue

        result.append(line)
        match = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line)
        if match:
            active_delimiter = match.group(2)

    return "\n".join(result)


def _contains_dangerous_word(command: str) -> str | None:
    """Return first dangerous command word matched as a standalone token."""
    lowered = command.lower()
    for word in sorted(DANGEROUS_COMMAND_WORDS, key=len, reverse=True):
        pattern = rf"(^|[\s;&|()]){re.escape(word)}(?=$|[\s;&|()])"
        if re.search(pattern, lowered):
            return word
    return None


def extract_base_command(command: str) -> str | None:
    """Extract the base command from a shell command string."""
    # Handle pipes - check first command
    if "|" in command:
        command = command.split("|")[0].strip()

    # Handle command chaining - check first command
    for sep in ["&&", ";", "||"]:
        if sep in command:
            command = command.split(sep)[0].strip()

    # Handle env vars at start (VAR=value cmd)
    parts = command.split()
    for _i, part in enumerate(parts):
        if "=" not in part:
            return part

    return parts[0] if parts else None


def is_command_allowed(command: str) -> tuple[bool, str]:
    """Check if a command is allowed in RESTRICTED mode. Returns (allowed, reason)."""
    command_for_checks = _strip_heredoc_bodies(command)

    # Check for dangerous patterns first
    for pattern in DANGEROUS_SUBSTRINGS:
        if pattern in command_for_checks:
            return False, f"Command contains dangerous pattern: {pattern}"

    dangerous_word = _contains_dangerous_word(command_for_checks)
    if dangerous_word:
        return False, f"Command contains dangerous pattern: {dangerous_word}"

    # Extract base command
    base_cmd = extract_base_command(command_for_checks)
    if not base_cmd:
        return False, "Could not parse command"

    # Check allowlist
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"Command '{base_cmd}' is not in the allowlist"

    return True, ""


class BashHandler(ToolHandler):
    """Execute shell commands with configurable restrictions.

    Standard agentic tool name: 'bash'

    Modes:
    - RESTRICTED: Only allowlisted commands, dangerous patterns blocked
    - UNRESTRICTED: Any command allowed (container provides sandbox)
    """

    def __init__(
        self,
        restricted: bool = True,
        working_dir: str | None = None,
        timeout: int | None = None,
        requires_approval: bool | None = None,
    ):
        """Initialize BashHandler.

        Args:
            restricted: If True, use command allowlist (default). If False, allow all commands.
            working_dir: Default working directory for commands.
            timeout: Command timeout in seconds. Defaults to 120s (restricted) or 300s (unrestricted).
            requires_approval: Override whether approval is required. Defaults to True for restricted,
                              False for unrestricted.
        """
        self._restricted = restricted
        self._working_dir = working_dir or os.getcwd()
        self._timeout = timeout or (
            DEFAULT_TIMEOUT_RESTRICTED if restricted else DEFAULT_TIMEOUT_UNRESTRICTED
        )
        # Default approval: not required for restricted (allowlist protects),
        # required for unrestricted outside sandboxes (but factory overrides for unrestricted profile)
        self._requires_approval = (
            requires_approval if requires_approval is not None else (not restricted)
        )

    @property
    def name(self) -> str:
        return "bash"

    @property
    def requires_approval(self) -> bool:
        return self._requires_approval

    @property
    def description(self) -> str:
        if self._restricted:
            return (
                "Execute a shell command to inspect files, logs, or system state. "
                "Use this for text-based inspection with tools like grep, cat, head, "
                "tail, find, jq, yq, etc. Commands are read-only."
            )
        else:
            return (
                "Execute a bash command. Use for running programs, installing packages, "
                "building code, file operations, and any other shell tasks."
            )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": f"Working directory for the command (default: {self._working_dir})",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {self._timeout})",
                },
            },
            "required": ["command"],
        }

    def _format_output(self, output: str, exit_code: int, duration_seconds: float) -> str:
        """Format output as Codex-style JSON."""
        return json.dumps(
            {
                "output": output,
                "metadata": {
                    "exit_code": exit_code,
                    "duration_seconds": round(duration_seconds, 1),
                },
            },
            indent=2,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolOutput:
        """Execute the shell command and return output."""
        command = invocation.arguments.get("command", "")
        working_dir = invocation.arguments.get("working_dir", self._working_dir)
        timeout = invocation.arguments.get("timeout", self._timeout)

        if not command:
            return ToolOutput(content="No command provided", success=False)

        # Check if command is allowed (only in restricted mode)
        if self._restricted:
            allowed, reason = is_command_allowed(command)
            if not allowed:
                return ToolOutput(
                    content=f"Command blocked: {reason}",
                    success=False,
                )

        start_time = time.perf_counter()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                start_new_session=os.name == "posix",
            )

            communication = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                # Children can retain the pipes after their shell dies.
                _kill_shell_group(process)
                await communication
                raise
            except TimeoutError:
                _kill_shell_group(process)
                partial_stdout, partial_stderr = await communication
                duration = time.perf_counter() - start_time

                # Combine stdout/stderr into a single output string
                output = partial_stdout[:50000].decode("utf-8", errors="replace")
                if partial_stderr:
                    stderr_text = partial_stderr[:50000].decode("utf-8", errors="replace")
                    output = f"{output}\n{stderr_text}" if output else stderr_text
                output += f"\n\n[Command timed out after {timeout}s and was killed]"

                content = self._format_output(output, exit_code=-1, duration_seconds=duration)

                return ToolOutput(
                    content=content,
                    success=False,
                    metadata={
                        "exit_code": -1,
                        "timed_out": True,
                        "duration_seconds": round(duration, 1),
                        "working_dir": working_dir,
                        "command": command,
                    },
                )

            duration = time.perf_counter() - start_time
            exit_code = process.returncode
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Combine stdout/stderr into single output
            if stdout_str and stderr_str:
                output = f"{stdout_str}\n{stderr_str}"
            elif stdout_str:
                output = stdout_str
            elif stderr_str:
                output = stderr_str
            else:
                output = ""

            content = self._format_output(output, exit_code, duration)

            return ToolOutput(
                content=content,
                success=exit_code == 0,
                metadata={
                    "exit_code": exit_code,
                    "duration_seconds": round(duration, 1),
                    "command": command,
                    "working_dir": working_dir,
                },
            )

        except FileNotFoundError:
            return ToolOutput(
                content=f"Working directory not found: {working_dir}",
                success=False,
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error executing command: {e}",
                success=False,
            )
