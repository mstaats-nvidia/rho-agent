"""ATIF trajectory builder for Harbor eval compatibility."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rho_agent.core.events import AgentEvent


@dataclass
class ToolCallEntry:
    """A tool call in ATIF format."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ObservationEntry:
    """A tool result/observation in ATIF format."""

    source_call_id: str
    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class Step:
    """A single step in the ATIF trajectory."""

    source: str  # "user" or "agent"
    message: str | None = None
    tool_calls: list[ToolCallEntry] = field(default_factory=list)
    observations: list[ObservationEntry] = field(default_factory=list)
    metrics: dict[str, Any] | None = None


class TrajectoryBuilder:
    """Builds ATIF-compliant trajectories from AgentEvent streams.

    Usage:
        builder = TrajectoryBuilder(model="gpt-5-mini")
        builder.add_user_step("What is 2+2?")
        builder.build_from_events(events)
        builder.save(path / "trajectory.json")
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize the trajectory builder.

        Args:
            model: Model name for metadata.
        """
        self._model = model
        self._steps: list[Step] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cached_tokens: int = 0
        self._total_reasoning_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._context_size: int = 0

    def add_user_step(self, message: str) -> None:
        """Add a user input as a step.

        Args:
            message: The user's input text.
        """
        self._steps.append(Step(source="user", message=message))

    def build_from_events(self, events: list[AgentEvent], user_input: str | None = None) -> None:
        """Convert AgentEvents to ATIF steps.

        Processes events from a single turn and appends steps to the trajectory.

        Args:
            events: List of AgentEvent objects from a turn.
            user_input: Optional user input to add before agent events.
        """
        if user_input:
            self.add_user_step(user_input)

        # Accumulate agent content for current step
        current_text = ""
        current_tool_calls: list[ToolCallEntry] = []
        pending_observations: list[ObservationEntry] = []
        step_prompt_tokens = 0
        step_completion_tokens = 0
        step_cached_tokens = 0
        step_cost_usd = 0.0
        step_reasoning_tokens = 0
        saw_api_call_metrics = False

        # Map tool_start events to call IDs for matching with tool_end when IDs are absent.
        tool_call_ids: dict[str, list[str]] = {}

        for event in events:
            if event.type == "text" and event.content:
                current_text += event.content

            elif event.type == "tool_start":
                # Generate a call ID for this tool invocation
                call_id = event.tool_call_id or str(uuid.uuid4())[:8]
                tool_name = event.tool_name or "unknown"
                tool_call_ids.setdefault(tool_name, []).append(call_id)

                current_tool_calls.append(
                    ToolCallEntry(
                        call_id=call_id,
                        name=tool_name,
                        arguments=event.tool_args or {},
                    )
                )

            elif event.type == "tool_end":
                tool_name = event.tool_name or "unknown"
                # Prefer event call ID; otherwise match earliest queued start for this tool.
                call_id = event.tool_call_id
                if not call_id:
                    queued_ids = tool_call_ids.get(tool_name, [])
                    call_id = queued_ids.pop(0) if queued_ids else str(uuid.uuid4())[:8]

                pending_observations.append(
                    ObservationEntry(
                        source_call_id=call_id,
                        content=event.tool_result or "",
                        metadata=event.tool_metadata,
                    )
                )

            elif event.type == "api_call_complete":
                # Aggregate per-call metrics for attachment to the agent step.
                if event.usage:
                    saw_api_call_metrics = True
                    step_prompt_tokens += event.usage.get("input_tokens", 0) or 0
                    step_completion_tokens += event.usage.get("output_tokens", 0) or 0
                    step_cached_tokens += event.usage.get("cached_tokens", 0) or 0
                    step_cost_usd += event.usage.get("cost_usd", 0.0) or 0.0
                    step_reasoning_tokens += event.usage.get("reasoning_tokens", 0) or 0

            elif event.type == "turn_complete":
                # Extract metrics from turn_complete
                if event.usage:
                    self._total_input_tokens = event.usage.get("total_input_tokens", 0)
                    self._total_output_tokens = event.usage.get("total_output_tokens", 0)
                    self._total_cached_tokens = event.usage.get("total_cached_tokens", 0)
                    self._total_reasoning_tokens = event.usage.get("total_reasoning_tokens", 0)
                    self._total_cost_usd = event.usage.get("total_cost_usd", 0.0)
                    self._context_size = event.usage.get("context_size", 0)

        # Create agent step(s) from accumulated content
        if current_text or current_tool_calls:
            step_metrics: dict[str, Any] | None = None
            if saw_api_call_metrics:
                step_metrics = {
                    "prompt_tokens": step_prompt_tokens,
                    "completion_tokens": step_completion_tokens,
                    "cached_tokens": step_cached_tokens,
                    "cost_usd": step_cost_usd,
                }
                if step_reasoning_tokens:
                    step_metrics["extra"] = {"reasoning_tokens": step_reasoning_tokens}

            step = Step(
                source="agent",
                message=current_text if current_text else None,
                tool_calls=current_tool_calls,
                observations=pending_observations,
                metrics=step_metrics,
            )
            self._steps.append(step)

    def to_trajectory(self) -> dict[str, Any]:
        """Export as ATIF-compliant dict.

        Returns:
            Dictionary in ATIF format.
        """
        steps_data = []
        for step in self._steps:
            step_dict: dict[str, Any] = {"source": step.source}

            if step.message:
                step_dict["message"] = step.message

            if step.tool_calls:
                step_dict["tool_calls"] = [
                    {
                        "call_id": tc.call_id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in step.tool_calls
                ]

            if step.observations:
                step_dict["observations"] = [
                    {
                        "source_call_id": obs.source_call_id,
                        "content": obs.content,
                        **({"metadata": obs.metadata} if obs.metadata else {}),
                    }
                    for obs in step.observations
                ]

            if step.metrics:
                step_dict["metrics"] = step.metrics

            steps_data.append(step_dict)

        metadata: dict[str, Any] = {
            "model": self._model,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cached_tokens": self._total_cached_tokens,
            "total_cost_usd": self._total_cost_usd,
            "context_size": self._context_size,
        }

        # Add reasoning tokens in extra field for ATIF compliance
        # Reasoning tokens are a subset of output_tokens, tracked separately for analysis
        if self._total_reasoning_tokens:
            metadata["extra"] = {
                "total_reasoning_tokens": self._total_reasoning_tokens,
            }

        return {
            "steps": steps_data,
            "metadata": metadata,
        }

    def save(self, path: Path | str, *, incomplete: bool = False) -> None:
        """Atomically write trajectory JSON, preserving the last complete checkpoint.

        Args:
            path: Output file path.
            incomplete: Mark an observed prefix that may omit later activity.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trajectory = self.to_trajectory()
        if incomplete:
            trajectory["metadata"]["incomplete"] = True
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(trajectory, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def checkpoint(self, path: Path | str, events: list[AgentEvent], *, user_input: str) -> None:
        """Save completed turns plus the observed prefix of the current turn.

        Keep the completed-turn builder unchanged, so later checkpoints and the
        final trajectory cannot duplicate an in-progress turn.
        """
        snapshot = deepcopy(self)
        snapshot.build_from_events(events, user_input=user_input)
        snapshot.save(path, incomplete=True)
