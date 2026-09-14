"""ATIF trajectory builder for Harbor eval compatibility."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rho_agent import __version__
from rho_agent.core.events import AgentEvent

_SCHEMA_VERSION = "ATIF-v1.7"


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
        self._session_id = str(uuid.uuid4())
        self._trajectory_id = str(uuid.uuid4())
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
        for step_id, step in enumerate(self._steps, start=1):
            step_dict: dict[str, Any] = {
                "step_id": step_id,
                "source": step.source,
                "message": step.message or "",
            }

            if step.tool_calls:
                step_dict["tool_calls"] = [
                    {
                        "tool_call_id": tc.call_id,
                        "function_name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in step.tool_calls
                ]

            if step.observations:
                step_dict["observation"] = {
                    "results": [
                        {
                            "source_call_id": obs.source_call_id,
                            "content": obs.content,
                            **({"extra": obs.metadata} if obs.metadata else {}),
                        }
                        for obs in step.observations
                    ]
                }

            if step.metrics:
                step_dict["metrics"] = step.metrics

            steps_data.append(step_dict)

        final_metrics: dict[str, Any] = {
            "total_prompt_tokens": self._total_input_tokens,
            "total_completion_tokens": self._total_output_tokens,
            "total_cached_tokens": self._total_cached_tokens,
            "total_cost_usd": self._total_cost_usd,
            "total_steps": len(steps_data),
        }

        final_extra: dict[str, Any] = {}
        if self._context_size:
            final_extra["context_size"] = self._context_size
        if self._total_reasoning_tokens:
            final_extra.update(
                {
                    "total_reasoning_tokens": self._total_reasoning_tokens,
                }
            )
        if final_extra:
            final_metrics["extra"] = final_extra

        agent: dict[str, Any] = {"name": "rho-agent", "version": __version__}
        if self._model:
            agent["model_name"] = self._model
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_id": self._session_id,
            "trajectory_id": self._trajectory_id,
            "agent": agent,
            "steps": steps_data,
            "final_metrics": final_metrics,
        }

    def save(self, path: Path | str) -> None:
        """Write trajectory to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trajectory = self.to_trajectory()
        path.write_text(json.dumps(trajectory, indent=2))
