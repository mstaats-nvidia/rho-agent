from rho_agent.core.events import AgentEvent
from rho_agent.eval.harbor.trajectory import TrajectoryBuilder


def test_step_metrics_aggregate_multiple_api_calls() -> None:
    builder = TrajectoryBuilder(model="gpt-5-mini")
    events = [
        AgentEvent(
            type="api_call_complete",
            usage={
                "input_tokens": 100,
                "output_tokens": 40,
                "cached_tokens": 10,
                "reasoning_tokens": 5,
                "cost_usd": 0.01,
            },
        ),
        AgentEvent(
            type="api_call_complete",
            usage={
                "input_tokens": 30,
                "output_tokens": 20,
                "cached_tokens": 0,
                "reasoning_tokens": 2,
                "cost_usd": 0.005,
            },
        ),
        AgentEvent(type="text", content="done"),
        AgentEvent(
            type="turn_complete",
            usage={
                "total_input_tokens": 130,
                "total_output_tokens": 60,
                "total_cached_tokens": 10,
                "total_reasoning_tokens": 7,
                "total_cost_usd": 0.015,
                "context_size": 130,
            },
        ),
    ]

    builder.build_from_events(events, user_input="task")
    trajectory = builder.to_trajectory()

    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"]
    assert trajectory["trajectory_id"]
    assert trajectory["agent"] == {
        "name": "rho-agent",
        "version": "0.1.0",
        "model_name": "gpt-5-mini",
    }
    assert trajectory["steps"][0]["step_id"] == 1
    agent_step = trajectory["steps"][1]
    assert agent_step["step_id"] == 2
    assert agent_step["metrics"]["prompt_tokens"] == 130
    assert agent_step["metrics"]["completion_tokens"] == 60
    assert agent_step["metrics"]["cached_tokens"] == 10
    assert agent_step["metrics"]["cost_usd"] == 0.015
    assert agent_step["metrics"]["extra"]["reasoning_tokens"] == 7
    assert trajectory["final_metrics"] == {
        "total_prompt_tokens": 130,
        "total_completion_tokens": 60,
        "total_cached_tokens": 10,
        "total_cost_usd": 0.015,
        "total_steps": 2,
        "extra": {"context_size": 130, "total_reasoning_tokens": 7},
    }


def test_tool_calls_and_observations_use_atif_field_names() -> None:
    builder = TrajectoryBuilder(model="gpt-5-mini")
    builder.build_from_events(
        [
            AgentEvent(
                type="tool_start",
                tool_call_id="call-1",
                tool_name="bash",
                tool_args={"command": "pwd"},
            ),
            AgentEvent(
                type="tool_end",
                tool_call_id="call-1",
                tool_name="bash",
                tool_result="/app",
                tool_metadata={"exit_code": 0},
            ),
        ],
        user_input="task",
    )

    agent_step = builder.to_trajectory()["steps"][1]

    assert agent_step["tool_calls"] == [
        {
            "tool_call_id": "call-1",
            "function_name": "bash",
            "arguments": {"command": "pwd"},
        }
    ]
    assert agent_step["observation"] == {
        "results": [
            {
                "source_call_id": "call-1",
                "content": "/app",
                "extra": {"exit_code": 0},
            }
        ]
    }
