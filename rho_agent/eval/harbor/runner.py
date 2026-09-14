"""Entry point for running rho-agent inside Harbor containers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rho_agent.client.litellm_client import LiteLLMClient
from rho_agent.core import Agent, AgentConfig, Session
from rho_agent.core.events import AgentEvent
from rho_agent.eval.harbor.trajectory import TrajectoryBuilder
from rho_agent.permissions import PermissionProfile
from rho_agent.prompts import load_prompt, prepare_prompt

# Load .env file - try multiple locations
# 1. Current directory
# 2. rho-agent package root (where this file lives, up 4 levels)
# 3. /rho-agent (Harbor container mount point)
_pkg_root = Path(__file__).parent.parent.parent.parent
for env_path in [Path.cwd() / ".env", _pkg_root / ".env", Path("/rho-agent/.env")]:
    if env_path.exists():
        load_dotenv(env_path)
        break

_EVAL_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "eval_terminal.md"
_REVIEWER_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "eval_reviewer.md"

# Reviewer system prompt (text-only, no tools)
_REVIEWER_SYSTEM = """\
You are a code reviewer evaluating whether an AI agent successfully completed a task.
Be concise and direct. Focus on whether the task requirements were met."""


def format_tool_call(name: str, args: dict[str, Any] | None) -> str:
    """Format tool name and args for console logging."""
    if not args:
        return f"[Tool: {name}]"
    # Create compact args summary
    args_preview = json.dumps(args, separators=(",", ":"))
    if len(args_preview) > 200:
        args_preview = args_preview[:200] + "..."
    return f"[Tool: {name}] {args_preview}"


def format_event_trace(events: list[AgentEvent]) -> str:
    """Format agent events into a readable trace for the reviewer."""
    lines = []
    for event in events:
        if event.type == "text":
            lines.append(f"[Agent Response]\n{event.content}")
        elif event.type == "tool_start":
            args_str = json.dumps(event.tool_args, indent=2) if event.tool_args else "{}"
            lines.append(f"[Tool Call: {event.tool_name}]\nArguments:\n{args_str}")
        elif event.type == "tool_end":
            # Truncate long results
            result = event.tool_result or ""
            if len(result) > 2000:
                result = result[:2000] + "\n... (truncated)"
            lines.append(f"[Tool Result: {event.tool_name}]\n{result}")
        elif event.type == "error":
            lines.append(f"[Error]\n{event.content}")
    return "\n\n".join(lines)


async def run_reviewer_phase(
    instruction: str,
    event_trace: list[AgentEvent],
    session: Session,
    client: LiteLLMClient,
    max_iterations: int,
) -> None:
    """Run post-execution review with optional revision loop.

    Args:
        instruction: Original task instruction.
        event_trace: Full event trace from the actor phase.
        session: The actor session (used for revision turns).
        client: Model client for reviewer calls.
        max_iterations: Maximum review-revise iterations (0 = review only).
    """
    # Load reviewer prompt template
    reviewer_prompt = load_prompt(_REVIEWER_PROMPT)

    for iteration in range(max_iterations + 1):  # +1 for initial review
        print(f"\n[Reviewer: iteration {iteration + 1}]", file=sys.stderr)

        # Format event trace for reviewer
        formatted_trace = format_event_trace(event_trace)

        # Build review prompt from template
        review_content, _ = prepare_prompt(
            reviewer_prompt,
            {
                "task_instruction": instruction,
                "agent_trace": formatted_trace,
            },
        )

        # Get review using raw completion (no tools)
        messages = [
            {"role": "system", "content": _REVIEWER_SYSTEM},
            {"role": "user", "content": review_content},
        ]
        review_text, _ = await client.complete(messages)

        # Log verdict preview
        preview = review_text[:100] + "..." if len(review_text) > 100 else review_text
        print(f"[Reviewer verdict: {preview}]", file=sys.stderr)

        # Parse verdict
        review_stripped = review_text.strip()
        if review_stripped.startswith("APPROVED"):
            print("[Reviewer: APPROVED]", file=sys.stderr)
            return

        if iteration >= max_iterations:
            print("[Reviewer: max iterations reached]", file=sys.stderr)
            return

        # Extract revision feedback
        if "REVISION_NEEDED:" in review_text:
            feedback = review_text.split("REVISION_NEEDED:", 1)[1].strip()
        else:
            feedback = review_text

        # Run revision with actor
        print(f"[Revision needed: {feedback[:100]}...]", file=sys.stderr)
        revision_prompt = (
            f"A reviewer found issues with your work:\n\n{feedback}\n\nPlease address these issues."
        )

        # Run actor revision turn and collect events for next review
        revision_events: list[AgentEvent] = []

        async def collect_revision(
            event: AgentEvent, _events: list[AgentEvent] = revision_events
        ) -> None:
            _events.append(event)
            if event.type == "text" and event.content:
                print(event.content, end="", flush=True)
            elif event.type == "tool_start":
                print(f"\n{format_tool_call(event.tool_name, event.tool_args)}", file=sys.stderr)

        await session.run(revision_prompt, on_event=collect_revision)

        # Use new trace for next review iteration
        event_trace = revision_events

        print()  # Newline after revision output


async def run_task(instruction: str, working_dir: str = "/app", bash_only: bool = False) -> None:
    """Run rho-agent on a TerminalBench task.

    Args:
        instruction: The task description/instruction.
        working_dir: Working directory for shell commands (default: /app).
        bash_only: If True, only provide bash tool (no Read, Grep, etc.).
    """
    # Load and render eval prompt template
    prompt = load_prompt(_EVAL_PROMPT)
    system_prompt, _ = prepare_prompt(
        prompt,
        {
            "platform": "Linux",
            "home_dir": str(Path.home()),
            "working_dir": working_dir,
        },
    )

    # Create client from environment
    # LiteLLM uses model names like "openai/gpt-5-mini" or "anthropic/claude-3-5-sonnet"
    model = os.environ.get("RHO_AGENT_MODEL") or os.environ.get("OPENAI_MODEL", "openai/gpt-5-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("RHO_AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    temperature = (
        float(os.environ["RHO_AGENT_TEMPERATURE"])
        if "RHO_AGENT_TEMPERATURE" in os.environ
        else None
    )
    reasoning_effort = os.environ.get("RHO_AGENT_REASONING_EFFORT")
    chunk_timeout = float(os.environ.get("RHO_AGENT_CHUNK_TIMEOUT", "180.0"))
    initial_timeout = float(os.environ.get("RHO_AGENT_INITIAL_TIMEOUT", "600.0"))
    cost_ceiling_usd = float(os.environ.get("RHO_AGENT_COST_CEILING_USD", "0.0"))

    client = LiteLLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        chunk_timeout=chunk_timeout,
        initial_timeout=initial_timeout,
    )

    # Determine context window for auto-compaction
    # Strip provider prefix for model detection (e.g., "openai/gpt-5-mini" -> "gpt-5-mini")
    model_name = model.split("/")[-1].lower()
    if "gpt-5" in model_name:
        context_window = 400_000  # GPT-5.x family
    elif "gpt-oss" in model_name:
        context_window = 128_000  # GPT-OSS-120B
    elif "claude" in model_name:
        context_window = 200_000  # Claude 3.x family
    else:
        context_window = 128_000  # conservative default

    # Build agent with unrestricted profile
    agent = Agent(
        AgentConfig(
            system_prompt=system_prompt,
            model=model,
            profile="unrestricted",
            working_dir=working_dir,
            auto_approve=True,
        )
    )

    # Replace registry with unrestricted tools
    profile = PermissionProfile.unrestricted(working_dir=working_dir)
    profile.bash_only = bash_only
    from rho_agent.permissions.factory import ToolFactory

    factory = ToolFactory(profile)
    agent._registry = factory.create_registry(working_dir=working_dir)

    # Log available tools for debugging
    tool_names = [h.name for h in agent.registry._handlers.values()]
    print(f"[rho-agent] Tools: {tool_names}", file=sys.stderr)

    # Create session with LiteLLM client (multi-provider support in containers)
    session = Session(agent, client=client)
    session.auto_compact = True
    session.context_window = context_window

    # Create trajectory builder for ATIF output
    trajectory_builder = TrajectoryBuilder(model=model)

    try:
        # Track events for reviewer (full trace, not just text output)
        event_trace: list[AgentEvent] = []

        async def run_turn(prompt_text: str) -> str:
            # Check cost ceiling before calling agent - end gracefully to allow grading
            if cost_ceiling_usd > 0 and session.state.usage["cost_usd"] >= cost_ceiling_usd:
                print(
                    f"\n[Cost ceiling reached: ${session.state.usage['cost_usd']:.2f} >= ${cost_ceiling_usd:.2f}]",
                    file=sys.stderr,
                )
                return ""

            text_content = ""
            turn_events: list[AgentEvent] = []

            def _write_tokens() -> None:
                """Write tokens/cost incrementally to mounted path (survives process kill)."""
                Path("/logs/agent/tokens.json").write_text(
                    json.dumps(
                        {
                            "input": session.state.usage["input_tokens"],
                            "output": session.state.usage["output_tokens"],
                            "cached": session.state.usage["cached_tokens"],
                            "reasoning": session.state.usage["reasoning_tokens"],
                            "cost_usd": session.state.usage["cost_usd"],
                        }
                    )
                )

            async def on_event(event: AgentEvent) -> None:
                nonlocal text_content
                event_trace.append(event)
                turn_events.append(event)
                if event.type == "text" and event.content:
                    text_content += event.content
                    print(event.content, end="", flush=True)
                elif event.type == "tool_start":
                    print(
                        f"\n{format_tool_call(event.tool_name, event.tool_args)}", file=sys.stderr
                    )
                elif event.type == "tool_end":
                    if os.environ.get("RHO_AGENT_DEBUG"):
                        result_preview = (
                            event.tool_result[:200] + "..."
                            if event.tool_result and len(event.tool_result) > 200
                            else event.tool_result
                        )
                        print(f"[Result: {result_preview}]", file=sys.stderr)
                    _write_tokens()
                elif event.type == "api_call_complete":
                    if os.environ.get("RHO_AGENT_DEBUG"):
                        usage = event.usage or {}
                        print(
                            f"[API call {usage.get('call_index')}: "
                            f"context={usage.get('input_tokens')}, "
                            f"out={usage.get('output_tokens')}, "
                            f"cost=${usage.get('cost_usd', 0):.4f}]",
                            file=sys.stderr,
                        )
                elif event.type == "compact_start":
                    print("\n[Compacting context...]", file=sys.stderr)
                elif event.type == "compact_end":
                    print(f"[{event.content}]", file=sys.stderr)
                elif event.type == "error":
                    print(f"\nError: {event.content}", file=sys.stderr)
                elif event.type == "turn_complete":
                    if event.usage:
                        cost = event.usage.get("total_cost_usd", 0.0)
                        reasoning = event.usage.get("total_reasoning_tokens", 0)
                        reasoning_str = f", reasoning={reasoning}" if reasoning else ""
                        print(
                            f"\n[Tokens: in={event.usage.get('total_input_tokens', 0)}, "
                            f"out={event.usage.get('total_output_tokens', 0)}{reasoning_str}, "
                            f"cost=${cost:.4f}]",
                            file=sys.stderr,
                        )
                    _write_tokens()

            await session.run(prompt_text, on_event=on_event)

            # Build trajectory from this turn's events
            trajectory_builder.build_from_events(turn_events, user_input=prompt_text)

            return text_content

        # Run initial actor turn
        await run_turn(instruction)

        # Completion confirmation gate (Terminus-style)
        enable_confirm = os.environ.get("RHO_AGENT_CONFIRM_DONE", "1") == "1"
        max_confirm = int(os.environ.get("RHO_AGENT_CONFIRM_DONE_MAX", "3"))
        confirm_token = "CONFIRM_DONE"
        confirm_prompt = (
            "Before finalizing, verify your solution:\n"
            "1. If test files exist, run them with the appropriate test runner\n"
            "2. Check actual test output - exit code 0 alone doesn't mean tests passed\n"
            "3. Confirm ALL requirements are met, including any performance/accuracy thresholds\n"
            "4. Re-read the task instructions and verify each requirement\n\n"
            f"If everything passes verification, reply with exactly {confirm_token}. "
            "Otherwise, fix issues first."
        )

        if enable_confirm:
            for _ in range(max_confirm):
                confirm_text = await run_turn(confirm_prompt)
                if confirm_text.strip() == confirm_token:
                    break

        # Reviewer phase (optional)
        enable_reviewer = os.environ.get("RHO_AGENT_ENABLE_REVIEWER") == "1"
        max_iterations = int(os.environ.get("RHO_AGENT_REVIEWER_MAX_ITERATIONS", "1"))

        if enable_reviewer:
            await run_reviewer_phase(
                instruction=instruction,
                event_trace=event_trace,
                session=session,
                client=client,
                max_iterations=max_iterations,
            )
    finally:
        # Save ATIF trajectory for Harbor analysis
        trajectory_builder.save("/logs/agent/trajectory.json")

    print()  # Final newline


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m rho_agent.eval.harbor.runner '<instruction>' [working_dir] [--bash-only]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse --bash-only flag
    bash_only = "--bash-only" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--bash-only"]

    instruction = args[0]
    working_dir = args[1] if len(args) > 1 else "/app"

    asyncio.run(run_task(instruction, working_dir, bash_only=bash_only))


if __name__ == "__main__":
    main()
