"""LiteLLM client for multi-provider model support with cost tracking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from .model import Prompt, StreamEvent, ToolCall


class LiteLLMClient:
    """Multi-provider model client via LiteLLM with cost tracking."""

    # Default chunk timeout: how long to wait for next chunk before aborting
    DEFAULT_CHUNK_TIMEOUT: float = 180.0  # 3 minutes between chunks

    # Default initial timeout: how long to wait for first chunk in streaming
    DEFAULT_INITIAL_TIMEOUT: float = 600.0  # 10 minutes for first chunk

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        base_url: str | None = None,
        chunk_timeout: float | None = None,
        initial_timeout: float | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        try:
            import litellm

            self._litellm = litellm
        except ImportError as err:
            raise ImportError(
                "litellm is required for LiteLLMClient. "
                "Install with: uv pip install rho-agent[evals]"
            ) from err

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._chunk_timeout = chunk_timeout or self.DEFAULT_CHUNK_TIMEOUT
        self._initial_timeout = initial_timeout or self.DEFAULT_INITIAL_TIMEOUT
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._response_format = response_format

        # Disable LiteLLM's internal logging to reduce noise
        litellm.suppress_debug_info = True

    def _build_messages(self, prompt: Prompt) -> list[dict[str, Any]]:
        """Build messages list from prompt."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt.system}]
        for msg in prompt.messages:
            m: dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                m["content"] = msg.content
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            messages.append(m)
        return messages

    def _build_kwargs(
        self,
        prompt: Prompt | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs for litellm.acompletion()."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(prompt) if prompt else messages,
            "stream": stream,
        }

        if stream:
            # Request usage in final chunk (OpenAI API feature)
            kwargs["stream_options"] = {"include_usage": True}
            # Time to wait for first chunk before aborting
            kwargs["stream_timeout"] = self._initial_timeout

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url

        # reasoning_effort and temperature are mutually exclusive
        # reasoning_effort is for OpenAI o-series/GPT-5 models
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        elif self._temperature is not None:
            kwargs["temperature"] = self._temperature

        if prompt and prompt.tools:
            kwargs["tools"] = prompt.tools

        if self._response_format:
            kwargs["response_format"] = self._response_format

        return kwargs

    def _extract_usage(self, usage_obj: Any, response: Any = None) -> dict[str, Any]:
        """Extract usage dict from usage object including cost."""
        usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if usage_obj:
            usage["input_tokens"] = getattr(usage_obj, "prompt_tokens", 0) or 0
            usage["output_tokens"] = getattr(usage_obj, "completion_tokens", 0) or 0

            # Extract cached tokens if available (prompt_tokens_details)
            if hasattr(usage_obj, "prompt_tokens_details"):
                details = usage_obj.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    usage["cached_tokens"] = details.cached_tokens or 0

            # Extract reasoning tokens if available (completion_tokens_details)
            # Reasoning tokens are a subset of completion_tokens, not additive
            if hasattr(usage_obj, "completion_tokens_details"):
                details = usage_obj.completion_tokens_details
                if details and hasattr(details, "reasoning_tokens"):
                    usage["reasoning_tokens"] = details.reasoning_tokens or 0

        # Compute cost using LiteLLM's pricing database
        # Pass the full response object for accurate cost calculation
        if response:
            try:
                cost = self._litellm.completion_cost(completion_response=response)
                if cost is not None:
                    usage["cost_usd"] = cost
            except Exception:
                pass

        # Fallback: compute from model + token counts (e.g. streaming chunks
        # where completion_cost on the chunk object fails)
        if "cost_usd" not in usage and usage["input_tokens"] + usage["output_tokens"] > 0:
            try:
                cost = self._litellm.completion_cost(
                    model=self._model,
                    prompt_tokens=usage["input_tokens"],
                    completion_tokens=usage["output_tokens"],
                )
                if cost is not None:
                    usage["cost_usd"] = cost
            except Exception:
                pass

        return usage

    async def stream(self, prompt: Prompt) -> AsyncIterator[StreamEvent]:
        """Stream a response from the model, with per-chunk timeout."""
        kwargs = self._build_kwargs(prompt=prompt, stream=True)

        try:
            response = await self._litellm.acompletion(**kwargs)

            # Accumulate tool calls (they come in pieces via deltas)
            tool_calls_in_progress: dict[int, dict[str, Any]] = {}
            usage: dict[str, Any] = {}

            async for chunk in self._iter_with_timeout(response):
                # Check for usage in chunk (comes in final chunk with stream_options)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = self._extract_usage(chunk.usage, response=chunk)

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Content delta
                if delta.content:
                    yield StreamEvent(type="text", content=delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_in_progress:
                            tool_calls_in_progress[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        tc = tool_calls_in_progress[idx]

                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["arguments"] += tc_delta.function.arguments

            # Emit completed tool calls
            for tc in tool_calls_in_progress.values():
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield StreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=args,
                    ),
                )

            # Emit done with usage
            yield StreamEvent(type="done", usage=usage)

        except TimeoutError:
            yield StreamEvent(
                type="error",
                content=f"Chunk timeout: no response received for {self._chunk_timeout}s",
            )
        except Exception as e:
            yield StreamEvent(type="error", content=str(e))

    async def _iter_with_timeout(self, response: Any) -> AsyncIterator[Any]:
        """Iterate over streaming response with chunk timeout."""
        aiter = response.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    aiter.__anext__(),
                    timeout=self._chunk_timeout,
                )
                yield chunk
            except StopAsyncIteration:
                break

    async def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """Non-streaming completion returning (content, usage_dict)."""
        try:
            kwargs = self._build_kwargs(messages=messages, stream=False)
            # Add a generous timeout for non-streaming calls
            kwargs["timeout"] = 600.0  # 10 minutes
            response = await self._litellm.acompletion(**kwargs)

            content = response.choices[0].message.content or ""
            usage = self._extract_usage(response.usage, response=response)

            return content, usage

        except Exception as e:
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0}
