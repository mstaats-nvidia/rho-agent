"""Tests for LiteLLMClient._extract_usage cost tracking.

Covers the bug where streaming chunks caused completion_cost() to fail,
leaving cost_usd unset and showing $0.0000 in the session summary.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rho_agent.client.litellm_client import LiteLLMClient


@pytest.fixture
def client():
    """Create a LiteLLMClient with a mocked litellm module."""
    c = object.__new__(LiteLLMClient)
    c._model = "openai/gpt-5-mini"
    c._litellm = MagicMock()
    return c


def _make_usage(prompt_tokens=1000, completion_tokens=500):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class TestExtractUsageCost:
    """Test _extract_usage cost computation paths."""

    def test_full_response_cost_is_set(self, client):
        """Normal non-streaming path: completion_cost works on full response."""
        client._litellm.completion_cost.return_value = 0.0042
        usage = client._extract_usage(_make_usage(), response=MagicMock())

        assert usage["cost_usd"] == 0.0042

    def test_streaming_chunk_fails_falls_back_to_model_tokens(self, client):
        """The core bug: streaming chunk causes completion_cost to raise,
        fallback should compute cost from model + token counts."""

        def side_effect(**kwargs):
            if "completion_response" in kwargs:
                # Streaming chunk fails
                raise Exception("Cannot compute cost from streaming chunk")
            # Fallback with model + tokens succeeds
            return 0.0035

        client._litellm.completion_cost.side_effect = side_effect
        usage = client._extract_usage(_make_usage(), response=MagicMock())

        assert usage["cost_usd"] == 0.0035

    def test_streaming_chunk_returns_none_falls_back(self, client):
        """completion_cost returns None for chunk, fallback should kick in."""

        def side_effect(**kwargs):
            if "completion_response" in kwargs:
                return None
            return 0.0035

        client._litellm.completion_cost.side_effect = side_effect
        usage = client._extract_usage(_make_usage(), response=MagicMock())

        assert usage["cost_usd"] == 0.0035

    def test_streaming_chunk_returns_zero_is_kept(self, client):
        """completion_cost returning 0 should be stored (not filtered as falsy)."""
        client._litellm.completion_cost.return_value = 0
        usage = client._extract_usage(_make_usage(), response=MagicMock())

        # 0 is a valid cost (e.g. free-tier models), should be set
        assert "cost_usd" in usage
        assert usage["cost_usd"] == 0

    def test_no_response_no_tokens_no_cost(self, client):
        """No response and no tokens: cost_usd should not be set."""
        usage = client._extract_usage(None, response=None)

        assert "cost_usd" not in usage
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_no_response_with_tokens_uses_fallback(self, client):
        """No response object but tokens present: fallback computes cost."""
        client._litellm.completion_cost.return_value = 0.001
        usage = client._extract_usage(_make_usage(), response=None)

        assert usage["cost_usd"] == 0.001
        client._litellm.completion_cost.assert_called_once_with(
            model="openai/gpt-5-mini",
            prompt_tokens=1000,
            completion_tokens=500,
        )

    def test_both_paths_fail_no_cost_key(self, client):
        """Both completion_cost calls raise: cost_usd should not be set."""
        client._litellm.completion_cost.side_effect = Exception("no pricing data")
        usage = client._extract_usage(_make_usage(), response=MagicMock())

        assert "cost_usd" not in usage
        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 500


def test_build_kwargs_forwards_custom_api_base() -> None:
    client = object.__new__(LiteLLMClient)
    client._model = "openai/test-model"
    client._api_key = "test-key"
    client._base_url = "http://model.example.test/v1"
    client._initial_timeout = 10.0
    client._temperature = None
    client._reasoning_effort = None
    client._response_format = None

    kwargs = client._build_kwargs(messages=[{"role": "user", "content": "hello"}])

    assert kwargs["api_base"] == "http://model.example.test/v1"
