"""Unit tests for Google ADK integration in agy-api."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from integrations.google_adk.google_adk_integration import (
    GoogleADKConfig,
    create_agent,
)


def test_google_adk_config_defaults():
    """Verify GoogleADKConfig default attributes match agy-api conventions."""
    config = GoogleADKConfig()

    assert config.base_url == "http://localhost:8000/v1"
    assert config.model == "Gemini 3.8 Flash"
    assert config.name == "root_agent"
    assert config.description is None
    assert config.instruction is None
    assert config.tools == []


def test_google_adk_config_custom_values():
    """Verify GoogleADKConfig preserves custom initialization parameters."""
    def sample_tool(city: str) -> str:
        return f"Weather in {city}: sunny"

    config = GoogleADKConfig(
        base_url="http://127.0.0.1:9000/v1",
        model="Claude Sonnet 4.6",
        name="weather_agent",
        description="Provides weather updates",
        instruction="Use sample_tool to look up weather.",
        tools=[sample_tool],
    )

    assert config.base_url == "http://127.0.0.1:9000/v1"
    assert config.model == "Claude Sonnet 4.6"
    assert config.name == "weather_agent"
    assert config.description == "Provides weather updates"
    assert config.instruction == "Use sample_tool to look up weather."
    assert config.tools == [sample_tool]


def test_google_adk_validation_empty_base_url():
    """Verify create_agent raises ValueError when base_url is empty."""
    with pytest.raises(ValueError, match="base_url cannot be empty"):
        create_agent(base_url="   ")


def test_google_adk_validation_empty_model():
    """Verify create_agent raises ValueError when model is empty."""
    with pytest.raises(ValueError, match="model cannot be empty"):
        create_agent(model="")


@patch("integrations.google_adk.google_adk_integration.Agent")
@patch("integrations.google_adk.google_adk_integration.OpenAILlm")
@patch("integrations.google_adk.google_adk_integration.AsyncOpenAI")
def test_google_adk_create_agent_with_mocks(
    mock_openai_cls: MagicMock,
    mock_llm_cls: MagicMock,
    mock_agent_cls: MagicMock,
):
    """Verify create_agent correctly wires Agent, OpenAILlm, and AsyncOpenAI."""
    def my_tool() -> str:
        return "ok"

    config = GoogleADKConfig(
        base_url="http://localhost:8000/v1",
        model="Gemini 3.8 Flash",
        name="test_agent",
        description="Agent description",
        instruction="System instruction",
        tools=[my_tool],
    )

    agent = create_agent(config)

    mock_openai_cls.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        api_key="agy",
    )
    mock_llm_cls.assert_called_once_with(
        model="Gemini 3.8 Flash",
        client=mock_openai_cls.return_value,
    )
    mock_agent_cls.assert_called_once_with(
        name="test_agent",
        model=mock_llm_cls.return_value,
        description="Agent description",
        instruction="System instruction",
        tools=[my_tool],
    )
    assert agent == mock_agent_cls.return_value


@patch("integrations.google_adk.google_adk_integration.Agent")
@patch("integrations.google_adk.google_adk_integration.OpenAILlm")
@patch("integrations.google_adk.google_adk_integration.AsyncOpenAI")
def test_google_adk_create_agent_direct_kwargs(
    mock_openai_cls: MagicMock,
    mock_llm_cls: MagicMock,
    mock_agent_cls: MagicMock,
):
    """Verify create_agent accepts direct kwargs without an explicit config."""
    agent = create_agent(
        base_url="http://127.0.0.1:8000/v1",
        model="Gemini 3.7 Flash",
        name="kwarg_agent",
        extra_param="custom_value",
    )

    mock_openai_cls.assert_called_once_with(
        base_url="http://127.0.0.1:8000/v1",
        api_key="agy",
    )
    mock_llm_cls.assert_called_once_with(
        model="Gemini 3.7 Flash",
        client=mock_openai_cls.return_value,
    )
    mock_agent_cls.assert_called_once_with(
        name="kwarg_agent",
        model=mock_llm_cls.return_value,
        tools=[],
        extra_param="custom_value",
    )
    assert agent == mock_agent_cls.return_value


@patch("integrations.google_adk.google_adk_integration.Agent")
@patch("integrations.google_adk.google_adk_integration.OpenAILlm")
@patch("integrations.google_adk.google_adk_integration.AsyncOpenAI")
def test_google_adk_config_create_agent_method(
    mock_openai_cls: MagicMock,
    mock_llm_cls: MagicMock,
    mock_agent_cls: MagicMock,
):
    """Verify GoogleADKConfig.create_agent method properly delegates."""
    cfg = GoogleADKConfig(name="delegated_agent")
    agent = cfg.create_agent()
    mock_agent_cls.assert_called_once()
    assert agent == mock_agent_cls.return_value


def test_top_level_integrations_exports():
    """Verify top-level integrations package exports GoogleADKConfig and create_agent."""
    import integrations
    import integrations.google_adk

    assert hasattr(integrations, "GoogleADKConfig")
    assert hasattr(integrations, "create_agent")
    assert hasattr(integrations.google_adk, "GoogleADKConfig")
    assert hasattr(integrations.google_adk, "create_agent")


def test_test_agent_get_weather(caplog: pytest.LogCaptureFixture):
    """Verify test_agent get_weather tool and its logging."""
    from integrations.google_adk.test_agent.agent import get_weather

    with caplog.at_level("INFO"):
        res_success = get_weather("New York")
        assert res_success["status"] == "success"
        assert "New York" in res_success["report"]

        res_unknown = get_weather("Atlantis")
        assert res_unknown["status"] == "error"
        assert "not available" in res_unknown["error_message"]

    assert any("get_weather called with city='New York'" in record.message for record in caplog.records)
    assert any("get_weather called with city='Atlantis'" in record.message for record in caplog.records)
    assert any("Weather information for 'Atlantis' is not available" in record.message for record in caplog.records)


def test_test_agent_get_current_time(caplog: pytest.LogCaptureFixture):
    """Verify test_agent get_current_time tool and its logging."""
    from integrations.google_adk.test_agent.agent import get_current_time

    with caplog.at_level("INFO"):
        res_success = get_current_time("New York")
        assert res_success["status"] == "success"
        assert "The current time in New York is" in res_success["report"]

        res_unknown = get_current_time("Atlantis")
        assert res_unknown["status"] == "error"
        assert "Sorry, I don't have timezone information" in res_unknown["error_message"]

    assert any("get_current_time called with city='New York'" in record.message for record in caplog.records)
    assert any("get_current_time called with city='Atlantis'" in record.message for record in caplog.records)
    assert any("Timezone information for 'Atlantis' is not available" in record.message for record in caplog.records)

