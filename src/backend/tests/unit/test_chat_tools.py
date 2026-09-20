"""Unit tests for OpenAI tool calling formatting and parsing in chat route."""
import json
from api_types import ChatCompletionMessage, ChatCompletionRequest, ChatMessage, DeltaContent
from routes.chat import (
    _extract_prompt_and_system,
    _format_tools_for_system_prompt,
    _parse_tool_calls_from_response,
)


def test_format_tools_for_system_prompt():
    """Verify tool formatting generates proper instructions and function signatures."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current time in city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    ]

    formatted = _format_tools_for_system_prompt(tools)
    assert "get_weather" in formatted
    assert "get_current_time" in formatted
    assert "tool_calls" in formatted
    assert "CRITICAL INSTRUCTION" in formatted


def test_parse_tool_calls_from_response_json():
    """Verify tool calls are parsed from raw JSON."""
    raw_json = json.dumps({
        "tool_calls": [
            {
                "name": "get_weather",
                "arguments": {"city": "New York"},
            }
        ]
    })

    tool_calls = _parse_tool_calls_from_response(raw_json)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "New York"}
    assert tool_calls[0]["id"].startswith("call_")


def test_parse_tool_calls_from_response_markdown():
    """Verify tool calls are parsed from markdown code block."""
    text = (
        "Here is the tool call:\n"
        "```json\n"
        '{\n  "tool_calls": [\n    {"name": "get_current_time", "arguments": {"city": "Toulouse"}}\n  ]\n}\n'
        "```"
    )

    tool_calls = _parse_tool_calls_from_response(text)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_current_time"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Toulouse"}


def test_parse_tool_calls_from_plain_text():
    """Verify plain text without tool calls returns None."""
    text = "The weather in New York is sunny and 75 degrees."
    assert _parse_tool_calls_from_response(text) is None


def test_extract_prompt_and_system_with_tools():
    """Verify _extract_prompt_and_system includes tool instructions when tools are present."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "lookup info",
                "parameters": {},
            },
        }
    ]
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Lookup Paris."),
        ],
        tools=tools,
    )

    prompt, system_message = _extract_prompt_and_system(request)
    assert "Be concise." in system_message
    assert "lookup" in system_message
    assert "tool_calls" in system_message
    assert "Lookup Paris." in prompt


def test_extract_prompt_and_system_with_tool_dialogue():
    """Verify dialogue history with tool calls and tool responses is formatted properly."""
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="Weather in NY?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[{"name": "get_weather", "arguments": {"city": "New York"}}],
            ),
            ChatMessage(
                role="tool",
                name="get_weather",
                content='{"status": "success", "report": "Sunny 25C"}',
            ),
        ]
    )

    prompt, _ = _extract_prompt_and_system(request)
    assert "Weather in NY?" in prompt
    assert "[Assistant called tools]" in prompt
    assert "[Tool Response (get_weather)]" in prompt
    assert "Sunny 25C" in prompt


def test_api_types_models_have_tool_calls():
    """Verify ChatCompletionMessage and DeltaContent models support tool_calls."""
    msg = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
    )
    assert msg.tool_calls is not None
    assert msg.tool_calls[0]["id"] == "call_1"

    delta = DeltaContent(
        role="assistant",
        tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
    )
    assert delta.tool_calls is not None
    assert delta.tool_calls[0]["index"] == 0
