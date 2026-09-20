"""Google ADK integration for NVIDIA NIM endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.labs.openai import OpenAILlm
from openai import AsyncOpenAI

# Ensure environment variables are loaded if .env exists
_env_path = Path(__file__).parent / ".env"
if _env_path.is_file():
    load_dotenv(dotenv_path=_env_path, override=False)


def create_agent(
    name: str = "root_agent",
    model: str = "nvidia/nemotron-3-ultra-550b-a55b",
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: str | None = None,
    description: str | None = None,
    instruction: str | None = None,
    tools: list[Callable[..., Any]] | None = None,
    **kwargs: Any,
) -> Agent:
    """Create a Google ADK Agent linked directly to the NVIDIA NIM endpoint."""
    resolved_api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not resolved_api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=resolved_api_key,
    )
    llm = OpenAILlm(
        model=model,
        client=client,
    )

    agent_kwargs: dict[str, Any] = {
        "name": name,
        "model": llm,
        "description": description,
        "instruction": instruction,
        "tools": tools or [],
    }
    agent_kwargs.update(kwargs)
    filtered = {k: v for k, v in agent_kwargs.items() if v is not None}
    return Agent(**filtered)
