"""Runtime configuration and provider construction for agent runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from analytics_agent.messages import generate_initial_messages
from analytics_agent.providers.base import (
    Provider,
    ToolLoopProvider,
    ToolLoopResponse,
)
from analytics_agent.providers.openai_provider import OpenAIProvider
from analytics_agent.providers.openai_provider import (
    list_available_models as list_openai_models,
)
from analytics_agent.tools import ToolChain

ProviderName = Provider


@dataclass(frozen=True)
class AgentRunConfig:
    """Validated configuration selected before an agent run starts."""

    provider: ProviderName
    model: str
    tool_chains: tuple[ToolChain, ...]
    system_prompt: str
    user_prompt: str
    verbose: bool = False

    def __post_init__(self) -> None:
        """Reject incomplete run configurations before provider construction."""
        if not self.model.strip():
            raise ValueError("A model must be selected.")
        if not self.tool_chains:
            raise ValueError("At least one tool chain must be selected.")
        if not self.system_prompt.strip():
            raise ValueError("A system prompt is required.")
        if not self.user_prompt.strip():
            raise ValueError("A user task is required.")


ModelLister = Callable[[str], list[str]]
ProviderFactory = Callable[
    [AgentRunConfig, str, list[dict[str, object]]],
    ToolLoopProvider[ToolLoopResponse],
]


@dataclass(frozen=True)
class ProviderDefinition:
    """Describe one provider available to the interactive runtime."""

    name: ProviderName
    label: str
    credential_env_var: str
    list_models: ModelLister
    create_provider: ProviderFactory


def create_openai_provider(
    config: AgentRunConfig,
    api_key: str,
    tools: list[dict[str, object]],
) -> OpenAIProvider:
    """Create an OpenAI provider from an interactive run configuration."""
    return OpenAIProvider(
        api_key=api_key,
        model=config.model,
        tools=tools,
        messages=generate_initial_messages(config.system_prompt, config.user_prompt),
    )


def available_providers() -> tuple[ProviderDefinition, ...]:
    """Return providers available for interactive agent runs."""
    return (
        ProviderDefinition(
            name="openai",
            label="OpenAI",
            credential_env_var="OPENAI_API_KEY",
            list_models=list_openai_models,
            create_provider=create_openai_provider,
        ),
    )
