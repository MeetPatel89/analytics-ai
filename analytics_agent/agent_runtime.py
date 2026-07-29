"""Runtime configuration and provider construction for agent runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from analytics_agent.messages import generate_initial_messages
from analytics_agent.providers.base import (
    Provider,
    ToolLoopProvider,
    ToolLoopResponse,
)
from analytics_agent.providers.generation import GenerationModel
from analytics_agent.providers.openai_provider import (
    OpenAIGenerationModel,
    OpenAIProvider,
)
from analytics_agent.providers.openai_provider import (
    list_available_models as list_openai_models,
)
from analytics_agent.tools.provider_factories import OpenAIToolSchema
from analytics_agent.tools.registry import ToolRegistry
from analytics_agent.tools.tool_chains import (
    ToolChain,
    ToolChainDependencies,
    build_tools_for_chains,
)


@dataclass(frozen=True)
class AgentRunConfig:
    """Validated configuration selected before an agent run starts."""

    provider: Provider
    model: str
    tool_chains: tuple[ToolChain, ...]
    system_prompt: str
    user_prompt: str
    verbose: bool = False
    data_path: Path | None = None

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
    [AgentRunConfig, str, list[OpenAIToolSchema]],
    ToolLoopProvider[ToolLoopResponse],
]


@dataclass(frozen=True)
class ProviderDefinition:
    """Describe one provider available to the interactive runtime."""

    name: Provider
    label: str
    credential_env_var: str
    list_models: ModelLister
    create_provider: ProviderFactory
    create_generation_model: Callable[[str, str], GenerationModel]


def create_openai_provider(
    config: AgentRunConfig,
    api_key: str,
    tools: list[OpenAIToolSchema],
) -> OpenAIProvider:
    """Create an OpenAI provider from an interactive run configuration."""
    return OpenAIProvider(
        api_key=api_key,
        model=config.model,
        tools=tools,
        messages=generate_initial_messages(config.system_prompt, config.user_prompt),
    )


def build_run_tools(
    definition: ProviderDefinition,
    config: AgentRunConfig,
    api_key: str,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Compose run tools with provider capabilities bound lazily."""
    dependencies = ToolChainDependencies(
        create_generation_model=lambda: definition.create_generation_model(
            api_key,
            config.model,
        ),
        data_path=config.data_path,
    )
    return build_tools_for_chains(config.tool_chains, dependencies)


def available_providers() -> tuple[ProviderDefinition, ...]:
    """Return providers available for interactive agent runs."""
    return (
        ProviderDefinition(
            name="openai",
            label="OpenAI",
            credential_env_var="OPENAI_API_KEY",
            list_models=list_openai_models,
            create_provider=create_openai_provider,
            create_generation_model=OpenAIGenerationModel,
        ),
    )
